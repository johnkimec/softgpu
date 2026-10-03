// The five baseline microbenchmarks. Each one is designed to expose one
// dimension of the driver stack:
//
//   submit  - fixed cost of one API call (async) and of one round trip (sync)
//   batch   - amortization: B commands then one sync, for increasing B
//   memcpy  - data-movement path (staging, chunking, copies per byte)
//   vadd    - a realistic tiny workload: copy in, compute, copy out
//   gemm    - compute-bound work, to see when submission cost stops mattering
//   mt      - scaling with concurrent submitters (the big lock)
//   alloc   - sgMalloc/sgFree cost while work is in flight (deferred frees)
//   fault   - first touch of managed memory (fault + migrate, prefetch)
//   tlb     - translation cost and TLB reach on resident memory
//   thrash  - managed working sets larger than the free VRAM frames

#include <atomic>
#include <time.h>
#include <cstring>
#include <thread>
#include <vector>

#include "bench.h"

namespace bench {

namespace {

sgDevPtr must_malloc(size_t bytes) {
    sgDevPtr p = 0;
    die_on(sgMalloc(&p, bytes), "sgMalloc");
    return p;
}

void latency_row(Row& r, std::vector<double>& samples) {
    r.metrics["p50_ns"] = percentile(samples, 50);
    r.metrics["p99_ns"] = percentile(samples, 99);
    r.metrics["max_ns"] = percentile(samples, 100);
}

} // namespace

void bench_submit(const Options& o, Report& rep) {
    const int iters = o.quick ? 5000 : 50000;
    sgDevPtr d = must_malloc(4096);

    struct Case { const char* name; sgError_t (*fn)(sgDevPtr); } cases[] = {
        {"nop", [](sgDevPtr) { return sgDeviceSynchronize(); }}, // API-only floor
        {"fill_64B", [](sgDevPtr p) { return sgMemset(p, 0, 64); }},
        {"fill_4KiB", [](sgDevPtr p) { return sgMemset(p, 0, 4096); }},
        {"fill_64B_sync", [](sgDevPtr p) {
             if (auto e = sgMemset(p, 0, 64)) return e;
             return sgDeviceSynchronize();
         }},
    };
    for (auto& c : cases) {
        std::vector<double> lat;
        lat.reserve(iters);
        for (int i = 0; i < 200; ++i) c.fn(d); // warm up
        Window w;
        w.begin();
        for (int i = 0; i < iters; ++i) {
            auto t0 = Clock::now();
            die_on(c.fn(d), c.name);
            lat.push_back(ns_since(t0));
        }
        die_on(sgDeviceSynchronize(), "sync");
        w.end();
        Row r{"submit", c.name, {}};
        latency_row(r, lat);
        r.metrics["ops_per_s"] = iters / (w.wall_ns() / 1e9);
        w.fill(r, iters);
        rep.rows.push_back(r);
    }
    sgFree(d);
}

void bench_batch(const Options& o, Report& rep) {
    const int total = o.quick ? 16384 : 262144;
    sgDevPtr d = must_malloc(4096);
    for (int B : {1, 4, 16, 64, 256, 1024}) {
        const int rounds = total / B;
        for (int i = 0; i < B; ++i) sgMemset(d, 0, 64);
        sgDeviceSynchronize();
        Window w;
        w.begin();
        for (int r = 0; r < rounds; ++r) {
            for (int i = 0; i < B; ++i) die_on(sgMemset(d, i, 64), "memset");
            die_on(sgDeviceSynchronize(), "sync");
        }
        w.end();
        char name[32];
        std::snprintf(name, sizeof name, "B=%d", B);
        Row r{"batch", name, {}};
        r.metrics["ns_per_op"] = w.wall_ns() / (double(rounds) * B);
        r.metrics["us_per_batch"] = w.wall_ns() / rounds / 1e3;
        w.fill(r, double(rounds) * B);
        rep.rows.push_back(r);
    }
    sgFree(d);
}

void bench_memcpy(const Options& o, Report& rep) {
    const size_t sizes[] = {4u << 10, 64u << 10, 1u << 20, 4u << 20, 16u << 20, 64u << 20};
    for (size_t bytes : sizes) {
        // Keep total traffic per case roughly constant so runs take similar time.
        const size_t budget = (o.quick ? 256u : 1024u) << 20;
        const int iters = std::max<int>(3, int(budget / bytes));
        std::vector<uint8_t> pageable(bytes, 0x5a);
        uint8_t* pinned = nullptr;
        die_on(sgMallocHost(reinterpret_cast<void**>(&pinned), bytes), "sgMallocHost");
        std::memset(pinned, 0x5a, bytes);
        sgDevPtr d = must_malloc(bytes);

        // Same synchronous API on pageable memory (staging pool) and pinned
        // memory (direct DMA, waited on); the difference is the data path.
        for (int mem = 0; mem < 2; ++mem) {
            uint8_t* host = mem == 0 ? pageable.data() : pinned;
            for (int dir = 0; dir < 2; ++dir) {
                auto op = [&] {
                    return dir == 0 ? sgMemcpyH2D(d, host, bytes) : sgMemcpyD2H(host, d, bytes);
                };
                op();
                Window w;
                w.begin();
                for (int i = 0; i < iters; ++i) die_on(op(), "memcpy");
                die_on(sgDeviceSynchronize(), "sync");
                w.end();
                char name[64];
                std::snprintf(name, sizeof name, "%s%s_%zuKiB", dir == 0 ? "h2d" : "d2h",
                              mem == 0 ? "" : "_pinned", bytes >> 10);
                Row r{"memcpy", name, {}};
                r.metrics["GBps"] = double(bytes) * iters / w.wall_ns();
                r.metrics["ns_per_op"] = w.wall_ns() / iters;
                w.fill(r, iters);
                rep.rows.push_back(r);
            }
        }
        sgFree(d);
        sgFreeHost(pinned);
    }
}

void bench_vadd(const Options& o, Report& rep) {
    const uint32_t ns[] = {1u << 10, 1u << 16, 1u << 20, 1u << 24};
    for (int mem = 0; mem < 2; ++mem) {
        const bool pinned = mem == 1;
        for (uint32_t n : ns) {
            const size_t bytes = size_t(n) * 4;
            const int iters = o.quick ? 3 : std::max<int>(5, int((256u << 20) / bytes));
            std::vector<float> ha, hb, hc;
            float *a, *b, *c;
            if (pinned) {
                die_on(sgMallocHost(reinterpret_cast<void**>(&a), bytes), "sgMallocHost");
                die_on(sgMallocHost(reinterpret_cast<void**>(&b), bytes), "sgMallocHost");
                die_on(sgMallocHost(reinterpret_cast<void**>(&c), bytes), "sgMallocHost");
                std::fill(a, a + n, 1.0f);
                std::fill(b, b + n, 2.0f);
            } else {
                ha.assign(n, 1.0f); hb.assign(n, 2.0f); hc.assign(n, 0.0f);
                a = ha.data(); b = hb.data(); c = hc.data();
            }
            sgDevPtr da = must_malloc(bytes), db = must_malloc(bytes), dc = must_malloc(bytes);

            // Pageable: the synchronous API, as an application would write it.
            // Pinned: async copies and one sync per iteration — the shape the
            // pinned contract exists to enable.
            double t_in = 0, t_k = 0, t_out = 0;
            Window w;
            w.begin();
            for (int i = 0; i < iters; ++i) {
                auto t0 = Clock::now();
                if (pinned) {
                    die_on(sgMemcpyH2DAsync(da, a, bytes, nullptr), "h2d");
                    die_on(sgMemcpyH2DAsync(db, b, bytes, nullptr), "h2d");
                } else {
                    die_on(sgMemcpyH2D(da, a, bytes), "h2d");
                    die_on(sgMemcpyH2D(db, b, bytes), "h2d");
                }
                auto t1 = Clock::now();
                die_on(sgVaddF32(dc, da, db, n), "vadd");
                auto t2 = Clock::now();
                if (pinned) {
                    die_on(sgMemcpyD2HAsync(c, dc, bytes, nullptr), "d2h");
                    die_on(sgDeviceSynchronize(), "sync");
                } else {
                    die_on(sgMemcpyD2H(c, dc, bytes), "d2h");
                }
                auto t3 = Clock::now();
                t_in += std::chrono::duration<double, std::nano>(t1 - t0).count();
                t_k += std::chrono::duration<double, std::nano>(t2 - t1).count();
                t_out += std::chrono::duration<double, std::nano>(t3 - t2).count();
            }
            die_on(sgDeviceSynchronize(), "sync");
            w.end();
            char name[32];
            std::snprintf(name, sizeof name, "%sn=%u", pinned ? "pinned_" : "", n);
            Row r{"vadd", name, {}};
            r.metrics["us_per_iter"] = w.wall_ns() / iters / 1e3;
            r.metrics["frac_h2d"] = t_in / (t_in + t_k + t_out);
            r.metrics["frac_kernel"] = t_k / (t_in + t_k + t_out);
            r.metrics["frac_d2h"] = t_out / (t_in + t_k + t_out);
            w.fill(r, iters);
            rep.rows.push_back(r);
            sgFree(da); sgFree(db); sgFree(dc);
            if (pinned) { sgFreeHost(a); sgFreeHost(b); sgFreeHost(c); }
        }
    }
}

void bench_alloc(const Options& o, Report& rep) {
    // Cost of one sgMalloc+sgFree pair (a) with nothing in flight and (b)
    // while a ~14 ms GEMM is executing. A driver that drains on free pays the
    // whole GEMM in (b); a deferring one pays a list append.
    const int rounds = o.quick ? 5 : 30;
    const uint32_t d = 512;
    const size_t gbytes = size_t(d) * d * 4;
    sgDevPtr ga = must_malloc(gbytes), gb = must_malloc(gbytes), gc = must_malloc(gbytes);
    die_on(sgMemset(ga, 0, gbytes), "memset");
    die_on(sgMemset(gb, 0, gbytes), "memset");
    for (int inflight = 0; inflight < 2; ++inflight) {
        std::vector<double> lat;
        Window w;
        w.begin();
        for (int r = 0; r < rounds; ++r) {
            if (inflight) die_on(sgGemmF32(gc, ga, gb, d, d, d), "gemm");
            auto t0 = Clock::now();
            sgDevPtr p = must_malloc(4096);
            die_on(sgFree(p), "free");
            lat.push_back(ns_since(t0));
            die_on(sgDeviceSynchronize(), "sync");
        }
        w.end();
        Row r{"alloc", inflight ? "malloc_free_inflight" : "malloc_free_idle", {}};
        latency_row(r, lat);
        w.fill(r, rounds);
        rep.rows.push_back(r);
    }
    sgFree(ga); sgFree(gb); sgFree(gc);
}

void bench_gemm(const Options& o, Report& rep) {
    const uint32_t dims[] = {32, 128, 512};
    for (uint32_t d : dims) {
        const size_t bytes = size_t(d) * d * 4;
        const int iters = o.quick ? 3 : (d >= 512 ? 5 : 200);
        sgDevPtr da = must_malloc(bytes), db = must_malloc(bytes), dc = must_malloc(bytes);
        die_on(sgMemset(da, 0, bytes), "memset");
        die_on(sgMemset(db, 0, bytes), "memset");
        sgGemmF32(dc, da, db, d, d, d);
        Window w;
        w.begin();
        for (int i = 0; i < iters; ++i) die_on(sgGemmF32(dc, da, db, d, d, d), "gemm");
        die_on(sgDeviceSynchronize(), "sync");
        w.end();
        char name[32];
        std::snprintf(name, sizeof name, "%ux%ux%u", d, d, d);
        Row r{"gemm", name, {}};
        r.metrics["GFLOPS"] = 2.0 * d * d * d * iters / w.wall_ns();
        r.metrics["us_per_op"] = w.wall_ns() / iters / 1e3;
        w.fill(r, iters);
        rep.rows.push_back(r);
        sgFree(da); sgFree(db); sgFree(dc);
    }
}

void bench_mt(const Options& o, Report& rep) {
    const int per_thread = o.quick ? 2000 : 20000;
    // Two variants: every thread on the default stream (they serialize on
    // the stream's own lock as well as the driver's), and one stream per
    // thread (only the driver lock is shared).
    for (int own_stream = 0; own_stream < 2; ++own_stream)
    for (int T = 1; T <= o.threads_max; T *= 2) {
        std::vector<sgDevPtr> bufs(T);
        for (auto& b : bufs) b = must_malloc(4096);
        std::vector<sgStream_t> streams(T, nullptr);
        if (own_stream)
            for (auto& st : streams) die_on(sgStreamCreate(&st), "sgStreamCreate");

        std::atomic<int> go{0};
        std::vector<std::thread> ts;
        std::vector<std::vector<double>> lats(T);
        std::vector<double> cpu(T, 0.0);
        Window w;
        w.begin();
        for (int t = 0; t < T; ++t)
            ts.emplace_back([&, t] {
                lats[t].reserve(per_thread);
                while (!go.load(std::memory_order_acquire)) {}
                const double c0 = thread_cpu_ns();
                for (int i = 0; i < per_thread; ++i) {
                    auto t0 = Clock::now();
                    die_on(sgMemsetAsync(bufs[t], i, 4096, streams[t]), "memset");
                    lats[t].push_back(ns_since(t0));
                }
                die_on(sgStreamSynchronize(streams[t]), "sync");
                cpu[t] = thread_cpu_ns() - c0;
            });
        go.store(1, std::memory_order_release);
        for (auto& th : ts) th.join();
        w.end();
        w.cpu_override = 0;
        for (double c : cpu) w.cpu_override += c;

        std::vector<double> all;
        for (auto& l : lats) all.insert(all.end(), l.begin(), l.end());
        char name[32];
        std::snprintf(name, sizeof name, "threads=%d%s", T, own_stream ? "_streams" : "");
        Row r{"mt", name, {}};
        r.metrics["ops_per_s"] = double(T) * per_thread / (w.wall_ns() / 1e9);
        latency_row(r, all);
        w.fill(r, double(T) * per_thread);
        rep.rows.push_back(r);
        for (auto b : bufs) sgFree(b);
        if (own_stream)
            for (auto st : streams) sgStreamDestroy(st);
    }
}

} // namespace bench

namespace bench {

// Streams: a chunked H2D -> compute -> D2H workload spread over S streams.
// With one stream nothing can overlap (in-order), even though the device has
// a copy engine and a compute engine; with two or more, copies of chunk i+1
// run while chunk i computes. R repetitions of the kernel per chunk set the
// copy:compute ratio (R=1 copy-heavy, R=4 compute-heavy).
void bench_pipeline(const Options& o, Report& rep) {
    const int chunks = o.quick ? 4 : 16;
    const uint32_t n = 1u << 20; // floats per chunk (4 MiB per array)
    const size_t bytes = size_t(n) * 4;

    std::vector<float*> ha(chunks), hb(chunks), hc(chunks);
    std::vector<sgDevPtr> da(chunks), db(chunks), dc(chunks);
    for (int i = 0; i < chunks; ++i) {
        die_on(sgMallocHost(reinterpret_cast<void**>(&ha[i]), bytes), "sgMallocHost");
        die_on(sgMallocHost(reinterpret_cast<void**>(&hb[i]), bytes), "sgMallocHost");
        die_on(sgMallocHost(reinterpret_cast<void**>(&hc[i]), bytes), "sgMallocHost");
        std::fill(ha[i], ha[i] + n, 1.0f);
        std::fill(hb[i], hb[i] + n, 2.0f);
        da[i] = must_malloc(bytes);
        db[i] = must_malloc(bytes);
        dc[i] = must_malloc(bytes);
    }

    for (int R : {1, 4}) {
        double t_s1 = 0;
        for (int S : {1, 2, 4}) {
            std::vector<sgStream_t> streams(S);
            for (auto& s : streams) die_on(sgStreamCreate(&s), "sgStreamCreate");
            auto run = [&] {
                for (int i = 0; i < chunks; ++i) {
                    sgStream_t s = streams[i % S];
                    die_on(sgMemcpyH2DAsync(da[i], ha[i], bytes, s), "h2d");
                    die_on(sgMemcpyH2DAsync(db[i], hb[i], bytes, s), "h2d");
                    die_on(sgVaddF32Async(dc[i], da[i], db[i], n, s), "vadd");
                    for (int r = 1; r < R; ++r) die_on(sgVaddF32Async(dc[i], dc[i], db[i], n, s), "vadd");
                    die_on(sgMemcpyD2HAsync(hc[i], dc[i], bytes, s), "d2h");
                }
                die_on(sgDeviceSynchronize(), "sync");
            };
            run(); // warm up
            const int iters = o.quick ? 1 : 3;
            Window w;
            w.begin();
            for (int it = 0; it < iters; ++it) run();
            w.end();
            for (auto& s : streams) sgStreamDestroy(s);

            char name[32];
            std::snprintf(name, sizeof name, "R=%d_S=%d", R, S);
            Row r{"pipeline", name, {}};
            const double us_per_chunk = w.wall_ns() / (double(iters) * chunks) / 1e3;
            if (S == 1) t_s1 = us_per_chunk;
            r.metrics["us_per_chunk"] = us_per_chunk;
            r.metrics["speedup_vs_S1"] = t_s1 / us_per_chunk;
            w.fill(r, double(iters) * chunks);
            rep.rows.push_back(r);
        }
    }

    for (int i = 0; i < chunks; ++i) {
        sgFree(da[i]); sgFree(db[i]); sgFree(dc[i]);
        sgFreeHost(ha[i]); sgFreeHost(hb[i]); sgFreeHost(hc[i]);
    }
}

} // namespace bench

namespace bench {

// Wait policy: one command of known duration, then synchronize, under each
// host-side policy the runtime can ask for (spin / block / driver default).
// The tradeoff is wall time (wake-up latency) against submitter CPU (power).
void bench_wait(const Options& o, Report& rep) {
    struct Work { const char* name; uint32_t dim; int iters; };
    const Work works[] = {
        {"fill4K", 0, o.quick ? 2000 : 20000},       // ~0.3 us
        {"gemm32", 32, o.quick ? 200 : 2000},        // ~10 us
        {"gemm128", 128, o.quick ? 50 : 500},        // ~200 us
        {"gemm512", 512, o.quick ? 3 : 20},          // ~14 ms
    };
    struct Pol { const char* name; sgSyncPolicy_t p; } pols[] = {
        {"spin", SG_SYNC_SPIN}, {"block", SG_SYNC_BLOCK}, {"default", SG_SYNC_DEFAULT}};

    for (const auto& wk : works) {
        const size_t bytes = wk.dim ? size_t(wk.dim) * wk.dim * 4 : 4096;
        sgDevPtr a = must_malloc(bytes), b = must_malloc(bytes), c = must_malloc(bytes);
        die_on(sgMemset(a, 0, bytes), "memset");
        die_on(sgMemset(b, 0, bytes), "memset");
        for (const auto& pol : pols) {
            die_on(sgSetSyncPolicy(pol.p), "sgSetSyncPolicy");
            auto op = [&] {
                return wk.dim ? sgGemmF32(c, a, b, wk.dim, wk.dim, wk.dim) : sgMemset(c, 1, bytes);
            };
            op(); sgDeviceSynchronize();
            std::vector<double> lat;
            lat.reserve(wk.iters);
            Window w;
            w.begin();
            for (int i = 0; i < wk.iters; ++i) {
                auto t0 = Clock::now();
                die_on(op(), wk.name);
                die_on(sgDeviceSynchronize(), "sync");
                lat.push_back(ns_since(t0));
            }
            w.end();
            char name[48];
            std::snprintf(name, sizeof name, "%s_%s", wk.name, pol.name);
            Row r{"wait", name, {}};
            latency_row(r, lat);
            r.metrics["us_per_op"] = w.wall_ns() / wk.iters / 1e3;
            r.metrics["cpu_us_per_op"] = w.cpu_ns() / wk.iters / 1e3;
            w.fill(r, wk.iters);
            rep.rows.push_back(r);
        }
        die_on(sgSetSyncPolicy(SG_SYNC_DEFAULT), "sgSetSyncPolicy");
        sgFree(a); sgFree(b); sgFree(c);
    }
}

} // namespace bench

namespace bench {

// Idle gating: leave the device idle for a gap, then submit one tiny command
// and spin-wait for it. Under a spinning engine every gap costs the same
// round trip; under gating, gaps longer than the idle budget pay the
// engine's wake-up. dev_cpu_cores over the run is the idle power.
void bench_wake(const Options& o, Report& rep) {
    die_on(sgSetSyncPolicy(SG_SYNC_SPIN), "sgSetSyncPolicy");
    sgDevPtr d = must_malloc(4096);
    struct Gap { const char* name; long ns; int iters; } gaps[] = {
        {"1us", 1000, o.quick ? 500 : 5000},
        {"30us", 30000, o.quick ? 300 : 3000},
        {"300us", 300000, o.quick ? 100 : 1000},
        {"3ms", 3000000, o.quick ? 20 : 200},
        {"30ms", 30000000, o.quick ? 5 : 30},
    };
    for (const auto& g : gaps) {
        std::vector<double> lat;
        lat.reserve(g.iters);
        Window w;
        w.begin();
        for (int i = 0; i < g.iters; ++i) {
            // nanosleep() has ~50 us of timer slack on Linux, so short gaps
            // are produced by busy-waiting on the clock instead.
            if (g.ns < 100000) {
                auto until = Clock::now() + std::chrono::nanoseconds(g.ns);
                while (Clock::now() < until) {}
            } else {
                timespec ts{g.ns / 1000000000L, g.ns % 1000000000L};
                nanosleep(&ts, nullptr);
            }
            auto t0 = Clock::now();
            die_on(sgMemset(d, i, 64), "memset");
            die_on(sgDeviceSynchronize(), "sync");
            lat.push_back(ns_since(t0));
        }
        w.end();
        char name[32];
        std::snprintf(name, sizeof name, "gap=%s", g.name);
        Row r{"wake", name, {}};
        latency_row(r, lat);
        w.fill(r, g.iters);
        rep.rows.push_back(r);
    }
    sgFree(d);
    die_on(sgSetSyncPolicy(SG_SYNC_DEFAULT), "sgSetSyncPolicy");
}

} // namespace bench

namespace bench {

namespace {

constexpr size_t kPage = 4096;

// Fill VRAM with sgMalloc so exactly `frames` frames stay free for managed
// pages. Returns the blocks to free afterwards.
std::vector<sgDevPtr> squeeze_vram(size_t frames) {
    std::vector<std::pair<sgDevPtr, size_t>> held;
    for (size_t chunk : {size_t{64} << 20, size_t{1} << 20, kPage})
        for (sgDevPtr p = 0; sgMalloc(&p, chunk) == SG_OK;) held.push_back({p, chunk});
    size_t freed = 0;
    while (freed < frames * kPage && !held.empty()) {
        die_on(sgFree(held.back().first), "sgFree");
        freed += held.back().second;
        held.pop_back();
    }
    for (size_t extra = freed / kPage - frames; extra--;) held.push_back({must_malloc(kPage), kPage});
    std::vector<sgDevPtr> out;
    for (auto& h : held) out.push_back(h.first);
    return out;
}

} // namespace

// First touch of managed memory. Every page of a fresh allocation starts in
// host memory, so one device pass over it is fault-bound: seq_* walk forward
// in one command (prefetch applies); strided_* touches one page per command
// in steps of 7 pages, so no two faults look sequential and every page pays
// a full fault round trip. Run once per SG_PREFETCH_PAGES setting.
void bench_fault(const Options& o, Report& rep) {
    const size_t sizes[] = {1u << 20, (o.quick ? 4u : 16u) << 20};
    const int iters = o.quick ? 3 : 10;
    enum Kind { kSeqD2H, kSeqFill, kStrided };
    for (size_t bytes : sizes) {
        const size_t pages = bytes / kPage;
        uint8_t* pinned = nullptr;
        die_on(sgMallocHost(reinterpret_cast<void**>(&pinned), bytes), "sgMallocHost");
        for (Kind kind : {kSeqD2H, kSeqFill, kStrided}) {
            double touch_ns = 0;
            Window w;
            w.begin();
            for (int i = 0; i < iters; ++i) {
                sgDevPtr d = 0;
                void* host = nullptr;
                die_on(sgMallocManaged(&d, &host, bytes), "sgMallocManaged");
                auto t0 = Clock::now();
                if (kind == kSeqD2H) {
                    die_on(sgMemcpyD2H(pinned, d, bytes), "d2h");
                } else if (kind == kSeqFill) {
                    die_on(sgMemset(d, i, bytes), "memset");
                } else {
                    for (size_t k = 0; k < pages; ++k) {
                        const size_t p = (k * 7) % pages;
                        die_on(sgMemcpyD2HAsync(pinned + p * kPage, d + p * kPage, kPage, nullptr), "d2h");
                    }
                }
                die_on(sgDeviceSynchronize(), "sync");
                touch_ns += ns_since(t0);
                die_on(sgFree(d), "sgFree");
            }
            w.end();
            const char* kname = kind == kSeqD2H ? "seq_d2h" : kind == kSeqFill ? "seq_fill" : "strided_d2h";
            char name[48];
            std::snprintf(name, sizeof name, "%s_%zuKiB", kname, bytes >> 10);
            Row r{"fault", name, {}};
            r.metrics["MBps"] = double(bytes) * iters / touch_ns * 1e3;
            r.metrics["us_per_page"] = touch_ns / (double(pages) * iters) / 1e3;
            if (w.faults() > 0) r.metrics["us_per_fault"] = touch_ns / w.faults() / 1e3;
            w.fill(r, double(pages) * iters);
            rep.rows.push_back(r);
        }
        sgFreeHost(pinned);
    }
}

// Translation on resident memory: sgMalloc only, no faults, just the
// compute engine's 64-entry TLB. stream_* FILLs the whole working set per
// pass; sparse_* FILLs 64 bytes at the start of each page, one command per
// page, so translation is a bigger share of device time. busy_ns_per_page
// is device execute time; the cliff should sit at 64 pages (256 KiB).
void bench_tlb(const Options& o, Report& rep) {
    std::vector<size_t> kib = o.quick ? std::vector<size_t>{64, 256, 1024}
                                      : std::vector<size_t>{64, 128, 256, 512, 1024, 4096, 16384};
    const size_t stream_budget = (o.quick ? 32u : 256u) << 20; // bytes filled per stream case
    const size_t sparse_budget = o.quick ? 1u << 15 : 1u << 18; // commands per sparse case
    for (int sparse = 0; sparse < 2; ++sparse)
        for (size_t k : kib) {
            const size_t bytes = k << 10, pages = bytes / kPage;
            const int passes = int(std::max<size_t>(sparse ? 2 : 4, (sparse ? sparse_budget / pages : stream_budget / bytes)));
            sgDevPtr d = must_malloc(bytes);
            auto pass = [&](int v) {
                if (!sparse) {
                    die_on(sgMemset(d, v, bytes), "memset");
                } else {
                    for (size_t p = 0; p < pages; ++p) die_on(sgMemset(d + p * kPage, v, 64), "memset");
                }
            };
            pass(0); // warm the TLB
            Window w;
            w.begin();
            for (int i = 0; i < passes; ++i) pass(i);
            die_on(sgDeviceSynchronize(), "sync");
            w.end();
            char name[48];
            std::snprintf(name, sizeof name, "%s_%zuKiB", sparse ? "sparse" : "stream", k);
            Row r{"tlb", name, {}};
            const double touched = double(pages) * passes;
            r.metrics["ns_per_page"] = w.wall_ns() / touched;
            r.metrics["busy_ns_per_page"] = w.busy(0) / touched;
            w.fill(r, touched);
            rep.rows.push_back(r);
            sgFree(d);
        }
}

// Working sets around the managed frame pool. VRAM is filled with sgMalloc
// except `pool` frames; a managed buffer of ratio x pool is walked in one
// command per pass, after a warm-up pass. At 0.5x it stays resident; past
// 1x, a cyclic walk under CLOCK evicts each page before it comes round
// again (LRU's worst case), so every pass migrates the whole set. read_*
// copies out (clean evictions), write_* fills (every eviction writes back).
void bench_thrash(const Options& o, Report& rep) {
    const size_t pool = o.quick ? 256 : 1024; // frames: 1 MiB / 4 MiB
    const int passes = o.quick ? 2 : 5;
    auto held = squeeze_vram(pool);
    for (int write = 0; write < 2; ++write)
        for (double ratio : {0.5, 1.5, 2.0, 4.0}) {
            const size_t pages = size_t(ratio * double(pool)), bytes = pages * kPage;
            sgDevPtr d = 0;
            void* host = nullptr;
            die_on(sgMallocManaged(&d, &host, bytes), "sgMallocManaged");
            uint8_t* pinned = nullptr;
            die_on(sgMallocHost(reinterpret_cast<void**>(&pinned), bytes), "sgMallocHost");
            auto pass = [&](int v) {
                if (write) die_on(sgMemset(d, v, bytes), "memset");
                else die_on(sgMemcpyD2H(pinned, d, bytes), "d2h");
                die_on(sgDeviceSynchronize(), "sync");
            };
            pass(0);
            Window w;
            w.begin();
            for (int i = 0; i < passes; ++i) pass(i + 1);
            w.end();
            char name[48];
            std::snprintf(name, sizeof name, "%s_%.1fx", write ? "write" : "read", ratio);
            Row r{"thrash", name, {}};
            const double touched = double(pages) * passes;
            r.metrics["GBps"] = double(bytes) * passes / w.wall_ns();
            r.metrics["us_per_page"] = w.wall_ns() / touched / 1e3;
            w.fill(r, touched);
            rep.rows.push_back(r);
            sgFreeHost(pinned);
            die_on(sgFree(d), "sgFree");
        }
    for (sgDevPtr p : held) sgFree(p);
}

} // namespace bench
