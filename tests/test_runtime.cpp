// Correctness tests for the runtime -> driver -> device path. No framework:
// a CHECK macro and a process exit code are all ctest needs.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <time.h>
#include <vector>

#include "softgpu/sg_runtime.h"

static int g_failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            ++g_failures;                                                                \
        }                                                                                \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == SG_OK)

static void test_memcpy_roundtrip(size_t bytes) {
    std::vector<uint8_t> src(bytes), dst(bytes, 0);
    for (size_t i = 0; i < bytes; ++i) src[i] = static_cast<uint8_t>(i * 31 + 7);
    sgDevPtr d = 0;
    CHECK_OK(sgMalloc(&d, bytes));
    CHECK_OK(sgMemcpyH2D(d, src.data(), bytes));
    CHECK_OK(sgMemcpyD2H(dst.data(), d, bytes));
    CHECK(src == dst);
    CHECK_OK(sgFree(d));
}

static void test_memset_and_d2d() {
    const size_t n = 1 << 16;
    sgDevPtr a = 0, b = 0;
    CHECK_OK(sgMalloc(&a, n));
    CHECK_OK(sgMalloc(&b, n));
    CHECK_OK(sgMemset(a, 0xAB, n));
    CHECK_OK(sgMemcpyD2D(b, a, n));
    std::vector<uint8_t> out(n);
    CHECK_OK(sgMemcpyD2H(out.data(), b, n));
    CHECK(std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == 0xAB; }));
    CHECK_OK(sgFree(a));
    CHECK_OK(sgFree(b));
}

static void test_vadd() {
    const uint32_t n = 100003; // not a multiple of anything convenient
    std::vector<float> a(n), b(n), c(n);
    for (uint32_t i = 0; i < n; ++i) { a[i] = float(i) * 0.5f; b[i] = 1000.0f - float(i); }
    sgDevPtr da = 0, db = 0, dc = 0;
    CHECK_OK(sgMalloc(&da, n * 4));
    CHECK_OK(sgMalloc(&db, n * 4));
    CHECK_OK(sgMalloc(&dc, n * 4));
    CHECK_OK(sgMemcpyH2D(da, a.data(), n * 4));
    CHECK_OK(sgMemcpyH2D(db, b.data(), n * 4));
    CHECK_OK(sgVaddF32(dc, da, db, n));
    CHECK_OK(sgMemcpyD2H(c.data(), dc, n * 4));
    bool ok = true;
    for (uint32_t i = 0; i < n && ok; ++i) ok = (c[i] == a[i] + b[i]);
    CHECK(ok);
    CHECK_OK(sgFree(da)); CHECK_OK(sgFree(db)); CHECK_OK(sgFree(dc));
}

static void test_gemm() {
    const uint32_t m = 37, n = 53, k = 29;
    std::vector<float> a(m * k), b(k * n), c(m * n), ref(m * n, 0.0f);
    for (size_t i = 0; i < a.size(); ++i) a[i] = float((i * 7) % 13) - 6.0f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = float((i * 5) % 11) - 5.0f;
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            float s = 0;
            for (uint32_t p = 0; p < k; ++p) s += a[i * k + p] * b[p * n + j];
            ref[i * n + j] = s;
        }
    sgDevPtr da = 0, db = 0, dc = 0;
    CHECK_OK(sgMalloc(&da, a.size() * 4));
    CHECK_OK(sgMalloc(&db, b.size() * 4));
    CHECK_OK(sgMalloc(&dc, c.size() * 4));
    CHECK_OK(sgMemcpyH2D(da, a.data(), a.size() * 4));
    CHECK_OK(sgMemcpyH2D(db, b.data(), b.size() * 4));
    CHECK_OK(sgGemmF32(dc, da, db, m, n, k));
    CHECK_OK(sgMemcpyD2H(c.data(), dc, c.size() * 4));
    bool ok = true;
    for (size_t i = 0; i < c.size() && ok; ++i) ok = std::fabs(c[i] - ref[i]) < 1e-3f;
    CHECK(ok);
    CHECK_OK(sgFree(da)); CHECK_OK(sgFree(db)); CHECK_OK(sgFree(dc));
}

static void test_validation() {
    sgDevPtr d = 0;
    CHECK_OK(sgMalloc(&d, 4096));
    uint8_t buf[16] = {};
    // Past the end of the allocation.
    CHECK(sgMemset(d, 0, 4097) == SG_ERR_INVALID_ADDRESS);
    CHECK(sgMemcpyH2D(d + 4090, buf, 16) == SG_ERR_INVALID_ADDRESS);
    // Never allocated.
    CHECK(sgMemcpyD2H(buf, 100 << 20, 16) == SG_ERR_INVALID_ADDRESS);
    // Double free / bogus free.
    CHECK_OK(sgFree(d));
    CHECK(sgFree(d) == SG_ERR_INVALID_VALUE);
    CHECK(sgFree(12345) == SG_ERR_INVALID_VALUE);
    // Zero-size and null.
    CHECK(sgMalloc(&d, 0) == SG_ERR_INVALID_VALUE);
    CHECK(sgMalloc(nullptr, 16) == SG_ERR_INVALID_VALUE);
}

static void test_alloc_reuse_and_oom() {
    // Fill VRAM with 64 MiB chunks until it refuses, then check that freeing
    // two *adjacent* chunks coalesces into a block big enough for both. The
    // exact count depends on placement: first-fit with fence-deferred frees
    // lands earlier tests' blocks differently depending on device timing, so
    // the test asks for adjacency by address rather than by allocation order.
    const size_t chunk = 64u << 20;
    std::vector<sgDevPtr> p;
    for (;;) {
        sgDevPtr x = 0;
        sgError_t e = sgMalloc(&x, chunk);
        if (e == SG_ERR_OUT_OF_MEMORY) break;
        CHECK_OK(e);
        p.push_back(x);
        CHECK(p.size() <= 4);
    }
    CHECK(p.size() >= 3);
    sgDevPtr extra = 0;
    CHECK(sgMalloc(&extra, chunk + 1) == SG_ERR_OUT_OF_MEMORY); // never room for a 5th
    std::sort(p.begin(), p.end());
    CHECK(p[1] == p[0] + chunk); // first-fit packs the first two back to back
    CHECK_OK(sgFree(p[0]));
    CHECK_OK(sgFree(p[1]));
    CHECK_OK(sgMalloc(&extra, 2 * chunk)); // only satisfiable if coalesced
    CHECK(extra <= p[0]); // the merged block may also absorb a free fragment just below p[0]
    CHECK_OK(sgFree(extra));
    for (size_t i = 2; i < p.size(); ++i) CHECK_OK(sgFree(p[i]));
}

static void test_async_ordering() {
    // Thousands of async fills — far more than any ring depth — then one
    // sync. The device must apply them in order and the last one must win.
    const size_t n = 4096;
    sgDevPtr d = 0;
    CHECK_OK(sgMalloc(&d, n));
    for (int i = 0; i < 20000; ++i) CHECK_OK(sgMemset(d, i & 0xff, n));
    CHECK_OK(sgDeviceSynchronize());
    std::vector<uint8_t> out(n);
    CHECK_OK(sgMemcpyD2H(out.data(), d, n));
    CHECK(std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == (19999 & 0xff); }));

    // Data dependency through the queue without an explicit sync in between.
    sgDevPtr a = 0, b = 0, c = 0;
    const uint32_t m = 1 << 14;
    CHECK_OK(sgMalloc(&a, m * 4)); CHECK_OK(sgMalloc(&b, m * 4)); CHECK_OK(sgMalloc(&c, m * 4));
    std::vector<float> ha(m, 3.0f), hb(m, 4.0f), hc(m);
    CHECK_OK(sgMemcpyH2D(a, ha.data(), m * 4));
    CHECK_OK(sgMemcpyH2D(b, hb.data(), m * 4));
    CHECK_OK(sgVaddF32(c, a, b, m));
    CHECK_OK(sgMemcpyD2D(a, c, m * 4));   // reuse a as scratch: a = c
    CHECK_OK(sgVaddF32(c, a, b, m));      // c = (3+4)+4
    CHECK_OK(sgMemcpyD2H(hc.data(), c, m * 4));
    CHECK(std::all_of(hc.begin(), hc.end(), [](float v) { return v == 11.0f; }));

    // Sync with nothing outstanding is a no-op that does not block.
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgDeviceSynchronize());
    CHECK_OK(sgGetStats(&s1));
    CHECK(s1.driver_waits == s0.driver_waits);

    CHECK_OK(sgFree(d)); CHECK_OK(sgFree(a)); CHECK_OK(sgFree(b)); CHECK_OK(sgFree(c));
}

static void test_pinned_roundtrip(size_t bytes) {
    uint8_t *src = nullptr, *dst = nullptr;
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&src), bytes));
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&dst), bytes));
    for (size_t i = 0; i < bytes; ++i) src[i] = static_cast<uint8_t>(i * 13 + 5);
    std::memset(dst, 0, bytes);
    sgDevPtr d = 0;
    CHECK_OK(sgMalloc(&d, bytes));
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgMemcpyH2D(d, src, bytes));
    CHECK_OK(sgMemcpyD2H(dst, d, bytes));
    CHECK_OK(sgGetStats(&s1));
    CHECK(std::memcmp(src, dst, bytes) == 0);
    CHECK(s1.bytes_direct - s0.bytes_direct == 2 * bytes); // both went direct
    CHECK(s1.bytes_staged == s0.bytes_staged);
    CHECK_OK(sgFree(d));
    CHECK_OK(sgFreeHost(src));
    CHECK_OK(sgFreeHost(dst));
}

static void test_host_register() {
    const size_t n = 1 << 20;
    std::vector<uint8_t> buf(n + 4096), out(n);
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = uint8_t(i);
    sgDevPtr d = 0;
    CHECK_OK(sgMalloc(&d, n));
    CHECK_OK(sgHostRegister(buf.data(), n)); // pin only the first n bytes
    CHECK(sgHostRegister(buf.data() + 100, 10) == SG_ERR_INVALID_VALUE); // overlap

    sgStats_t s0{}, s1{}, s2{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgMemcpyH2D(d, buf.data(), n)); // fully inside the pin: direct
    CHECK_OK(sgGetStats(&s1));
    CHECK(s1.bytes_direct - s0.bytes_direct == n);
    // Straddles the end of the pinned range: must fall back to staging and
    // still be correct.
    CHECK_OK(sgMemcpyH2D(d, buf.data() + 4096, n));
    CHECK_OK(sgGetStats(&s2));
    CHECK(s2.bytes_staged - s1.bytes_staged == n);
    CHECK_OK(sgMemcpyD2H(out.data(), d, n));
    CHECK(std::memcmp(out.data(), buf.data() + 4096, n) == 0);

    // Unregister while a DMA from the range is in flight, then reuse it.
    CHECK_OK(sgMemcpyH2DAsync(d, buf.data(), n, nullptr));
    CHECK_OK(sgHostUnregister(buf.data()));
    CHECK(sgHostUnregister(buf.data()) == SG_ERR_INVALID_VALUE);
    CHECK_OK(sgDeviceSynchronize());
    CHECK_OK(sgMemcpyD2H(out.data(), d, n));
    CHECK(std::memcmp(out.data(), buf.data(), n) == 0);
    CHECK_OK(sgFree(d));
}

static void test_async_pinned_pipeline() {
    // H2D a, H2D b (async, pinned), VADD, D2H c (async, pinned), one sync.
    const uint32_t n = 1 << 18;
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&a), n * 4));
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&b), n * 4));
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&c), n * 4));
    for (uint32_t i = 0; i < n; ++i) { a[i] = float(i); b[i] = 2.0f * float(i); c[i] = -1.0f; }
    sgDevPtr da = 0, db = 0, dc = 0;
    CHECK_OK(sgMalloc(&da, n * 4)); CHECK_OK(sgMalloc(&db, n * 4)); CHECK_OK(sgMalloc(&dc, n * 4));
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgMemcpyH2DAsync(da, a, n * 4, nullptr));
    CHECK_OK(sgMemcpyH2DAsync(db, b, n * 4, nullptr));
    CHECK_OK(sgVaddF32(dc, da, db, n));
    CHECK_OK(sgMemcpyD2HAsync(c, dc, n * 4, nullptr));
    CHECK_OK(sgDeviceSynchronize());
    CHECK_OK(sgGetStats(&s1));
    bool ok = true;
    for (uint32_t i = 0; i < n && ok; ++i) ok = (c[i] == 3.0f * float(i));
    CHECK(ok);
    CHECK(s1.driver_waits - s0.driver_waits <= 1); // at most the final sync blocked
    CHECK_OK(sgFree(da)); CHECK_OK(sgFree(db)); CHECK_OK(sgFree(dc));
    CHECK_OK(sgFreeHost(a)); CHECK_OK(sgFreeHost(b)); CHECK_OK(sgFreeHost(c));
}

static void test_deferred_free() {
    // A long GEMM is writing `c` when we free it and immediately reallocate.
    // The allocator must not hand the same memory out until the GEMM retired,
    // otherwise the GEMM's late writes would clobber our memset.
    const uint32_t d = 512;
    const size_t bytes = size_t(d) * d * 4;
    sgDevPtr a = 0, b = 0, c = 0;
    CHECK_OK(sgMalloc(&a, bytes)); CHECK_OK(sgMalloc(&b, bytes)); CHECK_OK(sgMalloc(&c, bytes));
    CHECK_OK(sgMemset(a, 0, bytes)); CHECK_OK(sgMemset(b, 0, bytes));
    CHECK_OK(sgGemmF32(c, a, b, d, d, d)); // ~14 ms, async
    CHECK_OK(sgFree(c));
    sgDevPtr c2 = 0;
    CHECK_OK(sgMalloc(&c2, bytes));
    // With first-fit and everything else live, c2 either reuses c's slot
    // (only safe if the driver deferred the free) or takes fresh space.
    CHECK_OK(sgMemset(c2, 0x7f, bytes));
    CHECK_OK(sgDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    CHECK_OK(sgMemcpyD2H(out.data(), c2, bytes));
    CHECK(std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == 0x7f; }));
    // Now that the GEMM has retired, the original c has been reclaimed and,
    // being the lowest free block, is what first-fit hands out next.
    CHECK_OK(sgFree(c2));
    sgDevPtr c3 = 0;
    CHECK_OK(sgMalloc(&c3, bytes));
    CHECK(c3 == c);
    CHECK_OK(sgFree(a)); CHECK_OK(sgFree(b)); CHECK_OK(sgFree(c3));
}

static void test_concurrent_pinned_submitters() {
    const int T = 8, iters = 100;
    std::vector<std::thread> ts;
    std::vector<int> bad(T, 0);
    for (int t = 0; t < T; ++t)
        ts.emplace_back([t, &bad] {
            const size_t n = 1 << 16;
            uint8_t *src = nullptr, *dst = nullptr;
            if (sgMallocHost(reinterpret_cast<void**>(&src), n) != SG_OK ||
                sgMallocHost(reinterpret_cast<void**>(&dst), n) != SG_OK) { bad[t] = 1; return; }
            sgDevPtr d = 0;
            if (sgMalloc(&d, n) != SG_OK) { bad[t] = 1; return; }
            for (int i = 0; i < iters && !bad[t]; ++i) {
                std::memset(src, t * 16 + (i & 15), n);
                if (sgMemcpyH2D(d, src, n) != SG_OK || sgMemcpyD2H(dst, d, n) != SG_OK ||
                    std::memcmp(src, dst, n) != 0)
                    bad[t] = 1;
            }
            sgFree(d);
            sgFreeHost(src);
            sgFreeHost(dst);
        });
    for (auto& th : ts) th.join();
    CHECK(std::accumulate(bad.begin(), bad.end(), 0) == 0);
}

// ---- stage 3b: engines, streams, events -----------------------------------

static void test_stream_cross_engine_ordering() {
    // 64 chunks of H2D -> VADD -> D2H, all async on one stream: copies run on
    // the copy engine, VADD on the compute engine; the stream must keep them
    // ordered via device-side waits, and the host only syncs once.
    const int chunks = 64;
    const uint32_t n = 1 << 12;
    float *a, *b, *c;
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&a), n * 4));
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&b), n * 4));
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&c), size_t(chunks) * n * 4));
    sgDevPtr da = 0, db = 0, dc = 0;
    CHECK_OK(sgMalloc(&da, n * 4)); CHECK_OK(sgMalloc(&db, n * 4)); CHECK_OK(sgMalloc(&dc, n * 4));
    sgStream_t s = nullptr;
    CHECK_OK(sgStreamCreate(&s));
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    for (int i = 0; i < chunks; ++i) {
        // Each chunk overwrites the same host inputs, so the copy for chunk i
        // must land before VADD i, and VADD i before D2H i — and the host must
        // not touch a/b again until the stream is done with them. We keep
        // a[] constant and vary b[] via a device-side memset instead.
        if (i == 0) {
            std::fill(a, a + n, 1.0f);
            CHECK_OK(sgMemcpyH2DAsync(da, a, n * 4, s));
        }
        CHECK_OK(sgMemsetAsync(db, i, n * 4, s)); // bytes = i -> a known float pattern
        CHECK_OK(sgVaddF32Async(dc, da, db, n, s));
        CHECK_OK(sgMemcpyD2HAsync(c + size_t(i) * n, dc, n * 4, s));
    }
    CHECK_OK(sgStreamSynchronize(s));
    CHECK_OK(sgGetStats(&s1));
    CHECK(s1.driver_waits - s0.driver_waits <= 1); // at most the final sync blocks
    bool ok = true;
    for (int i = 0; i < chunks && ok; ++i) {
        uint32_t bits = uint32_t(i) * 0x01010101u;
        float bf;
        std::memcpy(&bf, &bits, 4);
        const float expect = 1.0f + bf;
        for (uint32_t j = 0; j < n && ok; ++j) ok = (c[size_t(i) * n + j] == expect);
    }
    CHECK(ok);
    CHECK_OK(sgStreamDestroy(s));
    CHECK_OK(sgFree(da)); CHECK_OK(sgFree(db)); CHECK_OK(sgFree(dc));
    CHECK_OK(sgFreeHost(a)); CHECK_OK(sgFreeHost(b)); CHECK_OK(sgFreeHost(c));
}

static void test_two_streams_independent() {
    const uint32_t n = 1 << 16;
    struct Chain { sgStream_t s; float *a, *b, *c; sgDevPtr da, db, dc; float va, vb; };
    Chain ch[2] = {{nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 1.0f, 2.0f},
                   {nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 10.0f, 20.0f}};
    for (auto& k : ch) {
        CHECK_OK(sgStreamCreate(&k.s));
        CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&k.a), n * 4));
        CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&k.b), n * 4));
        CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&k.c), n * 4));
        std::fill(k.a, k.a + n, k.va);
        std::fill(k.b, k.b + n, k.vb);
        CHECK_OK(sgMalloc(&k.da, n * 4)); CHECK_OK(sgMalloc(&k.db, n * 4)); CHECK_OK(sgMalloc(&k.dc, n * 4));
    }
    for (int rep = 0; rep < 50; ++rep)
        for (auto& k : ch) {
            CHECK_OK(sgMemcpyH2DAsync(k.da, k.a, n * 4, k.s));
            CHECK_OK(sgMemcpyH2DAsync(k.db, k.b, n * 4, k.s));
            CHECK_OK(sgVaddF32Async(k.dc, k.da, k.db, n, k.s));
            CHECK_OK(sgMemcpyD2HAsync(k.c, k.dc, n * 4, k.s));
        }
    CHECK_OK(sgDeviceSynchronize());
    for (auto& k : ch) {
        CHECK(std::all_of(k.c, k.c + n, [&](float v) { return v == k.va + k.vb; }));
        CHECK_OK(sgStreamDestroy(k.s));
        CHECK_OK(sgFree(k.da)); CHECK_OK(sgFree(k.db)); CHECK_OK(sgFree(k.dc));
        CHECK_OK(sgFreeHost(k.a)); CHECK_OK(sgFreeHost(k.b)); CHECK_OK(sgFreeHost(k.c));
    }
}

static void test_event_dependency() {
    // Stream A produces x (a slow-ish GEMM then a memset marker); stream B
    // must see A's result only if it waited on the event. Repeated to catch
    // races: without the wait, B's D2D would usually copy stale data.
    const uint32_t d = 128;
    const size_t bytes = size_t(d) * d * 4;
    sgStream_t A = nullptr, B = nullptr;
    sgEvent_t ev = nullptr;
    CHECK_OK(sgStreamCreate(&A)); CHECK_OK(sgStreamCreate(&B)); CHECK_OK(sgEventCreate(&ev));
    sgDevPtr ga = 0, gb = 0, x = 0, y = 0;
    CHECK_OK(sgMalloc(&ga, bytes)); CHECK_OK(sgMalloc(&gb, bytes));
    CHECK_OK(sgMalloc(&x, bytes)); CHECK_OK(sgMalloc(&y, bytes));
    CHECK_OK(sgMemset(ga, 0, bytes)); CHECK_OK(sgMemset(gb, 0, bytes));
    std::vector<uint8_t> out(bytes);
    // Waiting on an event that was never recorded is a no-op.
    CHECK_OK(sgStreamWaitEvent(B, ev));
    int bad = 0;
    for (int i = 0; i < 200; ++i) {
        const uint8_t marker = uint8_t(i + 1);
        CHECK_OK(sgMemsetAsync(x, 0, bytes, A));
        CHECK_OK(sgGemmF32Async(x, ga, gb, d, d, d, A)); // ~200 us of work writing x
        CHECK_OK(sgMemsetAsync(x, marker, bytes, A));
        CHECK_OK(sgEventRecord(ev, A));
        CHECK_OK(sgStreamWaitEvent(B, ev));
        CHECK_OK(sgMemcpyD2DAsync(y, x, bytes, B));
        CHECK_OK(sgMemcpyD2HAsync(out.data(), y, bytes, B)); // pageable: synchronous
        CHECK_OK(sgStreamSynchronize(B));
        if (!std::all_of(out.begin(), out.end(), [&](uint8_t v) { return v == marker; })) ++bad;
    }
    CHECK(bad == 0);
    CHECK_OK(sgEventSynchronize(ev));
    CHECK_OK(sgEventDestroy(ev)); CHECK_OK(sgStreamDestroy(A)); CHECK_OK(sgStreamDestroy(B));
    CHECK_OK(sgFree(ga)); CHECK_OK(sgFree(gb)); CHECK_OK(sgFree(x)); CHECK_OK(sgFree(y));
}

static void test_stream_sync_is_per_stream() {
    // Synchronizing stream B must not wait for a long GEMM queued on A.
    const uint32_t d = 512;
    const size_t bytes = size_t(d) * d * 4;
    sgStream_t A = nullptr, B = nullptr;
    CHECK_OK(sgStreamCreate(&A)); CHECK_OK(sgStreamCreate(&B));
    sgDevPtr ga = 0, gb = 0, gc = 0, t = 0;
    CHECK_OK(sgMalloc(&ga, bytes)); CHECK_OK(sgMalloc(&gb, bytes)); CHECK_OK(sgMalloc(&gc, bytes));
    CHECK_OK(sgMalloc(&t, 4096));
    CHECK_OK(sgMemset(ga, 0, bytes)); CHECK_OK(sgMemset(gb, 0, bytes));
    CHECK_OK(sgDeviceSynchronize());
    CHECK_OK(sgGemmF32Async(gc, ga, gb, d, d, d, A)); // ~14 ms on the compute engine
    uint8_t* h = nullptr;
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&h), 4096));
    CHECK_OK(sgMemcpyH2DAsync(t, h, 4096, B)); // copy engine only
    auto t0 = std::chrono::steady_clock::now();
    CHECK_OK(sgStreamSynchronize(B));
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(ms < 5.0); // did not wait for A's GEMM
    CHECK_OK(sgDeviceSynchronize());
    CHECK_OK(sgStreamDestroy(A)); CHECK_OK(sgStreamDestroy(B));
    CHECK_OK(sgFree(ga)); CHECK_OK(sgFree(gb)); CHECK_OK(sgFree(gc)); CHECK_OK(sgFree(t));
    CHECK_OK(sgFreeHost(h));
}

static void test_pageable_on_streams_and_deferred_copy_free() {
    // Pageable copies on a non-default stream go through staging slots whose
    // fences now live on the copy engine; and freeing VRAM that an in-flight
    // *copy* references must be deferred just like compute.
    const size_t n = 3u << 20; // 3 MiB: many staging chunks
    std::vector<uint8_t> src(n), dst(n);
    for (size_t i = 0; i < n; ++i) src[i] = uint8_t(i * 7);
    sgStream_t s = nullptr;
    CHECK_OK(sgStreamCreate(&s));
    sgDevPtr d = 0;
    CHECK_OK(sgMalloc(&d, n));
    CHECK_OK(sgMemcpyH2DAsync(d, src.data(), n, s)); // pageable: staged, host buffer reusable
    CHECK_OK(sgMemcpyD2HAsync(dst.data(), d, n, s)); // pageable: synchronous
    CHECK(src == dst);
    // Queue a large copy out of d, free d, reallocate, overwrite; the copy
    // must have finished reading before the memory is reused.
    uint8_t* h = nullptr;
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&h), n));
    CHECK_OK(sgMemcpyD2HAsync(h, d, n, s)); // direct DMA, in flight
    CHECK_OK(sgFree(d));
    sgDevPtr d2 = 0;
    CHECK_OK(sgMalloc(&d2, n));
    CHECK_OK(sgMemsetAsync(d2, 0xEE, n, s));
    CHECK_OK(sgDeviceSynchronize());
    CHECK(std::memcmp(h, src.data(), n) == 0); // the DMA read d, not our memset
    CHECK_OK(sgFree(d2)); CHECK_OK(sgFreeHost(h)); CHECK_OK(sgStreamDestroy(s));
}

static void test_engine_rejections() {
    sgStats_t st{};
    CHECK_OK(sgGetStats(&st));
    CHECK(st.num_engines >= 2);
    // The public API cannot express "compute on a copy engine", so exercise
    // the driver's checks through the wait path instead: sgStreamWaitEvent
    // on a recorded event followed by a submit is the only WAIT producer and
    // is validated by construction. What we can check publicly: a stream's
    // synchronize with nothing submitted, and destroying NULL.
    CHECK(sgStreamDestroy(nullptr) == SG_ERR_INVALID_VALUE);
    CHECK(sgEventDestroy(nullptr) == SG_ERR_INVALID_VALUE);
    CHECK_OK(sgStreamSynchronize(nullptr));
    sgStream_t s = nullptr;
    CHECK_OK(sgStreamCreate(&s));
    CHECK_OK(sgStreamSynchronize(s));
    CHECK_OK(sgStreamDestroy(s));
}

static void test_concurrent_streams() {
    const int T = 8, iters = 50;
    std::vector<std::thread> ts;
    std::vector<int> bad(T, 0);
    for (int t = 0; t < T; ++t)
        ts.emplace_back([t, &bad] {
            const uint32_t n = 1 << 14;
            sgStream_t s = nullptr;
            float *a, *b, *c;
            if (sgStreamCreate(&s) != SG_OK ||
                sgMallocHost(reinterpret_cast<void**>(&a), n * 4) != SG_OK ||
                sgMallocHost(reinterpret_cast<void**>(&b), n * 4) != SG_OK ||
                sgMallocHost(reinterpret_cast<void**>(&c), n * 4) != SG_OK) { bad[t] = 1; return; }
            sgDevPtr da = 0, db = 0, dc = 0;
            if (sgMalloc(&da, n * 4) != SG_OK || sgMalloc(&db, n * 4) != SG_OK || sgMalloc(&dc, n * 4) != SG_OK) { bad[t] = 1; return; }
            for (int i = 0; i < iters && !bad[t]; ++i) {
                std::fill(a, a + n, float(t));
                std::fill(b, b + n, float(i));
                if (sgMemcpyH2DAsync(da, a, n * 4, s) != SG_OK || sgMemcpyH2DAsync(db, b, n * 4, s) != SG_OK ||
                    sgVaddF32Async(dc, da, db, n, s) != SG_OK || sgMemcpyD2HAsync(c, dc, n * 4, s) != SG_OK ||
                    sgStreamSynchronize(s) != SG_OK)
                    bad[t] = 1;
                else if (!std::all_of(c, c + n, [&](float v) { return v == float(t) + float(i); }))
                    bad[t] = 1;
            }
            sgFree(da); sgFree(db); sgFree(dc);
            sgFreeHost(a); sgFreeHost(b); sgFreeHost(c);
            sgStreamDestroy(s);
        });
    for (auto& th : ts) th.join();
    CHECK(std::accumulate(bad.begin(), bad.end(), 0) == 0);
}

// ---- stage 2: wait policy ---------------------------------------------------

static void test_wait_policy_stats() {
    // A 14 ms GEMM synchronized under each policy: blocking must sleep (one
    // blocked wait, tiny CPU), spinning must not (one spun wait, cpu ~ wall).
    const uint32_t d = 512;
    const size_t bytes = size_t(d) * d * 4;
    sgDevPtr a = 0, b = 0, c = 0;
    CHECK_OK(sgMalloc(&a, bytes)); CHECK_OK(sgMalloc(&b, bytes)); CHECK_OK(sgMalloc(&c, bytes));
    CHECK_OK(sgMemset(a, 0, bytes)); CHECK_OK(sgMemset(b, 0, bytes)); CHECK_OK(sgDeviceSynchronize());
    auto cpu_ns = [] {
        timespec ts{};
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
        return double(ts.tv_sec) * 1e9 + double(ts.tv_nsec);
    };
    for (int pol = 0; pol < 2; ++pol) {
        CHECK_OK(sgSetSyncPolicy(pol == 0 ? SG_SYNC_BLOCK : SG_SYNC_SPIN));
        sgStats_t s0{}, s1{};
        CHECK_OK(sgResetStats());
        CHECK_OK(sgGetStats(&s0));
        CHECK_OK(sgGemmF32(c, a, b, d, d, d));
        const double c0 = cpu_ns();
        auto t0 = std::chrono::steady_clock::now();
        CHECK_OK(sgDeviceSynchronize());
        const double wall = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
        const double cpu = cpu_ns() - c0;
        CHECK_OK(sgGetStats(&s1));
        CHECK(s1.driver_waits - s0.driver_waits == 1);
        if (pol == 0) {
            CHECK(s1.waits_blocked - s0.waits_blocked == 1);
            CHECK(cpu < 0.2 * wall); // slept most of the time
            CHECK(s1.engine_irqs[0] - s0.engine_irqs[0] >= 1);
        } else {
            CHECK(s1.waits_spun - s0.waits_spun == 1);
            CHECK(cpu > 0.8 * wall); // spun the whole time
        }
    }
    CHECK_OK(sgSetSyncPolicy(SG_SYNC_DEFAULT));
    CHECK_OK(sgFree(a)); CHECK_OK(sgFree(b)); CHECK_OK(sgFree(c));
}

static void test_blocking_lost_wakeup_stress() {
    // Many tiny waits under the blocking policy from several threads. Each
    // wait arms an interrupt for a fence that is often already about to
    // retire — the exact window a lost-wakeup bug would hang in. The ctest
    // TIMEOUT is the assertion; here we just check results.
    CHECK_OK(sgSetSyncPolicy(SG_SYNC_BLOCK));
    const int T = 8, iters = 2000;
    std::vector<std::thread> ts;
    std::vector<int> bad(T, 0);
    for (int t = 0; t < T; ++t)
        ts.emplace_back([t, &bad] {
            sgStream_t s = nullptr;
            sgDevPtr d = 0;
            uint8_t* h = nullptr;
            if (sgStreamCreate(&s) != SG_OK || sgMalloc(&d, 4096) != SG_OK ||
                sgMallocHost(reinterpret_cast<void**>(&h), 4096) != SG_OK) { bad[t] = 1; return; }
            for (int i = 0; i < iters && !bad[t]; ++i) {
                if (sgMemsetAsync(d, i & 0xff, 4096, s) != SG_OK ||
                    sgMemcpyD2HAsync(h, d, 4096, s) != SG_OK || sgStreamSynchronize(s) != SG_OK ||
                    h[0] != uint8_t(i & 0xff) || h[4095] != uint8_t(i & 0xff))
                    bad[t] = 1;
            }
            sgFreeHost(h); sgFree(d); sgStreamDestroy(s);
        });
    for (auto& th : ts) th.join();
    CHECK(std::accumulate(bad.begin(), bad.end(), 0) == 0);
    CHECK_OK(sgSetSyncPolicy(SG_SYNC_DEFAULT));
}

// ---- stage 4: no big lock -------------------------------------------------

static void test_shared_stream_many_producers() {
    // 8 threads submit to ONE stream (hence one channel), each on its own
    // device buffer: memset(value) then D2H. Stream order must hold for each
    // thread's own pair regardless of how producers interleave.
    const int T = 8, iters = 500;
    sgStream_t s = nullptr;
    CHECK_OK(sgStreamCreate(&s));
    std::vector<std::thread> ts;
    std::vector<int> bad(T, 0);
    for (int t = 0; t < T; ++t)
        ts.emplace_back([t, s, &bad] {
            sgDevPtr d = 0;
            uint8_t* h = nullptr;
            if (sgMalloc(&d, 4096) != SG_OK || sgMallocHost(reinterpret_cast<void**>(&h), 4096) != SG_OK) { bad[t] = 1; return; }
            for (int i = 0; i < iters && !bad[t]; ++i) {
                const uint8_t v = uint8_t(t * 32 + (i & 31));
                if (sgMemsetAsync(d, v, 4096, s) != SG_OK || sgMemcpyD2HAsync(h, d, 4096, s) != SG_OK ||
                    sgStreamSynchronize(s) != SG_OK || h[0] != v || h[4095] != v)
                    bad[t] = 1;
            }
            sgFreeHost(h); sgFree(d);
        });
    for (auto& th : ts) th.join();
    CHECK(std::accumulate(bad.begin(), bad.end(), 0) == 0);
    CHECK_OK(sgStreamDestroy(s));
}

static void test_alloc_storm_during_submits() {
    // Half the threads allocate/free continuously, half run pinned copy
    // chains on their own streams; validation must never see a torn
    // allocator and no thread may observe another's memory.
    const int T = 8, iters = 300;
    std::vector<std::thread> ts;
    std::vector<int> bad(T, 0);
    for (int t = 0; t < T; ++t)
        ts.emplace_back([t, &bad] {
            if (t % 2 == 0) {
                for (int i = 0; i < iters * 4 && !bad[t]; ++i) {
                    sgDevPtr p = 0;
                    if (sgMalloc(&p, 4096 << (i % 6)) != SG_OK) { bad[t] = 1; break; }
                    if (sgMemsetAsync(p, i, 4096, nullptr) != SG_OK) { bad[t] = 1; break; }
                    if (sgFree(p) != SG_OK) { bad[t] = 1; break; }
                }
            } else {
                sgStream_t s = nullptr;
                sgDevPtr d = 0;
                uint8_t *src = nullptr, *dst = nullptr;
                if (sgStreamCreate(&s) != SG_OK || sgMalloc(&d, 65536) != SG_OK ||
                    sgMallocHost(reinterpret_cast<void**>(&src), 65536) != SG_OK ||
                    sgMallocHost(reinterpret_cast<void**>(&dst), 65536) != SG_OK) { bad[t] = 1; return; }
                for (int i = 0; i < iters && !bad[t]; ++i) {
                    std::memset(src, t + i, 65536);
                    if (sgMemcpyH2DAsync(d, src, 65536, s) != SG_OK || sgMemcpyD2HAsync(dst, d, 65536, s) != SG_OK ||
                        sgStreamSynchronize(s) != SG_OK || std::memcmp(src, dst, 65536) != 0)
                        bad[t] = 1;
                }
                sgFreeHost(src); sgFreeHost(dst); sgFree(d); sgStreamDestroy(s);
            }
        });
    for (auto& th : ts) th.join();
    CHECK(std::accumulate(bad.begin(), bad.end(), 0) == 0);
    CHECK_OK(sgDeviceSynchronize());
}

static void test_pin_storm_during_copies() {
    // Threads register/unregister ranges while others copy from pinned
    // memory; a copy must be either fully direct or fully staged, never a
    // mix that reads a range mid-unpin.
    const int T = 6, iters = 200;
    std::vector<std::thread> ts;
    std::vector<int> bad(T, 0);
    for (int t = 0; t < T; ++t)
        ts.emplace_back([t, &bad] {
            std::vector<uint8_t> buf(1 << 16), out(1 << 16);
            sgDevPtr d = 0;
            if (sgMalloc(&d, buf.size()) != SG_OK) { bad[t] = 1; return; }
            for (int i = 0; i < iters && !bad[t]; ++i) {
                std::fill(buf.begin(), buf.end(), uint8_t(t + i));
                const bool pin = (i % 3) != 0;
                if (pin && sgHostRegister(buf.data(), buf.size()) != SG_OK) { bad[t] = 1; break; }
                if (sgMemcpyH2D(d, buf.data(), buf.size()) != SG_OK || sgMemcpyD2H(out.data(), d, buf.size()) != SG_OK ||
                    out != buf)
                    bad[t] = 1;
                if (pin && sgHostUnregister(buf.data()) != SG_OK) bad[t] = 1;
            }
            sgDeviceSynchronize();
            sgFree(d);
        });
    for (auto& th : ts) th.join();
    CHECK(std::accumulate(bad.begin(), bad.end(), 0) == 0);
}

static void test_ticket_publish_storm() {
    // Two producers per channel (16 streams over 8 channels), each thread
    // alternating between its two streams, hammering tiny commands. In
    // ticket mode this is the interleaving that regressed the device's PUT
    // in the first cut and hung the engine; the test must terminate and
    // every thread must read back its own values.
    const int T = 8, iters = 3000;
    std::vector<sgStream_t> streams(2 * T);
    for (auto& s : streams) CHECK_OK(sgStreamCreate(&s));
    std::vector<std::thread> ts;
    std::vector<int> bad(T, 0);
    for (int t = 0; t < T; ++t)
        ts.emplace_back([t, &streams, &bad] {
            sgDevPtr d[2] = {0, 0};
            uint8_t* h[2] = {nullptr, nullptr};
            for (int k = 0; k < 2; ++k)
                if (sgMalloc(&d[k], 4096) != SG_OK || sgMallocHost(reinterpret_cast<void**>(&h[k]), 4096) != SG_OK) { bad[t] = 1; return; }
            for (int i = 0; i < iters && !bad[t]; ++i) {
                const int k = i & 1;
                sgStream_t s = streams[t * 2 + k];
                const uint8_t v = uint8_t(t * 8 + (i & 7));
                if (sgMemsetAsync(d[k], v, 4096, s) != SG_OK || sgMemcpyD2HAsync(h[k], d[k], 4096, s) != SG_OK) bad[t] = 1;
                if (i % 16 == 15 && (sgStreamSynchronize(s) != SG_OK || h[k][0] != v || h[k][4095] != v)) bad[t] = 1;
            }
            for (int k = 0; k < 2; ++k) { sgStreamSynchronize(streams[t * 2 + k]); sgFreeHost(h[k]); sgFree(d[k]); }
        });
    for (auto& th : ts) th.join();
    CHECK(std::accumulate(bad.begin(), bad.end(), 0) == 0);
    sgStats_t st{};
    CHECK_OK(sgGetStats(&st));
    for (auto& s : streams) CHECK_OK(sgStreamDestroy(s));
    CHECK_OK(sgDeviceSynchronize()); // would report the sticky error if PUT ever regressed
}

// ---- stage 7: device idle gating ---------------------------------------------

static void test_gating_no_missed_doorbells() {
    // Many single tiny commands each followed by a host spin-wait: under a
    // sleeping engine every submit hits the arm/re-check window. A lost
    // doorbell shows up as a 1 ms timeout wake that finds work (counted by
    // the device) and as a >= 1 ms round trip.
    CHECK_OK(sgSetSyncPolicy(SG_SYNC_SPIN));
    const int T = 8, iters = 3000;
    sgStats_t s0{}, s1{};
    CHECK_OK(sgResetStats());
    CHECK_OK(sgGetStats(&s0));
    std::vector<std::thread> ts;
    std::vector<int> bad(T, 0);
    std::vector<double> worst(T, 0);
    for (int t = 0; t < T; ++t)
        ts.emplace_back([t, &bad, &worst] {
            sgStream_t s = nullptr;
            sgDevPtr d = 0;
            if (sgStreamCreate(&s) != SG_OK || sgMalloc(&d, 64) != SG_OK) { bad[t] = 1; return; }
            for (int i = 0; i < iters && !bad[t]; ++i) {
                auto t0 = std::chrono::steady_clock::now();
                if (sgMemsetAsync(d, i, 64, s) != SG_OK || sgStreamSynchronize(s) != SG_OK) bad[t] = 1;
                double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                if (us > worst[t]) worst[t] = us;
            }
            sgFree(d); sgStreamDestroy(s);
        });
    for (auto& th : ts) th.join();
    CHECK(std::accumulate(bad.begin(), bad.end(), 0) == 0);
    CHECK_OK(sgGetStats(&s1));
    uint64_t missed = 0;
    for (uint32_t e = 0; e < s1.num_engines; ++e) missed += s1.engine_missed_doorbells[e] - s0.engine_missed_doorbells[e];
    CHECK(missed == 0);
    CHECK_OK(sgSetSyncPolicy(SG_SYNC_DEFAULT));
}

static void test_gating_power_when_idle() {
    // An idle device should cost (almost) no CPU under a gating policy and
    // two full cores under `spin`. Only assert when the environment tells us
    // which policy is in effect; the numbers are recorded either way.
    CHECK_OK(sgDeviceSynchronize());
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK_OK(sgGetStats(&s1));
    double cpu = 0, sleep = 0;
    for (uint32_t e = 0; e < s1.num_engines; ++e) {
        cpu += double(s1.engine_cpu_ns[e] - s0.engine_cpu_ns[e]);
        sleep += double(s1.engine_sleep_cycles[e] - s0.engine_sleep_cycles[e]);
    }
    const double cores = cpu / 100e6; // engine CPU per wall second, in cores
    const char* pol = std::getenv("SG_ENGINE_IDLE");
    if (pol && !std::strcmp(pol, "spin")) CHECK(cores > 1.5);
    else if (pol && !std::strcmp(pol, "sleep")) { CHECK(cores < 0.2); CHECK(sleep > 50e6 * s1.num_engines); }
    std::printf("      [idle 100 ms: engines used %.2f cores, slept %.0f%%]\n", cores,
                std::min(100.0, 100.0 * sleep / (100e6 * s1.num_engines)));
}

static void test_concurrent_submitters() {
    // Many threads hammering the driver; every thread verifies its own data.
    const int T = 8, iters = 200;
    std::vector<std::thread> ts;
    std::vector<int> bad(T, 0);
    for (int t = 0; t < T; ++t)
        ts.emplace_back([t, &bad] {
            const size_t n = 4096;
            std::vector<uint8_t> src(n, uint8_t(t + 1)), dst(n);
            sgDevPtr d = 0;
            if (sgMalloc(&d, n) != SG_OK) { bad[t] = 1; return; }
            for (int i = 0; i < iters; ++i) {
                src[i % n] = uint8_t(i);
                if (sgMemcpyH2D(d, src.data(), n) != SG_OK ||
                    sgMemcpyD2H(dst.data(), d, n) != SG_OK || src != dst) { bad[t] = 1; break; }
            }
            sgFree(d);
        });
    for (auto& th : ts) th.join();
    CHECK(std::accumulate(bad.begin(), bad.end(), 0) == 0);
}

static void test_managed_migrate_in() {
    // Two pages, so the command faults, retries, and faults again.
    const size_t n = 8192;
    sgDevPtr d = 0;
    void* host_void = nullptr;
    CHECK_OK(sgMallocManaged(&d, &host_void, n));
    auto* host = static_cast<uint8_t*>(host_void);
    for (size_t i = 0; i < n; ++i) host[i] = static_cast<uint8_t>(i);
    std::vector<uint8_t> out(n, 0);
    CHECK_OK(sgMemcpyD2H(out.data(), d, n));
    CHECK(std::memcmp(out.data(), host, n) == 0);
    // Device writes stay in VRAM. D2H observes them; the host pointer does not.
    CHECK_OK(sgMemset(d, 0x5A, n));
    CHECK_OK(sgMemcpyD2H(out.data(), d, n));
    CHECK(std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == 0x5A; }));
    // Without userfaultfd the host page is stale. With it, the load migrates
    // the device's write back.
    CHECK(host[1] == (sgManagedCoherent() ? 0x5A : 1));
    CHECK_OK(sgFree(d));
    CHECK(sgMallocManaged(nullptr, &host_void, 16) == SG_ERR_INVALID_VALUE);
    CHECK(sgMallocManaged(&d, nullptr, 16) == SG_ERR_INVALID_VALUE);
    CHECK(sgMallocManaged(&d, &host_void, 0) == SG_ERR_INVALID_VALUE);
}

static void test_managed_two_channels() {
    sgStream_t a = nullptr, b = nullptr;
    CHECK_OK(sgStreamCreate(&a));
    CHECK_OK(sgStreamCreate(&b));
    const size_t n = 4096;
    sgDevPtr da = 0, db = 0;
    void *ha_void = nullptr, *hb_void = nullptr;
    CHECK_OK(sgMallocManaged(&da, &ha_void, n));
    CHECK_OK(sgMallocManaged(&db, &hb_void, n));
    auto* ha = static_cast<uint8_t*>(ha_void);
    auto* hb = static_cast<uint8_t*>(hb_void);
    std::memset(ha, 0x11, n);
    std::memset(hb, 0x22, n);
    uint8_t *pa = nullptr, *pb = nullptr;
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&pa), n));
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&pb), n));
    CHECK_OK(sgMemcpyD2HAsync(pa, da, n, a));
    CHECK_OK(sgMemcpyD2HAsync(pb, db, n, b));
    CHECK_OK(sgStreamSynchronize(a));
    CHECK_OK(sgStreamSynchronize(b));
    CHECK(std::all_of(pa, pa + n, [](uint8_t v) { return v == 0x11; }));
    CHECK(std::all_of(pb, pb + n, [](uint8_t v) { return v == 0x22; }));
    CHECK_OK(sgFree(da));
    CHECK_OK(sgFree(db));
    CHECK_OK(sgFreeHost(pa));
    CHECK_OK(sgFreeHost(pb));
    CHECK_OK(sgStreamDestroy(a));
    CHECK_OK(sgStreamDestroy(b));
}

// Fill VRAM with sgMalloc so exactly `frames` 4 KiB frames stay free for
// managed pages. Returns the blocks to free afterwards.
static std::vector<sgDevPtr> squeeze_vram(size_t frames) {
    const size_t pg = 4096;
    std::vector<std::pair<sgDevPtr, size_t>> held;
    for (size_t chunk : {size_t{64} << 20, size_t{1} << 20, pg})
        for (sgDevPtr p = 0; sgMalloc(&p, chunk) == SG_OK;) held.push_back({p, chunk});
    // VRAM is full. Give back the newest blocks until enough is free, then
    // take back the excess a page at a time.
    size_t freed = 0;
    while (freed < frames * pg && !held.empty()) {
        CHECK_OK(sgFree(held.back().first));
        freed += held.back().second;
        held.pop_back();
    }
    for (size_t extra = freed / pg - frames; extra--;) {
        sgDevPtr p = 0;
        CHECK_OK(sgMalloc(&p, pg));
        held.push_back({p, pg});
    }
    std::vector<sgDevPtr> out;
    for (auto& h : held) out.push_back(h.first);
    return out;
}

// Read device memory through the compute engine's TLB (x + 0.0f leaves
// these byte patterns unchanged) rather than the copy engine's.
static void read_via_compute(uint8_t* out, sgDevPtr src, sgDevPtr zeros, sgDevPtr tmp, size_t n) {
    CHECK_OK(sgVaddF32(tmp, src, zeros, uint32_t(n / 4)));
    CHECK_OK(sgMemcpyD2H(out, tmp, n));
}

static void test_managed_evict_writeback() {
    // 64 managed pages through 16 free frames: every fault past the 16th
    // evicts, and a page the device wrote must reach the host backing.
    const size_t frames = 16, pages = 64, pg = 4096, n = pages * pg;
    sgDevPtr d = 0;
    void* host_void = nullptr;
    CHECK_OK(sgMallocManaged(&d, &host_void, n));
    auto* host = static_cast<uint8_t*>(host_void);
    auto held = squeeze_vram(frames);
    for (size_t i = 0; i < pages; ++i) CHECK_OK(sgMemset(d + i * pg, int(i + 1), pg));
    CHECK_OK(sgDeviceSynchronize());
    // At most `frames` pages can still be resident; all earlier ones were
    // evicted dirty.
    bool ok = true;
    for (size_t i = 0; i + frames < pages && ok; ++i)
        ok = std::all_of(host + i * pg, host + (i + 1) * pg, [&](uint8_t v) { return v == uint8_t(i + 1); });
    CHECK(ok);
    // Reading through the device migrates the evicted pages back in.
    std::vector<uint8_t> out(n, 0);
    CHECK_OK(sgMemcpyD2H(out.data(), d, n));
    ok = true;
    for (size_t i = 0; i < n && ok; ++i) ok = out[i] == uint8_t(i / pg + 1);
    CHECK(ok);
    for (sgDevPtr p : held) CHECK_OK(sgFree(p));
    CHECK_OK(sgFree(d));
}

static void test_managed_oversubscribed_compute() {
    // Three 32-page arrays through 8 free frames (12x oversubscribed): VADD
    // cannot hold its operands resident at once and must stream them.
    // GEMM likewise with 12 operand pages.
    const size_t frames = 8;
    const uint32_t n = 32 * 1024;
    const size_t bytes = size_t{n} * 4;
    sgDevPtr a = 0, b = 0, c = 0;
    void *ha = nullptr, *hb = nullptr, *hc = nullptr;
    CHECK_OK(sgMallocManaged(&a, &ha, bytes));
    CHECK_OK(sgMallocManaged(&b, &hb, bytes));
    CHECK_OK(sgMallocManaged(&c, &hc, bytes));
    for (uint32_t i = 0; i < n; ++i) {
        static_cast<float*>(ha)[i] = float(i);
        static_cast<float*>(hb)[i] = 2.0f * float(i);
    }
    const uint32_t m = 64, k = 64, w = 64;
    sgDevPtr ga = 0, gb = 0, gc = 0;
    void *hga = nullptr, *hgb = nullptr, *hgc = nullptr;
    CHECK_OK(sgMallocManaged(&ga, &hga, m * k * 4));
    CHECK_OK(sgMallocManaged(&gb, &hgb, k * w * 4));
    CHECK_OK(sgMallocManaged(&gc, &hgc, m * w * 4));
    auto* fa = static_cast<float*>(hga);
    auto* fb = static_cast<float*>(hgb);
    for (uint32_t i = 0; i < m * k; ++i) fa[i] = float((i * 7) % 13) - 6.0f;
    for (uint32_t i = 0; i < k * w; ++i) fb[i] = float((i * 5) % 11) - 5.0f;
    std::vector<float> ref(m * w, 0.0f);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < w; ++j)
            for (uint32_t p = 0; p < k; ++p) ref[i * w + j] += fa[i * k + p] * fb[p * w + j];

    auto held = squeeze_vram(frames);
    CHECK_OK(sgVaddF32(c, a, b, n));
    std::vector<float> out(n, -1.0f);
    CHECK_OK(sgMemcpyD2H(out.data(), c, bytes));
    bool ok = true;
    for (uint32_t i = 0; i < n && ok; ++i) ok = out[i] == 3.0f * float(i);
    CHECK(ok);
    CHECK_OK(sgGemmF32(gc, ga, gb, m, w, k));
    std::vector<float> gout(m * w, 0.0f);
    CHECK_OK(sgMemcpyD2H(gout.data(), gc, gout.size() * 4));
    ok = true;
    for (size_t i = 0; i < gout.size() && ok; ++i) ok = std::fabs(gout[i] - ref[i]) < 1e-3f;
    CHECK(ok);
    for (sgDevPtr p : held) CHECK_OK(sgFree(p));
    for (sgDevPtr p : {a, b, c, ga, gb, gc}) CHECK_OK(sgFree(p));
}

static void test_managed_tlb_shootdown() {
    // Every page here fits in the compute engine's 64-entry TLB at once, so
    // if eviction skipped its shootdown (or unmap and the remap at the same
    // VA both did), the stale entry would hit and read a frame that now
    // holds another page, or one nothing was migrated into.
    const size_t frames = 16, pages = 48, pg = 4096;
    sgDevPtr zeros = 0, tmp = 0, d = 0;
    CHECK_OK(sgMalloc(&zeros, pg));
    CHECK_OK(sgMalloc(&tmp, pg));
    CHECK_OK(sgMemset(zeros, 0, pg));
    void* host_void = nullptr;
    CHECK_OK(sgMallocManaged(&d, &host_void, pages * pg));
    auto held = squeeze_vram(frames);

    // Eviction: writing pages 16.. reuses the frames of pages 0..
    for (size_t i = 0; i < pages; ++i) CHECK_OK(sgMemset(d + i * pg, int(i + 1), pg));
    std::vector<uint8_t> out(pg);
    bool ok = true;
    for (size_t i = 0; i < pages && ok; ++i) {
        read_via_compute(out.data(), d + i * pg, zeros, tmp, pg);
        ok = std::all_of(out.begin(), out.end(), [&](uint8_t v) { return v == uint8_t(i + 1); });
    }
    CHECK(ok);

    // Unmap: a new allocation at the freed VA must fault in its own bytes,
    // not hit the old translation.
    sgDevPtr x = 0, y = 0;
    void *hx = nullptr, *hy = nullptr;
    CHECK_OK(sgMallocManaged(&x, &hx, pg));
    CHECK_OK(sgMemset(x, 0xAA, pg));
    read_via_compute(out.data(), x, zeros, tmp, pg);
    CHECK(std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == 0xAA; }));
    CHECK_OK(sgFree(x));
    CHECK_OK(sgMallocManaged(&y, &hy, pg));
    CHECK(y == x); // first-fit hands the VA straight back
    std::memset(hy, 0x55, pg);
    read_via_compute(out.data(), y, zeros, tmp, pg);
    CHECK(std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == 0x55; }));

    for (sgDevPtr p : held) CHECK_OK(sgFree(p));
    for (sgDevPtr p : {y, d, zeros, tmp}) CHECK_OK(sgFree(p));
}

// ---- stage 5 slice 4: prefetch and VM counters ---------------------------------

static uint64_t prefetch_window() {
    const char* e = std::getenv("SG_PREFETCH_PAGES");
    return e ? std::strtoull(e, nullptr, 10) : 16;
}

struct VmDelta {
    uint64_t faults = 0, tlb_hits = 0, tlb_misses = 0;
    uint64_t migrations = 0, prefetches = 0, evictions = 0, writebacks = 0, bytes = 0;
};

static VmDelta vm_delta(const sgStats_t& a, const sgStats_t& b) {
    VmDelta d;
    for (uint32_t e = 0; e < b.num_engines; ++e) {
        d.faults += b.engine_faults[e] - a.engine_faults[e];
        d.tlb_hits += b.engine_tlb_hits[e] - a.engine_tlb_hits[e];
        d.tlb_misses += b.engine_tlb_misses[e] - a.engine_tlb_misses[e];
    }
    d.migrations = b.um_migrations - a.um_migrations;
    d.prefetches = b.um_prefetches - a.um_prefetches;
    d.evictions = b.um_evictions - a.um_evictions;
    d.writebacks = b.um_writebacks - a.um_writebacks;
    d.bytes = b.um_bytes_migrated - a.um_bytes_migrated;
    return d;
}

static void test_managed_stats() {
    // Everything fits: each page migrates exactly once however prefetch
    // split the work, nothing is evicted, and a second pass is fault-free.
    const size_t pages = 32, pg = 4096, n = pages * pg;
    sgDevPtr d = 0;
    void* hv = nullptr;
    CHECK_OK(sgMallocManaged(&d, &hv, n));
    auto* host = static_cast<uint8_t*>(hv);
    for (size_t i = 0; i < n; ++i) host[i] = uint8_t(i * 3 + 1);
    // Snapshot: comparing against `host` later would fault every page back
    // out of VRAM once userfaultfd is on, and the second pass below expects
    // those pages to still be resident.
    const std::vector<uint8_t> expect(host, host + n);
    uint8_t* out = nullptr;
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&out), n));
    sgStats_t s0{}, s1{}, s2{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgMemcpyD2H(out, d, n));
    CHECK_OK(sgGetStats(&s1));
    CHECK(std::memcmp(out, expect.data(), n) == 0);
    VmDelta v = vm_delta(s0, s1);
    CHECK(v.faults >= 1 && v.faults <= pages);
    CHECK(v.migrations == pages);
    CHECK(v.prefetches <= v.migrations);
    CHECK(v.migrations - v.prefetches <= v.faults); // every demand migration was a fault
    CHECK(v.evictions == 0 && v.writebacks == 0);
    CHECK(v.bytes == n);
    CHECK(v.tlb_misses >= 1);
    if (prefetch_window() == 0) CHECK(v.prefetches == 0 && v.faults == pages);

    CHECK_OK(sgMemcpyD2H(out, d, n));
    CHECK_OK(sgGetStats(&s2));
    v = vm_delta(s1, s2);
    CHECK(v.faults == 0 && v.migrations == 0 && v.bytes == 0);
    CHECK(v.tlb_hits + v.tlb_misses >= pages);

    // Under pressure: evictions happen, dirty ones are written back, and
    // bytes_migrated counts both directions.
    auto held = squeeze_vram(8);
    sgStats_t s3{}, s4{};
    CHECK_OK(sgGetStats(&s3));
    CHECK_OK(sgMemset(d, 0x3C, n)); // one command over all 32 pages, 8 frames
    CHECK_OK(sgDeviceSynchronize());
    CHECK_OK(sgGetStats(&s4));
    v = vm_delta(s3, s4);
    CHECK(v.evictions >= pages - 8);
    CHECK(v.writebacks >= 1 && v.writebacks <= v.evictions);
    // userfaultfd copies every evicted page back, clean or not. Without it
    // only dirty pages move.
    if (sgManagedCoherent()) CHECK(v.bytes == (v.migrations + v.evictions) * pg);
    else CHECK(v.bytes == (v.migrations + v.writebacks) * pg);
    CHECK_OK(sgMemcpyD2H(out, d, n));
    CHECK(std::all_of(out, out + n, [](uint8_t x) { return x == 0x3C; }));
    for (sgDevPtr p : held) CHECK_OK(sgFree(p));
    CHECK_OK(sgFreeHost(out));
    CHECK_OK(sgFree(d));
}

static void test_managed_prefetch_sequential() {
    // One copy walking 256 pages forward. With a window of P, each window
    // costs the fault that started it plus at most one more (the engine can
    // overtake a prefetch still in progress).
    const size_t pages = 256, pg = 4096, n = pages * pg;
    const uint64_t P = prefetch_window();
    sgDevPtr d = 0;
    void* hv = nullptr;
    CHECK_OK(sgMallocManaged(&d, &hv, n));
    auto* host = static_cast<uint8_t*>(hv);
    for (size_t i = 0; i < n; ++i) host[i] = uint8_t(i / pg + i);
    uint8_t* out = nullptr;
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&out), n));
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgMemcpyD2H(out, d, n));
    CHECK_OK(sgGetStats(&s1));
    CHECK(std::memcmp(out, host, n) == 0);
    const VmDelta v = vm_delta(s0, s1);
    CHECK(v.migrations == pages);
    if (P == 0) {
        CHECK(v.faults == pages);
        CHECK(v.prefetches == 0);
    } else {
        CHECK(v.faults <= 2 + 2 * (pages / (P + 1) + 1));
        CHECK(v.prefetches >= pages / 2);
    }
    std::printf("      [256-page walk, window %llu: %llu faults, %llu prefetched]\n",
                (unsigned long long)P, (unsigned long long)v.faults, (unsigned long long)v.prefetches);
    CHECK_OK(sgFreeHost(out));
    CHECK_OK(sgFree(d));
}

static void test_managed_prefetch_strided() {
    // Visit pages 0, 7, 14, ... (mod 256): page x-1 was touched 73 faults
    // before x, far outside the streams the driver tracks, so nothing may
    // look sequential and every page is its own fault.
    const size_t pages = 256, pg = 4096, n = pages * pg;
    sgDevPtr d = 0;
    void* hv = nullptr;
    CHECK_OK(sgMallocManaged(&d, &hv, n));
    auto* host = static_cast<uint8_t*>(hv);
    for (size_t i = 0; i < n; ++i) host[i] = uint8_t(i / pg);
    uint8_t* out = nullptr;
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&out), n));
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    for (size_t k = 0; k < pages; ++k) {
        const size_t p = (k * 7) % pages;
        CHECK_OK(sgMemcpyD2HAsync(out + p * pg, d + p * pg, pg, nullptr));
    }
    CHECK_OK(sgDeviceSynchronize());
    CHECK_OK(sgGetStats(&s1));
    CHECK(std::memcmp(out, host, n) == 0);
    const VmDelta v = vm_delta(s0, s1);
    CHECK(v.prefetches == 0);
    CHECK(v.faults == pages);
    CHECK(v.migrations == pages);
    CHECK_OK(sgFreeHost(out));
    CHECK_OK(sgFree(d));
}

static void test_managed_prefetch_stays_in_allocation() {
    // Two managed allocations back to back in VA. Walking all of `a` must
    // not migrate any page of `b`, however large the window.
    const size_t pages = 8, pg = 4096, n = pages * pg;
    sgDevPtr a = 0, b = 0;
    void *ha = nullptr, *hb = nullptr;
    CHECK_OK(sgMallocManaged(&a, &ha, n));
    CHECK_OK(sgMallocManaged(&b, &hb, n));
    std::memset(ha, 0x61, n);
    std::memset(hb, 0x62, n);
    std::vector<uint8_t> out(n);
    sgStats_t s0{}, s1{}, s2{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgMemcpyD2H(out.data(), a, n));
    CHECK_OK(sgGetStats(&s1));
    CHECK(std::all_of(out.begin(), out.end(), [](uint8_t x) { return x == 0x61; }));
    CHECK(vm_delta(s0, s1).migrations == pages);
    CHECK_OK(sgMemcpyD2H(out.data(), b, n));
    CHECK_OK(sgGetStats(&s2));
    CHECK(std::all_of(out.begin(), out.end(), [](uint8_t x) { return x == 0x62; }));
    CHECK(vm_delta(s1, s2).migrations == pages);
    CHECK(vm_delta(s1, s2).faults >= 1);
    CHECK_OK(sgFree(a));
    CHECK_OK(sgFree(b));
}

static void test_managed_prefetch_under_pressure() {
    // Three channels stream 48-page buffers through 6 free frames at once.
    // Prefetch may evict, but never so much that a channel's demand page is
    // gone before it resumes; a livelock here is a ctest timeout.
    const size_t frames = 6, pages = 48, pg = 4096, n = pages * pg;
    const int S = 3;
    sgDevPtr d[S] = {};
    void* h[S] = {};
    uint8_t* out[S] = {};
    sgStream_t st[S] = {};
    for (int s = 0; s < S; ++s) {
        CHECK_OK(sgMallocManaged(&d[s], &h[s], n));
        for (size_t i = 0; i < n; ++i) static_cast<uint8_t*>(h[s])[i] = uint8_t(s * 50 + i / pg);
        CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&out[s]), n));
        CHECK_OK(sgStreamCreate(&st[s]));
    }
    auto held = squeeze_vram(frames);
    // One stream alone: a window larger than the free frames would evict
    // its own prefetched pages before the copy reached them, and they would
    // migrate twice. Capped, every page moves in exactly once.
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgMemcpyD2HAsync(out[0], d[0], n, st[0]));
    CHECK_OK(sgStreamSynchronize(st[0]));
    CHECK_OK(sgGetStats(&s1));
    CHECK(std::memcmp(out[0], h[0], n) == 0);
    CHECK(vm_delta(s0, s1).migrations == pages);
    for (int rep = 0; rep < 3; ++rep) {
        for (int s = 0; s < S; ++s) {
            std::memset(out[s], 0, n);
            CHECK_OK(sgMemcpyD2HAsync(out[s], d[s], n, st[s]));
        }
        for (int s = 0; s < S; ++s) {
            CHECK_OK(sgStreamSynchronize(st[s]));
            CHECK(std::memcmp(out[s], h[s], n) == 0);
        }
    }
    for (sgDevPtr p : held) CHECK_OK(sgFree(p));
    for (int s = 0; s < S; ++s) {
        CHECK_OK(sgStreamDestroy(st[s]));
        CHECK_OK(sgFreeHost(out[s]));
        CHECK_OK(sgFree(d[s]));
    }
}

// ---- stage 5 slice 5: host faults migrate a page back --------------------

static thread_local sigjmp_buf tl_bus_jmp;
static thread_local volatile sig_atomic_t tl_bus_armed = 0;

static void on_sigbus(int) {
    if (tl_bus_armed) siglongjmp(tl_bus_jmp, 1);
}

struct HoldHostFaults {
    HoldHostFaults() { sgHoldHostFaults(1); }
    ~HoldHostFaults() { sgHoldHostFaults(0); }
};

static void test_managed_pingpong() {
    // Host write after a device write is visible to the next device read,
    // and a device write is visible to a host read with no pressure eviction.
    if (!sgManagedCoherent()) return;
    const size_t n = 4096;
    sgDevPtr d = 0;
    void* hv = nullptr;
    CHECK_OK(sgMallocManaged(&d, &hv, n));
    auto* host = static_cast<uint8_t*>(hv);
    std::memset(host, 0x01, n);
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgMemset(d, 0x11, n));
    CHECK_OK(sgDeviceSynchronize()); // sgMemset only queues the fill
    CHECK(host[0] == 0x11);
    host[0] = 0x22;
    std::vector<uint8_t> out(n);
    CHECK_OK(sgMemcpyD2H(out.data(), d, n));
    CHECK(out[0] == 0x22);
    bool rest = true;
    for (size_t i = 1; i < n && rest; ++i) rest = out[i] == 0x11;
    CHECK(rest);
    CHECK_OK(sgMemset(d, 0x33, n));
    CHECK_OK(sgDeviceSynchronize());
    CHECK(host[3] == 0x33);
    CHECK_OK(sgGetStats(&s1));
    CHECK(s1.um_host_faults - s0.um_host_faults >= 2);
    CHECK(s1.um_evictions - s0.um_evictions >= 2);
    CHECK_OK(sgFree(d));
}

static void test_managed_host_other_page() {
    // A host fault on page 0 while commands run on page 1 of the same
    // allocation. The two pages must not trade bytes.
    if (!sgManagedCoherent()) return;
    const size_t pg = 4096;
    sgDevPtr d = 0;
    void* hv = nullptr;
    CHECK_OK(sgMallocManaged(&d, &hv, pg * 2));
    auto* host = static_cast<uint8_t*>(hv);
    std::memset(host, 0x11, pg * 2);
    CHECK_OK(sgMemset(d, 0x11, pg * 2));
    CHECK_OK(sgDeviceSynchronize());
    sgStream_t s = nullptr;
    CHECK_OK(sgStreamCreate(&s));
    std::atomic<int> started{0};
    std::thread th([&] {
        started.store(1, std::memory_order_release);
        host[0] = 0x42;
        host[1] = 0x42;
    });
    while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
    for (int i = 0; i < 200; ++i) CHECK_OK(sgMemsetAsync(d + pg, 0x44, pg, s));
    th.join();
    CHECK_OK(sgStreamSynchronize(s));
    std::vector<uint8_t> out(pg);
    CHECK_OK(sgMemcpyD2H(out.data(), d, pg));
    CHECK(out[0] == 0x42 && out[1] == 0x42);
    CHECK_OK(sgMemcpyD2H(out.data(), d + pg, pg));
    bool page1 = true;
    for (uint8_t v : out) page1 = page1 && v == 0x44;
    CHECK(page1);
    CHECK_OK(sgStreamDestroy(s));
    CHECK_OK(sgFree(d));
}

static void test_managed_free_during_host_fault() {
    // sgFree while a host thread is blocked in the fault. The handler is
    // held off mem_lock (sgHoldHostFaults) so recycle's unregister runs
    // first and has to wake that thread. A signal from the racing munmap
    // is a successful wake; a hang is not.
    if (!sgManagedCoherent()) return;
    const size_t n = 4096;
    sgDevPtr d = 0;
    void* hv = nullptr;
    CHECK_OK(sgMallocManaged(&d, &hv, n));
    auto* host = static_cast<uint8_t*>(hv);
    CHECK_OK(sgMemset(d, 0xAB, n));
    CHECK_OK(sgDeviceSynchronize());
    HoldHostFaults hold;

    // A racing munmap delivers SIGSEGV; a fault on a file mapping would be
    // SIGBUS. Either one means the blocked reader was woken.
    struct sigaction sa{}, old_bus{}, old_segv{};
    sa.sa_handler = on_sigbus;
    sigaction(SIGBUS, &sa, &old_bus);
    sigaction(SIGSEGV, &sa, &old_segv);

    sgStats_t s0{};
    CHECK_OK(sgGetStats(&s0));
    std::atomic<int> value{-1};
    std::atomic<int> saw_bus{0};
    std::thread th([&] {
        if (sigsetjmp(tl_bus_jmp, 1) == 0) {
            tl_bus_armed = 1;
            value.store(host[0], std::memory_order_release);
        } else {
            saw_bus.store(1, std::memory_order_release);
        }
        tl_bus_armed = 0;
    });

    bool pending = false;
    for (int i = 0; i < 1000000 && !pending; ++i) {
        sgStats_t s{};
        if (sgGetStats(&s) == SG_OK && s.um_host_faults > s0.um_host_faults) pending = true;
        else std::this_thread::yield();
    }
    CHECK(pending);
    CHECK_OK(sgFree(d));
    sgHoldHostFaults(0);
    th.join();
    sigaction(SIGBUS, &old_bus, nullptr);
    sigaction(SIGSEGV, &old_segv, nullptr);
    if (!saw_bus.load()) {
        const int v = value.load();
        CHECK(v == 0xAB || v == 0);
    }
}

static void test_tlb_reach() {
    // The compute TLB has 64 entries with LRU replacement. A working set
    // inside its reach hits on every later pass; a cyclic walk over twice
    // that misses on every page (LRU's worst case).
    for (size_t pages : {size_t{32}, size_t{128}}) {
        const size_t n = pages * 4096;
        sgDevPtr d = 0;
        CHECK_OK(sgMalloc(&d, n));
        CHECK_OK(sgMemset(d, 1, n));
        CHECK_OK(sgDeviceSynchronize());
        sgStats_t s0{}, s1{};
        CHECK_OK(sgGetStats(&s0));
        CHECK_OK(sgMemset(d, 2, n));
        CHECK_OK(sgDeviceSynchronize());
        CHECK_OK(sgGetStats(&s1));
        const uint64_t hits = s1.engine_tlb_hits[0] - s0.engine_tlb_hits[0];
        const uint64_t misses = s1.engine_tlb_misses[0] - s0.engine_tlb_misses[0];
        if (pages <= 64) CHECK(hits == pages && misses == 0);
        else CHECK(hits == 0 && misses == pages);
        CHECK_OK(sgFree(d));
    }
}

int main() {
    CHECK(sgInit() == SG_OK);
    CHECK(sgInit() == SG_ERR_ALREADY_INITIALIZED);

    struct { const char* name; void (*fn)(); } tests[] = {
        {"memcpy_roundtrip_small", [] { test_memcpy_roundtrip(1); test_memcpy_roundtrip(4096); }},
        {"memcpy_roundtrip_multi_chunk", [] { test_memcpy_roundtrip((4u << 20) + 12345); }},
        {"memcpy_roundtrip_large", [] { test_memcpy_roundtrip(64u << 20); }},
        {"memset_and_d2d", test_memset_and_d2d},
        {"vadd", test_vadd},
        {"gemm", test_gemm},
        {"validation", test_validation},
        {"alloc_reuse_and_oom", test_alloc_reuse_and_oom},
        {"async_ordering", test_async_ordering},
        {"pinned_roundtrip", [] { test_pinned_roundtrip(1); test_pinned_roundtrip(4096); test_pinned_roundtrip(64u << 20); }},
        {"host_register", test_host_register},
        {"async_pinned_pipeline", test_async_pinned_pipeline},
        {"deferred_free", test_deferred_free},
        {"concurrent_pinned_submitters", test_concurrent_pinned_submitters},
        {"stream_cross_engine_ordering", test_stream_cross_engine_ordering},
        {"two_streams_independent", test_two_streams_independent},
        {"event_dependency", test_event_dependency},
        {"stream_sync_is_per_stream", test_stream_sync_is_per_stream},
        {"pageable_on_streams_and_deferred_copy_free", test_pageable_on_streams_and_deferred_copy_free},
        {"engine_rejections", test_engine_rejections},
        {"concurrent_streams", test_concurrent_streams},
        {"wait_policy_stats", test_wait_policy_stats},
        {"blocking_lost_wakeup_stress", test_blocking_lost_wakeup_stress},
        {"shared_stream_many_producers", test_shared_stream_many_producers},
        {"alloc_storm_during_submits", test_alloc_storm_during_submits},
        {"pin_storm_during_copies", test_pin_storm_during_copies},
        {"ticket_publish_storm", test_ticket_publish_storm},
        {"gating_no_missed_doorbells", test_gating_no_missed_doorbells},
        {"gating_power_when_idle", test_gating_power_when_idle},
        {"concurrent_submitters", test_concurrent_submitters},
        {"managed_migrate_in", test_managed_migrate_in},
        {"managed_two_channels", test_managed_two_channels},
        {"managed_evict_writeback", test_managed_evict_writeback},
        {"managed_oversubscribed_compute", test_managed_oversubscribed_compute},
        {"managed_tlb_shootdown", test_managed_tlb_shootdown},
        {"managed_stats", test_managed_stats},
        {"managed_prefetch_sequential", test_managed_prefetch_sequential},
        {"managed_prefetch_strided", test_managed_prefetch_strided},
        {"managed_prefetch_stays_in_allocation", test_managed_prefetch_stays_in_allocation},
        {"managed_prefetch_under_pressure", test_managed_prefetch_under_pressure},
        {"managed_pingpong", test_managed_pingpong},
        {"managed_host_other_page", test_managed_host_other_page},
        {"managed_free_during_host_fault", test_managed_free_during_host_fault},
        {"tlb_reach", test_tlb_reach},
    };
    for (auto& t : tests) {
        int before = g_failures;
        t.fn();
        std::printf("[%s] %s\n", g_failures == before ? " OK " : "FAIL", t.name);
    }

    CHECK(sgShutdown() == SG_OK);
    CHECK(sgShutdown() == SG_ERR_NOT_INITIALIZED);
    std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
