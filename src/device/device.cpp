#include "device/device.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "common/clock.h"
#include "common/cpu.h"

namespace softgpu::device {

namespace {

bool is_copy_op(uint32_t op) {
    return op == SG_OP_COPY_H2D || op == SG_OP_COPY_D2H || op == SG_OP_COPY_D2D;
}
bool is_compute_op(uint32_t op) {
    return op == SG_OP_FILL || op == SG_OP_VADD_F32 || op == SG_OP_GEMM_F32;
}

uint64_t thread_cpu_ns() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

} // namespace

Device::Device(uint64_t vram_bytes, uint32_t copy_engines, uint32_t channels)
    : num_engines_(1 + copy_engines), num_channels_(channels), vram_(new uint8_t[vram_bytes]),
      vram_size_(vram_bytes), mmu_(vram_bytes) {
    // Touch every page so first-use page faults do not show up as device
    // "busy" time in the first benchmark that runs.
    std::memset(vram_.get(), 0, vram_bytes);
}

Device::~Device() { power_off(); }

void Device::power_on(int cpu) {
    if (running_.exchange(true)) return;
    for (uint32_t e = 0; e < num_engines_; ++e) engines_.emplace_back(&Device::run, this, e);
#ifdef __linux__
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(engines_[0].native_handle(), sizeof set, &set);
    }
#else
    (void)cpu;
#endif
}

void Device::power_off() {
    if (!running_.exchange(false)) return;
    for (uint32_t e = 0; e < num_engines_; ++e) bell_[e].signal(); // wake gated engines so they exit
    for (auto& t : engines_)
        if (t.joinable()) t.join();
    engines_.clear();
}

void Device::wake_blocked_sleepers(uint32_t self) {
    if (!no_fence_) std::atomic_thread_fence(std::memory_order_seq_cst);
    if (num_asleep_blocked_.load(std::memory_order_relaxed) == 0) return;
    for (uint32_t e = 0; e < num_engines_; ++e)
        if (e != self && asleep_blocked_[e].load(std::memory_order_relaxed)) {
            wake_stamp_[e].store(now_cycles(), std::memory_order_relaxed);
            bell_[e].signal();
        }
}

void Device::reset_stats() {
    for (uint32_t e = 0; e < num_engines_; ++e) {
        auto& st = stats_[e];
        st.busy_cycles.store(0, std::memory_order_relaxed);
        st.wait_cycles.store(0, std::memory_order_relaxed);
        st.idle_cycles.store(0, std::memory_order_relaxed);
        st.cmds_executed.store(0, std::memory_order_relaxed);
        st.batches.store(0, std::memory_order_relaxed);
        st.irqs.store(0, std::memory_order_relaxed);
        st.sleep_cycles.store(0, std::memory_order_relaxed);
        st.wakeups.store(0, std::memory_order_relaxed);
        st.missed_doorbells.store(0, std::memory_order_relaxed);
        st.stats_gen.fetch_add(1, std::memory_order_release);
    }
    mmu_.reset_stats();
}

void Device::push_fault(const Fault& f) {
    {
        std::lock_guard<std::mutex> g(fault_mu_);
        faults_.push_back(f);
    }
    fault_irq_.signal();
}

bool Device::pop_fault(Fault* out) {
    std::lock_guard<std::mutex> g(fault_mu_);
    if (faults_.empty()) return false;
    *out = faults_.front();
    faults_.pop_front();
    return true;
}

void Device::unpark(uint32_t engine, uint32_t channel) {
    ch_[engine][channel].parked.store(false, std::memory_order_seq_cst);
    doorbell(engine);
}

int Device::migrate_in(uint64_t va) {
    va = page_floor(va);
    const PageInfo pg = mmu_.page(va);
    if (pg.kind == PageInfo::Vram) return 0;
    if (pg.kind != PageInfo::Host || pg.host == 0) return -EFAULT;
    if (va >= vram_size_) return -EFAULT;
    std::memcpy(vram_.get() + va, reinterpret_cast<const void*>(pg.host), SG_PAGE_SIZE);
    return mmu_.make_resident(va, va);
}

// Translate [va, va+len) through the MMU and require a contiguous VRAM
// mapping (true for identity-mapped allocations). A host-resident page is a
// fault: the caller parks the channel and retries the command after the
// driver migrates that page. Not-present is a hard -EFAULT.
int Device::resolve(uint32_t engine, uint64_t va, uint64_t len, bool write, uint8_t** out,
                    uint64_t* fault_va) {
    if (len == 0) {
        *out = nullptr;
        return 0;
    }
    uint64_t base = 0;
    uint64_t left = len;
    uint64_t cur = va;
    while (left) {
        const Translate t = mmu_.translate(engine, cur, write);
        if (t.status == Translate::HostResident) {
            *fault_va = page_floor(cur);
            return kPageFault;
        }
        if (t.status != Translate::Ok) return -EFAULT;
        const uint64_t off = page_offset(cur);
        const uint64_t pa = t.pa + off;
        if (cur == va) base = pa;
        else if (pa != base + (cur - va)) return -EFAULT; // non-contiguous
        const uint64_t chunk = std::min(left, uint64_t{SG_PAGE_SIZE} - off);
        cur += chunk;
        left -= chunk;
    }
    if (base > vram_size_ || len > vram_size_ - base) return -EFAULT;
    *out = vram_.get() + base;
    return 0;
}

// One engine's loop: a runlist over its channels. Each round visits every
// channel with published work and runs it until it empties or its head is a
// WAIT_FENCE that is not yet satisfied — then moves on rather than blocking,
// so one stream's semaphore never stalls another stream's commands.
//
// STAGE 7: when no round has made progress for longer than the idle budget
// the engine power-gates: it marks itself asleep, re-checks every channel
// (Dekker: the driver stores PUT then loads `asleep`; we store `asleep`
// then load the PUTs, seq_cst fences on both sides), and sleeps on its
// doorbell futex. A 1 ms timeout is the safety net; a timeout wake that
// finds work is counted as a missed doorbell, which must never happen.
//
// Time accounting: executing = busy; a round that made no progress is
// charged to `wait` if some channel had pending work (all heads blocked) or
// to `idle` if none did; time asleep is `sleep` (and also idle or wait).
void Device::run(uint32_t engine) {
    EngineStats& st = stats_[engine];
    struct Cursor { const sg_cmd* ring; uint64_t mask; uint64_t get; };
    Cursor cur[SG_MAX_CHANNELS];
    for (uint32_t c = 0; c < num_channels_; ++c) {
        Channel& ch = ch_[engine][c];
        cur[c] = {reinterpret_cast<const sg_cmd*>(ch.ring_base), ch.ring_mask,
                  ch.get.load(std::memory_order_relaxed)};
    }
    uint64_t mark = now_cycles();
    uint32_t gen = st.stats_gen.load(std::memory_order_relaxed);
    bool was_active = false;
    uint64_t gap_start = mark;       // when the current no-progress stretch began
    uint64_t gap_ewma = 0;           // adaptive: typical idle gap length (ns)
    uint64_t seen_put[SG_MAX_CHANNELS] = {};
    uint64_t last_cpu_pub = mark;
    bool woke = false; // the previous round ended a sleep that was signalled
    st.cpu_ns.store(thread_cpu_ns(), std::memory_order_relaxed);

    while (running_.load(std::memory_order_relaxed)) {
        bool progress = false, pending = false;
        for (uint32_t c = 0; c < num_channels_; ++c) {
            Channel& ch = ch_[engine][c];
            Cursor& k = cur[c];
            // Acquire pairs with the driver's release store of `put` and
            // makes every slot below it visible to this thread.
            uint64_t put = ch.put.load(std::memory_order_acquire);
            seen_put[c] = put;
            if (ch.parked.load(std::memory_order_acquire)) {
                // Head command is waiting on a migration. Other channels
                // still run. Unpark rings the doorbell if we sleep.
                if (put != k.get) pending = true;
                continue;
            }
            const int32_t fail = ch.fault_fail.exchange(0, std::memory_order_acquire);
            if (fail != 0) {
                int expected = 0;
                ch.sticky_error.compare_exchange_strong(expected, fail, std::memory_order_relaxed);
                if (put != k.get) {
                    ++k.get;
                    ch.get.store(k.get, std::memory_order_release);
                    progress = true;
                    const uint64_t tgt = ch.irq_target.load(std::memory_order_acquire);
                    if (tgt != 0 && k.get >= tgt) {
                        ch.irq_target.store(0, std::memory_order_relaxed);
                        const uint64_t t = now_cycles();
                        ch.last_irq_cycles.store(t, std::memory_order_relaxed);
                        note_irq(t);
                        st.irqs.fetch_add(1, std::memory_order_relaxed);
                        irq_.signal();
                    }
                }
                continue;
            }
            if (put == k.get) continue;
            if (put < k.get) {
                // PUT behind GET is a driver bug (a doorbell went backwards).
                // Real hardware would fault the channel; so do we, rather
                // than executing whatever is in the ring.
                int expected = 0;
                ch.sticky_error.compare_exchange_strong(expected, -EIO, std::memory_order_relaxed);
                if (std::getenv("SG_TRACE")) std::fprintf(stderr, "[dev] engine %u ch %u PUT %llu < GET %llu\n", engine, c, (unsigned long long)put, (unsigned long long)k.get);
                continue;
            }
            pending = true;
            while (k.get != put) {
                const sg_cmd& cmd = k.ring[k.get & k.mask];
                int rc = 0;
                if (cmd.opcode == SG_OP_WAIT_FENCE) {
                    // Semaphore acquire. Cannot deadlock: the driver only
                    // publishes waits on fences whose commands were enqueued
                    // earlier (ADR 003).
                    if (cmd.arg0 >= num_engines_ || cmd.arg1 >= num_channels_) {
                        rc = -EINVAL;
                    } else if (ch_[cmd.arg0][cmd.arg1].get.load(std::memory_order_acquire) < cmd.src1) {
                        break; // blocked: leave this channel for now
                    }
                } else {
                    const uint64_t t0 = now_cycles();
                    uint64_t fault_va = 0;
                    rc = execute(engine, cmd, &fault_va);
                    st.busy_cycles.fetch_add(now_cycles() - t0, std::memory_order_relaxed);
                    if (rc == kPageFault) {
                        // Park before publishing the fault, so the handler
                        // cannot unpark a channel we then park again.
                        ch.parked.store(true, std::memory_order_seq_cst);
                        push_fault(Fault{fault_va, engine, c});
                        break;
                    }
                }
                if (rc != 0) {
                    int expected = 0;
                    ch.sticky_error.compare_exchange_strong(expected, rc, std::memory_order_relaxed);
                    if (std::getenv("SG_TRACE")) std::fprintf(stderr, "[dev] engine %u ch %u cmd#%llu opcode %u rc %d\n", engine, c, (unsigned long long)k.get, cmd.opcode, rc);
                }
                ++k.get;
                progress = true;
                st.cmds_executed.fetch_add(1, std::memory_order_relaxed);
                // Retire each command individually (release publishes its
                // results) so a host or another channel waiting on an early
                // fence is not held up by the rest. Also frees the slot.
                ch.get.store(k.get, std::memory_order_release);
                // A gated engine may be waiting on this fence at the head of
                // one of its channels.
                wake_blocked_sleepers(engine);
                // Interrupt: only when the driver armed one for a fence we
                // just passed. Clearing before signalling means a waiter that
                // re-arms after waking cannot miss a later fence.
                const uint64_t tgt = ch.irq_target.load(std::memory_order_acquire);
                if (tgt != 0 && k.get >= tgt) {
                    ch.irq_target.store(0, std::memory_order_relaxed);
                    const uint64_t t = now_cycles();
                    ch.last_irq_cycles.store(t, std::memory_order_relaxed);
                    note_irq(t);
                    st.irqs.fetch_add(1, std::memory_order_relaxed);
                    irq_.signal();
                }
                if (k.get == put) put = ch.put.load(std::memory_order_acquire);
            }
        }

        const uint64_t now = now_cycles();
        // Keep the CPU-time export fresh (about once a millisecond, busy or
        // idle) so the power proxy is accurate over any window. This is a
        // syscall on Linux, hence the rate limit.
        if (now - last_cpu_pub > 1000000) {
            st.cpu_ns.store(thread_cpu_ns(), std::memory_order_relaxed);
            last_cpu_pub = now;
        }
        if (progress) {
            if (!was_active) {
                st.batches.fetch_add(1, std::memory_order_relaxed);
                // Adaptive: remember how long we were idle before this work.
                // If we were asleep, measure to the doorbell's timestamp, not
                // to now: the wake-up latency is the policy's own penalty and
                // feeding it back made the estimator lock itself into
                // sleeping (the same failure stage 2's wait estimator had).
                // (No CPU-time publish here: CLOCK_THREAD_CPUTIME_ID is a
                // syscall on Linux and would tax every command after a gap.)
                uint64_t end = now;
                if (woke) {
                    const uint64_t stamp = wake_stamp_[engine].load(std::memory_order_relaxed);
                    if (stamp > gap_start && stamp < now) end = stamp;
                }
                const uint64_t gap = end - gap_start;
                gap_ewma = gap_ewma == 0 ? gap : (3 * gap_ewma + gap) / 4;
            }
            was_active = true;
            woke = false;
        } else {
            // A stats reset while idle restarts the timer, so time from
            // before the reset is not charged to the new window.
            const uint32_t g = st.stats_gen.load(std::memory_order_acquire);
            if (g != gen) {
                gen = g;
                mark = now;
            }
            (pending ? st.wait_cycles : st.idle_cycles).fetch_add(now - mark, std::memory_order_relaxed);
            if (was_active) gap_start = now;
            was_active = false;

            // Idle gating: how long to keep spinning before sleeping.
            uint64_t budget;
            switch (idle_policy_) {
            case SG_IDLE_SPIN:  budget = ~0ull; break;
            case SG_IDLE_SLEEP: budget = 0; break;
            case SG_IDLE_ADAPTIVE:
                // Gaps have been long: sleep soon. Short: spin through them.
                budget = gap_ewma == 0 ? idle_ns_
                       : gap_ewma > idle_ns_ ? 2000
                       : std::max<uint64_t>(2000, std::min<uint64_t>(2 * gap_ewma, idle_ns_));
                break;
            default: budget = idle_ns_; break;
            }
            if (budget != ~0ull && now - gap_start >= budget) {
                // Arm, then re-check (Dekker with Device::doorbell()).
                const uint32_t seen = bell_[engine].seq();
                asleep_[engine].store(true, no_fence_ ? std::memory_order_relaxed : std::memory_order_seq_cst);
                if (pending) {
                    asleep_blocked_[engine].store(true, std::memory_order_seq_cst);
                    num_asleep_blocked_.fetch_add(1, std::memory_order_seq_cst);
                }
                if (!no_fence_) std::atomic_thread_fence(std::memory_order_seq_cst);
                bool work = false;
                for (uint32_t c = 0; c < num_channels_ && !work; ++c) {
                    if (ch_[engine][c].parked.load(std::memory_order_seq_cst)) continue;
                    const uint64_t p = ch_[engine][c].put.load(no_fence_ ? std::memory_order_relaxed : std::memory_order_seq_cst);
                    if (p != seen_put[c]) work = true; // a doorbell arrived meanwhile
                    else if (p != cur[c].get) {
                        // Blocked head: has its fence passed since we looked?
                        const sg_cmd& head = cur[c].ring[cur[c].get & cur[c].mask];
                        if (head.opcode != SG_OP_WAIT_FENCE || head.arg0 >= num_engines_ ||
                            head.arg1 >= num_channels_ ||
                            ch_[head.arg0][head.arg1].get.load(std::memory_order_seq_cst) >= head.src1)
                            work = true;
                    }
                }
                if (!work) {
                    const uint64_t t_sleep = now_cycles();
                    st.cpu_ns.store(thread_cpu_ns(), std::memory_order_relaxed);
                    bell_[engine].wait(seen, 1000000); // 1 ms safety net
                    const uint64_t t_wake = now_cycles();
                    st.sleep_cycles.fetch_add(t_wake - t_sleep, std::memory_order_relaxed);
                    st.wakeups.fetch_add(1, std::memory_order_relaxed);
                    woke = bell_[engine].seq() != seen;
                    if (!woke) {
                        // Timeout, not a signal. If there is work, someone
                        // rang a doorbell we did not hear: a lost wake-up.
                        for (uint32_t c = 0; c < num_channels_; ++c)
                            if (ch_[engine][c].put.load(std::memory_order_acquire) != seen_put[c]) {
                                st.missed_doorbells.fetch_add(1, std::memory_order_relaxed);
                                break;
                            }
                    }
                    (pending ? st.wait_cycles : st.idle_cycles).fetch_add(t_wake - t_sleep, std::memory_order_relaxed);
                    mark = t_wake;
                }
                asleep_[engine].store(false, std::memory_order_seq_cst);
                if (pending) {
                    asleep_blocked_[engine].store(false, std::memory_order_seq_cst);
                    num_asleep_blocked_.fetch_sub(1, std::memory_order_seq_cst);
                }
                if (!work) { continue; }
            }
            cpu_relax();
        }
        mark = now;
    }
}

int Device::execute(uint32_t engine, const sg_cmd& c, uint64_t* fault_va) {
    // Engine class check: a real copy engine has no ALUs and a compute engine
    // no DMA. The driver validates this too; the device is the last line.
    if (engine == SG_ENGINE_COMPUTE ? is_copy_op(c.opcode) : is_compute_op(c.opcode))
        return -EINVAL;

    switch (c.opcode) {
    case SG_OP_NOP:
        return 0;

    case SG_OP_FILL: {
        uint8_t* dst = nullptr;
        if (int rc = resolve(engine, c.dst, c.size, true, &dst, fault_va)) return rc;
        std::memset(dst, static_cast<int>(c.value & 0xff), c.size);
        return 0;
    }

    // Host addresses in COPY_H2D/COPY_D2H are trusted, exactly as a DMA
    // engine trusts the bus addresses its driver programs. The driver is
    // responsible for only ever handing it its staging buffers or pinned
    // user memory. Device addresses go through the MMU.
    case SG_OP_COPY_H2D: {
        uint8_t* dst = nullptr;
        if (int rc = resolve(engine, c.dst, c.size, true, &dst, fault_va)) return rc;
        std::memcpy(dst, reinterpret_cast<const void*>(c.src0), c.size);
        return 0;
    }

    case SG_OP_COPY_D2H: {
        uint8_t* src = nullptr;
        if (int rc = resolve(engine, c.src0, c.size, false, &src, fault_va)) return rc;
        std::memcpy(reinterpret_cast<void*>(c.dst), src, c.size);
        return 0;
    }

    case SG_OP_COPY_D2D: {
        uint8_t *dst = nullptr, *src = nullptr;
        if (int rc = resolve(engine, c.dst, c.size, true, &dst, fault_va)) return rc;
        if (int rc = resolve(engine, c.src0, c.size, false, &src, fault_va)) return rc;
        std::memmove(dst, src, c.size);
        return 0;
    }

    case SG_OP_VADD_F32: {
        const uint64_t bytes = uint64_t{c.arg0} * sizeof(float);
        uint8_t *dst = nullptr, *s0 = nullptr, *s1 = nullptr;
        if (int rc = resolve(engine, c.dst, bytes, true, &dst, fault_va)) return rc;
        if (int rc = resolve(engine, c.src0, bytes, false, &s0, fault_va)) return rc;
        if (int rc = resolve(engine, c.src1, bytes, false, &s1, fault_va)) return rc;
        const float* a = reinterpret_cast<const float*>(s0);
        const float* b = reinterpret_cast<const float*>(s1);
        float* out = reinterpret_cast<float*>(dst);
        for (uint32_t i = 0; i < c.arg0; ++i) out[i] = a[i] + b[i];
        return 0;
    }

    case SG_OP_GEMM_F32: {
        const uint64_t m = c.arg0, n = c.arg1, k = c.arg2;
        uint8_t *dst = nullptr, *s0 = nullptr, *s1 = nullptr;
        if (int rc = resolve(engine, c.src0, m * k * sizeof(float), false, &s0, fault_va)) return rc;
        if (int rc = resolve(engine, c.src1, k * n * sizeof(float), false, &s1, fault_va)) return rc;
        if (int rc = resolve(engine, c.dst, m * n * sizeof(float), true, &dst, fault_va)) return rc;
        const float* a = reinterpret_cast<const float*>(s0);
        const float* b = reinterpret_cast<const float*>(s1);
        float* out = reinterpret_cast<float*>(dst);
        // i-k-j ordering keeps the inner loop streaming over contiguous rows
        // of B and C. Deliberately no blocking; this is the reference engine.
        for (uint64_t i = 0; i < m; ++i) {
            float* crow = out + i * n;
            for (uint64_t j = 0; j < n; ++j) crow[j] = 0.0f;
            for (uint64_t p = 0; p < k; ++p) {
                const float aip = a[i * k + p];
                const float* brow = b + p * n;
                for (uint64_t j = 0; j < n; ++j) crow[j] += aip * brow[j];
            }
        }
        return 0;
    }

    default:
        return -EINVAL;
    }
}

} // namespace softgpu::device
