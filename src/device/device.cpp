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

// resolve(): resident but not one contiguous run. Internal; never retired.
constexpr int kSplit = 2;

Device::Device(uint64_t vram_bytes, uint64_t va_bytes, uint32_t copy_engines, uint32_t channels)
    : num_engines_(1 + copy_engines), num_channels_(channels), vram_(new uint8_t[vram_bytes]),
      vram_size_(vram_bytes), mmu_(va_bytes, vram_bytes) {
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
        st.faults.store(0, std::memory_order_relaxed);
        st.preempts.store(0, std::memory_order_relaxed);
        st.stats_gen.fetch_add(1, std::memory_order_release);
    }
    mmu_.reset_stats();
}

int Device::set_channel_priority(uint32_t engine, uint32_t channel, int32_t priority) {
    if (engine >= num_engines_ || channel >= num_channels_) return -EINVAL;
    if (priority > SG_PRIORITY_LEAST || priority < SG_PRIORITY_GREATEST) return -EINVAL;
    ch_[engine][channel].priority.store(priority, std::memory_order_relaxed);
    bool mixed = false;
    for (uint32_t e = 0; e < num_engines_ && !mixed; ++e)
        for (uint32_t c = 0; c < num_channels_; ++c)
            if (ch_[e][c].priority.load(std::memory_order_relaxed) != SG_PRIORITY_LEAST) mixed = true;
    // Release pairs with the run loop's acquire: the priority stores above
    // are visible once an engine observes mixed == true.
    mixed_priority_.store(mixed, std::memory_order_release);
    return 0;
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

int Device::copy_in(uint64_t pa, const void* host) {
    if (page_offset(pa) || pa >= vram_size_ || !host) return -EFAULT;
    std::memcpy(vram_.get() + pa, host, SG_PAGE_SIZE);
    return 0;
}

int Device::copy_out(uint64_t pa, void* host) {
    if (page_offset(pa) || pa >= vram_size_ || !host) return -EFAULT;
    std::memcpy(host, vram_.get() + pa, SG_PAGE_SIZE);
    return 0;
}

// Dekker with the run loop: the caller has already changed the PTE and
// cleared the TLB tags; the engine marks itself in-command, fences, then
// translates. Either the engine's translate sees the shootdown, or this
// load sees an odd sequence and waits for that command to finish.
void Device::quiesce() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for (uint32_t e = 0; e < num_engines_; ++e) {
        const uint64_t s = access_[e].v.load(std::memory_order_acquire);
        if (!(s & 1)) continue;
        while (access_[e].v.load(std::memory_order_acquire) == s) cpu_relax();
    }
}

// Translate page by page from `va` while frames stay contiguous. The page
// after the run is translated too (that is how the run ends); its fault, if
// any, is reported by the next call, once the run before it is applied.
int Device::run_at(uint32_t engine, uint64_t va, uint64_t len, bool write, uint8_t** out,
                   uint64_t* n, uint64_t* fault_va) {
    uint64_t base = 0, got = 0;
    while (got < len) {
        const uint64_t cur = va + got;
        const Translate t = mmu_.translate(engine, cur, write);
        if (t.status != Translate::Ok) {
            if (got) break;
            if (t.status == Translate::HostResident) {
                *fault_va = page_floor(cur);
                return kPageFault;
            }
            return -EFAULT;
        }
        const uint64_t pa = t.pa + page_offset(cur);
        if (got == 0) base = pa;
        else if (pa != base + got) break;
        got += std::min(len - got, uint64_t{SG_PAGE_SIZE} - page_offset(cur));
    }
    if (base > vram_size_ || got > vram_size_ - base) return -EFAULT;
    *out = vram_.get() + base;
    *n = got;
    return 0;
}

int Device::resolve(uint32_t engine, uint64_t va, uint64_t len, bool write, uint8_t** out,
                    uint64_t* fault_va) {
    if (len == 0) {
        *out = nullptr;
        return 0;
    }
    uint64_t n = 0;
    if (int rc = run_at(engine, va, len, write, out, &n, fault_va)) return rc;
    return n == len ? 0 : kSplit;
}

template <class Op>
int Device::for_runs(uint32_t engine, Replay& r, uint64_t va, uint64_t len, bool write,
                     uint64_t* fault_va, Op op) {
    while (r.done < len) {
        uint8_t* p = nullptr;
        uint64_t n = 0;
        if (int rc = run_at(engine, va + r.done, len - r.done, write, &p, &n, fault_va)) return rc;
        op(p, r.done, n);
        r.done += n;
    }
    return 0;
}

// One engine's loop: a runlist over its channels. Each visit runs the
// highest-priority channel that can execute (a WAIT_FENCE whose target has
// not passed is skipped, so one stream's semaphore never stalls another
// stream's commands). Equal priorities are round-robin. A visit ends at a
// command boundary when the timeslice expires and someone else can run, or
// when a strictly higher-priority channel becomes runnable. A lone channel
// is not sliced: there is nothing to switch to.
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
    uint64_t aseq = access_[engine].v.load(std::memory_order_relaxed);
    st.cpu_ns.store(thread_cpu_ns(), std::memory_order_relaxed);
    uint32_t rr = 0; // next equal-priority channel to prefer

    // Head command cannot execute yet: a WAIT whose target fence has not passed.
    // An out-of-range WAIT is runnable so the visit can retire it as -EINVAL.
    auto wait_blocked = [&](const sg_cmd& cmd) {
        return cmd.opcode == SG_OP_WAIT_FENCE && cmd.arg0 < num_engines_ && cmd.arg1 < num_channels_ &&
               ch_[cmd.arg0][cmd.arg1].get.load(std::memory_order_acquire) < cmd.src1;
    };
    // Another channel can execute now, and its priority compares as asked.
    // `better` selects a strictly higher priority (smaller value); otherwise
    // only the same priority, which is who a timeslice yields to.
    auto other_runnable = [&](uint32_t self, int32_t self_prio, bool better) {
        for (uint32_t c = 0; c < num_channels_; ++c) {
            if (c == self) continue;
            Channel& o = ch_[engine][c];
            const int32_t prio = o.priority.load(std::memory_order_relaxed);
            if (better ? prio >= self_prio : prio != self_prio) continue;
            if (o.parked.load(std::memory_order_acquire)) continue;
            const uint64_t put = o.put.load(std::memory_order_acquire);
            if (put <= cur[c].get) continue;
            if (wait_blocked(cur[c].ring[cur[c].get & cur[c].mask])) continue;
            return true;
        }
        return false;
    };

    while (running_.load(std::memory_order_relaxed)) {
        bool progress = false, pending = false;
        int chosen = -1;
        int32_t chosen_prio = 0;
        for (uint32_t i = 0; i < num_channels_; ++i) {
            const uint32_t c = (rr + i) % num_channels_;
            Channel& ch = ch_[engine][c];
            Cursor& k = cur[c];
            // Acquire pairs with the driver's release store of `put` and
            // makes every slot below it visible to this thread.
            uint64_t put = ch.put.load(std::memory_order_acquire);
            seen_put[c] = put;
            const int32_t fail = ch.fault_fail.exchange(0, std::memory_order_acquire);
            if (fail != 0) {
                int expected = 0;
                ch.sticky_error.compare_exchange_strong(expected, fail, std::memory_order_relaxed);
                if (put != k.get) {
                    replay_[engine][c] = Replay{};
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
                        raise_host_irq();
                    }
                }
                continue;
            }
            if (ch.parked.load(std::memory_order_acquire)) {
                // Head command is waiting on a migration. Other channels
                // still run. Unpark rings the doorbell if we sleep.
                if (put != k.get) pending = true;
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
            if (wait_blocked(k.ring[k.get & k.mask])) {
                pending = true;
                continue;
            }
            pending = true;
            const int32_t prio = ch.priority.load(std::memory_order_relaxed);
            // Smaller priority value wins. The scan starts at `rr`, so the
            // first channel of the winning priority is the round-robin one.
            if (chosen < 0 || prio < chosen_prio) {
                chosen = int(c);
                chosen_prio = prio;
            }
        }

        bool retired = false;
        if (chosen >= 0) {
            const uint32_t c = uint32_t(chosen);
            rr = (c + 1) % num_channels_;
            Channel& ch = ch_[engine][c];
            Cursor& k = cur[c];
            uint64_t put = seen_put[c];
            uint64_t slice_start = now_cycles();
            while (k.get != put) {
                const sg_cmd& cmd = k.ring[k.get & k.mask];
                int rc = 0;
                if (cmd.opcode == SG_OP_WAIT_FENCE) {
                    // Semaphore acquire. Cannot deadlock: the driver only
                    // publishes waits on fences whose commands were enqueued
                    // earlier (ADR 003).
                    if (cmd.arg0 >= num_engines_ || cmd.arg1 >= num_channels_) {
                        rc = -EINVAL;
                    } else if (wait_blocked(cmd)) {
                        break; // blocked: leave this channel for now
                    }
                } else {
                    const uint64_t t0 = now_cycles();
                    uint64_t fault_va = 0;
                    Replay& r = replay_[engine][c];
                    access_[engine].v.store(++aseq, std::memory_order_relaxed);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    rc = execute(engine, r, cmd, &fault_va);
                    access_[engine].v.store(++aseq, std::memory_order_release);
                    st.busy_cycles.fetch_add(now_cycles() - t0, std::memory_order_relaxed);
                    if (rc == kPageFault) {
                        // Park before publishing the fault, so the handler
                        // cannot unpark a channel we then park again.
                        ch.parked.store(true, std::memory_order_seq_cst);
                        st.faults.fetch_add(1, std::memory_order_relaxed);
                        push_fault(Fault{fault_va, engine, c});
                        break;
                    }
                    if (r.gather) r = Replay{}; // drop the gather buffers too
                    else r.done = 0;
                }
                if (rc != 0) {
                    int expected = 0;
                    ch.sticky_error.compare_exchange_strong(expected, rc, std::memory_order_relaxed);
                    if (std::getenv("SG_TRACE")) std::fprintf(stderr, "[dev] engine %u ch %u cmd#%llu opcode %u rc %d\n", engine, c, (unsigned long long)k.get, cmd.opcode, rc);
                }
                ++k.get;
                retired = true;
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
                    raise_host_irq();
                }
                if (k.get == put) put = ch.put.load(std::memory_order_acquire);
                if (k.get == put) break;
                // Command boundary. A higher priority preempts immediately.
                // The timeslice yields only to an equal priority; a lower one
                // waits, and a lone channel restarts its slice instead of
                // switching to nobody.
                const bool mixed = mixed_priority_.load(std::memory_order_acquire);
                const bool slice_due = timeslice_ns_ != 0 && now_cycles() - slice_start >= timeslice_ns_;
                if (mixed || slice_due) {
                    const int32_t self = ch.priority.load(std::memory_order_relaxed);
                    if (mixed && other_runnable(c, self, true)) {
                        st.preempts.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                    if (slice_due) {
                        if (other_runnable(c, self, false)) {
                            st.preempts.fetch_add(1, std::memory_order_relaxed);
                            break;
                        }
                        slice_start = now_cycles();
                    }
                }
            }
        }
        if (retired) progress = true;
        // Faulted before retiring anything, and no sibling was retired via
        // fault_fail either: look again so another channel runs before sleep.
        if (chosen >= 0 && !retired && !progress) continue;

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
                        // Timeout, not a signal. A new PUT only counts as a
                        // lost wake-up once every producer has finished
                        // doorbell(). Otherwise the safety net woke us in
                        // the gap between the PUT store and the ring, which
                        // is not a missed doorbell — and under TSan that gap
                        // can be longer than this timeout.
                        bool arrived = false;
                        for (uint32_t c = 0; c < num_channels_; ++c)
                            if (ch_[engine][c].put.load(std::memory_order_acquire) != seen_put[c]) {
                                arrived = true;
                                break;
                            }
                        if (arrived && doorbell_inflight(engine) == 0)
                            st.missed_doorbells.fetch_add(1, std::memory_order_relaxed);
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

int Device::execute(uint32_t engine, Replay& r, const sg_cmd& c, uint64_t* fault_va) {
    // Engine class check: a real copy engine has no ALUs and a compute engine
    // no DMA. The driver validates this too; the device is the last line.
    if (engine == SG_ENGINE_COMPUTE ? is_copy_op(c.opcode) : is_compute_op(c.opcode))
        return -EINVAL;

    // Every command below may fault part-way. Work already applied stays
    // applied and `r` remembers where to resume, so a retry never needs
    // more pages resident at once than the run it is working on.
    auto gather = [&](uint64_t va, uint64_t len, std::vector<uint8_t>& buf) {
        if (buf.size() < len) buf.resize(len);
        return for_runs(engine, r, va, len, false, fault_va,
                        [&](uint8_t* p, uint64_t off, uint64_t n) { std::memcpy(buf.data() + off, p, n); });
    };
    auto scatter = [&](uint64_t va, uint64_t len, const std::vector<uint8_t>& buf) {
        return for_runs(engine, r, va, len, true, fault_va,
                        [&](uint8_t* p, uint64_t off, uint64_t n) { std::memcpy(p, buf.data() + off, n); });
    };
    // Two sources, one destination. Fast path: each operand is one resident
    // contiguous run. Otherwise, and for every retry after a fault, move the
    // operands through scratch a run at a time and compute there.
    auto compute = [&](uint64_t b0, uint64_t b1, uint64_t bd, auto kernel) -> int {
        if (!r.gather) {
            uint8_t *dst = nullptr, *s0 = nullptr, *s1 = nullptr;
            int rc = resolve(engine, c.src0, b0, false, &s0, fault_va);
            if (rc == 0) rc = resolve(engine, c.src1, b1, false, &s1, fault_va);
            if (rc == 0) rc = resolve(engine, c.dst, bd, true, &dst, fault_va);
            if (rc == 0) {
                kernel(dst, s0, s1);
                return 0;
            }
            if (rc != kPageFault && rc != kSplit) return rc;
            r.gather = true;
            if (rc == kPageFault) return rc;
        }
        if (r.phase == 0) {
            if (int rc = gather(c.src0, b0, r.a)) return rc;
            r.phase = 1;
            r.done = 0;
        }
        if (r.phase == 1) {
            if (int rc = gather(c.src1, b1, r.b)) return rc;
            r.phase = 2;
            r.done = 0;
        }
        if (r.phase == 2) {
            r.c.resize(bd);
            kernel(r.c.data(), r.a.data(), r.b.data());
            r.phase = 3;
        }
        return scatter(c.dst, bd, r.c);
    };

    switch (c.opcode) {
    case SG_OP_NOP:
        return 0;

    case SG_OP_FILL: {
        const int v = static_cast<int>(c.value & 0xff);
        return for_runs(engine, r, c.dst, c.size, true, fault_va,
                        [&](uint8_t* p, uint64_t, uint64_t n) { std::memset(p, v, n); });
    }

    // Host addresses in COPY_H2D/COPY_D2H are trusted, exactly as a DMA
    // engine trusts the bus addresses its driver programs. The driver is
    // responsible for only ever handing it its staging buffers or pinned
    // user memory. Device addresses go through the MMU.
    case SG_OP_COPY_H2D: {
        const auto* src = reinterpret_cast<const uint8_t*>(c.src0);
        return for_runs(engine, r, c.dst, c.size, true, fault_va,
                        [&](uint8_t* p, uint64_t off, uint64_t n) { std::memcpy(p, src + off, n); });
    }

    case SG_OP_COPY_D2H: {
        auto* dst = reinterpret_cast<uint8_t*>(c.dst);
        return for_runs(engine, r, c.src0, c.size, false, fault_va,
                        [&](uint8_t* p, uint64_t off, uint64_t n) { std::memcpy(dst + off, p, n); });
    }

    case SG_OP_COPY_D2D: {
        const bool overlap = c.size && c.dst < c.src0 + c.size && c.src0 < c.dst + c.size;
        if (!overlap) {
            // Distinct VAs never share a frame, so a forward copy is safe;
            // each step goes as far as both sides' runs reach.
            while (r.done < c.size) {
                const uint64_t left = c.size - r.done;
                uint8_t *dst = nullptr, *src = nullptr;
                uint64_t nd = 0, ns = 0;
                if (int rc = run_at(engine, c.dst + r.done, left, true, &dst, &nd, fault_va)) return rc;
                if (int rc = run_at(engine, c.src0 + r.done, left, false, &src, &ns, fault_va)) return rc;
                const uint64_t n = std::min(nd, ns);
                std::memcpy(dst, src, n);
                r.done += n;
            }
            return 0;
        }
        // Overlap needs memmove semantics: one run per side, or via scratch.
        if (!r.gather) {
            uint8_t *dst = nullptr, *src = nullptr;
            int rc = resolve(engine, c.dst, c.size, true, &dst, fault_va);
            if (rc == 0) rc = resolve(engine, c.src0, c.size, false, &src, fault_va);
            if (rc == 0) {
                std::memmove(dst, src, c.size);
                return 0;
            }
            if (rc != kPageFault && rc != kSplit) return rc;
            r.gather = true;
            if (rc == kPageFault) return rc;
        }
        if (r.phase == 0) {
            if (int rc = gather(c.src0, c.size, r.a)) return rc;
            r.phase = 1;
            r.done = 0;
        }
        return scatter(c.dst, c.size, r.a);
    }

    case SG_OP_VADD_F32: {
        const uint32_t n = c.arg0;
        const uint64_t bytes = uint64_t{n} * sizeof(float);
        return compute(bytes, bytes, bytes, [n](uint8_t* d, const uint8_t* s0, const uint8_t* s1) {
            const float* a = reinterpret_cast<const float*>(s0);
            const float* b = reinterpret_cast<const float*>(s1);
            float* out = reinterpret_cast<float*>(d);
            for (uint32_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
        });
    }

    case SG_OP_GEMM_F32: {
        const uint64_t m = c.arg0, n = c.arg1, k = c.arg2;
        return compute(m * k * sizeof(float), k * n * sizeof(float), m * n * sizeof(float),
                       [m, n, k](uint8_t* d, const uint8_t* s0, const uint8_t* s1) {
            const float* a = reinterpret_cast<const float*>(s0);
            const float* b = reinterpret_cast<const float*>(s1);
            float* out = reinterpret_cast<float*>(d);
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
        });
    }

    default:
        return -EINVAL;
    }
}

} // namespace softgpu::device
