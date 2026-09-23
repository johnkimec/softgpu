// STAGE 4 (built after 2): no big lock. Submission holds only its channel's
// lock (or, in ticket mode, no lock at all); the shared structures behind it
// — VRAM allocator and deferred-release list, pin registry, staging pool —
// each have a small lock of their own, and every counter is atomic. PUT
// mirrors are atomics so snapshots ("everything submitted so far") need no
// lock either.
//
// Lock order, where two are ever held: pin_lock -> alloc_lock (UNPIN pushes
// onto the pending list). Channel locks nest inside nothing and nothing
// nests inside them except the staging lock (briefly, for slot bookkeeping).
//
// STAGE 2: interrupts and a wait policy. Every place the host waits for a
// fence goes through wait_until(): spin for a budget, then arm the channel's
// interrupt and sleep on the device's IRQ line until it fires.
//
// STAGE 3b: engines, channels, cross-channel fences. SUBMIT names the
// (engine, channel); SG_OP_WAIT_FENCE lets a channel block on another's
// fence; every stored fence is (engine, channel, value) or a PUT snapshot.
//
// Deadlock freedom: a WAIT_FENCE is accepted only if its target value is
// <= that channel's PUT at submission time, i.e. the command it waits for was
// enqueued before the WAIT itself. By induction on enqueue order the
// dependency graph is a DAG, so no engine can wait on something behind it.

#include "driver/sg_driver.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "common/clock.h"
#include "common/cpu.h"
#include "device/device.h"
#include "driver/vram_alloc.h"
#include "softgpu/sg_ioctl.h"

namespace softgpu::driver {
namespace {

constexpr int kFd = 3; // the one and only descriptor we hand out
constexpr uint32_t kDefaultRingDepth = 1024; // chosen from the depth sweep in ADR 001

// Tuning knobs for the decision records, not user-facing settings.
uint64_t env_pow2(const char* name, uint64_t def, uint64_t lo, uint64_t hi) {
    const char* e = std::getenv(name);
    if (!e) return def;
    unsigned long long v = std::strtoull(e, nullptr, 10);
    if (v < lo || v > hi || (v & (v - 1)) != 0) return def;
    return v;
}
uint64_t env_int(const char* name, uint64_t def, uint64_t lo, uint64_t hi) {
    const char* e = std::getenv(name);
    if (!e) return def;
    unsigned long long v = std::strtoull(e, nullptr, 10);
    return (v < lo || v > hi) ? def : v;
}
uint32_t ring_depth_from_env() { return uint32_t(env_pow2("SG_RING_DEPTH", kDefaultRingDepth, 2, 65536)); }
uint32_t staging_slots_from_env() { return uint32_t(env_pow2("SG_STAGING_SLOTS", SG_STAGING_SLOTS, 1, 64)); }
uint64_t staging_chunk_from_env() { return env_pow2("SG_STAGING_CHUNK", SG_STAGING_CHUNK, 4096, 64ull << 20); }
uint32_t copy_engines_from_env() { return uint32_t(env_int("SG_COPY_ENGINES", SG_COPY_ENGINES, 1, SG_MAX_ENGINES - 1)); }
uint32_t channels_from_env() { return uint32_t(env_int("SG_CHANNELS", SG_CHANNELS, 1, SG_MAX_CHANNELS)); }
constexpr uint64_t kDefaultSpinNs = 30000; // hybrid budget / adaptive cap: ~ the measured wake-up latency (ADR 004)
uint64_t spin_ns_from_env() { return env_int("SG_SPIN_NS", kDefaultSpinNs, 0, 1000000000ull); }
uint32_t policy_from_env() {
    const char* e = std::getenv("SG_WAIT_POLICY");
    if (!e) return SG_POLICY_ADAPTIVE; // default chosen from the sweep in ADR 004
    if (!std::strcmp(e, "spin")) return SG_POLICY_SPIN;
    if (!std::strcmp(e, "block")) return SG_POLICY_BLOCK;
    if (!std::strcmp(e, "hybrid")) return SG_POLICY_HYBRID;
    return SG_POLICY_ADAPTIVE;
}
constexpr uint64_t kDefaultIdleNs = 50000; // engine idle budget / adaptive cap
uint64_t idle_ns_from_env() { return env_int("SG_ENGINE_IDLE_NS", kDefaultIdleNs, 0, 10000000000ull); }
uint32_t idle_policy_from_env() {
    const char* e = std::getenv("SG_ENGINE_IDLE");
    if (!e) return SG_IDLE_HYBRID; // fixed 50 us hysteresis, chosen in ADR 006
    if (!std::strcmp(e, "spin")) return SG_IDLE_SPIN;
    if (!std::strcmp(e, "sleep")) return SG_IDLE_SLEEP;
    if (!std::strcmp(e, "hybrid")) return SG_IDLE_HYBRID;
    return SG_IDLE_ADAPTIVE;
}
uint32_t submit_mode_from_env() {
    const char* e = std::getenv("SG_SUBMIT_MODE");
    return (e && !std::strcmp(e, "ticket")) ? SG_SUBMIT_TICKET : SG_SUBMIT_MUTEX;
}
// SG_DEVICE_CPU=<n> pins the compute engine's thread; -1 (default) leaves it to the OS.
int device_cpu_from_env() {
    const char* e = std::getenv("SG_DEVICE_CPU");
    return e ? std::atoi(e) : -1;
}

struct FreeDeleter { void operator()(void* p) const { std::free(p); } };

struct Fence {
    uint32_t engine = 0;
    uint32_t channel = 0;
    uint64_t value = 0;
};

// A snapshot of every channel's PUT: "everything submitted up to now".
using PutSnapshot = uint64_t[SG_MAX_ENGINES][SG_MAX_CHANNELS];

// Something the user has released that in-flight commands may still touch.
// Safe to recycle once every channel has retired the PUT snapshot.
struct Pending {
    enum Kind { kVram, kPin } kind;
    uint64_t addr;
    uint64_t size;
    PutSnapshot fence;
};

// Per-channel producer state, on its own cache line.
struct alignas(64) Producer {
    std::mutex lock;                 // SG_SUBMIT_MUTEX: one producer at a time
    std::atomic<uint64_t> reserve{0}; // SG_SUBMIT_TICKET: next slot to claim
    std::atomic<uint64_t> put{0};    // driver-side mirror of the PUT register
};

struct Slot {
    Fence fence;       // DMA that last used the slot; reuse once retired
    bool busy = false; // acquired by a copy in progress
};

struct Driver {
    uint32_t num_engines = 0;
    uint32_t num_channels = 0;
    uint32_t submit_mode = SG_SUBMIT_MUTEX;
    device::Device dev{SG_VRAM_SIZE, copy_engines_from_env(), channels_from_env()};

    std::unique_ptr<sg_cmd, FreeDeleter> ring[SG_MAX_ENGINES][SG_MAX_CHANNELS];
    uint32_t depth = 0;
    Producer prod[SG_MAX_ENGINES][SG_MAX_CHANNELS];

    // VRAM allocator and the deferred-release list share one lock; the
    // submit path only reads (owns_range) and takes it shared.
    std::shared_mutex alloc_lock;
    VramAllocator vram{SG_VRAM_SIZE, SG_ALLOC_ALIGN};
    std::vector<Pending> pending;

    // Pinned host ranges: looked up on every copy submit (shared), changed
    // rarely (exclusive).
    std::shared_mutex pin_lock;
    std::map<uint64_t, uint64_t> pins; // addr -> size

    // Staging pool for pageable copies: `slots` chunks of `chunk` bytes. The
    // lock covers slot bookkeeping only; the memcpy and the wait happen
    // outside it.
    std::mutex staging_lock;
    std::vector<uint8_t> staging;
    std::vector<Slot> slot;
    uint32_t slots = 0;
    uint64_t chunk = 0;
    uint32_t next_slot = 0;

    // Wait policy (ADR 004).
    uint32_t policy = SG_POLICY_ADAPTIVE;
    uint64_t spin_ns = kDefaultSpinNs;
    uint32_t idle_policy = SG_IDLE_HYBRID;
    uint64_t idle_ns = 50000;
    std::atomic<uint64_t> ewma[SG_MAX_ENGINES][SG_MAX_CHANNELS] = {};
    std::atomic<uint64_t> ewma_all{0};

    // Counters: all atomic, all relaxed; nothing here needs a lock.
    std::atomic<uint64_t> waits{0}, waits_spun{0}, waits_blocked{0}, wake_latency_ns{0};
    std::atomic<uint64_t> submits{0}, stalls{0}, staging_waits{0}, lock_wait_ns{0};
    std::atomic<uint64_t> bytes_h2d{0}, bytes_d2h{0}, bytes_direct{0}, bytes_staged{0};

    Driver() {
        num_engines = dev.num_engines();
        num_channels = dev.num_channels();
        submit_mode = submit_mode_from_env();
        depth = ring_depth_from_env();
        for (uint32_t e = 0; e < num_engines; ++e)
            for (uint32_t c = 0; c < num_channels; ++c) {
                void* mem = nullptr;
                if (posix_memalign(&mem, 64, size_t{depth} * sizeof(sg_cmd)) != 0) std::abort();
                std::memset(mem, 0, size_t{depth} * sizeof(sg_cmd));
                ring[e][c].reset(static_cast<sg_cmd*>(mem));
                dev.channel(e, c).ring_base = reinterpret_cast<uint64_t>(ring[e][c].get());
                dev.channel(e, c).ring_mask = depth - 1;
            }
        slots = staging_slots_from_env();
        chunk = staging_chunk_from_env();
        staging.assign(size_t{slots} * chunk, 0);
        slot.assign(slots, Slot{});
        policy = policy_from_env();
        spin_ns = spin_ns_from_env();
        idle_policy = idle_policy_from_env();
        idle_ns = idle_ns_from_env();
        dev.set_idle_policy(idle_policy, idle_ns);
        if (const char* e = std::getenv("SG_EXPERIMENT_NO_FENCE")) dev.set_no_fence(*e == '1');
    }

    uint8_t* slot_ptr(uint32_t i) { return staging.data() + size_t{i} * chunk; }
};

std::mutex g_open_lock;
std::unique_ptr<Driver> g_drv;
int g_refs = 0;

// ---- locks with contention accounting ----------------------------------------
// The uncontended path is a plain try_lock; only a contended acquisition
// reads the clock, so `lock_wait_ns` is the cost of contention and nothing else.
template <class M>
void lock_timed(Driver& d, M& m) {
    if (m.try_lock()) return;
    const uint64_t t0 = now_cycles();
    m.lock();
    d.lock_wait_ns.fetch_add(now_cycles() - t0, std::memory_order_relaxed);
}
void lock_shared_timed(Driver& d, std::shared_mutex& m) {
    if (m.try_lock_shared()) return;
    const uint64_t t0 = now_cycles();
    m.lock_shared();
    d.lock_wait_ns.fetch_add(now_cycles() - t0, std::memory_order_relaxed);
}
struct Guard {
    Driver& d; std::mutex& m;
    Guard(Driver& d_, std::mutex& m_) : d(d_), m(m_) { lock_timed(d, m); }
    ~Guard() { m.unlock(); }
};
struct SharedGuard {
    std::shared_mutex& m;
    SharedGuard(Driver& d, std::shared_mutex& m_) : m(m_) { lock_shared_timed(d, m); }
    ~SharedGuard() { m.unlock_shared(); }
};
struct ExclusiveGuard {
    std::shared_mutex& m;
    ExclusiveGuard(Driver& d, std::shared_mutex& m_) : m(m_) { lock_timed(d, m); }
    ~ExclusiveGuard() { m.unlock(); }
};

// ---- fences and waiting --------------------------------------------------------

inline int sticky(Driver& d) {
    for (uint32_t e = 0; e < d.num_engines; ++e)
        for (uint32_t c = 0; c < d.num_channels; ++c)
            if (int rc = d.dev.channel(e, c).sticky_error.load(std::memory_order_relaxed)) return rc;
    return 0;
}
inline uint64_t retired(Driver& d, uint32_t e, uint32_t c) {
    return d.dev.channel(e, c).get.load(std::memory_order_acquire);
}
inline uint64_t published(Driver& d, uint32_t e, uint32_t c) {
    return d.prod[e][c].put.load(std::memory_order_acquire);
}

// Arm the interrupt on a channel for `target`, keeping the lowest armed
// value so no waiter's fence can be skipped. The engine clears it on fire;
// a still-waiting thread re-arms after it wakes.
void arm_irq(Driver& d, uint32_t e, uint32_t c, uint64_t target) {
    auto& t = d.dev.channel(e, c).irq_target;
    uint64_t cur = t.load(std::memory_order_relaxed);
    while ((cur == 0 || cur > target) &&
           !t.compare_exchange_weak(cur, target, std::memory_order_release, std::memory_order_relaxed)) {}
}

constexpr uint64_t kMinSpinNs = 200;        // always spin at least one round trip
constexpr uint64_t kAdaptiveFloorNs = 2000; // adaptive never predicts below this: cheap insurance
                                            // against blocking (and a 30 us wake) on sub-us chains
constexpr uint64_t kBlockTimeoutNs = 1000000; // safety net: re-check every 1 ms even without an IRQ

// The one wait primitive. `pred` is true when the wait is over; `arm` arms
// the interrupt(s) the waiter depends on. Spins for a budget chosen by the
// policy, then arms and sleeps on the IRQ line. The arm-then-check order is
// what closes the race between "fence not yet passed" and "interrupt fired
// before we slept": the engine clears irq_target only after storing get, so
// a check after arming either sees the fence or is guaranteed a signal.
// Returns 0 if no wait was needed, 1 if satisfied while spinning, 2 if it
// slept. Callers account the wait to the right counter.
enum { kNoWait = 0, kSpun = 1, kBlocked = 2 };
template <class Pred, class Arm>
int wait_until(Driver& d, Pred pred, Arm arm, uint32_t flags, std::atomic<uint64_t>* ewma) {
    if (pred()) return kNoWait;

    uint64_t budget;
    uint32_t policy = d.policy;
    if (flags & SG_WAIT_SPIN) policy = SG_POLICY_SPIN;
    else if (flags & SG_WAIT_BLOCK) policy = SG_POLICY_BLOCK;
    switch (policy) {
    case SG_POLICY_SPIN:  budget = ~0ull; break;
    case SG_POLICY_BLOCK: budget = 0; break;
    case SG_POLICY_ADAPTIVE: {
        // Expect this wait to look like recent ones: spin for about twice
        // the typical wait if that fits the cap, otherwise go straight to
        // sleep after the floor.
        const uint64_t typical = ewma ? ewma->load(std::memory_order_relaxed) : 0;
        budget = typical == 0 ? d.spin_ns
               : typical > d.spin_ns ? kAdaptiveFloorNs
               : std::max<uint64_t>(kAdaptiveFloorNs, std::min<uint64_t>(2 * typical, d.spin_ns));
        break;
    }
    default: budget = d.spin_ns; break;
    }

    // Spin phase. The clock is read only after the first round of spinning
    // fails; the first round is long enough (~1 us) to cover a device round
    // trip, so a wait that is satisfied that quickly costs the same as the
    // old pure spin — the fast path pays for no timing at all.
    bool done = false;
    uint64_t t0 = 0;
    for (int round = 0;; ++round) {
        const int iters = round == 0 ? 512 : 32;
        for (int i = 0; i < iters; ++i) {
            if (pred()) { done = true; break; }
            cpu_relax();
        }
        if (done) break;
        if (t0 == 0) t0 = now_cycles();
        else if (budget != ~0ull && now_cycles() - t0 >= budget) break;
    }
    if (done) {
        d.waits_spun.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Block phase.
        d.waits_blocked.fetch_add(1, std::memory_order_relaxed);
        Event& irq = d.dev.irq();
        for (;;) {
            const uint32_t seen = irq.seq();
            arm();
            if (pred()) break;
            irq.wait(seen, kBlockTimeoutNs);
            if (pred()) break;
        }
    }
    // Only the adaptive policy needs to know how long this took. For a
    // blocked wait, measure until the fence *passed* (the interrupt
    // timestamp), not until this thread woke: the wake-up latency is the
    // policy's own penalty, and feeding it back into the estimate makes one
    // blocked wait predict "long" forever — the first version of this code
    // did exactly that (ADR 004).
    if (policy == SG_POLICY_ADAPTIVE && ewma) {
        uint64_t dur = kMinSpinNs;
        if (t0) {
            const uint64_t end = done ? now_cycles() : d.dev.last_irq_cycles();
            dur = end > t0 ? end - t0 : kMinSpinNs;
        }
        const uint64_t old = ewma->load(std::memory_order_relaxed);
        ewma->store(old == 0 ? dur : (3 * old + dur) / 4, std::memory_order_relaxed);
    }
    return done ? kSpun : kBlocked;
}

// Wake-up latency: from the last interrupt pulse to this waiter running.
void note_wake(Driver& d) {
    const uint64_t fired = d.dev.last_irq_cycles();
    const uint64_t now = now_cycles();
    if (fired && now > fired) d.wake_latency_ns.fetch_add(now - fired, std::memory_order_relaxed);
}

// Block until (engine, channel) has retired everything up to `value`.
int wait_fence(Driver& d, Fence f, std::atomic<uint64_t>* counter = nullptr,
               uint32_t flags = SG_WAIT_DEFAULT) {
    const int r = wait_until(
        d, [&] { return retired(d, f.engine, f.channel) >= f.value; },
        [&] { arm_irq(d, f.engine, f.channel, f.value); }, flags, &d.ewma[f.engine][f.channel]);
    if (r == kBlocked) note_wake(d);
    if (r != kNoWait) (counter ? *counter : d.waits).fetch_add(1, std::memory_order_relaxed);
    return sticky(d) ? -EIO : 0;
}

// Snapshot of every channel's PUT. Lock-free: monotonic counters read one at
// a time give "at least everything whose submit had returned before now",
// which is exactly what "everything submitted so far" promises.
void snapshot_puts(Driver& d, PutSnapshot out) {
    for (uint32_t e = 0; e < SG_MAX_ENGINES; ++e)
        for (uint32_t c = 0; c < SG_MAX_CHANNELS; ++c)
            out[e][c] = (e < d.num_engines && c < d.num_channels) ? published(d, e, c) : 0;
}
bool all_retired(Driver& d, const PutSnapshot fence) {
    for (uint32_t e = 0; e < d.num_engines; ++e)
        for (uint32_t c = 0; c < d.num_channels; ++c)
            if (retired(d, e, c) < fence[e][c]) return false;
    return true;
}

// Block until every channel has retired a PUT snapshot.
int wait_snapshot(Driver& d, const PutSnapshot fence, uint32_t flags = SG_WAIT_DEFAULT) {
    const int r = wait_until(
        d, [&] { return all_retired(d, fence); },
        [&] {
            for (uint32_t e = 0; e < d.num_engines; ++e)
                for (uint32_t c = 0; c < d.num_channels; ++c)
                    if (fence[e][c] && retired(d, e, c) < fence[e][c]) arm_irq(d, e, c, fence[e][c]);
        },
        flags, &d.ewma_all);
    if (r == kBlocked) note_wake(d);
    if (r != kNoWait) d.waits.fetch_add(1, std::memory_order_relaxed);
    return sticky(d) ? -EIO : 0;
}

// Wait for everything submitted so far.
int drain(Driver& d) {
    PutSnapshot snap;
    snapshot_puts(d, snap);
    return wait_snapshot(d, snap);
}

// ---- submission ------------------------------------------------------------------

// Append one command to a channel's ring and ring its doorbell. Returns the
// fence. In mutex mode the caller holds the channel's lock and is the single
// producer. In ticket mode any number of producers race: each claims a slot
// with fetch_add, writes it, then publishes in ticket order — waiting for
// the previous ticket's publish, since PUT must advance contiguously.
uint64_t enqueue(Driver& d, uint32_t engine, uint32_t channel, const sg_cmd& cmd) {
    Producer& p = d.prod[engine][channel];
    auto& regs = d.dev.channel(engine, channel);
    uint64_t slot;
    if (d.submit_mode == SG_SUBMIT_TICKET) {
        slot = p.reserve.fetch_add(1, std::memory_order_relaxed);
    } else {
        slot = p.put.load(std::memory_order_relaxed);
    }
    // Backpressure: the slot we are about to write must have been retired.
    if (slot - retired(d, engine, channel) >= d.depth) {
        d.stalls.fetch_add(1, std::memory_order_relaxed);
        const uint64_t need = slot - d.depth + 1;
        wait_until(d, [&] { return retired(d, engine, channel) >= need; },
                   [&] { arm_irq(d, engine, channel, need); }, SG_WAIT_DEFAULT, nullptr);
    }
    d.ring[engine][channel].get()[slot & (d.depth - 1)] = cmd;
    if (d.submit_mode == SG_SUBMIT_TICKET) {
        // Publish in order. The hazard is the classic one: a producer
        // descheduled between reserve and here holds up everyone behind it —
        // and if the waiters spin, they are what keeps it descheduled. With
        // more producers than cores a pure spin here livelocked for good
        // (ADR 005); after a short spin, yield so the head ticket can run.
        for (int i = 0; p.put.load(std::memory_order_acquire) != slot; ++i) {
            if (i < 256) cpu_relax();
            else std::this_thread::yield();
        }
    }
    // Doorbell first, then hand the ticket on. The first version did these
    // in the other order and the next producer could ring its doorbell
    // before ours landed, so the device's PUT went backwards and the engine
    // ran off the end of the ring forever (ADR 005). Every access here is
    // atomic, so TSan had nothing to say; the hang did.
    regs.put.store(slot + 1, std::memory_order_release); // release: slot before doorbell
    p.put.store(slot + 1, std::memory_order_release);
    d.dev.doorbell(engine); // wake the engine if it is power-gated
    return slot + 1;
}

// Recycle everything the device has finished with. Caller holds alloc_lock.
void reclaim_locked(Driver& d) {
    if (d.pending.empty()) return;
    auto keep = d.pending.begin();
    for (auto& p : d.pending) {
        if (all_retired(d, p.fence)) {
            if (p.kind == Pending::kVram) d.vram.release(p.addr, p.size);
            // kPin: nothing to give back in the model; a real driver would
            // drop its page references / IOMMU mappings here.
        } else {
            *keep++ = p;
        }
    }
    d.pending.erase(keep, d.pending.end());
}

bool owns(Driver& d, uint64_t addr, uint64_t len) {
    SharedGuard g(d, d.alloc_lock);
    return d.vram.owns_range(addr, len);
}

// Is [addr, addr+size) entirely inside one pinned range?
bool pinned(Driver& d, uint64_t addr, uint64_t size) {
    SharedGuard g(d, d.pin_lock);
    auto it = d.pins.upper_bound(addr);
    if (it == d.pins.begin()) return false;
    --it;
    return addr - it->first <= it->second && size <= it->second - (addr - it->first);
}

// Staging slots. acquire marks a free slot busy and hands back the fence its
// previous user left; the caller waits for that fence *outside* the lock.
uint32_t acquire_slot(Driver& d, Fence* prev) {
    for (;;) {
        {
            Guard g(d, d.staging_lock);
            for (uint32_t i = 0; i < d.slots; ++i) {
                const uint32_t s = (d.next_slot + i) % d.slots;
                if (!d.slot[s].busy) {
                    d.slot[s].busy = true;
                    d.next_slot = (s + 1) % d.slots;
                    *prev = d.slot[s].fence;
                    return s;
                }
            }
        }
        cpu_relax(); // more concurrent pageable copies than slots: rare
    }
}
void release_slot(Driver& d, uint32_t s, Fence f) {
    Guard g(d, d.staging_lock);
    d.slot[s].fence = f;
    d.slot[s].busy = false;
}

bool is_copy_op(uint32_t op) {
    return op == SG_OP_COPY_H2D || op == SG_OP_COPY_D2H || op == SG_OP_COPY_D2D;
}
bool is_compute_op(uint32_t op) {
    return op == SG_OP_FILL || op == SG_OP_VADD_F32 || op == SG_OP_GEMM_F32;
}

int do_submit(Driver& d, sg_submit_args& a) {
    sg_cmd cmd = a.cmd;
    const uint32_t eng = a.engine, chn = a.channel;
    if (cmd.opcode >= SG_OP_COUNT || cmd.flags != 0 || cmd.reserved != 0) return -EINVAL;
    if (eng >= d.num_engines || chn >= d.num_channels) return -EINVAL;
    // Engine classes: copies only on copy engines, compute only on engine 0.
    if (eng == SG_ENGINE_COMPUTE ? is_copy_op(cmd.opcode) : is_compute_op(cmd.opcode)) return -EINVAL;
    if (sticky(d)) return -EIO;
    d.submits.fetch_add(1, std::memory_order_relaxed);
    a.out_flags = 0;

    // Mutex mode: this channel's producers take turns for the whole submit
    // (all chunks of a copy stay contiguous). Ticket mode: no lock; chunks
    // from different producers may interleave, which is fine — each stream
    // only depends on its own commands' relative order.
    std::unique_lock<std::mutex> chan_guard;
    if (d.submit_mode == SG_SUBMIT_MUTEX) {
        Producer& p = d.prod[eng][chn];
        if (!p.lock.try_lock()) {
            const uint64_t t0 = now_cycles();
            p.lock.lock();
            d.lock_wait_ns.fetch_add(now_cycles() - t0, std::memory_order_relaxed);
        }
        chan_guard = std::unique_lock<std::mutex>(p.lock, std::adopt_lock);
    }

    switch (cmd.opcode) {
    case SG_OP_NOP:
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;

    case SG_OP_WAIT_FENCE:
        // Only fences already published may be waited on (see header note).
        if (cmd.arg0 >= d.num_engines || cmd.arg1 >= d.num_channels ||
            cmd.src1 > published(d, cmd.arg0, cmd.arg1))
            return -EINVAL;
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;

    case SG_OP_FILL:
        if (!owns(d, cmd.dst, cmd.size)) return -EFAULT;
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;

    case SG_OP_COPY_D2D:
        if (!owns(d, cmd.dst, cmd.size) || !owns(d, cmd.src0, cmd.size)) return -EFAULT;
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;

    case SG_OP_COPY_H2D: {
        if (a.host_ptr == 0 || !owns(d, cmd.dst, cmd.size)) return -EFAULT;
        const uint64_t dst = cmd.dst, total = cmd.size;
        d.bytes_h2d.fetch_add(total, std::memory_order_relaxed);

        if (pinned(d, a.host_ptr, total)) {
            // Direct DMA from the user's buffer. Asynchronous: the caller owns
            // the "don't touch it until the fence retires" contract.
            cmd.src0 = a.host_ptr;
            a.fence = enqueue(d, eng, chn, cmd);
            a.out_flags |= SG_SUBMIT_DIRECT;
            d.bytes_direct.fetch_add(total, std::memory_order_relaxed);
            return 0;
        }

        // Pageable: pipeline through the staging pool. memcpy of chunk i+1
        // proceeds while the DMA of chunk i is in flight; a slot is reused
        // only once the DMA that last used it has retired.
        d.bytes_staged.fetch_add(total, std::memory_order_relaxed);
        const auto* src = reinterpret_cast<const uint8_t*>(a.host_ptr);
        for (uint64_t off = 0; off < total; off += d.chunk) {
            Fence prev;
            const uint32_t s = acquire_slot(d, &prev);
            const uint64_t n = std::min<uint64_t>(d.chunk, total - off);
            if (int rc = wait_fence(d, prev, &d.staging_waits)) { release_slot(d, s, prev); return rc; }
            std::memcpy(d.slot_ptr(s), src + off, n);
            cmd.src0 = reinterpret_cast<uint64_t>(d.slot_ptr(s));
            cmd.dst = dst + off;
            cmd.size = n;
            a.fence = enqueue(d, eng, chn, cmd);
            release_slot(d, s, {eng, chn, a.fence});
        }
        return 0;
    }

    case SG_OP_COPY_D2H: {
        if (a.host_ptr == 0 || !owns(d, cmd.src0, cmd.size)) return -EFAULT;
        const uint64_t src = cmd.src0, total = cmd.size;
        d.bytes_d2h.fetch_add(total, std::memory_order_relaxed);

        if (pinned(d, a.host_ptr, total)) {
            cmd.dst = a.host_ptr;
            a.fence = enqueue(d, eng, chn, cmd);
            a.out_flags |= SG_SUBMIT_DIRECT;
            d.bytes_direct.fetch_add(total, std::memory_order_relaxed);
            return 0;
        }

        // Pageable: keep up to `slots` DMAs ahead of the host copy-out.
        d.bytes_staged.fetch_add(total, std::memory_order_relaxed);
        auto* dst = reinterpret_cast<uint8_t*>(a.host_ptr);
        const uint64_t nchunks = (total + d.chunk - 1) / d.chunk;
        struct Issued { uint32_t slot; uint64_t fence; };
        std::vector<Issued> window;
        window.reserve(d.slots);
        uint64_t issued = 0, done = 0;
        auto issue = [&]() -> int {
            Fence prev;
            const uint32_t s = acquire_slot(d, &prev);
            if (int rc = wait_fence(d, prev, &d.staging_waits)) { release_slot(d, s, prev); return rc; }
            cmd.src0 = src + issued * d.chunk;
            cmd.dst = reinterpret_cast<uint64_t>(d.slot_ptr(s));
            cmd.size = std::min<uint64_t>(d.chunk, total - issued * d.chunk);
            window.push_back({s, enqueue(d, eng, chn, cmd)});
            ++issued;
            return 0;
        };
        // Leave headroom in the pool for other threads' copies.
        const uint64_t depth = std::max<uint64_t>(1, d.slots / 2);
        while (issued < std::min<uint64_t>(depth, nchunks))
            if (int rc = issue()) return rc;
        while (done < nchunks) {
            Issued w = window.front();
            window.erase(window.begin());
            if (int rc = wait_fence(d, {eng, chn, w.fence})) { release_slot(d, w.slot, {eng, chn, w.fence}); return rc; }
            const uint64_t n = std::min<uint64_t>(d.chunk, total - done * d.chunk);
            std::memcpy(dst + done * d.chunk, d.slot_ptr(w.slot), n);
            release_slot(d, w.slot, {eng, chn, w.fence});
            ++done;
            if (issued < nchunks)
                if (int rc = issue()) return rc;
        }
        a.fence = published(d, eng, chn); // everything issued here has retired
        return 0;
    }

    case SG_OP_VADD_F32: {
        const uint64_t bytes = uint64_t{cmd.arg0} * sizeof(float);
        if (!owns(d, cmd.dst, bytes) || !owns(d, cmd.src0, bytes) || !owns(d, cmd.src1, bytes))
            return -EFAULT;
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;
    }

    case SG_OP_GEMM_F32: {
        const uint64_t m = cmd.arg0, n = cmd.arg1, k = cmd.arg2;
        if (!owns(d, cmd.src0, m * k * sizeof(float)) || !owns(d, cmd.src1, k * n * sizeof(float)) ||
            !owns(d, cmd.dst, m * n * sizeof(float)))
            return -EFAULT;
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;
    }

    default:
        return -EINVAL;
    }
}

} // namespace
} // namespace softgpu::driver

using namespace softgpu::driver;

extern "C" int sg_drv_open(void) {
    std::lock_guard<std::mutex> g(g_open_lock);
    if (!g_drv) {
        g_drv = std::make_unique<Driver>();
        g_drv->dev.power_on(device_cpu_from_env());
    }
    ++g_refs;
    return kFd;
}

extern "C" int sg_drv_close(int fd) {
    std::lock_guard<std::mutex> g(g_open_lock);
    if (fd != kFd || g_refs == 0) return -EBADF;
    if (--g_refs == 0) {
        drain(*g_drv); // drain before pulling the plug
        {
            ExclusiveGuard a(*g_drv, g_drv->alloc_lock);
            reclaim_locked(*g_drv);
        }
        g_drv->dev.power_off();
        g_drv.reset();
    }
    return 0;
}

extern "C" int sg_drv_ioctl(int fd, unsigned int req, void* arg) {
    if (fd != kFd) return -EBADF;
    Driver* dp;
    {
        std::lock_guard<std::mutex> g(g_open_lock);
        if (!g_drv) return -EBADF;
        dp = g_drv.get();
    }
    Driver& d = *dp;

    switch (req) {
    case SG_IOC_WAIT: {
        if (!arg) return -EINVAL;
        auto* w = static_cast<sg_wait_args*>(arg);
        if (w->engine == SG_WAIT_ALL) {
            PutSnapshot snap;
            snapshot_puts(d, snap);
            return wait_snapshot(d, snap, w->flags);
        }
        if (w->engine >= d.num_engines || w->channel >= d.num_channels) return -EINVAL;
        return wait_fence(d, {w->engine, w->channel, w->fence}, nullptr, w->flags);
    }
    case SG_IOC_QUERY: {
        if (!arg) return -EINVAL;
        auto* q = static_cast<sg_query_args*>(arg);
        q->abi_version = SG_ABI_VERSION;
        q->staging_slots = d.slots;
        q->vram_size = SG_VRAM_SIZE;
        q->staging_chunk = d.chunk;
        q->num_engines = d.num_engines;
        q->num_channels = d.num_channels;
        q->wait_policy = d.policy;
        q->submit_mode = d.submit_mode;
        q->spin_ns = d.spin_ns;
        q->idle_policy = d.idle_policy;
        q->reserved3 = 0;
        q->idle_ns = d.idle_ns;
        return 0;
    }
    case SG_IOC_ALLOC: {
        if (!arg) return -EINVAL;
        auto* a = static_cast<sg_alloc_args*>(arg);
        if (a->size == 0) return -EINVAL;
        ExclusiveGuard g(d, d.alloc_lock);
        reclaim_locked(d);
        auto addr = d.vram.alloc(a->size);
        if (!addr) {
            // Maybe everything we need is sitting in the deferred list.
            if (int rc = drain(d)) return rc;
            reclaim_locked(d);
            addr = d.vram.alloc(a->size);
            if (!addr) {
                if (std::getenv("SG_TRACE"))
                    std::fprintf(stderr, "[drv] ENOMEM size=%llu live=%llu pending=%zu\n",
                                 (unsigned long long)a->size, (unsigned long long)d.vram.bytes_live(), d.pending.size());
                return -ENOMEM;
            }
        }
        a->addr = *addr;
        return 0;
    }
    case SG_IOC_FREE: {
        if (!arg) return -EINVAL;
        const uint64_t addr = static_cast<sg_free_args*>(arg)->addr;
        ExclusiveGuard g(d, d.alloc_lock);
        reclaim_locked(d);
        uint64_t size = 0;
        if (!d.vram.detach(addr, &size)) return -EINVAL;
        // Nothing may be handed this memory until every command issued so
        // far, on any channel, has retired; recycle it then rather than
        // draining now.
        Pending p{Pending::kVram, addr, size, {}};
        snapshot_puts(d, p.fence);
        if (all_retired(d, p.fence)) d.vram.release(addr, size);
        else d.pending.push_back(p);
        return 0;
    }
    case SG_IOC_PIN: {
        if (!arg) return -EINVAL;
        auto* p = static_cast<sg_pin_args*>(arg);
        if (p->addr == 0 || p->size == 0 || p->addr + p->size < p->addr) return -EINVAL;
        {
            ExclusiveGuard g(d, d.alloc_lock);
            reclaim_locked(d);
        }
        ExclusiveGuard g(d, d.pin_lock);
        // Reject overlap with any existing pin.
        auto next = d.pins.lower_bound(p->addr);
        if (next != d.pins.end() && next->first < p->addr + p->size) return -EEXIST;
        if (next != d.pins.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second > p->addr) return -EEXIST;
        }
        d.pins.emplace(p->addr, p->size);
        return 0;
    }
    case SG_IOC_UNPIN: {
        if (!arg) return -EINVAL;
        auto* p = static_cast<sg_pin_args*>(arg);
        Pending pend{Pending::kPin, 0, 0, {}};
        {
            ExclusiveGuard g(d, d.pin_lock);
            auto it = d.pins.find(p->addr);
            if (it == d.pins.end()) return -EINVAL;
            pend.addr = it->first;
            pend.size = it->second;
            d.pins.erase(it); // new copies from this range go via staging from now on
        }
        snapshot_puts(d, pend.fence);
        ExclusiveGuard g(d, d.alloc_lock);
        d.pending.push_back(pend);
        return 0;
    }
    case SG_IOC_SUBMIT:
        if (!arg) return -EINVAL;
        return do_submit(d, *static_cast<sg_submit_args*>(arg));
    case SG_IOC_STATS: {
        if (!arg) return -EINVAL;
        auto* s = static_cast<sg_stats_args*>(arg);
        std::memset(s, 0, sizeof *s);
        s->num_engines = d.num_engines;
        for (uint32_t e = 0; e < d.num_engines; ++e) {
            auto& st = d.dev.stats(e);
            s->busy_cycles[e] = st.busy_cycles.load(std::memory_order_relaxed);
            s->wait_cycles[e] = st.wait_cycles.load(std::memory_order_relaxed);
            s->idle_cycles[e] = st.idle_cycles.load(std::memory_order_relaxed);
            s->cmds_executed[e] = st.cmds_executed.load(std::memory_order_relaxed);
            s->batches[e] = st.batches.load(std::memory_order_relaxed);
            s->irqs[e] = st.irqs.load(std::memory_order_relaxed);
            s->sleep_cycles[e] = st.sleep_cycles.load(std::memory_order_relaxed);
            s->wakeups[e] = st.wakeups.load(std::memory_order_relaxed);
            s->missed_doorbells[e] = st.missed_doorbells.load(std::memory_order_relaxed);
            s->cpu_ns[e] = st.cpu_ns.load(std::memory_order_relaxed);
        }
        auto ld = [](const std::atomic<uint64_t>& v) { return v.load(std::memory_order_relaxed); };
        s->submits = ld(d.submits);
        s->waits = ld(d.waits);
        s->waits_spun = ld(d.waits_spun);
        s->waits_blocked = ld(d.waits_blocked);
        s->wake_latency_ns = ld(d.wake_latency_ns);
        s->lock_wait_ns = ld(d.lock_wait_ns);
        s->stalls = ld(d.stalls);
        s->staging_waits = ld(d.staging_waits);
        s->bytes_h2d = ld(d.bytes_h2d);
        s->bytes_d2h = ld(d.bytes_d2h);
        s->bytes_direct = ld(d.bytes_direct);
        s->bytes_staged = ld(d.bytes_staged);
        return 0;
    }
    case SG_IOC_RESET_STATS:
        d.dev.reset_stats();
        for (auto* c : {&d.submits, &d.waits, &d.waits_spun, &d.waits_blocked, &d.wake_latency_ns,
                        &d.lock_wait_ns, &d.stalls, &d.staging_waits, &d.bytes_h2d, &d.bytes_d2h,
                        &d.bytes_direct, &d.bytes_staged})
            c->store(0, std::memory_order_relaxed);
        return 0;
    default:
        return -ENOTTY;
    }
}
