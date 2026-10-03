// STAGE 4 (built after 2): no big lock. Submission holds only its channel's
// lock (or, in ticket mode, no lock at all); the shared structures behind it
// — VRAM allocator and deferred-release list, pin registry, staging pool —
// each have a small lock of their own, and every counter is atomic. PUT
// mirrors are atomics so snapshots ("everything submitted so far") need no
// lock either.
//
// Lock order, where two are ever held: pin_lock -> alloc_lock (UNPIN pushes
// onto the pending list), alloc_lock -> mem_lock (allocation and recycling
// touch frames). Channel locks nest inside nothing and nothing nests inside
// them except the staging lock (briefly, for slot bookkeeping). The fault
// thread and the userfaultfd thread take mem_lock only: ALLOC drains while
// holding alloc_lock, and the drain may be waiting on a fault.
//
// STAGE 5: two address spaces. Device VAs [0, VRAM) are sgMalloc, mapped
// VA == PA and resident for life; [VRAM, SG_VA_SIZE) are managed. VRAM
// frames come from one allocator: sgMalloc takes contiguous blocks, a
// managed page takes one frame when it faults in and gives it back when it
// is evicted (CLOCK over resident managed pages) or freed. A fault that
// continues a channel's sequential walk also prefetches the next pages.
// On Linux, when a userfaultfd is open, migrating a page into VRAM drops
// the host copy; the next host touch faults and a second thread migrates
// the page back (ADR 007 slice 5). SG_UFFD=0 leaves the one-way model.
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

#if defined(__linux__)
#define _GNU_SOURCE 1
#endif

#include "driver/sg_driver.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/userfaultfd.h>
#ifndef UFFD_USER_MODE_ONLY
#define UFFD_USER_MODE_ONLY 1
#endif
#endif

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
constexpr uint64_t kDefaultPrefetchPages = 16; // ADR 007 slice 4; 0 disables
uint64_t prefetch_from_env() { return env_int("SG_PREFETCH_PAGES", kDefaultPrefetchPages, 0, 4096); }
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
    enum Kind { kVram, kManaged, kPin } kind;
    uint64_t addr;
    uint64_t size;
    void* host; // managed backing to free on recycle; nullptr otherwise
    PutSnapshot fence;
};

// A live user allocation. `host` is the managed backing; nullptr for sgMalloc.
struct Alloc {
    uint64_t size;
    void* host;
};

// Where a channel's faults are heading: `next` is the page after the last
// one migrated for this stream. Fault thread only.
struct FaultStream {
    uint64_t next = 0;
    uint64_t used = 0; // LRU stamp; 0 = empty
};
// A few per channel, so commands that walk several operands at once
// (non-overlapping D2D alternates dst and src pages) still look sequential.
constexpr int kFaultStreams = 4;

struct Driver;

void service_faults(Driver* self);
#if defined(__linux__)
void service_host_faults(Driver* self);
int open_userfaultfd();
#endif

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
    device::Device dev{SG_VRAM_SIZE, SG_VA_SIZE, copy_engines_from_env(), channels_from_env()};

    std::unique_ptr<sg_cmd, FreeDeleter> ring[SG_MAX_ENGINES][SG_MAX_CHANNELS];
    uint32_t depth = 0;
    Producer prod[SG_MAX_ENGINES][SG_MAX_CHANNELS];

    // Live allocations, the managed VA allocator, and the deferred-release
    // list share one lock; the submit path only reads (owns) and takes it
    // shared.
    std::shared_mutex alloc_lock;
    std::map<uint64_t, Alloc> allocs; // device VA -> allocation
    VramAllocator managed_va{SG_VRAM_SIZE, SG_VA_SIZE - SG_VRAM_SIZE, SG_ALLOC_ALIGN};
    std::vector<Pending> pending;

    // VRAM frames and the resident managed pages. The fault thread holds
    // this for a whole migration, including the eviction it may need.
    std::mutex mem_lock;
    VramAllocator vram{0, SG_VRAM_SIZE, SG_ALLOC_ALIGN};
    std::list<uint64_t> resident; // managed page VAs in VRAM; CLOCK hand at the front
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> resident_pos;
    // Mapped managed ranges, device VA start -> end and host base, so
    // prefetch and the host-fault thread stay inside an allocation without
    // taking alloc_lock. Host pages are contiguous from `host`.
    struct ManagedSpan {
        uint64_t end = 0;
        uint64_t host = 0;
    };
    std::map<uint64_t, ManagedSpan> managed_ranges;
    // Bounce for UFFDIO_COPY. Callers hold mem_lock. Page-aligned so the
    // ioctl accepts it as a source.
    alignas(SG_PAGE_SIZE) uint8_t migrate_bounce[SG_PAGE_SIZE]{};

    // Prefetch (fault thread only, apart from the knob).
    uint64_t prefetch_pages = kDefaultPrefetchPages;
    FaultStream streams[SG_MAX_ENGINES][SG_MAX_CHANNELS][kFaultStreams];
    uint64_t stream_clock = 0;

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
    std::atomic<uint64_t> migrations{0}, prefetches{0}, evictions{0}, writebacks{0}, bytes_migrated{0};
    std::atomic<uint64_t> host_faults{0}, host_fault_ns{0};
    // SG_IOC_UFFD_HOLD. The handler spins on this before mem_lock so a test
    // can free an allocation while a host thread sits in the kernel fault.
    // An atomic, not an environment variable: setenv races with getenv.
    std::atomic<int> uffd_hold{0};

    std::atomic<bool> fault_stop{false};
    std::thread fault_th;
    // userfaultfd, or -1 when two-way coherence is off (SG_UFFD=0, macOS,
    // or the syscall was refused).
    int uffd = -1;
    int uffd_kick = -1;
    std::thread uffd_th;

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
        prefetch_pages = prefetch_from_env();
#if defined(__linux__)
        if (const char* e = std::getenv("SG_UFFD"); !(e && e[0] == '0' && e[1] == '\0')) {
            uffd = open_userfaultfd();
            if (uffd >= 0) {
                uffd_kick = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
                if (uffd_kick < 0) {
                    close(uffd);
                    uffd = -1;
                } else {
                    uffd_th = std::thread(service_host_faults, this);
                }
            } else if (std::getenv("SG_TRACE")) {
                std::fprintf(stderr, "[drv] userfaultfd unavailable: %s\n", std::strerror(errno));
            }
        }
#endif
        fault_th = std::thread(service_faults, this);
    }

    ~Driver() {
        fault_stop.store(true, std::memory_order_release);
        dev.fault_irq().signal();
        if (uffd_kick >= 0) {
            uint64_t one = 1;
            ssize_t n = ::write(uffd_kick, &one, sizeof one);
            (void)n;
        }
        if (fault_th.joinable()) fault_th.join();
        if (uffd_th.joinable()) uffd_th.join();
        if (uffd >= 0) ::close(uffd);
        if (uffd_kick >= 0) ::close(uffd_kick);
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

// ---- managed memory: residency and eviction (caller holds mem_lock) --------

#if defined(__linux__)
int open_userfaultfd() {
    // O_NONBLOCK is required: on a blocking userfaultfd, poll() returns
    // POLLERR and never reports the faults (kernel userfaultfd_poll).
    int fd = static_cast<int>(syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY));
    if (fd < 0) fd = static_cast<int>(syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK));
    if (fd < 0) return -1;
    uffdio_api api{};
    api.api = UFFD_API;
    if (ioctl(fd, UFFDIO_API, &api) != 0) {
        int e = errno;
        ::close(fd);
        errno = e;
        return -1;
    }
    return fd;
}

int uffd_register(Driver& d, void* host, uint64_t bytes) {
    uffdio_register reg{};
    reg.range.start = reinterpret_cast<uint64_t>(host);
    reg.range.len = bytes;
    reg.mode = UFFDIO_REGISTER_MODE_MISSING;
    if (ioctl(d.uffd, UFFDIO_REGISTER, &reg) != 0) return -errno;
    return 0;
}

void uffd_unregister(Driver& d, void* host, uint64_t bytes) {
    if (d.uffd < 0 || !host || bytes == 0) return;
    uffdio_range range{};
    range.start = reinterpret_cast<uint64_t>(host);
    range.len = bytes;
    ioctl(d.uffd, UFFDIO_UNREGISTER, &range);
}

// Populate one missing host page and wake anyone blocked on it. EEXIST
// means a racer already did; that copy woke the waiter.
int uffd_copy_page(Driver& d, uint64_t dst, const void* src) {
    uffdio_copy cp{};
    cp.src = reinterpret_cast<uint64_t>(src);
    cp.dst = dst;
    cp.len = SG_PAGE_SIZE;
    if (ioctl(d.uffd, UFFDIO_COPY, &cp) != 0) return errno == EEXIST ? 0 : -errno;
    if (cp.copy == static_cast<__s64>(SG_PAGE_SIZE)) return 0;
    const int err = cp.copy < 0 ? static_cast<int>(-cp.copy) : EIO;
    return err == EEXIST ? 0 : -err;
}

void uffd_wake(Driver& d, uint64_t dst) {
    if (d.uffd < 0) return;
    uffdio_range range{};
    range.start = dst;
    range.len = SG_PAGE_SIZE;
    ioctl(d.uffd, UFFDIO_WAKE, &range);
}
#endif

// Install the frame's bytes into the host page. With userfaultfd the host
// page is missing, so this is UFFDIO_COPY (a memcpy would fault in the
// caller and deadlock on mem_lock). Without it, only a dirty page is
// written back; a clean one still has its host copy.
int publish_host(Driver& d, uint64_t host, uint64_t pa, bool dirty) {
    if (d.uffd < 0) {
        if (!dirty) return 0;
        if (int rc = d.dev.copy_out(pa, reinterpret_cast<void*>(host))) return rc;
        d.writebacks.fetch_add(1, std::memory_order_relaxed);
        d.bytes_migrated.fetch_add(SG_PAGE_SIZE, std::memory_order_relaxed);
        return 0;
    }
#if defined(__linux__)
    if (int rc = d.dev.copy_out(pa, d.migrate_bounce)) return rc;
    if (int rc = uffd_copy_page(d, host, d.migrate_bounce)) return rc;
    if (dirty) d.writebacks.fetch_add(1, std::memory_order_relaxed);
    d.bytes_migrated.fetch_add(SG_PAGE_SIZE, std::memory_order_relaxed);
    return 0;
#else
    (void)host;
    (void)pa;
    return -ENOSYS;
#endif
}

void forget_resident(Driver& d, uint64_t va) {
    auto it = d.resident_pos.find(va);
    if (it == d.resident_pos.end()) return;
    d.resident.erase(it->second);
    d.resident_pos.erase(it);
}

// Take one resident managed page out of VRAM and return its frame. The
// shootdown, then quiesce(), guarantee no engine can still reach the frame
// through this VA; only then is the host page filled and the frame reused.
// nullopt leaves the page in VRAM (the publish failed and was rolled back).
std::optional<uint64_t> evict_locked(Driver& d, uint64_t va) {
    auto& mmu = d.dev.mmu();
    const device::PageInfo pg = mmu.page(va);
    const uint64_t old = mmu.begin_evict(va);
    if (!(old & device::PTE_IN_VRAM) || pg.host == 0) return std::nullopt;
    d.dev.quiesce();
    const uint64_t late = mmu.end_evict(va);
    const uint64_t pa = device::pte_pa(old);
    const bool dirty = (old | late) & device::PTE_DIRTY;
    if (int rc = publish_host(d, pg.host, pa, dirty)) {
        mmu.make_resident(va, pa);
        if (std::getenv("SG_TRACE"))
            std::fprintf(stderr, "[drv] eviction publish failed va=%llx rc=%d\n",
                         (unsigned long long)va, rc);
        return std::nullopt;
    }
    d.evictions.fetch_add(1, std::memory_order_relaxed);
    forget_resident(d, va);
    return pa;
}

// CLOCK: a page referenced since the hand last passed gets a second chance.
// One full pass clears every reference bit, so this always picks someone.
std::optional<uint64_t> evict_one_locked(Driver& d) {
    if (d.resident.empty()) return std::nullopt;
    for (size_t chances = d.resident.size(); chances--;) {
        if (!d.dev.mmu().test_and_clear_accessed(d.resident.front())) break;
        d.resident.splice(d.resident.end(), d.resident, d.resident.begin());
    }
    return evict_locked(d, d.resident.front());
}

// Make every managed page host-resident again and return the frames, so
// an sgMalloc can use them.
void evict_all_locked(Driver& d) {
    while (!d.resident.empty()) {
        std::optional<uint64_t> pa = evict_locked(d, d.resident.front());
        if (!pa) break;
        d.vram.free(*pa);
    }
}

// Migrate one page: take a free frame or evict for one, copy the host
// backing in, and publish the PTE. Already resident (two channels faulted
// on the same page, or prefetch got there first) is success with *moved
// left false.
int migrate_locked(Driver& d, uint64_t va, bool* moved) {
    auto& mmu = d.dev.mmu();
    const device::PageInfo pg = mmu.page(va);
    if (pg.kind == device::PageInfo::Vram) return 0;
    if (pg.kind != device::PageInfo::Host || pg.host == 0) return -EFAULT;
    std::optional<uint64_t> frame = d.vram.alloc(SG_PAGE_SIZE);
    if (!frame) frame = evict_one_locked(d);
    if (!frame) return -ENOMEM; // VRAM is all sgMalloc
    if (int rc = d.dev.copy_in(*frame, reinterpret_cast<const void*>(pg.host))) {
        d.vram.free(*frame);
        return rc;
    }
    if (int rc = mmu.make_resident(va, *frame)) {
        d.vram.free(*frame);
        return rc;
    }
    // The host copy is stale the moment the device can write the frame.
    // Drop it so the next host touch is a missing fault. After the PTE
    // publish: a failed make_resident never gets here, and the host page
    // is still populated.
    if (d.uffd >= 0 && madvise(reinterpret_cast<void*>(pg.host), SG_PAGE_SIZE, MADV_DONTNEED) != 0 &&
        std::getenv("SG_TRACE"))
        std::fprintf(stderr, "[drv] MADV_DONTNEED failed va=%llx: %s\n",
                     (unsigned long long)va, std::strerror(errno));
    d.resident.push_back(va);
    d.resident_pos[va] = std::prev(d.resident.end());
    d.migrations.fetch_add(1, std::memory_order_relaxed);
    d.bytes_migrated.fetch_add(SG_PAGE_SIZE, std::memory_order_relaxed);
    *moved = true;
    return 0;
}

// Is a fault at `va` where one of this channel's recent fault streams
// expected the next one? Returns that stream, or recycles the least
// recently used one for a new stream starting here.
FaultStream& match_stream(Driver& d, uint32_t engine, uint32_t channel, uint64_t va, bool* sequential) {
    FaultStream* set = d.streams[engine][channel];
    FaultStream* lru = &set[0];
    for (int i = 0; i < kFaultStreams; ++i) {
        if (set[i].used && set[i].next == va) {
            *sequential = true;
            set[i].used = ++d.stream_clock;
            return set[i];
        }
        if (set[i].used < lru->used) lru = &set[i];
    }
    *sequential = false;
    lru->used = ++d.stream_clock;
    return *lru;
}

// Speculatively migrate pages from `va` on, up to the end of the managed
// allocation that contains it. Returns the first page not covered.
//
// One service never migrates more than half of the frames managed memory
// can use (demand page included). Pages migrated in this service sit
// behind every older page in CLOCK order, so with that cap the evictions
// it causes only take older pages, and the newest half (other channels'
// just-migrated demand pages) survives until they retry. Uncapped, a
// window wider than the free frames evicts its own pages before the
// channel reaches them, so they migrate twice (managed_prefetch_under_
// pressure), and at worst the demand page goes before its channel resumes.
uint64_t prefetch_locked(Driver& d, uint64_t va) {
    auto it = d.managed_ranges.upper_bound(va);
    if (it == d.managed_ranges.begin()) return va;
    --it;
    const uint64_t end = it->second.end;
    const uint64_t capacity = (SG_VRAM_SIZE - d.vram.bytes_live()) / SG_PAGE_SIZE + d.resident.size();
    const uint64_t budget = std::min<uint64_t>(d.prefetch_pages, capacity / 2 > 0 ? capacity / 2 - 1 : 0);
    for (uint64_t i = 0; i < budget && va < end; ++i, va += SG_PAGE_SIZE) {
        bool moved = false;
        if (migrate_locked(d, va, &moved) != 0) break;
        if (moved) d.prefetches.fetch_add(1, std::memory_order_relaxed);
    }
    return va;
}

// Recycle a detached allocation. Caller holds alloc_lock. Unmap first so a
// stale TLB cannot observe the frame after it returns to the free list.
void recycle(Driver& d, const Pending& p) {
    if (p.kind == Pending::kPin) return;
    const uint64_t pages = p.size / SG_PAGE_SIZE;
    auto& mmu = d.dev.mmu();
    {
        Guard g(d, d.mem_lock);
        if (p.kind == Pending::kVram) {
            mmu.unmap(p.addr, pages);
            d.vram.free(p.addr);
        } else {
            d.managed_ranges.erase(p.addr);
            std::vector<uint64_t> frames;
            for (uint64_t i = 0; i < pages; ++i) {
                const uint64_t va = p.addr + i * SG_PAGE_SIZE;
                const device::PageInfo pg = mmu.page(va);
                if (pg.kind != device::PageInfo::Vram) continue;
                forget_resident(d, va);
                frames.push_back(pg.pa);
            }
            mmu.unmap(p.addr, pages);
            for (uint64_t f : frames) d.vram.free(f);
            // Unregister wakes a thread blocked in a host fault on this
            // range, then the mapping goes away. The handler may already
            // hold a fault message; it drops the message when the range
            // is gone rather than copying into the unmapped address.
#if defined(__linux__)
            if (d.uffd >= 0) uffd_unregister(d, p.host, p.size);
#endif
            if (d.uffd >= 0) ::munmap(p.host, p.size);
            else std::free(p.host);
        }
    }
    if (p.kind == Pending::kManaged) d.managed_va.free(p.addr);
}

// Recycle everything the device has finished with. Caller holds alloc_lock.
void reclaim_locked(Driver& d) {
    if (d.pending.empty()) return;
    auto keep = d.pending.begin();
    for (auto& p : d.pending) {
        if (all_retired(d, p.fence)) recycle(d, p);
        else *keep++ = p;
    }
    d.pending.erase(keep, d.pending.end());
}

// Is [addr, addr+len) entirely inside one live allocation?
bool owns(Driver& d, uint64_t addr, uint64_t len) {
    SharedGuard g(d, d.alloc_lock);
    auto it = d.allocs.upper_bound(addr);
    if (it == d.allocs.begin()) return false;
    --it;
    return addr - it->first <= it->second.size && len <= it->second.size - (addr - it->first);
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

#if defined(__linux__)
// Device VA of a host address inside some managed allocation, if any.
// Caller holds mem_lock.
std::optional<uint64_t> va_for_host(const Driver& d, uint64_t host) {
    for (const auto& [va, sp] : d.managed_ranges) {
        const uint64_t bytes = sp.end - va;
        if (host >= sp.host && host - sp.host < bytes) return va + (host - sp.host);
    }
    return std::nullopt;
}

// Host page faults. Polls the userfaultfd beside an eventfd used only to
// wake this thread on shutdown. Takes mem_lock, never alloc_lock.
void service_host_faults(Driver* self) {
    Driver& d = *self;
    while (!d.fault_stop.load(std::memory_order_acquire)) {
        pollfd pf[2] = {{d.uffd, POLLIN, 0}, {d.uffd_kick, POLLIN, 0}};
        if (poll(pf, 2, -1) < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if ((pf[1].revents) || d.fault_stop.load(std::memory_order_acquire)) break;
        if (!(pf[0].revents & POLLIN)) continue;
        uffd_msg msg{};
        const ssize_t nread = ::read(d.uffd, &msg, sizeof msg);
        if (nread != static_cast<ssize_t>(sizeof msg)) continue; // EAGAIN, or a short read
        if (msg.event != UFFD_EVENT_PAGEFAULT) continue;
        const uint64_t host = msg.arg.pagefault.address & ~(SG_PAGE_SIZE - 1);
        // Count before the test latch so a blocked fault is visible, and
        // before mem_lock so sgFree can run while this thread spins.
        d.host_faults.fetch_add(1, std::memory_order_release);
        if (d.uffd_hold.load(std::memory_order_acquire)) {
            while (d.uffd_hold.load(std::memory_order_acquire) &&
                   !d.fault_stop.load(std::memory_order_acquire))
                std::this_thread::yield();
            if (d.fault_stop.load(std::memory_order_acquire)) break;
        }
        const uint64_t t0 = now_cycles();
        {
            std::lock_guard<std::mutex> g(d.mem_lock);
            const std::optional<uint64_t> va = va_for_host(d, host);
            if (!va) {
                uffd_wake(d, host); // range already unregistered
            } else {
                const device::PageInfo pg = d.dev.mmu().page(*va);
                if (pg.kind == device::PageInfo::Vram) {
                    if (std::optional<uint64_t> pa = evict_locked(d, *va)) d.vram.free(*pa);
                    else uffd_wake(d, host);
                } else {
                    uffd_wake(d, host); // already populated, or a stale message
                }
            }
        }
        d.host_fault_ns.fetch_add(now_cycles() - t0, std::memory_order_relaxed);
    }
}
#endif

// UVM thread: migrate host-resident pages and unpark the channel that
// faulted. Takes mem_lock but never alloc_lock — ALLOC can drain while
// holding alloc_lock, and drain waits for this thread to finish the fault.
//
// The channel is unparked as soon as its own page is in; a prefetch runs
// after that, so the channel computes on the demand page while the next
// ones copy in. If it reaches a page the prefetch has not got to yet, it
// faults, and that fault finds the page resident once the window is done.
void service_faults(Driver* self) {
    Driver& d = *self;
    while (!d.fault_stop.load(std::memory_order_acquire)) {
        device::Device::Fault f;
        const uint32_t seen = d.dev.fault_irq().seq();
        if (!d.dev.pop_fault(&f)) {
            if (d.fault_stop.load(std::memory_order_acquire)) break;
            d.dev.fault_irq().wait(seen, 1000000000ull);
            continue;
        }
        const uint64_t va = device::page_floor(f.va);
        bool moved = false;
        int rc;
        {
            std::lock_guard<std::mutex> g(d.mem_lock);
            rc = migrate_locked(d, va, &moved);
        }
        if (rc != 0) d.dev.channel(f.engine, f.channel).fault_fail.store(rc, std::memory_order_release);
        d.dev.unpark(f.engine, f.channel);
        if (!moved) continue;
        bool sequential = false;
        FaultStream& s = match_stream(d, f.engine, f.channel, va, &sequential);
        s.next = va + SG_PAGE_SIZE;
        if (sequential && d.prefetch_pages) {
            std::lock_guard<std::mutex> g(d.mem_lock);
            s.next = prefetch_locked(d, s.next);
        }
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
        q->uffd = d.uffd >= 0 ? 1u : 0u;
        q->idle_ns = d.idle_ns;
        return 0;
    }
    case SG_IOC_ALLOC: {
        if (!arg) return -EINVAL;
        auto* a = static_cast<sg_alloc_args*>(arg);
        if (a->size == 0) return -EINVAL;
        ExclusiveGuard g(d, d.alloc_lock);
        reclaim_locked(d);
        auto take = [&]() -> std::optional<uint64_t> {
            Guard m(d, d.mem_lock);
            return d.vram.alloc(a->size);
        };
        auto addr = take();
        if (!addr) {
            // Maybe everything we need is sitting in the deferred list.
            if (int rc = drain(d)) return rc;
            reclaim_locked(d);
            addr = take();
        }
        if (!addr) {
            // Then managed pages give way: they can live in host memory.
            {
                Guard m(d, d.mem_lock);
                evict_all_locked(d);
            }
            addr = take();
        }
        if (!addr) {
            if (std::getenv("SG_TRACE")) {
                Guard m(d, d.mem_lock);
                std::fprintf(stderr, "[drv] ENOMEM size=%llu live=%llu pending=%zu\n",
                             (unsigned long long)a->size, (unsigned long long)d.vram.bytes_live(), d.pending.size());
            }
            return -ENOMEM;
        }
        const uint64_t pages = (a->size + SG_PAGE_SIZE - 1) / SG_PAGE_SIZE;
        if (int rc = d.dev.mmu().map_vram(*addr, *addr, pages)) {
            Guard m(d, d.mem_lock);
            d.vram.free(*addr);
            return rc;
        }
        d.allocs.emplace(*addr, Alloc{pages * SG_PAGE_SIZE, nullptr});
        a->addr = *addr;
        return 0;
    }
    case SG_IOC_ALLOC_MANAGED: {
        if (!arg) return -EINVAL;
        auto* a = static_cast<sg_alloc_managed_args*>(arg);
        if (a->size == 0) return -EINVAL;
        ExclusiveGuard g(d, d.alloc_lock);
        reclaim_locked(d);
        // Only VA is reserved here; frames are taken one page at a time on
        // first touch.
        auto addr = d.managed_va.alloc(a->size);
        if (!addr) {
            if (int rc = drain(d)) return rc;
            reclaim_locked(d);
            addr = d.managed_va.alloc(a->size);
            if (!addr) return -ENOMEM;
        }
        const uint64_t pages = (a->size + SG_PAGE_SIZE - 1) / SG_PAGE_SIZE;
        const uint64_t bytes = pages * SG_PAGE_SIZE;
        void* host = nullptr;
        if (d.uffd >= 0) {
            // Populate first, then register: a missing-mode registration
            // does not fault on pages that already have PTEs, and the
            // handler is not ready to service faults for a range it has
            // not recorded yet.
            host = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (host == MAP_FAILED) {
                d.managed_va.free(*addr);
                return -ENOMEM;
            }
            std::memset(host, 0, bytes);
#if defined(__linux__)
            if (int rc = uffd_register(d, host, bytes)) {
                ::munmap(host, bytes);
                d.managed_va.free(*addr);
                return rc;
            }
#endif
        } else if (posix_memalign(&host, SG_PAGE_SIZE, bytes) != 0) {
            d.managed_va.free(*addr);
            return -ENOMEM;
        } else {
            std::memset(host, 0, bytes);
        }
        if (int rc = d.dev.mmu().map_host(*addr, reinterpret_cast<uint64_t>(host), pages)) {
#if defined(__linux__)
            if (d.uffd >= 0) uffd_unregister(d, host, bytes);
#endif
            if (d.uffd >= 0) ::munmap(host, bytes);
            else std::free(host);
            d.managed_va.free(*addr);
            return rc;
        }
        {
            Guard m(d, d.mem_lock);
            d.managed_ranges.emplace(*addr, Driver::ManagedSpan{*addr + bytes, reinterpret_cast<uint64_t>(host)});
        }
        d.allocs.emplace(*addr, Alloc{bytes, host});
        a->addr = *addr;
        a->host_ptr = reinterpret_cast<uint64_t>(host);
        return 0;
    }
    case SG_IOC_FREE: {
        if (!arg) return -EINVAL;
        const uint64_t addr = static_cast<sg_free_args*>(arg)->addr;
        ExclusiveGuard g(d, d.alloc_lock);
        reclaim_locked(d);
        auto it = d.allocs.find(addr);
        if (it == d.allocs.end()) return -EINVAL;
        const Alloc al = it->second;
        d.allocs.erase(it);
        // Nothing may be handed this memory until every command issued so
        // far, on any channel, has retired; recycle it then rather than
        // draining now. The mapping stays until then, so an in-flight fault
        // can still copy out of the host backing.
        Pending p{al.host ? Pending::kManaged : Pending::kVram, addr, al.size, al.host, {}};
        snapshot_puts(d, p.fence);
        if (all_retired(d, p.fence)) recycle(d, p);
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
        Pending pend{Pending::kPin, 0, 0, nullptr, {}};
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
            s->faults[e] = st.faults.load(std::memory_order_relaxed);
            s->tlb_hits[e] = d.dev.mmu().tlb_hits(e);
            s->tlb_misses[e] = d.dev.mmu().tlb_misses(e);
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
        s->migrations = ld(d.migrations);
        s->prefetches = ld(d.prefetches);
        s->evictions = ld(d.evictions);
        s->writebacks = ld(d.writebacks);
        s->bytes_migrated = ld(d.bytes_migrated);
        s->host_faults = ld(d.host_faults);
        s->host_fault_ns = ld(d.host_fault_ns);
        return 0;
    }
    case SG_IOC_UFFD_HOLD:
        if (!arg) return -EINVAL;
        d.uffd_hold.store(*static_cast<const int*>(arg) ? 1 : 0, std::memory_order_release);
        return 0;
    case SG_IOC_RESET_STATS:
        d.dev.reset_stats();
        for (auto* c : {&d.submits, &d.waits, &d.waits_spun, &d.waits_blocked, &d.wake_latency_ns,
                        &d.lock_wait_ns, &d.stalls, &d.staging_waits, &d.bytes_h2d, &d.bytes_d2h,
                        &d.bytes_direct, &d.bytes_staged, &d.migrations, &d.prefetches, &d.evictions,
                        &d.writebacks, &d.bytes_migrated, &d.host_faults, &d.host_fault_ns})
            c->store(0, std::memory_order_relaxed);
        return 0;
    default:
        return -ENOTTY;
    }
}
