#pragma once
// The "hardware": a software model of a simple accelerator.
//
// It owns a region of VRAM, an MMU (page table + per-engine TLB), and a
// register file, and runs one execution thread per engine that pulls
// commands out of that engine's ring and executes them. Nothing above this
// layer touches VRAM or registers except through the paths real hardware
// would expose: MMIO registers and DMA to host addresses the driver hands it.
//
// STAGE 5: device addresses are virtual. Engines translate through the MMU.
// sgMalloc pages are identity-mapped and always resident. A host-resident
// page (managed memory) parks the channel, records a fault, and lets the
// rest of the runlist proceed; the driver migrates the page and unparks.
// Faults are replayable: a command records how far it got and resumes there,
// so one command may touch more pages than VRAM has frames.
//
// STAGE 7: idle gating. An engine whose runlist has been empty for longer
// than its idle budget goes to sleep on a per-engine futex instead of
// spinning; a doorbell write or another engine's retirement wakes it. The
// engine's own CPU time is exported as the device power proxy.
//
// STAGE 3b: several engines, each serving several channels. Engine 0 is
// the compute engine (FILL, VADD, GEMM); engines 1.. are copy engines
// (COPY_*). A channel is a ring with its own PUT/GET and fence, executed in
// order. An engine's runlist runs the highest-priority runnable channel and
// skips one whose head is a SG_OP_WAIT_FENCE that is not yet satisfied — the
// semaphore acquire blocks that channel, not the engine.
//
// STAGE 6: priority and timeslice. A channel's `priority` uses CUDA's sign
// (more negative runs first; 0 is the default). Equal priorities take
// turns of `timeslice_ns` (0 = run until the channel blocks or empties).
// The engine yields at a command boundary, never in the middle of one.

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/clock.h"
#include "common/event.h"
#include "device/mmu.h"
#include "softgpu/sg_ioctl.h"

namespace softgpu::device {

// execute() returns this when a page is present but still in host memory.
// The channel parks; the command is not retired.
inline constexpr int kPageFault = 1;

// One channel's memory-mapped registers. Each group lives on its own cache
// line so the producer and consumer sides do not false-share.
//
// Commands live in a ring in system memory that the driver allocates and
// programs into `ring_base`/`ring_mask`. The driver advances `put` (the
// doorbell: "commands up to here are valid"); the engine advances `get` as
// it retires them. `get` therefore doubles as the fence register: command
// number N has retired once get >= N + 1. This is the GPFIFO PUT/GET scheme
// real GPUs use.
struct alignas(64) Channel {
    // ---- configuration: written by the driver before power_on() ------------
    uint64_t ring_base = 0; // address of sg_cmd[ring_mask + 1]
    uint32_t ring_mask = 0; // depth - 1; depth is a power of two

    // ---- driver -> engine -------------------------------------------------
    alignas(64) std::atomic<uint64_t> put{0};

    // ---- engine -> driver -------------------------------------------------
    alignas(64) std::atomic<uint64_t> get{0};
    std::atomic<int32_t> sticky_error{0}; // first -errno since reset, or 0

    // ---- interrupt --------------------------------------------------------
    // STAGE 2: the driver arms an interrupt for a fence value ("tell me when
    // get >= irq_target"); the engine clears it and pulses the device's IRQ
    // line when the fence passes. 0 = disarmed. last_irq_cycles is when it
    // fired, so a woken waiter can measure its wake-up latency.
    alignas(64) std::atomic<uint64_t> irq_target{0};
    std::atomic<uint64_t> last_irq_cycles{0};

    // STAGE 5: set while the head command is waiting on a page fault. The
    // engine skips the channel; the driver clears it after migrating.
    // fault_fail is a sticky -errno if the migration itself failed, so the
    // engine retires the command instead of faulting forever.
    alignas(64) std::atomic<bool> parked{false};
    std::atomic<int32_t> fault_fail{0};
    // CUDA sign: smaller runs first. 0 is the default and the lowest.
    // The driver writes it; the engine reads it at command boundaries.
    std::atomic<int32_t> priority{0};
};

// One engine's telemetry (the engine thread accounts across its channels).
struct alignas(64) EngineStats {
    std::atomic<uint64_t> busy_cycles{0}; // executing commands
    std::atomic<uint64_t> wait_cycles{0}; // work pending but every channel head blocked on a WAIT
    std::atomic<uint64_t> idle_cycles{0}; // nothing queued on any channel
    std::atomic<uint64_t> cmds_executed{0};
    std::atomic<uint64_t> batches{0};   // idle -> busy transitions
    std::atomic<uint64_t> irqs{0};      // interrupts raised
    std::atomic<uint64_t> sleep_cycles{0};     // time spent power-gated
    std::atomic<uint64_t> wakeups{0};          // sleeps ended
    std::atomic<uint64_t> missed_doorbells{0}; // timeout wake that found work waiting
    std::atomic<uint64_t> cpu_ns{0};           // this engine thread's CPU time (absolute)
    std::atomic<uint64_t> faults{0};           // commands parked on a host-resident page
    std::atomic<uint64_t> preempts{0};         // left a channel that still had commands queued
    std::atomic<uint32_t> stats_gen{0}; // bumped by reset_stats(); engine restarts its idle timer
};

class Device {
public:
    // One compute engine plus `copy_engines` copy engines, `channels` rings
    // each, and a `va_bytes` device address space over `vram_bytes` of VRAM.
    Device(uint64_t vram_bytes, uint64_t va_bytes, uint32_t copy_engines, uint32_t channels);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    uint32_t num_engines() const { return num_engines_; }
    uint32_t num_channels() const { return num_channels_; }
    Channel& channel(uint32_t engine, uint32_t ch) { return ch_[engine][ch]; }
    EngineStats& stats(uint32_t engine) { return stats_[engine]; }
    // The device's single interrupt line (one MSI): every armed channel
    // pulses it; the "ISR" on the driver side works out which fence fired.
    Event& irq() { return irq_; }
    // When the IRQ line last pulsed (device cycles); waiters use it to
    // measure their wake-up latency.
    uint64_t last_irq_cycles() const { return last_irq_.load(std::memory_order_relaxed); }
    void note_irq(uint64_t t) { last_irq_.store(t, std::memory_order_relaxed); }
    uint64_t vram_size() const { return vram_size_; }
    Mmu& mmu() { return mmu_; }

    // Page-fault sideband. Distinct from irq() so a migration does not wake
    // fence waiters. One fault per parked channel; the queue holds them.
    struct Fault {
        uint64_t va = 0; // page base
        uint32_t engine = 0;
        uint32_t channel = 0;
    };
    Event& fault_irq() { return fault_irq_; }
    void push_fault(const Fault& f);
    bool pop_fault(Fault* out);
    // Clear parked and wake the engine if it slept on the fault. Dekker
    // with the run loop: store parked, then doorbell (which fences and
    // loads asleep).
    void unpark(uint32_t engine, uint32_t channel);
    // Migration DMA: one page between a VRAM frame and host memory. The
    // driver's fault thread programs these; engines never call them.
    int copy_in(uint64_t pa, const void* host);
    int copy_out(uint64_t pa, void* host);
    // Shootdown acknowledgement: returns once every engine that was inside
    // a command when this was called has finished it. After an MMU
    // shootdown, nothing can still be touching the old frame.
    void quiesce();

    // Idle gating policy; set before power_on().
    void set_idle_policy(uint32_t policy, uint64_t idle_ns) { idle_policy_ = policy; idle_ns_ = idle_ns; }
    // 0: a visit runs until the channel blocks or empties. Set before power_on().
    void set_timeslice_ns(uint64_t ns) { timeslice_ns_ = ns; }
    // Out-of-range priority returns -EINVAL. Visible to the engine at its
    // next command boundary.
    int set_channel_priority(uint32_t engine, uint32_t channel, int32_t priority);

    // Bring the engines up / down. power_off() blocks until every engine
    // thread has exited; commands in flight are completed first. `cpu` >= 0
    // pins the compute engine's thread to that CPU (Linux only).
    void power_on(int cpu = -1);
    void power_off();

    // The driver rings this after storing a channel's PUT. Wakes the engine
    // if it is power-gated. Dekker pattern with the engine's arm/re-check:
    // both sides store then load, with seq_cst fences in between.
    void doorbell(uint32_t engine) {
        if (!no_fence_) std::atomic_thread_fence(std::memory_order_seq_cst);
        if (asleep_[engine].load(std::memory_order_relaxed)) {
            // Timestamp the wake so the engine's idle-gap estimator measures
            // the gap to the doorbell, not to its own (slow) wake-up.
            wake_stamp_[engine].store(now_cycles(), std::memory_order_relaxed);
            bell_[engine].signal();
        }
    }
    // SG_EXPERIMENT_NO_FENCE=1: drop the Dekker fences on both sides so the
    // lost-doorbell race can be *measured* (ADR 006). Never on by default.
    void set_no_fence(bool v) { no_fence_ = v; }

    void reset_stats();

private:
    // Progress of a channel's head command across page faults. Engine
    // thread only; reset when the command retires.
    struct Replay {
        uint64_t done = 0;   // bytes of the current phase already applied
        uint32_t phase = 0;  // gather path: which operand is being moved
        bool gather = false; // operands are not one contiguous resident run
        std::vector<uint8_t> a, b, c;
    };

    void run(uint32_t engine);
    int execute(uint32_t engine, Replay& r, const sg_cmd& cmd, uint64_t* fault_va);
    // The longest resident, physically contiguous run at `va`, at most
    // `len` bytes. A host-resident first page returns kPageFault and
    // writes the page base to fault_va.
    int run_at(uint32_t engine, uint64_t va, uint64_t len, bool write, uint8_t** out, uint64_t* n,
               uint64_t* fault_va);
    // Resolve a whole range to one contiguous VRAM pointer, or kPageFault,
    // or kSplit if it is resident but not one run.
    int resolve(uint32_t engine, uint64_t va, uint64_t len, bool write, uint8_t** out,
                uint64_t* fault_va);
    // Apply `op(ptr, offset, n)` run by run over [va, va+len), resuming at
    // r.done. A fault returns with r.done at the first unapplied byte.
    template <class Op>
    int for_runs(uint32_t engine, Replay& r, uint64_t va, uint64_t len, bool write,
                 uint64_t* fault_va, Op op);

    // Wake gated engines that went to sleep with work pending behind a
    // WAIT_FENCE (a retirement may have satisfied it). Engines sleeping
    // with nothing queued are only ever woken by a doorbell.
    void wake_blocked_sleepers(uint32_t self);

    Channel ch_[SG_MAX_ENGINES][SG_MAX_CHANNELS];
    Replay replay_[SG_MAX_ENGINES][SG_MAX_CHANNELS];
    EngineStats stats_[SG_MAX_ENGINES];
    // Odd while the engine is inside execute(); see quiesce().
    struct alignas(64) AccessSeq { std::atomic<uint64_t> v{0}; };
    AccessSeq access_[SG_MAX_ENGINES];
    Event irq_;
    std::atomic<uint64_t> last_irq_{0};
    Event fault_irq_;
    std::mutex fault_mu_;
    std::deque<Fault> faults_;
    Event bell_[SG_MAX_ENGINES];
    std::atomic<bool> asleep_[SG_MAX_ENGINES] = {};
    std::atomic<bool> asleep_blocked_[SG_MAX_ENGINES] = {}; // asleep with work pending behind a WAIT_FENCE
    std::atomic<uint64_t> wake_stamp_[SG_MAX_ENGINES] = {}; // when the last wake was requested
    std::atomic<uint32_t> num_asleep_blocked_{0};
    uint32_t idle_policy_ = SG_IDLE_HYBRID;
    uint64_t idle_ns_ = 50000;
    uint64_t timeslice_ns_ = 0;
    // True once any channel priority is not the default. The run loop
    // skips the priority scan while this is false.
    std::atomic<bool> mixed_priority_{false};
    bool no_fence_ = false;
    uint32_t num_engines_;
    uint32_t num_channels_;
    std::unique_ptr<uint8_t[]> vram_;
    uint64_t vram_size_;
    Mmu mmu_;
    std::atomic<bool> running_{false};
    std::vector<std::thread> engines_;
};

} // namespace softgpu::device
