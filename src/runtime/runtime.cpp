// User-mode runtime: implements sg_runtime.h on top of the driver's
// ioctl-shaped interface. It owns no device state — only a descriptor plus
// the stream and event objects — and never touches VRAM or registers.
//
// STAGE 3b: streams. The device has a compute engine and copy engines, each
// serving several channels (in-order rings). A stream owns one channel index
// and keeps its commands in order *across* engines: before submitting to
// engine E, if the stream's previous command ran on E' != E, it first
// submits a SG_OP_WAIT_FENCE on (E, channel) for that command's fence. Waiting on the immediate
// predecessor is enough because rings are in order and the predecessor
// already waited on its own predecessor. Events are (engine, fence)
// snapshots; sgStreamWaitEvent turns them into extra WAITs ahead of the
// stream's next command. This is how CUDA streams sit on top of channels and
// semaphores.

#include "softgpu/sg_runtime.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "driver/sg_driver.h"
#include "softgpu/sg_ioctl.h"

struct sgFence {
    uint32_t engine = 0;
    uint32_t channel = 0;
    uint64_t value = 0;
};

struct sgStream {
    std::mutex m;
    uint32_t id = 0;
    uint32_t ce = 1;        // copy engine this stream's copies go to
    uint32_t channel = 0;   // ring index used on every engine
    bool has_tail = false;  // has any command been submitted?
    sgFence tail;           // the most recent command
    std::vector<sgFence> extra_waits; // from sgStreamWaitEvent
};

struct sgEvent {
    std::mutex m;
    bool recorded = false;
    sgFence at;
};

namespace {

int g_fd = -1;
int g_managed_coherent = 0;
uint32_t g_num_engines = 1;
uint32_t g_num_ce = 0;
uint32_t g_num_channels = 1;
int g_prio_least = SG_PRIORITY_LEAST;
int g_prio_greatest = SG_PRIORITY_GREATEST;
uint64_t g_timeslice_ns = 0;
std::atomic<uint32_t> g_sync_flags{SG_WAIT_DEFAULT}; // from sgSetSyncPolicy
std::atomic<uint32_t> g_next_stream_id{1};
sgStream g_default_stream; // NULL maps here; id 0

// Which priority each channel index is programmed to, and how many streams
// use it. The same index is used on every engine. The default stream holds
// channel 0 at priority 0 for the life of the context, so a higher-priority
// stream takes a different channel and the two can be scheduled apart.
struct ChanBind {
    int priority = 0;
    int users = 0;
};
std::mutex g_bind_mu;
ChanBind g_bind[SG_MAX_CHANNELS];

sgError_t from_errno(int rc);

int clamp_priority(int priority) {
    if (priority < g_prio_greatest) return g_prio_greatest;
    if (priority > g_prio_least) return g_prio_least;
    return priority;
}

int set_priority_all(uint32_t channel, int priority) {
    for (uint32_t e = 0; e < g_num_engines; ++e) {
        sg_priority_args a{};
        a.engine = e;
        a.channel = channel;
        a.priority = priority;
        if (int rc = sg_drv_ioctl(g_fd, SG_IOC_SET_PRIORITY, &a)) return rc;
    }
    return 0;
}

// Caller holds g_bind_mu. On success the stream's channel is set and that
// slot's user count is incremented.
sgError_t bind_channel(sgStream* s, int priority) {
    int found = -1;
    int best_users = 0;
    for (uint32_t c = 0; c < g_num_channels; ++c) {
        if (g_bind[c].priority != priority) continue;
        if (found < 0 || g_bind[c].users < best_users) {
            found = int(c);
            best_users = g_bind[c].users;
        }
    }
    if (found < 0) {
        for (uint32_t c = 0; c < g_num_channels; ++c) {
            if (g_bind[c].users == 0) {
                found = int(c);
                break;
            }
        }
        if (found < 0) return SG_ERR_INVALID_VALUE;
        if (int rc = set_priority_all(uint32_t(found), priority)) {
            set_priority_all(uint32_t(found), SG_PRIORITY_LEAST);
            return from_errno(rc);
        }
        g_bind[found].priority = priority;
    }
    g_bind[found].users++;
    s->channel = uint32_t(found);
    return SG_OK;
}

void unbind_channel(sgStream* s) {
    std::lock_guard<std::mutex> g(g_bind_mu);
    ChanBind& slot = g_bind[s->channel];
    if (slot.users > 0) slot.users--;
    if (slot.users == 0 && slot.priority != SG_PRIORITY_LEAST) {
        // In-flight commands stay on the ring; they drop to the default
        // priority, which is the only one left that can claim this channel.
        if (set_priority_all(s->channel, SG_PRIORITY_LEAST) == 0) slot.priority = SG_PRIORITY_LEAST;
    }
}

sgError_t create_stream(sgStream_t* out, int priority) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!out) return SG_ERR_INVALID_VALUE;
    auto* s = new sgStream();
    s->id = g_next_stream_id.fetch_add(1, std::memory_order_relaxed);
    // Spread copies across copy engines. The channel is chosen separately
    // so two priorities never have to share a ring.
    s->ce = g_num_ce ? 1 + (s->id % g_num_ce) : 0;
    std::lock_guard<std::mutex> g(g_bind_mu);
    if (sgError_t e = bind_channel(s, clamp_priority(priority))) {
        delete s;
        return e;
    }
    *out = s;
    return SG_OK;
}

sgError_t from_errno(int rc) {
    switch (rc) {
    case 0:       return SG_OK;
    case -EINVAL: return SG_ERR_INVALID_VALUE;
    case -ENOMEM: return SG_ERR_OUT_OF_MEMORY;
    case -EFAULT: return SG_ERR_INVALID_ADDRESS;
    case -EBADF:  return SG_ERR_NOT_INITIALIZED;
    case -EIO:    return SG_ERR_DEVICE;
    case -EEXIST: return SG_ERR_INVALID_VALUE;
    default:      return SG_ERR_UNKNOWN;
    }
}

sgError_t wait_fence(const sgFence& f) {
    sg_wait_args w{f.engine, f.channel, f.value, g_sync_flags.load(std::memory_order_relaxed), 0};
    return from_errno(sg_drv_ioctl(g_fd, SG_IOC_WAIT, &w));
}

// Everything submitted so far by any thread, on every engine and channel.
sgError_t wait_all() {
    sg_wait_args w{SG_WAIT_ALL, 0, 0, g_sync_flags.load(std::memory_order_relaxed), 0};
    return from_errno(sg_drv_ioctl(g_fd, SG_IOC_WAIT, &w));
}

sg_cmd make_cmd(sg_opcode op) {
    sg_cmd c{};
    c.opcode = op;
    return c;
}

sgStream* resolve(sgStream_t s) { return s ? s : &g_default_stream; }

// Raw submit of one command on one (engine, channel). `direct` (optional)
// reports whether the driver DMA'd straight to/from the user buffer.
int raw_submit(uint32_t engine, uint32_t channel, const sg_cmd& cmd, uint64_t host_ptr,
               uint64_t* fence, bool* direct) {
    sg_submit_args a{};
    a.cmd = cmd;
    a.host_ptr = host_ptr;
    a.engine = engine;
    a.channel = channel;
    int rc = sg_drv_ioctl(g_fd, SG_IOC_SUBMIT, &a);
    if (rc == 0) {
        if (fence) *fence = a.fence;
        if (direct) *direct = (a.out_flags & SG_SUBMIT_DIRECT) != 0;
    }
    return rc;
}

// Submit on a stream: flush the stream's pending cross-engine dependencies
// as WAIT_FENCE commands on `engine`, then the command itself. Holding the
// stream mutex across the ioctls makes the recorded order the submission
// order.
sgError_t stream_submit(sgStream* s, uint32_t engine, sg_cmd cmd, uint64_t host_ptr = 0,
                        uint64_t* fence_out = nullptr, bool* direct = nullptr) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> g(s->m);

    auto wait_on = [&](const sgFence& f) -> int {
        if (f.engine == engine && f.channel == s->channel) return 0; // same ring, already ordered
        sg_cmd w = make_cmd(SG_OP_WAIT_FENCE);
        w.arg0 = f.engine;
        w.arg1 = f.channel;
        w.src1 = f.value;
        return raw_submit(engine, s->channel, w, 0, nullptr, nullptr);
    };
    for (auto& f : s->extra_waits)
        if (int rc = wait_on(f)) return from_errno(rc);
    s->extra_waits.clear();
    if (s->has_tail)
        if (int rc = wait_on(s->tail)) return from_errno(rc);

    uint64_t fence = 0;
    int rc = raw_submit(engine, s->channel, cmd, host_ptr, &fence, direct);
    if (rc == 0) {
        s->has_tail = true;
        s->tail = {engine, s->channel, fence};
        if (fence_out) *fence_out = fence;
    }
    return from_errno(rc);
}

// Synchronous copy contract: if the driver went direct, the user buffer is
// still being read/written by the device until the fence retires.
sgError_t copy_sync(sgStream* s, sg_cmd cmd, uint64_t host_ptr) {
    uint64_t fence = 0;
    bool direct = false;
    sgError_t e = stream_submit(s, s->ce, cmd, host_ptr, &fence, &direct);
    if (e != SG_OK || !direct) return e;
    return wait_fence({s->ce, s->channel, fence});
}

} // namespace

extern "C" {

sgError_t sgInit(void) {
    if (g_fd >= 0) return SG_ERR_ALREADY_INITIALIZED;
    int fd = sg_drv_open();
    if (fd < 0) return from_errno(fd);
    sg_query_args q{};
    if (int rc = sg_drv_ioctl(fd, SG_IOC_QUERY, &q); rc != 0 || q.abi_version != SG_ABI_VERSION) {
        sg_drv_close(fd);
        return rc ? from_errno(rc) : SG_ERR_DEVICE;
    }
    g_fd = fd;
    g_managed_coherent = q.uffd ? 1 : 0;
    g_num_engines = q.num_engines;
    g_num_ce = q.num_engines - 1;
    g_num_channels = q.num_channels;
    g_prio_least = q.priority_least;
    g_prio_greatest = q.priority_greatest;
    g_timeslice_ns = q.timeslice_ns;
    {
        std::lock_guard<std::mutex> g(g_bind_mu);
        for (uint32_t c = 0; c < SG_MAX_CHANNELS; ++c) g_bind[c] = {};
        g_bind[0].users = 1; // the default stream
        std::lock_guard<std::mutex> d(g_default_stream.m);
        g_default_stream.has_tail = false;
        g_default_stream.extra_waits.clear();
        g_default_stream.ce = g_num_ce ? 1 : 0;
        g_default_stream.channel = 0;
    }
    return SG_OK;
}

sgError_t sgShutdown(void) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    int rc = sg_drv_close(g_fd);
    g_fd = -1;
    g_managed_coherent = 0;
    return from_errno(rc);
}

int sgManagedCoherent(void) {
    return g_fd >= 0 && g_managed_coherent ? 1 : 0;
}

void sgHoldHostFaults(int hold) {
    if (g_fd < 0) return;
    sg_drv_ioctl(g_fd, SG_IOC_UFFD_HOLD, &hold);
}

sgError_t sgMalloc(sgDevPtr* out, size_t bytes) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!out || bytes == 0) return SG_ERR_INVALID_VALUE;
    sg_alloc_args a{};
    a.size = bytes;
    int rc = sg_drv_ioctl(g_fd, SG_IOC_ALLOC, &a);
    if (rc == 0) *out = a.addr;
    return from_errno(rc);
}

sgError_t sgFree(sgDevPtr ptr) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    sg_free_args a{ptr};
    return from_errno(sg_drv_ioctl(g_fd, SG_IOC_FREE, &a));
}

sgError_t sgMallocManaged(sgDevPtr* dev, void** host, size_t bytes) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!dev || !host || bytes == 0) return SG_ERR_INVALID_VALUE;
    sg_alloc_managed_args a{};
    a.size = bytes;
    int rc = sg_drv_ioctl(g_fd, SG_IOC_ALLOC_MANAGED, &a);
    if (rc == 0) {
        *dev = a.addr;
        *host = reinterpret_cast<void*>(a.host_ptr);
    }
    return from_errno(rc);
}

sgError_t sgMallocHost(void** out, size_t bytes) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!out || bytes == 0) return SG_ERR_INVALID_VALUE;
    const size_t page = 4096;
    const size_t rounded = (bytes + page - 1) / page * page;
    void* p = nullptr;
    if (posix_memalign(&p, page, rounded) != 0) return SG_ERR_OUT_OF_MEMORY;
    sg_pin_args a{};
    a.addr = reinterpret_cast<uint64_t>(p);
    a.size = rounded;
    int rc = sg_drv_ioctl(g_fd, SG_IOC_PIN, &a);
    if (rc != 0) {
        std::free(p);
        return from_errno(rc);
    }
    *out = p;
    return SG_OK;
}

sgError_t sgFreeHost(void* ptr) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!ptr) return SG_ERR_INVALID_VALUE;
    sg_pin_args a{};
    a.addr = reinterpret_cast<uint64_t>(ptr);
    int rc = sg_drv_ioctl(g_fd, SG_IOC_UNPIN, &a);
    if (rc != 0) return from_errno(rc);
    // The device may still be DMAing to/from this memory; like cudaFreeHost,
    // wait for everything queued so far before returning it to the allocator.
    sgError_t e = wait_all();
    std::free(ptr);
    return e;
}

sgError_t sgHostRegister(void* ptr, size_t bytes) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!ptr || bytes == 0) return SG_ERR_INVALID_VALUE;
    sg_pin_args a{};
    a.addr = reinterpret_cast<uint64_t>(ptr);
    a.size = bytes;
    return from_errno(sg_drv_ioctl(g_fd, SG_IOC_PIN, &a));
}

sgError_t sgHostUnregister(void* ptr) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!ptr) return SG_ERR_INVALID_VALUE;
    sg_pin_args a{};
    a.addr = reinterpret_cast<uint64_t>(ptr);
    // The memory stays valid (the caller owns it), so no need to wait for
    // in-flight DMAs here; the driver defers the unpin itself.
    return from_errno(sg_drv_ioctl(g_fd, SG_IOC_UNPIN, &a));
}

// ---- streams and events ------------------------------------------------

sgError_t sgStreamCreate(sgStream_t* out) { return create_stream(out, SG_PRIORITY_LEAST); }

sgError_t sgStreamCreateWithPriority(sgStream_t* out, int priority) {
    return create_stream(out, priority);
}

void sgDeviceGetStreamPriorityRange(int* least, int* greatest) {
    if (least) *least = g_fd >= 0 ? g_prio_least : 0;
    if (greatest) *greatest = g_fd >= 0 ? g_prio_greatest : 0;
}

uint64_t sgTimesliceNs(void) { return g_fd >= 0 ? g_timeslice_ns : 0; }

uint32_t sgNumChannels(void) { return g_fd >= 0 ? g_num_channels : 0; }

sgError_t sgStreamDestroy(sgStream_t stream) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!stream) return SG_ERR_INVALID_VALUE;
    unbind_channel(stream); // in-flight commands do not reference the object
    delete stream;
    return SG_OK;
}

sgError_t sgStreamSynchronize(sgStream_t stream) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    sgStream* s = resolve(stream);
    sgFence f;
    {
        std::lock_guard<std::mutex> g(s->m);
        if (!s->has_tail) return SG_OK;
        f = s->tail;
    }
    return wait_fence(f);
}

sgError_t sgEventCreate(sgEvent_t* out) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!out) return SG_ERR_INVALID_VALUE;
    *out = new sgEvent();
    return SG_OK;
}

sgError_t sgEventDestroy(sgEvent_t event) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!event) return SG_ERR_INVALID_VALUE;
    delete event;
    return SG_OK;
}

sgError_t sgEventRecord(sgEvent_t event, sgStream_t stream) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!event) return SG_ERR_INVALID_VALUE;
    sgStream* s = resolve(stream);
    // Recording is itself a command in the stream (a NOP), so it flushes any
    // pending cross-stream waits and the event captures everything before it.
    uint32_t engine, channel;
    {
        std::lock_guard<std::mutex> g(s->m);
        engine = s->has_tail ? s->tail.engine : SG_ENGINE_COMPUTE;
        channel = s->channel;
    }
    uint64_t fence = 0;
    sgError_t e = stream_submit(s, engine, make_cmd(SG_OP_NOP), 0, &fence);
    if (e != SG_OK) return e;
    std::lock_guard<std::mutex> g(event->m);
    event->recorded = true;
    event->at = {engine, channel, fence};
    return SG_OK;
}

sgError_t sgEventSynchronize(sgEvent_t event) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!event) return SG_ERR_INVALID_VALUE;
    sgFence f;
    {
        std::lock_guard<std::mutex> g(event->m);
        if (!event->recorded) return SG_OK;
        f = event->at;
    }
    return wait_fence(f);
}

sgError_t sgStreamWaitEvent(sgStream_t stream, sgEvent_t event) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!event) return SG_ERR_INVALID_VALUE;
    sgStream* s = resolve(stream);
    sgFence f;
    {
        std::lock_guard<std::mutex> g(event->m);
        if (!event->recorded) return SG_OK; // nothing to wait for
        f = event->at;
    }
    std::lock_guard<std::mutex> g(s->m);
    s->extra_waits.push_back(f);
    return SG_OK;
}

// ---- copies --------------------------------------------------------------

sgError_t sgMemsetAsync(sgDevPtr dst, int value, size_t bytes, sgStream_t stream) {
    sg_cmd c = make_cmd(SG_OP_FILL);
    c.dst = dst;
    c.size = bytes;
    c.value = static_cast<uint32_t>(value) & 0xffu;
    return stream_submit(resolve(stream), SG_ENGINE_COMPUTE, c);
}

sgError_t sgMemset(sgDevPtr dst, int value, size_t bytes) {
    return sgMemsetAsync(dst, value, bytes, nullptr);
}

sgError_t sgMemcpyH2D(sgDevPtr dst, const void* src, size_t bytes) {
    if (!src) return SG_ERR_INVALID_VALUE;
    sg_cmd c = make_cmd(SG_OP_COPY_H2D);
    c.dst = dst;
    c.size = bytes;
    return copy_sync(&g_default_stream, c, reinterpret_cast<uint64_t>(src));
}

sgError_t sgMemcpyD2H(void* dst, sgDevPtr src, size_t bytes) {
    if (!dst) return SG_ERR_INVALID_VALUE;
    sg_cmd c = make_cmd(SG_OP_COPY_D2H);
    c.src0 = src;
    c.size = bytes;
    return copy_sync(&g_default_stream, c, reinterpret_cast<uint64_t>(dst));
}

sgError_t sgMemcpyD2DAsync(sgDevPtr dst, sgDevPtr src, size_t bytes, sgStream_t stream) {
    sgStream* s = resolve(stream);
    sg_cmd c = make_cmd(SG_OP_COPY_D2D);
    c.dst = dst;
    c.src0 = src;
    c.size = bytes;
    return stream_submit(s, s->ce, c);
}

sgError_t sgMemcpyD2D(sgDevPtr dst, sgDevPtr src, size_t bytes) {
    return sgMemcpyD2DAsync(dst, src, bytes, nullptr);
}

sgError_t sgMemcpyH2DAsync(sgDevPtr dst, const void* src, size_t bytes, sgStream_t stream) {
    if (!src) return SG_ERR_INVALID_VALUE;
    sgStream* s = resolve(stream);
    sg_cmd c = make_cmd(SG_OP_COPY_H2D);
    c.dst = dst;
    c.size = bytes;
    return stream_submit(s, s->ce, c, reinterpret_cast<uint64_t>(src));
}

sgError_t sgMemcpyD2HAsync(void* dst, sgDevPtr src, size_t bytes, sgStream_t stream) {
    if (!dst) return SG_ERR_INVALID_VALUE;
    sgStream* s = resolve(stream);
    sg_cmd c = make_cmd(SG_OP_COPY_D2H);
    c.src0 = src;
    c.size = bytes;
    return stream_submit(s, s->ce, c, reinterpret_cast<uint64_t>(dst));
}

// ---- compute -------------------------------------------------------------

sgError_t sgVaddF32Async(sgDevPtr c_, sgDevPtr a, sgDevPtr b, uint32_t n, sgStream_t stream) {
    sg_cmd c = make_cmd(SG_OP_VADD_F32);
    c.dst = c_;
    c.src0 = a;
    c.src1 = b;
    c.arg0 = n;
    return stream_submit(resolve(stream), SG_ENGINE_COMPUTE, c);
}

sgError_t sgVaddF32(sgDevPtr c, sgDevPtr a, sgDevPtr b, uint32_t n) {
    return sgVaddF32Async(c, a, b, n, nullptr);
}

sgError_t sgGemmF32Async(sgDevPtr c_, sgDevPtr a, sgDevPtr b, uint32_t m, uint32_t n, uint32_t k,
                         sgStream_t stream) {
    sg_cmd c = make_cmd(SG_OP_GEMM_F32);
    c.dst = c_;
    c.src0 = a;
    c.src1 = b;
    c.arg0 = m;
    c.arg1 = n;
    c.arg2 = k;
    return stream_submit(resolve(stream), SG_ENGINE_COMPUTE, c);
}

sgError_t sgGemmF32(sgDevPtr c, sgDevPtr a, sgDevPtr b, uint32_t m, uint32_t n, uint32_t k) {
    return sgGemmF32Async(c, a, b, m, n, k, nullptr);
}

// ---- synchronization and telemetry -----------------------------------------

sgError_t sgSetSyncPolicy(sgSyncPolicy_t policy) {
    switch (policy) {
    case SG_SYNC_DEFAULT: g_sync_flags.store(SG_WAIT_DEFAULT, std::memory_order_relaxed); return SG_OK;
    case SG_SYNC_SPIN:    g_sync_flags.store(SG_WAIT_SPIN, std::memory_order_relaxed); return SG_OK;
    case SG_SYNC_BLOCK:   g_sync_flags.store(SG_WAIT_BLOCK, std::memory_order_relaxed); return SG_OK;
    default:              return SG_ERR_INVALID_VALUE;
    }
}

sgError_t sgDeviceSynchronize(void) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    return wait_all();
}

sgError_t sgGetStats(sgStats_t* out) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!out) return SG_ERR_INVALID_VALUE;
    sg_stats_args s{};
    int rc = sg_drv_ioctl(g_fd, SG_IOC_STATS, &s);
    if (rc == 0) {
        std::memset(out, 0, sizeof *out);
        out->num_engines = s.num_engines;
        for (uint32_t e = 0; e < s.num_engines && e < SG_MAX_ENGINES_RT; ++e) {
            out->engine_busy_cycles[e] = s.busy_cycles[e];
            out->engine_wait_cycles[e] = s.wait_cycles[e];
            out->engine_idle_cycles[e] = s.idle_cycles[e];
            out->engine_cmds[e] = s.cmds_executed[e];
            out->engine_batches[e] = s.batches[e];
            out->engine_irqs[e] = s.irqs[e];
            out->engine_sleep_cycles[e] = s.sleep_cycles[e];
            out->engine_wakeups[e] = s.wakeups[e];
            out->engine_missed_doorbells[e] = s.missed_doorbells[e];
            out->engine_cpu_ns[e] = s.cpu_ns[e];
            out->engine_faults[e] = s.faults[e];
            out->engine_preempts[e] = s.preempts[e];
            out->engine_tlb_hits[e] = s.tlb_hits[e];
            out->engine_tlb_misses[e] = s.tlb_misses[e];
        }
        out->driver_submits = s.submits;
        out->driver_waits = s.waits;
        out->waits_spun = s.waits_spun;
        out->waits_blocked = s.waits_blocked;
        out->wake_latency_ns = s.wake_latency_ns;
        out->lock_wait_ns = s.lock_wait_ns;
        out->driver_stalls = s.stalls;
        out->staging_waits = s.staging_waits;
        out->bytes_h2d = s.bytes_h2d;
        out->bytes_d2h = s.bytes_d2h;
        out->bytes_direct = s.bytes_direct;
        out->bytes_staged = s.bytes_staged;
        out->um_migrations = s.migrations;
        out->um_prefetches = s.prefetches;
        out->um_evictions = s.evictions;
        out->um_writebacks = s.writebacks;
        out->um_bytes_migrated = s.bytes_migrated;
        out->um_host_faults = s.host_faults;
        out->um_host_fault_ns = s.host_fault_ns;
    }
    return from_errno(rc);
}

sgError_t sgResetStats(void) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    return from_errno(sg_drv_ioctl(g_fd, SG_IOC_RESET_STATS, nullptr));
}

const char* sgErrorString(sgError_t err) {
    switch (err) {
    case SG_OK:                      return "no error";
    case SG_ERR_NOT_INITIALIZED:     return "runtime not initialized";
    case SG_ERR_ALREADY_INITIALIZED: return "runtime already initialized";
    case SG_ERR_INVALID_VALUE:       return "invalid value";
    case SG_ERR_OUT_OF_MEMORY:       return "out of device memory";
    case SG_ERR_INVALID_ADDRESS:     return "invalid device address";
    case SG_ERR_DEVICE:              return "device error";
    default:                         return "unknown error";
    }
}

} // extern "C"
