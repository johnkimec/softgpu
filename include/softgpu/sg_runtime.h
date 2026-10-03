#ifndef SOFTGPU_SG_RUNTIME_H
#define SOFTGPU_SG_RUNTIME_H
/*
 * softgpu runtime API — the user-facing library, shaped like the CUDA runtime.
 *
 * C ABI so it can be consumed from any language and so the boundary between
 * "application" and "driver stack" is explicit.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SG_MAX_ENGINES_RT 4 /* mirrors SG_MAX_ENGINES in the driver ABI */

typedef uint64_t sgDevPtr; /* opaque device address (VRAM offset) */

/*
 * Streams and events.
 *
 * Work submitted to one stream executes in submission order, even when it is
 * spread across the device's engines (copies run on a copy engine, kernels on
 * the compute engine). Work on different streams may overlap. The default
 * stream (NULL) is an ordinary stream: it does NOT implicitly synchronize
 * with other streams (CUDA's per-thread-default-stream semantics). Events
 * record a point in a stream; sgStreamWaitEvent makes another stream wait
 * for it on the device, without blocking the host.
 */
typedef struct sgStream* sgStream_t;
typedef struct sgEvent* sgEvent_t;

typedef enum sgError {
    SG_OK = 0,
    SG_ERR_NOT_INITIALIZED,
    SG_ERR_ALREADY_INITIALIZED,
    SG_ERR_INVALID_VALUE,
    SG_ERR_OUT_OF_MEMORY,
    SG_ERR_INVALID_ADDRESS,
    SG_ERR_DEVICE,
    SG_ERR_UNKNOWN
} sgError_t;

typedef struct sgStats {
    uint32_t num_engines;      /* engine 0 = compute, 1.. = copy engines       */
    uint32_t reserved;
    uint64_t engine_busy_cycles[SG_MAX_ENGINES_RT];  /* executing              */
    uint64_t engine_wait_cycles[SG_MAX_ENGINES_RT];  /* blocked on a semaphore */
    uint64_t engine_idle_cycles[SG_MAX_ENGINES_RT];  /* nothing queued         */
    uint64_t engine_cmds[SG_MAX_ENGINES_RT];
    uint64_t engine_batches[SG_MAX_ENGINES_RT];      /* idle->busy wakes       */
    uint64_t engine_irqs[SG_MAX_ENGINES_RT];         /* interrupts raised      */
    uint64_t engine_sleep_cycles[SG_MAX_ENGINES_RT]; /* asleep (power-gated)   */
    uint64_t engine_wakeups[SG_MAX_ENGINES_RT];
    uint64_t engine_missed_doorbells[SG_MAX_ENGINES_RT];
    uint64_t engine_cpu_ns[SG_MAX_ENGINES_RT];       /* engine thread CPU, absolute */
    uint64_t engine_faults[SG_MAX_ENGINES_RT];       /* page faults (managed memory) */
    uint64_t engine_tlb_hits[SG_MAX_ENGINES_RT];
    uint64_t engine_tlb_misses[SG_MAX_ENGINES_RT];
    uint64_t driver_submits;
    uint64_t driver_waits;     /* times the host had to wait on the device     */
    uint64_t waits_spun;       /* ... satisfied while spinning                 */
    uint64_t waits_blocked;    /* ... after sleeping for an interrupt          */
    uint64_t wake_latency_ns;  /* summed over blocked waits: retire -> awake   */
    uint64_t lock_wait_ns;     /* time submitters spent contended on driver locks */
    uint64_t driver_stalls;    /* times submission blocked on a full queue     */
    uint64_t staging_waits;    /* times the host blocked for a staging slot    */
    uint64_t bytes_h2d;
    uint64_t bytes_d2h;
    uint64_t bytes_direct;     /* copied straight to/from pinned host memory   */
    uint64_t bytes_staged;     /* bounced through the staging pool             */
    uint64_t um_migrations;    /* managed pages moved into VRAM                */
    uint64_t um_prefetches;    /* ... of which prefetched, not demanded        */
    uint64_t um_evictions;     /* managed pages moved out of VRAM              */
    uint64_t um_writebacks;    /* ... of which dirty (copied back to host)     */
    uint64_t um_bytes_migrated; /* migration traffic, both directions          */
} sgStats_t;

/* Lifecycle. Not thread-safe with respect to each other. */
sgError_t sgInit(void);
sgError_t sgShutdown(void);

/* Device memory. sgFree is safe while work referencing the memory is in
 * flight: the memory is recycled only after that work retires. */
sgError_t sgMalloc(sgDevPtr* out, size_t bytes);
sgError_t sgFree(sgDevPtr ptr);

/*
 * Managed memory. `host` is ordinary memory the CPU may read and write until
 * the device first touches the allocation; that access copies the pages into
 * VRAM. Managed memory may exceed VRAM: under pressure, pages move back to
 * `host` (with the device's writes) and fault in again on the next access.
 * sgFree releases both sides.
 */
sgError_t sgMallocManaged(sgDevPtr* dev, void** host, size_t bytes);

/*
 * Pinned host memory. Copies to/from pinned memory are DMA'd directly (no
 * staging copy) and may be asynchronous. sgMallocHost allocates and pins;
 * sgHostRegister pins memory you already own (page-aligned ranges recommended).
 */
sgError_t sgMallocHost(void** out, size_t bytes);
sgError_t sgFreeHost(void* ptr);
sgError_t sgHostRegister(void* ptr, size_t bytes);
sgError_t sgHostUnregister(void* ptr);

/* Streams and events. Destroying a stream or event does not wait for work. */
sgError_t sgStreamCreate(sgStream_t* out);
sgError_t sgStreamDestroy(sgStream_t stream);
sgError_t sgStreamSynchronize(sgStream_t stream); /* host waits for this stream's work */
sgError_t sgEventCreate(sgEvent_t* out);
sgError_t sgEventDestroy(sgEvent_t event);
sgError_t sgEventRecord(sgEvent_t event, sgStream_t stream);
sgError_t sgEventSynchronize(sgEvent_t event);    /* host waits for the recorded point */
sgError_t sgStreamWaitEvent(sgStream_t stream, sgEvent_t event); /* device-side wait */

/*
 * Copies. The synchronous forms run on the default stream and return when
 * the host buffer may be reused (H2D) or the data has arrived (D2H). The
 * *Async forms return as soon as the copy is queued when the host buffer is
 * pinned — do not touch it until the stream has synchronized — and behave
 * like the synchronous forms for pageable memory.
 */
sgError_t sgMemset(sgDevPtr dst, int value, size_t bytes);
sgError_t sgMemcpyH2D(sgDevPtr dst, const void* src, size_t bytes);
sgError_t sgMemcpyD2H(void* dst, sgDevPtr src, size_t bytes);
sgError_t sgMemcpyD2D(sgDevPtr dst, sgDevPtr src, size_t bytes);
sgError_t sgMemsetAsync(sgDevPtr dst, int value, size_t bytes, sgStream_t stream);
sgError_t sgMemcpyH2DAsync(sgDevPtr dst, const void* src, size_t bytes, sgStream_t stream);
sgError_t sgMemcpyD2HAsync(void* dst, sgDevPtr src, size_t bytes, sgStream_t stream);
sgError_t sgMemcpyD2DAsync(sgDevPtr dst, sgDevPtr src, size_t bytes, sgStream_t stream);

/* Compute. All operands are f32 in device memory. Asynchronous. */
sgError_t sgVaddF32(sgDevPtr c, sgDevPtr a, sgDevPtr b, uint32_t n);
sgError_t sgGemmF32(sgDevPtr c, sgDevPtr a, sgDevPtr b, uint32_t m, uint32_t n, uint32_t k);
sgError_t sgVaddF32Async(sgDevPtr c, sgDevPtr a, sgDevPtr b, uint32_t n, sgStream_t stream);
sgError_t sgGemmF32Async(sgDevPtr c, sgDevPtr a, sgDevPtr b, uint32_t m, uint32_t n, uint32_t k,
                         sgStream_t stream);

/*
 * How host-side waits (sgDeviceSynchronize, sgStreamSynchronize,
 * sgEventSynchronize, synchronous copies) behave, like cudaSetDeviceFlags'
 * schedule modes. DEFAULT defers to the driver's policy (spin, block, hybrid
 * or adaptive); SPIN never sleeps (lowest latency, a core per waiter); BLOCK
 * sleeps until the device's interrupt (lowest CPU, wake-up latency added).
 */
typedef enum sgSyncPolicy { SG_SYNC_DEFAULT = 0, SG_SYNC_SPIN = 1, SG_SYNC_BLOCK = 2 } sgSyncPolicy_t;
sgError_t sgSetSyncPolicy(sgSyncPolicy_t policy);

/* Wait for all submitted work on every stream and engine to finish. */
sgError_t sgDeviceSynchronize(void);

/* Telemetry. */
sgError_t sgGetStats(sgStats_t* out);
sgError_t sgResetStats(void);
const char* sgErrorString(sgError_t err);

#ifdef __cplusplus
}
#endif
#endif /* SOFTGPU_SG_RUNTIME_H */
