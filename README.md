# softgpu

A software model of a small GPU, built so the driver stack can be measured one bottleneck at a time. The device is threads and `memcpy` on a 1 GHz clock. The object of the project is everything above it: command rings, fences, interrupts, channels, a page table, a runlist, and a Linux char device.

The runtime looks like CUDA (`sgMalloc`, `sgMemcpy*`, `sgStream*`, `sgEvent*`). It talks to the driver through `sg_drv_open` / `sg_drv_ioctl` / `sg_drv_close`. On Linux, when `/dev/softgpu` is loaded, the rings and the doorbell page are kernel memory mapped into the process, and a blocked wait sleeps on a kernel wait queue. The device threads stay in the process.

```
application
  │  include/softgpu/sg_runtime.h
runtime     src/runtime/runtime.cpp
  │  sg_drv_open / sg_drv_ioctl / sg_drv_close
driver      src/driver/driver.cpp          validation, VRAM, pins, staging, faults
  │  mmap rings + doorbell; ioctl only to sleep or raise an IRQ
kernel      kmod/softgpu.c                 /dev/softgpu
device      src/device/device.cpp          engines, runlist, MMU, idle gating
```

A stage-by-stage walk of the measurements, with the idle-power curves, is at <https://jvkec.github.io/softgpu/>. The same page is [`site/index.html`](site/index.html). It has no build and no network requests. The decision records are in `docs/decisions/`.

## Stages

| stage | what changed | the number that mattered |
|---|---|---|
| 0 | mailbox, one lock, busy-poll | the baseline everything else is compared to |
| 1 | command ring, depth 1024 | round trip 208 ns on that week's fast cluster |
| 3a | pinned memory, 8×256 KiB staging | pinned 1 MiB copy at 66 GB/s |
| 3b | copy engine, channels, streams | cross-engine order via `WAIT_FENCE` |
| 2 | interrupts, adaptive host wait | spin, then sleep; 30 µs cap |
| 4 | per-channel submit, no global lock | ticket mode yields, or it livelocks |
| 7 | device idle gating, hybrid 50 µs | idle cores 2.00 → 0.03; wake ~48 µs |
| 5 | device virtual memory, prefetch 16 | sequential fault 29 → 469 MB/s; pinned copy 40 → 9 GB/s |
| 6 | channel priority and timeslice | 5 µs slice cuts pair throughput by about a third |
| 8 | `/dev/softgpu` | user-mode submit; host waits in the kernel |

Stages are numbered in the order they were designed. Idle gating (7) was built before virtual memory (5).

## Build and test

Mac is the edit loop. Do not quote Mac timings: `steady_clock` resolves to 1 µs.

```sh
cmake --preset release && cmake --build --preset release
./build/release/sgtest
```

The VM is the source of truth for numbers. Ubuntu 24.04 arm64 under UTM, kernel 6.8.0-139, the repo mounted at `/mnt/softgpu`. Build trees stay on the VM disk.

```sh
scripts/vm.sh build          # release
scripts/vm.sh test           # ctest, every knob variant
scripts/vm.sh test tsan
scripts/vm.sh test asan
scripts/vm.sh canary         # one round trip; fast cluster or the slow one
```

`scripts/vm.sh bench <tag> all --threads 8 --repeat 3` writes `results/<tag>.json`. It refuses to run when the canary is over `SG_CANARY_NS` (default 400). Placement of the VM's vCPUs on the host is bimodal, around 300 ns or around 750 ns, and the guest cannot see which one it got.

```sh
scripts/report.py results/a.json results/b.json -m p50_ns ops_per_s -b submit
```

## Kernel module

`kmod/softgpu.c` is the char device. The 9p share is not a place to build it.

```sh
scripts/vm.sh kmod
ssh -t softgpu-vm 'sudo insmod ~/build/softgpu-kmod/kmod/softgpu.ko && ls -l /dev/softgpu'
```

`SG_KMOD=require` fails `sgInit` if the device node is missing. `SG_KMOD=0` keeps the in-process rings and the futex interrupt. With the module loaded, `sgtest` is the same binary: submit stores into the mapping, and a wait that outlasts its spin sleeps in the kernel.

Loading the module needs root on the VM.

## Knobs

Read once, at `sgInit`.

| variable | default | |
|---|---|---|
| `SG_RING_DEPTH` | 1024 | commands per ring, power of two |
| `SG_COPY_ENGINES` | 1 | |
| `SG_CHANNELS` | 8 | per engine |
| `SG_WAIT_POLICY` | adaptive | `spin` `block` `hybrid` `adaptive` |
| `SG_SPIN_NS` | 30000 | host spin budget |
| `SG_ENGINE_IDLE` | hybrid | `spin` `sleep` `hybrid` `adaptive` |
| `SG_ENGINE_IDLE_NS` | 50000 | engine idle budget |
| `SG_SUBMIT_MODE` | mutex | or `ticket` |
| `SG_PREFETCH_PAGES` | 16 | managed pages pulled ahead of a sequential fault |
| `SG_TIMESLICE_NS` | 0 | 0 runs until the channel blocks or empties |
| `SG_KMOD` | try | `0` never, `require` must open `/dev/softgpu` |
| `SG_UFFD` | on if it opens | `0` is one-way managed memory. Linux only |
