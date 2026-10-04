# 006 — Device idle gating

## Context

Stage 2 ([004](004-interrupts-and-wait-policy.md)) fixed the host side of
"who burns a core waiting": a thread waiting on the device sleeps until an
interrupt. The device side never changed. Both engine threads have spun at
100% on empty runlists since stage 0 — an idle `softgpu` costs two full
cores — and every `dev_util` figure so far has been honest about *work*
and silent about *power*. Real GPUs clock-gate and power-gate idle engines
behind a hysteresis timer and pay for it with wake-up latency on the first
command after an idle period.

This stage builds that, in the same shape as stage 2 pointed the other way:
an engine spins briefly on an empty runlist, then sleeps on a futex until a
doorbell — or another engine's fence — wakes it.

## Problem

The device's idle power is 2.0 cores by construction, and nothing in the
model can trade it against latency.

## Options considered

1. **Leave the engines spinning.** Zero wake-up latency; two cores forever.
2. **`sched_yield()` on empty rounds.** Cheaper under oversubscription, still
   a busy loop, unbounded wake-up under load.
3. **Sleep after an idle budget, wake on doorbell** (this record), with the
   budget fixed (*hybrid*) or predicted from recent idle-gap lengths
   (*adaptive*), plus `spin` and `sleep` as the two ends of the curve.

## Decision

**Device** (`src/device`): each engine gets a doorbell `Event` (the stage 2
futex primitive) and an `asleep` flag. In the runlist's no-progress branch
the engine tracks how long the current idle stretch has lasted; past the
budget it **arms** (`asleep = true`), **re-checks** every channel's PUT and
every blocked head's fence, and only then sleeps (1 ms safety timeout). On
wake it clears the flag and resumes. Two wake sources: the driver's
`Device::doorbell(engine)` after every PUT store, and `wake_sleepers()`
after every retirement (a sleeping engine may be blocked on that fence).
`power_off` wakes everyone so they can exit.

**Dekker.** Driver: store PUT, fence, load `asleep`. Engine: store `asleep`,
fence, load PUTs. Both are store-then-load; without `seq_cst` fences on
both sides ARM64 may reorder either, and a doorbell rung between the
engine's check and its sleep is lost until the timeout. The device counts
such events (`missed_doorbells`: a timeout wake that finds work), the
`gating_no_missed_doorbells` test asserts zero, and
`SG_EXPERIMENT_NO_FENCE=1` removes the fences so the race can be measured
rather than asserted.

**Policy** (`SG_ENGINE_IDLE`, `SG_ENGINE_IDLE_NS`): `spin`, `sleep`,
`hybrid` (fixed budget), `adaptive` (EWMA of recent idle-gap lengths —
long gaps → sleep after a 2 µs floor, short gaps → spin up to 2× the
typical gap, capped). **Default: `hybrid` with a 50 µs budget** — a fixed
hysteresis timer, which is also what real GPUs do. Adaptive is kept as a
knob: even after fixing its estimator to measure gaps to the doorbell
timestamp rather than to its own wake-up (the same self-feedback bug stage
2's wait estimator had), it still sleeps on some sub-2 µs gaps and costs
~3× the round trip there (see *Results*). A fixed budget has no state to
get wrong.

**Wake only the sleepers that need it.** The first cut woke *every*
sleeping engine on every retirement, so an idle copy engine was kicked
awake by each memset the compute engine retired: it never got back to sleep
at short gaps (0.4–0.8 cores of pure overhead) and at long gaps every
retirement paid a `FUTEX_WAKE` syscall (+600 ns on the round trip). Only
engines that went to sleep *with work pending behind a WAIT_FENCE* are
woken by retirements; an idle sleeper is woken by its doorbell alone.

**CPU-time accounting is a syscall.** `CLOCK_THREAD_CPUTIME_ID` is not a
vDSO read on Linux; publishing it on every idle→busy transition taxed the
first command after any gap by ~130 ns. It is published at most once a
millisecond instead.

**Telemetry:** per-engine `sleep_cycles`, `wakeups`, `missed_doorbells`, and
`cpu_ns` — the engine thread's own `CLOCK_THREAD_CPUTIME_ID`, published on
every idle→busy transition, before each sleep, and about once a millisecond
while spinning. The benchmark reports it as `dev_cpu_cores` (engine CPU
per wall second; 2.0 = both engines spinning) — the device power proxy,
which until now was 2.0 in every row without being shown.

**ABI v7**: the stats above plus `QUERY.idle_policy`/`idle_ns`. No API
change.

**Not changed:** host wait policy, channels, locks, staging, allocator.

## Hypotheses (written before measuring)

* **H1** Idle device: engine CPU falls from 2.0 cores to < 0.05 under
  gating; during `wait`'s 14 ms GEMM the copy engine's CPU goes to ~0 while
  the compute engine stays at 1.0.
* **H2** The first command after a long idle pays a device wake-up of
  30–60 µs (futex wake plus a cold vCPU); after gaps shorter than the budget
  it costs nothing.
* **H3** Throughput benchmarks are unchanged within noise under the default
  — their inter-command gaps never reach the budget.
* **H4** The knee of the budget sweep sits at the wake-up latency again
  (~30 µs); adaptive matches the best fixed budget without tuning.
* **H5** Dropping the fences produces measurable lost doorbells on ARM64
  (kept as a negative experiment, `results/vm7-nofence.json`).

## Results

Stage 4 (`results/nolock.json`) vs stage 7 default (`results/gated.json`,
hybrid 50 µs), plus `gate-spin.json` and `gate-sleep.json`. Same-cluster
mode, best of 15 passes across 5 processes per benchmark, canary-gated.
ASan clean on all 10 variants; TSan clean on 9. `runtime_ch1_ticket`
(lock-free ticket publishing, one channel, 8 producers) used to either
time out or fail `gating_no_missed_doorbells` under TSan with no race
reported. A producer descheduled for longer than the engine's 1 ms
timeout, between the PUT store and `doorbell()`, made that timeout find
work and count a miss. The counter now ignores a PUT whose doorbell is
still in flight. On 2026-10-04, TSan `runtime_ch1_ticket` passed in 31 s
and `runtime_gated` in 46 s, with no race report. The fixed-budget sweep and the no-fence experiment were
run later, canary-gated, fast cluster 375 ns (`results/vm7-idle10k.json`,
`vm7-idle200k.json`, `vm7-nofence.json`).

### First-command latency vs idle power (`wake`: idle gap, then one 64 B fill + spin-sync)

| gap before the command | `spin` p50 · cores | **`hybrid 50 µs` p50 · cores** | `sleep` p50 · cores |
|---:|---:|---:|---:|
| 1 µs | 333 ns · 1.95 | **333 ns · 0.99** | 21.5 µs · 0.27 |
| 30 µs | 334 ns · 1.99 | **333 ns · 0.98** | 29.4 µs · 0.15 |
| 300 µs | 459 ns · 2.00 | **49 µs · 0.16** | 49 µs · 0.04 |
| 3 ms | 2.9 µs · 2.00 | **48 µs · 0.04** | 52 µs · 0.03 |
| 30 ms | 5.4 µs · 2.00 | **50 µs · 0.03** | 67 µs · 0.03 |

"cores" is `dev_cpu_cores`: engine-thread CPU per wall second, both engines.
Under `spin` it is 2.0 whatever happens — the number this project had been
paying since stage 0 without printing it.

* **H1 confirmed.** An idle device now costs 0.03 cores instead of 2.0
  (−99%). The remaining ~1.0 at short gaps is the *compute* engine legitimately
  spinning through sub-budget gaps; the copy engine sleeps (its idle cost went
  from 1.0 to ~0).
* **H2 confirmed.** The device wake-up is ~48 µs on this VM: a futex wake plus
  a vCPU that has to be rescheduled onto a host core. Gaps shorter than the
  budget cost exactly what `spin` costs. Note the `spin` column itself climbs
  to 2.9–5.4 µs at long gaps: that is the *host* thread waking from
  `nanosleep` cold; the gating cost is the difference, ~45 µs.
* **H4, the knee: confirmed. The adaptive half: not.** A budget shorter
  than the wake pays it too early. At 10 µs the 30 µs gap is already a
  38 µs command and 0.28 cores (`vm7-idle10k.json`, canary 375 ns). At
  50 µs that same gap is still the spin canary, 333 ns and ~1 core, and
  the 300 µs gap is the ~49 µs wake. At 200 µs the 30 µs gap is still the
  canary (375 ns) and the 300 µs gap is still a ~49 µs wake, but the
  engine has spun through its budget, so that row costs 0.51 cores
  instead of 0.16 (`vm7-idle200k.json`, canary 375 ns before and after).
  Lengthening the budget past the wake does not make the first command
  faster. Adaptive was not re-run; the original measurement stands, and
  it does not match the best fixed budget (917 ns vs 333 ns at a 1 µs gap).

| gap | 10 µs p50 · cores | **50 µs** p50 · cores | 200 µs p50 · cores |
|---:|---:|---:|---:|
| 1 µs | 375 ns · 0.97 | **333 ns · 0.99** | 375 ns · 0.94 |
| 30 µs | 38 µs · 0.28 | **333 ns · 0.98** | 375 ns · 1.01 |
| 300 µs | 50 µs · 0.07 | **49 µs · 0.16** | 49 µs · 0.51 |
| 3 ms | 47 µs · 0.03 | **48 µs · 0.04** | 49 µs · 0.08 |
| 30 ms | 53 µs · 0.02 | **50 µs · 0.03** | 54 µs · 0.03 |

  The 50 µs column is the earlier fast cluster (333 ns). 10 µs and 200 µs
  are the 375 ns cluster from 2026-10-04. Read latency against that run's
  own canary, not against 333.

* **H5 not confirmed.** `SG_EXPERIMENT_NO_FENCE=1` with the engine in
  `sleep` and the host spinning, on `wake` and `mt`, five processes each,
  canary 375 ns before and after (`vm7-nofence.json`). `missed_doorbells`
  was 0 on every row of every process, not only on the fastest pass.
  The fences stay. One clean run that did not lose a doorbell is not a
  reason to delete the barrier the memory model requires; it is a reason
  not to claim the race showed up.

* **H3 confirmed with one exception.** `submit` (166/292 ns), `memcpy`,
  `gemm`, `pipeline` wall time (+5–10%) and per-stream `mt` (+2%) are within
  noise. Default-stream `mt` at 8 threads fell 23%: with all threads serialized
  on one stream, the compute engine sees gaps long enough to gate and then
  pays wake-ups — `sleep_frac` confirms it. That workload is pathological by
  construction (stage 5's ADR); per-stream submission is unaffected.
* **The `wait` benchmark shows the power story end to end:** during a 14 ms
  GEMM the host now sleeps (stage 2) *and* the copy engine sleeps (this stage):
  total device CPU 2.0 → 1.0 cores, i.e. only the engine doing the work.

### What was found and fixed on the way (all measured, see *Decision*)

1. Waking every sleeper on every retirement kept the idle copy engine awake
   at short gaps (0.4–0.8 cores) and added a `FUTEX_WAKE` syscall (+600 ns)
   to every round trip at long gaps. Only fence-blocked sleepers are woken now.
2. Publishing `CLOCK_THREAD_CPUTIME_ID` on each idle→busy transition cost
   ~130 ns per first-command-after-gap: it is a syscall on Linux.
3. Adaptive's gap estimator measured its own wake-up latency and locked
   itself into sleeping (52 µs at 1 µs gaps); timestamping the doorbell fixed
   the lock-in but adaptive still trails the fixed budget at short gaps
   (917 vs 333 ns), so the fixed budget is the default.
4. `nanosleep(1 µs)` on Linux sleeps ~50 µs (timer slack); the benchmark's
   short gaps are host busy-waits so the rows mean what they say.

## Consequences

* The device power proxy is now a real number in every row (`dev_cpu_cores`)
  and an idle `softgpu` costs 0.03 cores. Together with stage 2 the whole
  stack now spends CPU proportional to work, not to time.
* The price is a ~48 µs first-command latency after any idle stretch longer
  than 50 µs. Latency-critical callers can set `SG_ENGINE_IDLE=spin`; a
  per-channel "keep awake" hint is the natural next refinement.
* The doorbell is now a real wake-up path with a Dekker-style arm/re-check.
  The stage-8 module's wait queue is the same protocol, with the kernel's
  barrier in place of the explicit fence. The no-fence run
  (`vm7-nofence.json`) did not lose a doorbell; the fences stay anyway.
* A missed doorbell is a doorbell that finished and was not heard. A PUT
  whose `doorbell()` has not returned yet is the safety net, which is what
  TSan was counting.
* Two stages in a row, an adaptive estimator failed by measuring its own
  penalty. Rule recorded: an estimator's input must come from the workload's
  timeline (doorbell or interrupt timestamps), never from the policy's.
