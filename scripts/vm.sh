#!/usr/bin/env bash
# Build, test and benchmark on the Linux VM from the Mac.
#
# The repo is shared into the VM at $SG_VM_SRC; build trees live on the VM's
# own disk ($SG_VM_BUILD) so artifacts never touch the shared mount or git.
# Results are copied back into results/ so they can be committed.
#
#   scripts/vm.sh build  [preset]        # configure + build (default: release)
#   scripts/vm.sh test   [preset]        # build + ctest
#   scripts/vm.sh bench  <tag> [args..]  # build release, run sgbench, save results/<tag>.json
#                                        #   SG_ENV="SG_RING_DEPTH=16" to set env vars for the run
#                                        #   SG_RUNS=5 separate processes are merged (best pass per row)
#                                        #   SG_FORCE=1 to bench even on a loaded host
#   scripts/vm.sh canary                 # print the round-trip canary (ns); ~210 / ~590 on a quiet host
#   scripts/vm.sh kmod                   # build kmod/softgpu.ko on the VM (load needs sudo)
#   scripts/vm.sh shell                  # ssh into the VM at the source dir
set -euo pipefail

VM=${SG_VM:-softgpu-vm}
SRC=${SG_VM_SRC:-/mnt/softgpu}
BUILD=${SG_VM_BUILD:-\$HOME/build/softgpu}
HERE=$(cd "$(dirname "$0")/.." && pwd)

remote() { ssh -o LogLevel=error "$VM" "$@"; }

preset_flags() {
  case "$1" in
    release) echo "-DCMAKE_BUILD_TYPE=Release" ;;
    debug)   echo "-DCMAKE_BUILD_TYPE=Debug" ;;
    tsan)    echo "-DCMAKE_BUILD_TYPE=RelWithDebInfo -DSG_SANITIZE=thread -DCMAKE_CXX_COMPILER=clang++" ;;
    asan)    echo "-DCMAKE_BUILD_TYPE=RelWithDebInfo -DSG_SANITIZE=address,undefined -DCMAKE_CXX_COMPILER=clang++" ;;
    *) echo "unknown preset: $1" >&2; exit 1 ;;
  esac
}

configure_and_build() {
  local preset=$1
  # Mirrors CMakePresets.json, but with the build dir on VM-local disk.
  remote "set -e; cd $SRC && \
    cmake -S . -B $BUILD/$preset -G Ninja $(preset_flags "$preset") >/dev/null && \
    cmake --build $BUILD/$preset --parallel"
}

# The VM's vCPUs are ordinary host threads: a busy Mac makes every number
# worse, and macOS load average does not predict it well. The guard is a
# canary: one device round trip. Placement is bimodal. As of stage 5 the
# fast cluster reads ~333 ns, sometimes ~375; across clusters it is ~750.
# The limit sits above the fast readings and rejects the slow one.
# Re-measure when the fast path changes — a stale limit idles the job.
# Placement noise inside a mode is handled by sgbench --repeat.
host_load() { sysctl -n vm.loadavg | awk '{print $2}'; }

canary_ns() {
  # Pure spin on both sides: the canary measures host thread placement, not
  # whichever wait/idle policy is the current default.
  remote "cd $BUILD/release && SG_WAIT_POLICY=spin SG_ENGINE_IDLE=spin ./sgbench submit --quick 2>/dev/null" \
    | awk '/fill_64B_sync/ { for (i = 1; i <= NF; i++) if ($i ~ /^p50_ns=/) { sub("p50_ns=", "", $i); print $i } }'
}

check_host_quiet() {
  local ns limit=${SG_CANARY_NS:-400}
  ns=$(canary_ns)
  if [ -z "${SG_FORCE:-}" ] && awk -v v="$ns" -v l="$limit" 'BEGIN{exit !(v+0 > l+0)}'; then
    echo "canary round trip is ${ns} ns (limit ${limit}); host is busy, numbers would be noise." >&2
    echo "quiet the host, or SG_FORCE=1 to override." >&2
    exit 2
  fi
  echo "canary round trip ${ns} ns, host load $(host_load): ok"
}

# Merge runs into one results file. Files are grouped by benchmark (the
# first word of the filename); within a group, per row, every metric comes
# from the run with the lowest wall_ms (see bench/main.cpp on bimodal
# placement); groups are concatenated. Provenance goes in "meta".
merge() {
  python3 - "$@" <<'PYEOF'
import json, sys, datetime, os, collections
out, load, env, runs, files = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:]
groups = collections.OrderedDict()
for f in sorted(files):
    groups.setdefault(os.path.basename(f).split("-")[0], []).append(json.load(open(f)))
rows, tag = [], None
for _, ds in groups.items():
    tag = ds[0]["tag"]
    best = ds[0]["rows"]
    for d in ds[1:]:
        for i, row in enumerate(d["rows"]):
            if row["wall_ms"] < best[i]["wall_ms"]:
                best[i] = row
    rows += best
json.dump({"tag": tag, "meta": {"host_loadavg_1m": float(load), "env": env, "runs": runs,
           "one_process_per_bench": True,
           "date": datetime.datetime.now().isoformat(timespec="seconds")}, "rows": rows},
          open(out, "w"), indent=1)
PYEOF
}

cmd=${1:-build}; shift || true
case "$cmd" in
  build) configure_and_build "${1:-release}" ;;
  test)  p=${1:-release}; configure_and_build "$p"; remote "cd $BUILD/$p && ctest --output-on-failure" ;;
  bench)
    tag=${1:?usage: vm.sh bench <tag> [sgbench args]}; shift || true
    configure_and_build release
    check_host_quiet
    mkdir -p "$HERE/results"
    runs=${SG_RUNS:-5}
    # Split benchmarks from flags; expand "all". Each benchmark runs in its
    # own process so none inherits the previous one's thread placement.
    sels=(); flags=()
    while [ $# -gt 0 ]; do
      case "$1" in
        all) sels+=(submit batch memcpy vadd gemm mt alloc pipeline wait wake fault tlb thrash fair) ;;
        submit|batch|memcpy|vadd|gemm|mt|alloc|pipeline|wait|wake|fault|tlb|thrash|fair) sels+=("$1") ;;
        --json|--tag|--threads|--repeat) flags+=("$1" "$2"); shift ;;
        *) flags+=("$1") ;;
      esac
      shift
    done
    tmp=$(mktemp -d)
    for s in "${sels[@]}"; do
      for k in $(seq 1 "$runs"); do
        remote "cd $SRC && env ${SG_ENV:-} $BUILD/release/sgbench $s ${flags[*]} --tag $tag --json /tmp/sg-$tag-$s-$k.json >/dev/null 2>&1; sleep 1"
        scp -q "$VM:/tmp/sg-$tag-$s-$k.json" "$tmp/$s-$k.json"
      done
    done
    merge "$HERE/results/$tag.json" "$(host_load)" "${SG_ENV:-}" "$runs" "$tmp"/*.json
    rm -rf "$tmp"
    echo "saved results/$tag.json ($runs runs per benchmark, host load $(host_load))"
    ;;
  canary) configure_and_build release >/dev/null; echo "$(canary_ns) ns" ;;
  kmod)
    # The 9p share is not a build tree: the module is compiled on the VM disk.
    remote "set -euo pipefail
      rm -rf \$HOME/build/softgpu-kmod
      mkdir -p \$HOME/build/softgpu-kmod
      cp -a $SRC/kmod \$HOME/build/softgpu-kmod/kmod
      cp -a $SRC/include \$HOME/build/softgpu-kmod/include
      make -C \$HOME/build/softgpu-kmod/kmod -j\$(nproc)"
    echo "built ~/build/softgpu-kmod/kmod/softgpu.ko"
    echo "load: ssh -t $VM 'sudo insmod ~/build/softgpu-kmod/kmod/softgpu.ko && ls -l /dev/softgpu'"
    ;;
  shell) ssh -t "$VM" "cd $SRC && exec \$SHELL -l" ;;
  *) echo "unknown command: $cmd" >&2; exit 1 ;;
esac
