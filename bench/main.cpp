#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#include "bench.h"

namespace bench {
void die_on(sgError_t e, const char* what) {
    if (e != SG_OK) {
        std::fprintf(stderr, "%s: %s\n", what, sgErrorString(e));
        std::exit(2);
    }
}
} // namespace bench

static void usage() {
    std::fprintf(stderr,
                 "usage: sgbench [submit|batch|memcpy|vadd|gemm|mt|alloc|pipeline|wait|wake|fault|tlb|thrash|fair|all]... "
                 "[--quick] [--json FILE] "
                 "[--tag NAME] [--threads N] [--repeat N]\n");
    std::exit(1);
}

int main(int argc, char** argv) {
    bench::Options opt;
    std::vector<std::string> which;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--quick") opt.quick = true;
        else if (a == "--json" && i + 1 < argc) opt.json = argv[++i];
        else if (a == "--tag" && i + 1 < argc) opt.tag = argv[++i];
        else if (a == "--threads" && i + 1 < argc) opt.threads_max = std::atoi(argv[++i]);
        else if (a == "--repeat" && i + 1 < argc) opt.repeat = std::max(1, std::atoi(argv[++i]));
        else if (a[0] == '-') usage();
        else which.push_back(a);
    }
    if (which.empty()) which.push_back("all");

    for (const auto& w : which)
        if (w != "all" && w != "submit" && w != "batch" && w != "memcpy" && w != "vadd" &&
            w != "gemm" && w != "mt" && w != "alloc" && w != "pipeline" && w != "wait" && w != "wake" &&
            w != "fault" && w != "tlb" && w != "thrash" && w != "fair")
            usage();

    bench::die_on(sgInit(), "sgInit");
    // Timings on the VM are bimodal: when the submitter and the device thread
    // sit on different host CPU clusters, a cache-line handoff costs ~3x, and
    // the guest cannot see or control which it got. Each pass re-creates the
    // device thread; per row we keep every metric from the fastest pass. The
    // harness (scripts/vm.sh) additionally merges several separate processes,
    // because placement tends to be sticky for a process lifetime.
    std::vector<bench::Report> runs(opt.repeat);
    for (int r = 0; r < opt.repeat; ++r) {
        if (r > 0) {
            sgShutdown();
            bench::die_on(sgInit(), "sgInit");
        }
        bench::Report& rep = runs[r];
        for (const auto& w : which) {
            bool all = w == "all";
            if (all || w == "submit") bench::bench_submit(opt, rep);
            if (all || w == "batch") bench::bench_batch(opt, rep);
            if (all || w == "memcpy") bench::bench_memcpy(opt, rep);
            if (all || w == "vadd") bench::bench_vadd(opt, rep);
            if (all || w == "gemm") bench::bench_gemm(opt, rep);
            if (all || w == "mt") bench::bench_mt(opt, rep);
            if (all || w == "alloc") bench::bench_alloc(opt, rep);
            if (all || w == "pipeline") bench::bench_pipeline(opt, rep);
            if (all || w == "wait") bench::bench_wait(opt, rep);
            if (all || w == "wake") bench::bench_wake(opt, rep);
            if (all || w == "fault") bench::bench_fault(opt, rep);
            if (all || w == "tlb") bench::bench_tlb(opt, rep);
            if (all || w == "thrash") bench::bench_thrash(opt, rep);
            if (all || w == "fair") bench::bench_fair(opt, rep);
        }
        if (opt.repeat > 1) std::fprintf(stderr, "pass %d/%d done\n", r + 1, opt.repeat);
    }
    sgShutdown();

    bench::Report rep = runs[0];
    for (size_t i = 0; i < rep.rows.size(); ++i) {
        size_t best = 0;
        for (size_t r = 1; r < runs.size(); ++r)
            if (runs[r].rows[i].metrics.at("wall_ms") < runs[best].rows[i].metrics.at("wall_ms")) best = r;
        rep.rows[i] = runs[best].rows[i];
        rep.rows[i].metrics["passes"] = double(runs.size());
    }

    rep.print();
    if (!opt.json.empty()) {
        if (!rep.write_json(opt.json, opt.tag)) { std::perror("write_json"); return 3; }
        std::fprintf(stderr, "wrote %s\n", opt.json.c_str());
    }
    return 0;
}
