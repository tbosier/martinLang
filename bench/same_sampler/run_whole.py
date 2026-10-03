#!/usr/bin/env python3
"""Whole sampling runs with identical sampler settings: every implementation
of a problem runs under the Martin runtime's NUTS (mint_sample) with the same
draws, warmup, chains and seed, so the sampler code, warmup, metric
adaptation, initialisation and random number stream are the same; only the
log density and gradient code differ.

Runs are interleaved: for each seed, every (problem, implementation) once,
in a shuffled order. Before each run the load average and every CPU's busy
fraction are recorded, because other work shares the machine. Results are
appended to the output file after every run, so a partial run keeps what
finished.

Reported per run (from the runtime's own report, identical code for all):
sampler wall time, gradients (warmup and sampling), time per gradient per
chain (wall x chains / gradients, which includes the sampler), the lowest
bulk ESS and highest split R-hat over all parameters, step sizes,
divergences and leapfrog steps per draw.

usage: python bench/same_sampler/run_whole.py --out results/whole_small.json
           --problems dynpois_small,logistic,eight_schools --seeds 1 2 3 [--impls mint,stan]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--problems", default="dynpois_small,logistic,eight_schools")
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--impls", default=None, help="comma-separated subset of implementations")
ap.add_argument("--draws", type=int, default=1000)
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--chains", type=int, default=4)
args = ap.parse_args()

out_path = args.out if os.path.isabs(args.out) else os.path.join(common.ROOT, "bench", "same_sampler", args.out)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
res = json.load(open(out_path)) if os.path.exists(out_path) else {
    "what": "whole runs under the shared Martin runtime NUTS, identical settings, interleaved",
    "settings": {"draws": args.draws, "warmup": args.warmup, "chains": args.chains}, "runs": []}
assert res["settings"] == {"draws": args.draws, "warmup": args.warmup, "chains": args.chains}, \
    "settings differ from the runs already in this file"

jobs = []
for p in args.problems.split(","):
    for impl in common.implementations(p):
        if args.impls is None or impl in args.impls.split(","):
            jobs.append((p, impl))
done = {(r["problem"], r["impl"], r["seed"]) for r in res["runs"]}
rng = random.Random(sum(args.seeds))
for seed in args.seeds:
    order = jobs[:]
    rng.shuffle(order)
    for p, impl in order:
        if (p, impl, seed) in done:
            continue
        argv, env = common.command(p, impl, args.draws, args.warmup, args.chains, seed)
        before = {"loadavg": common.loadavg(), "cpu_busy_percent": common.cpu_busy(1.0)}
        out, wall = common.run(argv, env)
        r = common.parse_run(out)
        assert (r["chains"], r["draws"], r["warmup_iters"]) == (args.chains, args.draws, args.warmup), r
        if "MINT_BASELINE_SEED" in env:
            assert f"seed={seed}" in out, "the baseline did not take the seed"
        r.update({"problem": p, "impl": impl, "seed": seed, "process_wall_seconds": wall,
                  "us_per_gradient_per_chain": 1e6 * r["sampling_seconds"] * args.chains / r["gradients"],
                  "min_ess_per_second": r["min_ess"] / r["sampling_seconds"],
                  "min_ess_per_1k_gradients": 1e3 * r["min_ess"] / r["gradients"],
                  "before": before, "loadavg_after": common.loadavg()})
        res["runs"].append(r)
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"seed {seed} {p:14s} {impl:9s} {r['sampling_seconds']:8.2f} s  {r['gradients']:8d} grads  "
              f"{r['us_per_gradient_per_chain']:8.2f} us/grad/chain  min ESS {r['min_ess']:7.0f}  "
              f"max R-hat {r['max_rhat']:.3f}  load {before['loadavg'][0]:.1f}", flush=True)
print(f"wrote {out_path}")
