#!/usr/bin/env python3
"""Compares the sampler's metric adaptations.

  stan        the default: regularised variance of the warmup draws
  grad        MINT_METRIC=grad: sqrt(var(draws) / var(gradients)), starting
              from 1 / |gradient| at the initial point
  grad-init0  MINT_METRIC=grad MINT_METRIC_INIT=0: the same, starting from the
              identity, which separates the adaptation from the initialisation
  lowrank     MINT_METRIC=lowrank: Stan's diagonal plus up to 24 directions
              estimated from the gradients of each window's draws
  lowrank-k16 the same with at most 16 directions (MINT_LOWRANK_K=16)
  lowrank-k32 the same with at most 32 directions (MINT_LOWRANK_K=32)

Each model runs 4 chains, 1000 warmup + 1000 draws, for several seeds, serial
sampler (MINT_THREADS_PER_CHAIN=1). The efficiency measure is the runtime's
lowest ESS over all parameters divided by the number of gradients, which does
not depend on machine load. ESS per second does: the runs of one model and
seed are made back to back, but other work on the machine moves it.

usage: python3 bench/metric_experiment.py [variant ...]
  (from the repository root, after bench/setup.sh; default: all variants)
writes bench/metric_results.json (merged with what is there for other
variants) and prints a summary table and a per-seed table
"""
import json
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINTC = os.path.join(ROOT, "compiler", "target", "release", "mintc")
MODELS = {"eight_schools": range(1, 6), "logistic_bayes": range(1, 6), "dynamic_poisson": range(1, 7)}
VARIANTS = {
    "stan": {},
    "grad": {"MINT_METRIC": "grad"},
    "grad-init0": {"MINT_METRIC": "grad", "MINT_METRIC_INIT": "0"},
    "lowrank": {"MINT_METRIC": "lowrank"},
    "lowrank-k16": {"MINT_METRIC": "lowrank", "MINT_LOWRANK_K": "16"},
    "lowrank-k32": {"MINT_METRIC": "lowrank", "MINT_LOWRANK_K": "32"},
}
OUT = os.path.join(ROOT, "bench", "metric_results.json")


def run(model, seed, variant):
    src = open(os.path.join(ROOT, "examples", model + ".mint")).read()
    src = re.sub(r"draws = \d+, warmup = \d+, chains = \d+, seed = \d+",
                 f"draws = 1000, warmup = 1000, chains = 4, seed = {seed}", src)
    prog = os.path.join(ROOT, "build", f"metric_{model}_{seed}")
    if not os.path.exists(prog):
        open(prog + ".mint", "w").write(src)
        subprocess.run([MINTC, "build", prog + ".mint", "-o", prog], check=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MINT_", "OMP_"))}
    env.update(VARIANTS[variant], MINT_THREADS_PER_CHAIN="1")
    r = subprocess.run([prog], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    out = r.stdout + r.stderr
    m = re.search(r"highest rhat (\S+) \(.*\), lowest ess (\S+) ", out)
    lf = [float(x) for x in re.search(r"leapfrog/draw=(\S+)", out).group(1).split(",")]
    rank = re.search(r"\(rank (\d+) to (\d+)\)", out)
    return {
        "max_rhat": float(m.group(1)), "min_ess": float(m.group(2)),
        "gradients": int(re.search(r"gradients=(\d+)", out).group(1)),
        "seconds": float(re.search(r"sampling took (\S+) s", out).group(1)),
        "divergences": int(re.search(r"divergences=(\d+)", out).group(1)),
        "leapfrog_per_draw": statistics.mean(lf),
        "rank": [int(rank.group(1)), int(rank.group(2))] if rank else None,
    }


variants = sys.argv[1:] or list(VARIANTS)
for v in variants:
    if v not in VARIANTS:
        sys.exit(f"unknown variant {v}; known: {', '.join(VARIANTS)}")
new = {m: {v: [] for v in variants} for m in MODELS}
for m, seeds in MODELS.items():
    for s in seeds:
        for v in variants:  # back to back for each seed, so load affects them alike
            new[m][v].append(run(m, s, v))
for m in MODELS:
    for s in MODELS[m]:
        for ext in ("", ".mint"):
            p = os.path.join(ROOT, "build", f"metric_{m}_{s}{ext}")
            if os.path.exists(p):
                os.remove(p)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for m in new:
    res.setdefault(m, {}).update(new[m])
json.dump(res, open(OUT, "w"), indent=2)

print("| model | metric | seeds | lowest ESS per 1000 gradients, median (range) | lowest ESS per second, median | lowest ESS, median | leapfrog per draw, median | max R-hat, worst | divergences, total |")
print("|---|---|---|---|---|---|---|---|---|")
for m, by in res.items():
    for v, runs in by.items():
        eff = [1000 * r["min_ess"] / r["gradients"] for r in runs]
        print(f"| {m} | {v} | {len(runs)} | {statistics.median(eff):.2f} ({min(eff):.2f} to {max(eff):.2f}) "
              f"| {statistics.median(r['min_ess'] / r['seconds'] for r in runs):.0f} "
              f"| {statistics.median(r['min_ess'] for r in runs):.0f} "
              f"| {statistics.median(r['leapfrog_per_draw'] for r in runs):.1f} "
              f"| {max(r['max_rhat'] for r in runs):.3f} | {sum(r['divergences'] for r in runs)} |")
print()
print("| model | seed | metric | lowest ESS | per 1000 gradients | per second | leapfrog per draw | max R-hat | rank |")
print("|---|---|---|---|---|---|---|---|---|")
for m, by in res.items():
    for i, s in enumerate(MODELS[m]):
        for v, runs in by.items():
            if i >= len(runs):
                continue
            r = runs[i]
            rank = "" if not r.get("rank") else f"{r['rank'][0]} to {r['rank'][1]}"
            print(f"| {m} | {s} | {v} | {r['min_ess']:.0f} | {1000 * r['min_ess'] / r['gradients']:.2f} "
                  f"| {r['min_ess'] / r['seconds']:.0f} | {r['leapfrog_per_draw']:.1f} | {r['max_rhat']:.3f} | {rank} |")
