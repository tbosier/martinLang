#!/usr/bin/env python3
"""Compares the sampler's two diagonal metric adaptations.

  stan       the default: regularised variance of the warmup draws
  grad       MINT_METRIC=grad: sqrt(var(draws) / var(gradients)), starting
             from 1 / |gradient| at the initial point
  grad-init0 MINT_METRIC=grad MINT_METRIC_INIT=0: the same, starting from the
             identity, which separates the adaptation from the initialisation

Each model runs 4 chains, 1000 warmup + 1000 draws, for several seeds, serial
sampler (MINT_THREADS_PER_CHAIN=1). The efficiency measure is the runtime's
lowest ESS over all parameters divided by the number of gradients, which does
not depend on machine load.

usage: python3 bench/metric_experiment.py   (from the repository root, after bench/setup.sh)
writes bench/metric_results.json and prints a table
"""
import json
import os
import re
import statistics
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINTC = os.path.join(ROOT, "compiler", "target", "release", "mintc")
MODELS = {"eight_schools": range(1, 6), "logistic_bayes": range(1, 6), "dynamic_poisson": range(1, 4)}
VARIANTS = {
    "stan": {},
    "grad": {"MINT_METRIC": "grad"},
    "grad-init0": {"MINT_METRIC": "grad", "MINT_METRIC_INIT": "0"},
}


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
    return {
        "max_rhat": float(m.group(1)), "min_ess": float(m.group(2)),
        "gradients": int(re.search(r"gradients=(\d+)", out).group(1)),
        "seconds": float(re.search(r"sampling took (\S+) s", out).group(1)),
        "divergences": int(re.search(r"divergences=(\d+)", out).group(1)),
        "leapfrog_per_draw": statistics.mean(lf),
    }


res = {m: {v: [run(m, s, v) for s in seeds] for v in VARIANTS} for m, seeds in MODELS.items()}
for m in MODELS:
    for s in MODELS[m]:
        for ext in ("", ".mint"):
            p = os.path.join(ROOT, "build", f"metric_{m}_{s}{ext}")
            if os.path.exists(p):
                os.remove(p)
json.dump(res, open(os.path.join(ROOT, "bench", "metric_results.json"), "w"), indent=2)

print("| model | metric | seeds | lowest ESS per 1000 gradients, median (range) | lowest ESS, median | leapfrog per draw, median | max R-hat, worst | divergences, total |")
print("|---|---|---|---|---|---|---|---|")
for m, by in res.items():
    for v, runs in by.items():
        eff = [1000 * r["min_ess"] / r["gradients"] for r in runs]
        print(f"| {m} | {v} | {len(runs)} | {statistics.median(eff):.2f} ({min(eff):.2f} to {max(eff):.2f}) "
              f"| {statistics.median(r['min_ess'] for r in runs):.0f} "
              f"| {statistics.median(r['leapfrog_per_draw'] for r in runs):.1f} "
              f"| {max(r['max_rhat'] for r in runs):.3f} | {sum(r['divergences'] for r in runs)} |")
