#!/usr/bin/env python3
"""Compares warmup schedules by gradients spent and effective draws obtained.

Each run is one compiled example with 1000 warmup (as written in the program)
+ 1000 kept draws and 4 chains (logistic_1chain: 1 chain, so the fast
warmup cannot pool across chains). The efficiency measure is the runtime's
lowest ESS over all parameters divided by the total number of gradients
(initialisation + warmup + sampling), which does not depend on machine load.
Wall time is reported too but is noisy on a shared machine. All binaries are
built before the first run.

usage: python3 bench/warmup_experiment.py [--models m,...] [--seeds 1,2,3]
           [--variants name=ENV=VAL:ENV=VAL,...] [--variants-file F]
           [--json out.json] [--draws-dir DIR]
  run from the repository root after bench/setup.sh. A variant is a name and
  the environment it sets, e.g. stan= fast=MINT_WARMUP=fast. Models:
  dynpois_small, dynpois_large, logistic, logistic_1chain, eight_schools.
"""
import argparse
import json
import os
import re
import statistics
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINTC = os.path.join(ROOT, "compiler", "target", "release", "mintc")
# model: (example, size substitution, draws, warmup, chains)
MODELS = {
    "dynpois_small": ("dynamic_poisson", "small", 1000, 1000, 4),
    "dynpois_large": ("dynamic_poisson", "large", 1000, 1000, 4),
    "logistic": ("logistic_bayes", None, 1000, 1000, 4),
    "logistic_1chain": ("logistic_bayes", None, 1000, 1000, 1),
    "eight_schools": ("eight_schools", None, 1000, 1000, 4),
}


def build(model, seed, warmup_override=None):
    ex, size, draws, warmup, chains = MODELS[model]
    if warmup_override is not None:
        warmup = warmup_override
    src = open(os.path.join(ROOT, "examples", ex + ".mint")).read()
    if size:
        src = src.replace("data_small", f"data_{size}")
    src = re.sub(r"draws = \d+, warmup = \d+, chains = \d+, seed = \d+",
                 f"draws = {draws}, warmup = {warmup}, chains = {chains}, seed = {seed}", src)
    prog = os.path.join(ROOT, "build", f"warmup_{model}_{seed}_{warmup}_{os.getpid()}")
    open(prog + ".mint", "w").write(src)
    subprocess.run([MINTC, "build", prog + ".mint", "-o", prog], check=True, capture_output=True)
    return prog


def run(prog, env_extra, draws_file=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MINT_", "OMP_"))}
    env.update(env_extra)
    if draws_file:
        env["MINT_DRAWS"] = draws_file
    r = subprocess.run([prog], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    out = r.stdout + r.stderr
    m = re.search(r"highest rhat (\S+) \(.*\), lowest ess (\S+) ", out)
    lf = [float(x) for x in re.search(r"leapfrog/draw=(\S+)", out).group(1).split(",")]
    means = {}
    for line in r.stdout.splitlines():
        f = line.split()
        if len(f) == 8 and f[0] in ("mu", "tau", "pop", "alpha"):
            means[f[0]] = (float(f[1]), float(f[2]), float(f[6]))  # mean, sd, ess
    return {
        "max_rhat": float(m.group(1)), "min_ess": float(m.group(2)),
        "gradients": int(re.search(r"gradients=(\d+)", out).group(1)),
        "warmup_gradients": int(re.search(r"gradients: warmup=(\d+)", out).group(1)),
        "seconds": float(re.search(r"sampling took (\S+) s", out).group(1)),
        "divergences": int(re.search(r"divergences=(\d+)", out).group(1)),
        "leapfrog_per_draw": statistics.mean(lf),
        "means": means,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="dynpois_small,logistic,eight_schools")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--variants", default="stan=,fast=MINT_WARMUP=fast")
    ap.add_argument("--variants-file", help="one variant per line: NAME [ENV=VAL ...]")
    ap.add_argument("--json")
    ap.add_argument("--warmup", type=int, help="warmup iterations written into the program (default 1000)")
    ap.add_argument("--draws-dir")
    a = ap.parse_args()
    variants = {}
    if a.variants_file:
        for line in open(a.variants_file):
            f = line.split()
            if f and not f[0].startswith("#"):
                variants[f[0]] = dict(e.split("=", 1) for e in f[1:])
    else:
        for v in a.variants.split(","):
            name, _, envs = v.partition("=")
            variants[name] = dict(e.split("=", 1) for e in envs.split(":") if e)
    res = []
    # build everything first, so that the runs use one runtime even if the source changes meanwhile
    jobs = [(model, int(seed)) for model in a.models.split(",") for seed in a.seeds.split(",")]
    progs = {j: build(j[0], j[1], a.warmup) for j in jobs}
    for (model, seed), prog in progs.items():
        for name, env in variants.items():
            df = os.path.join(a.draws_dir, f"{model}_{seed}_{name}.draws") if a.draws_dir else None
            r = run(prog, env, df)
            r.update(model=model, seed=seed, variant=name)
            res.append(r)
            if a.json:
                json.dump(res, open(a.json, "w"), indent=1)
            eff = 1000 * r["min_ess"] / r["gradients"]
            print(f"{model:14s} seed={seed} {name:10s} grads={r['gradients']:9d} "
                  f"(warmup {r['warmup_gradients']:9d}) ess={r['min_ess']:7.0f} rhat={r['max_rhat']:.3f} "
                  f"ess/1k grad={eff:6.3f} lf/draw={r['leapfrog_per_draw']:6.1f} "
                  f"div={r['divergences']} t={r['seconds']:.2f}s {r['means']}", flush=True)
        for ext in ("", ".mint"):
            if os.path.exists(prog + ext):
                os.remove(prog + ext)


if __name__ == "__main__":
    main()
