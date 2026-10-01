#!/usr/bin/env python3
"""Compares the sampler's metric adaptations.

  stan         the default: regularised variance of the warmup draws
  grad         MINT_METRIC=grad: sqrt(var(draws) / var(gradients)), starting
               from 1 / |gradient| at the initial point
  grad-init0   MINT_METRIC=grad MINT_METRIC_INIT=0: the same, starting from
               the identity, which separates the adaptation from the
               initialisation
  lowrank      MINT_METRIC=lowrank: Stan's diagonal plus up to MINT_LOWRANK_K
               directions (the runtime's default for the model's size)
               estimated from the gradients of each window's draws
  lowrank-kN   the same with at most N directions (MINT_LOWRANK_K=N)

Each run is 4 chains, 1000 warmup + 1000 draws. The models:

  eight_schools, logistic_bayes, dynamic_poisson (D = 3,171), and
  dynamic_poisson_large (D = 37,901: examples/dynamic_poisson.mint on
  bench/dynpois/data_large)

The efficiency measures are the runtime's lowest ESS over all parameters
divided by the number of gradients, which does not depend on machine load,
and divided by the sampling wall time, which does. The sampler's threads per
chain are the runtime's default unless --threads is given; the runs of one
model and seed are made back to back, so that load affects the variants
alike, but other work on the machine still moves the times.

usage: python3 bench/metric_experiment.py [--models a,b] [--seeds 1,2,3] [--threads N] [variant ...]
  (from the repository root, after bench/setup.sh; default: all models with
  their default seeds, variants stan and lowrank)
Each run is stored in bench/metric_results.json, replacing an earlier run of
the same model, variant, seed and thread setting; the summary covers every
stored run.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINTC = os.path.join(ROOT, "compiler", "target", "release", "mintc")
OUT = os.path.join(ROOT, "bench", "metric_results.json")
MODELS = {
    "eight_schools": ("eight_schools", range(1, 6)),
    "logistic_bayes": ("logistic_bayes", range(1, 6)),
    "dynamic_poisson": ("dynamic_poisson", range(1, 7)),
    "dynamic_poisson_large": ("dynamic_poisson", [11]),
}
FIXED = {
    "stan": {},
    "grad": {"MINT_METRIC": "grad"},
    "grad-init0": {"MINT_METRIC": "grad", "MINT_METRIC_INIT": "0"},
    "lowrank": {"MINT_METRIC": "lowrank"},
}


def variant_env(v):
    if v in FIXED:
        return FIXED[v]
    m = re.fullmatch(r"lowrank-k(\d+)", v)
    if m:
        return {"MINT_METRIC": "lowrank", "MINT_LOWRANK_K": m.group(1)}
    raise SystemExit(f"unknown variant {v}")


def build(model, seed):
    example, _ = MODELS[model]
    src = open(os.path.join(ROOT, "examples", example + ".mint")).read()
    if model.endswith("_large"):
        src = src.replace("bench/dynpois/data_small/", "bench/dynpois/data_large/")
    src = re.sub(r"draws = \d+, warmup = \d+, chains = \d+, seed = \d+",
                 f"draws = 1000, warmup = 1000, chains = 4, seed = {seed}", src)
    prog = os.path.join(ROOT, "build", f"metric_{model}_{seed}")
    open(prog + ".mint", "w").write(src)
    subprocess.run([MINTC, "build", prog + ".mint", "-o", prog], check=True, capture_output=True)
    return prog


def run(prog, variant, threads):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MINT_", "OMP_"))}
    env.update(variant_env(variant))
    if threads:
        env["MINT_THREADS_PER_CHAIN"] = str(threads)
    t0 = time.time()
    p = subprocess.Popen([prog], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout.read()
    _, status, ru = os.wait4(p.pid, 0)
    if status != 0:
        raise SystemExit(f"{prog} failed:\n{out}")
    m = re.search(r"highest rhat (\S+) \(.*\), lowest ess (\S+) ", out)
    lf = [float(x) for x in re.search(r"leapfrog/draw=(\S+)", out).group(1).split(",")]
    rank = re.search(r"\(rank (\d+) to (\d+)\)", out)
    return {
        "max_rhat": float(m.group(1)), "min_ess": float(m.group(2)),
        "gradients": int(re.search(r"gradients=(\d+)", out).group(1)),
        "seconds": float(re.search(r"sampling took (\S+) s", out).group(1)),
        "cpu_seconds": ru.ru_utime + ru.ru_stime,
        "load_average": os.getloadavg()[0],
        "divergences": int(re.search(r"divergences=(\d+)", out).group(1)),
        "leapfrog_per_draw": statistics.mean(lf),
        "threads_per_chain": int(re.search(r"threads per chain=(\d+)", out).group(1)),
        "rank": [int(rank.group(1)), int(rank.group(2))] if rank else None,
        "wall": time.time() - t0,
    }


def rng(xs, fmt):
    xs = list(xs)
    if len(xs) == 1:
        return fmt.format(xs[0])
    return (fmt + " ({} to {})").format(statistics.median(xs), fmt.format(min(xs)), fmt.format(max(xs)))


def summary(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["model"], r["variant"], r["threads"]), []).append(r)
    print("| model | metric | threads per chain | seeds | lowest ESS per 1000 gradients, median (range) "
          "| lowest ESS per second, median (range) | lowest ESS, median | leapfrog per draw, median "
          "| max R-hat, worst | divergences, total |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    order = list(MODELS)
    for (m, v, t), rs in sorted(groups.items(), key=lambda kv: (order.index(kv[0][0]) if kv[0][0] in order else 99,
                                                                 kv[0][1], str(kv[0][2]))):
        tp = sorted({r["threads_per_chain"] for r in rs})
        print(f"| {m} | {v} | {'/'.join(map(str, tp))} | {len(rs)} "
              f"| {rng((1000 * r['min_ess'] / r['gradients'] for r in rs), '{:.2f}')} "
              f"| {rng((r['min_ess'] / r['seconds'] for r in rs), '{:.0f}')} "
              f"| {statistics.median(r['min_ess'] for r in rs):.0f} "
              f"| {statistics.median(r['leapfrog_per_draw'] for r in rs):.1f} "
              f"| {max(r['max_rhat'] for r in rs):.3f} | {sum(r['divergences'] for r in rs)} |")
    print()
    print("| model | seed | metric | threads | lowest ESS | per 1000 gradients | per second | seconds | CPU seconds "
          "| load | leapfrog per draw | max R-hat | rank |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: (order.index(r["model"]) if r["model"] in order else 99, r["seed"],
                                         r["variant"], str(r["threads"]))):
        rank = "" if not r.get("rank") else f"{r['rank'][0]} to {r['rank'][1]}"
        print(f"| {r['model']} | {r['seed']} | {r['variant']} | {r['threads_per_chain']} | {r['min_ess']:.0f} "
              f"| {1000 * r['min_ess'] / r['gradients']:.2f} | {r['min_ess'] / r['seconds']:.0f} "
              f"| {r['seconds']:.2f} | {r.get('cpu_seconds', float('nan')):.1f} | {r.get('load_average', float('nan')):.1f} "
              f"| {r['leapfrog_per_draw']:.1f} | {r['max_rhat']:.3f} | {rank} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default=None, help="comma-separated; default: each model's own")
    ap.add_argument("--threads", type=int, default=0, help="threads per chain (default: the runtime's)")
    ap.add_argument("--summary", action="store_true", help="only print the summary of stored runs")
    ap.add_argument("variants", nargs="*", default=["stan", "lowrank"])
    a = ap.parse_args()
    stored = json.load(open(OUT))["runs"] if os.path.exists(OUT) else []
    if not a.summary:
        for v in a.variants:
            variant_env(v)
        for model in a.models.split(","):
            seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(MODELS[model][1])
            for seed in seeds:
                prog = build(model, seed)
                for v in a.variants:  # back to back for each seed, so load affects them alike
                    r = run(prog, v, a.threads)
                    r.update(model=model, variant=v, seed=seed, threads=a.threads or "default")
                    print(f"{model} seed {seed} {v}: lowest ESS {r['min_ess']:.0f}, "
                          f"{1000 * r['min_ess'] / r['gradients']:.2f} per 1000 gradients, "
                          f"{r['seconds']:.2f} s, rank {r['rank']}", flush=True)
                    stored = [s for s in stored if (s["model"], s["variant"], s["seed"], s["threads"]) !=
                              (model, v, seed, r["threads"])] + [r]
                    json.dump({"runs": stored}, open(OUT, "w"), indent=1)
                for ext in ("", ".mint"):
                    if os.path.exists(prog + ext):
                        os.remove(prog + ext)
    summary(stored)


if __name__ == "__main__":
    main()
