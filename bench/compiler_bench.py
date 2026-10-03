#!/usr/bin/env python3
"""Before/after benchmark of a compiler round against the max-effort Rust.

Each program runs REPS times, interleaved (every program once per round, in
an order shuffled per round with a fixed seed, so that warm-up and frequency
effects are not tied to one program) and pinned to one core. Gradient
programs report the runtime's ns_per_eval (MINT_BENCH_GRAD=K: K gradient
evaluations at one fixed point, so no seed is involved); Newton reports
fit_seconds. The table gives the median and the
range over the repetitions.

usage: python3 bench/compiler_bench.py OLD_DIR [REPS]
  OLD_DIR holds the "before" binaries old_<example> (built by the previous
  compiler, e.g. from `git archive <rev>`); the "after" binaries are
  build/cmp/new_<example>. Run from the repository root.
writes bench/compiler_bench.json
"""
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys

old = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 7
CORE = "2"

# (problem, metric, K, {label: command})
cases = [
    ("dynamic Poisson gradient, 3,171 parameters", "ns", 20000, {
        "Martin before": [f"{old}/old_dynamic_poisson"],
        "Martin after": ["build/cmp/new_dynamic_poisson"],
        "max-effort Rust": ["build/rs_dynpois_max", "bench/dynpois/data_small/y.f64"],
    }),
    ("dynamic Poisson gradient, 37,901 parameters", "ns", 3000, {
        "Martin before": [f"{old}/old_dpl"],
        "Martin after": ["build/cmp/new_dpl"],
        "max-effort Rust": ["build/rs_dynpois_max", "bench/dynpois/data_large/y.f64"],
    }),
    ("logistic gradient (n=5000, p=20)", "ns", 20000, {
        "Martin before": [f"{old}/old_logistic_bayes"],
        "Martin after": ["build/cmp/new_logistic_bayes"],
        "max-effort Rust": ["build/rs_logistic_bayes_max"],
    }),
    ("linear regression gradient (sufficient statistics)", "ns", 200000, {
        "Martin before": [f"{old}/old_linear_bayes"],
        "Martin after": ["build/cmp/new_linear_bayes"],
        "tuned Rust, same rewrite": ["build/rs_linear_bayes_suffstats"],
    }),
    ("Newton's method (n=200000, p=50), 10 iterations", "s", None, {
        "Martin before": [f"{old}/old_logistic_newton"],
        "Martin after": ["build/cmp/new_logistic_newton"],
        "max-effort Rust": ["build/rs_logistic_newton_max"],
    }),
]


def run(cmd, metric, k):
    env = {k2: v for k2, v in os.environ.items() if not k2.startswith(("MINT_", "OMP_"))}
    if metric == "ns":
        env["MINT_BENCH_GRAD"] = str(k)
    out = subprocess.run(["taskset", "-c", CORE] + cmd, capture_output=True, text=True, env=env, check=True).stdout
    if metric == "ns":
        return float(re.search(r"ns_per_eval=(\S+)", out).group(1))
    return float(re.search(r"fit_seconds (\S+)", out).group(1))


load0 = open("/proc/loadavg").read().split()[:3]
res = {name: {label: [] for label in progs} for name, _, _, progs in cases}
rng = random.Random(1)
for _ in range(reps):
    for name, metric, k, progs in cases:
        order = list(progs.items())
        rng.shuffle(order)
        for label, cmd in order:
            res[name][label].append(run(cmd, metric, k))
load1 = open("/proc/loadavg").read().split()[:3]
binaries = {cmd[0]: hashlib.sha256(open(cmd[0], "rb").read()).hexdigest()[:16]
            for _, _, _, progs in cases for cmd in progs.values()}
json.dump({"reps": reps, "loadavg_at_start": load0, "loadavg_at_end": load1, "core": CORE,
           "order": "shuffled per repetition (seed 1)", "binaries_sha256": binaries,
           "commands": {n: p for n, _, _, p in cases}, "results": res},
          open("bench/compiler_bench.json", "w"), indent=2)

print(f"{reps} interleaved repetitions, core {CORE}; load average {' '.join(load0)} at start, {' '.join(load1)} at end\n")
print("| problem | program | median | range |")
print("|---|---|---|---|")
for name, metric, _, progs in cases:
    for label in progs:
        xs = res[name][label]
        if metric == "ns":
            f = lambda v: f"{v / 1000:.2f} µs"
        else:
            f = lambda v: f"{v:.3f} s"
        print(f"| {name} | {label} | {f(statistics.median(xs))} | {f(min(xs))} to {f(max(xs))} |")
