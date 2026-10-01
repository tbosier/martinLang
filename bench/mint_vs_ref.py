#!/usr/bin/env python3
"""Times the current Mint build against the frozen Rust reference
(bench/rust_reference.json), for quick iteration on the compiler.

The Rust numbers were measured once; machine conditions drift by a few per
cent between sessions, so a final claim should come from an interleaved run
(bench/compiler_bench.py), not from this script.

usage: python3 bench/mint_vs_ref.py [REPS] [--flags "..."]
"""
import argparse
import json
import os
import re
import statistics
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("reps", nargs="?", type=int, default=5)
ap.add_argument("--flags", default="")
args = ap.parse_args()
ref = json.load(open("bench/rust_reference.json"))
MINTC = "compiler/target/release/mintc"
os.makedirs("build/cmp", exist_ok=True)
large = open("examples/dynamic_poisson.mint").read().replace("data_small", "data_large")
open("build/cmp/cur_dpl.mint", "w").write(large)
cases = [
    ("dynpois_small", "examples/dynamic_poisson.mint", 20000),
    ("dynpois_large", "build/cmp/cur_dpl.mint", 3000),
    ("logistic", "examples/logistic_bayes.mint", 20000),
    ("linear", "examples/linear_bayes.mint", 200000),
    ("newton", "examples/logistic_newton.mint", None),
]
for name, src, _ in cases:
    subprocess.run([MINTC, "build", src, "-o", f"build/cmp/cur_{name}"] + args.flags.split(), check=True, capture_output=True)


def run(name, k):
    env = {a: b for a, b in os.environ.items() if not a.startswith(("MINT_", "OMP_"))}
    if k:
        env["MINT_BENCH_GRAD"] = str(k)
    out = subprocess.run(["taskset", "-c", "2", f"build/cmp/cur_{name}"], capture_output=True, text=True, env=env, check=True).stdout
    return float(re.search(r"ns_per_eval=(\S+)" if k else r"fit_seconds (\S+)", out).group(1))


res = {name: [] for name, _, _ in cases}
for _ in range(args.reps):
    for name, _, k in cases:
        res[name].append(run(name, k))
print(f"load average {open('/proc/loadavg').read().split()[0]}; {args.reps} runs each")
print("| problem | Mint median (range) | Rust reference median | Rust / Mint |")
print("|---|---|---|---|")
for name, _, k in cases:
    xs = res[name]
    r = (ref["gradient_ns"].get(name) or ref["fit_seconds"].get(name))["median"]
    m = statistics.median(xs)
    unit = (lambda v: f"{v / 1000:.2f} µs") if k else (lambda v: f"{v:.3f} s")
    print(f"| {name} | {unit(m)} ({unit(min(xs))} to {unit(max(xs))}) | {unit(r)} | {r / m:.3f}x |")
