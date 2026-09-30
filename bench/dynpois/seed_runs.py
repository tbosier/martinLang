#!/usr/bin/env python3
"""Whole sampling runs of the dynamic Poisson model for several seeds:
Mint (built with the current compiler, optionally also a previous one) and
the max-effort Rust baseline, which shares Mint's sampler. 4 chains,
1000 warmup + 1000 draws. Reports the sampler's wall time, the number of
gradients, the time per gradient (wall x chains / gradients) and the
runtime's lowest ESS over all parameters.

usage: python3 bench/dynpois/seed_runs.py SIZE SEED... [--old MINTC_DIR]
  MINTC_DIR: a checkout whose compiler/target/release/mintc builds the
  "before" binary (optional). Run from the repository root.
"""
import argparse
import re
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("size", choices=["small", "large"])
ap.add_argument("seeds", nargs="+", type=int)
ap.add_argument("--old")
args = ap.parse_args()

src = open("examples/dynamic_poisson.mint").read().replace("data_small", f"data_{args.size}")
builds = {"Mint after": "compiler/target/release/mintc"}
if args.old:
    builds = {"Mint before": f"{args.old}/compiler/target/release/mintc", **builds}


def parse(out):
    return {
        "seconds": float(re.search(r"sampling took (\S+) s", out).group(1)),
        "gradients": int(re.search(r"gradients=(\d+)", out).group(1)),
        "min_ess": float(re.search(r"all \d+ parameters: .*lowest ess (\S+) ", out).group(1)),
        "max_rhat": float(re.search(r"all \d+ parameters: highest rhat (\S+) ", out).group(1)),
    }


rows = []
for seed in args.seeds:
    s = re.sub(r"seed = \d+", f"seed = {seed}", src)
    for label, mintc in builds.items():
        prog = f"build/seedrun_{label.split()[1]}_{args.size}_{seed}"
        open(prog + ".mint", "w").write(s)
        subprocess.run([mintc, "build", prog + ".mint", "-o", prog], check=True, capture_output=True)
        r = subprocess.run([prog], capture_output=True, text=True, check=True)
        rows.append((label, seed, parse(r.stdout + r.stderr)))
    r = subprocess.run(["build/rs_dynpois_max", f"bench/dynpois/data_{args.size}/y.f64", str(seed)],
                       capture_output=True, text=True, check=True)
    rows.append(("max-effort Rust", seed, parse(r.stdout + r.stderr)))

print("| program | seed | sampling s | gradients | µs per gradient per chain | lowest ESS | highest R-hat |")
print("|---|---|---|---|---|---|---|")
for label, seed, p in rows:
    per = 4 * p["seconds"] / p["gradients"] * 1e6
    print(f"| {label} | {seed} | {p['seconds']:.2f} | {p['gradients']} | {per:.1f} | {p['min_ess']:.0f} | {p['max_rhat']:.3f} |")
