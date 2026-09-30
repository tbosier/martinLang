#!/usr/bin/env python3
"""Runs the Mint dynamic Poisson model on one dataset and writes the SPEC
results JSON and draws npz.

usage: run_mint.py SIZE [WARMUP DRAWS] [--variant NAME --flags "..."]
Run from the repository root with a Python that has numpy.
"""
import argparse
import json
import os
import re
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "bench", "dynpois")

ap = argparse.ArgumentParser()
ap.add_argument("size", choices=["small", "large"])
ap.add_argument("warmup", nargs="?", type=int, default=1000)
ap.add_argument("draws", nargs="?", type=int, default=1000)
ap.add_argument("--variant", default="mint")
ap.add_argument("--flags", default="")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--rust", action="store_true", help="run the hand-written Rust baseline (build/rs_dynpois_max) instead")
args = ap.parse_args()

if args.rust:
    # Same NUTS runtime; warmup/draws/seed are fixed in the baseline (1000/1000, seed 7).
    assert (args.warmup, args.draws) == (1000, 1000), "the Rust baseline runs 1000 warmup + 1000 draws"
    prog = os.path.join(ROOT, "build", "rs_dynpois_max")
    cmd = [prog, os.path.join(HERE, f"data_{args.size}", "y.f64")]
    compile_s = 0.0
else:
    src = open(os.path.join(ROOT, "examples", "dynamic_poisson.mint")).read()
    src = src.replace("bench/dynpois/data_small/y.f64", f"bench/dynpois/data_{args.size}/y.f64")
    src = re.sub(r"draws = \d+, warmup = \d+, chains = \d+, seed = \d+",
                 f"draws = {args.draws}, warmup = {args.warmup}, chains = 4, seed = {args.seed}", src)
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    prog = os.path.join(ROOT, "build", f"dynpois_{args.variant}_{args.size}")
    open(prog + ".mint", "w").write(src)
    t = time.perf_counter()
    subprocess.run([os.path.join(ROOT, "compiler", "target", "release", "mintc"), "build", prog + ".mint", "-o", prog]
                   + args.flags.split(), check=True)
    compile_s = time.perf_counter() - t
    cmd = [prog]

draws_path = os.path.join(ROOT, "build", f"{args.variant}_{args.size}.draws")
env = dict(os.environ, MINT_DRAWS=draws_path)
t = time.perf_counter()
r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
wall = time.perf_counter() - t
out = r.stdout + r.stderr
print(r.stdout)
sampling = float(re.search(r"sampling took (\S+) s", out).group(1))
grads = int(re.search(r"gradients=(\d+)", out).group(1))

hdr = np.fromfile(draws_path, dtype="<u8", count=3)
C, N, D = (int(x) for x in hdr)
dr = np.memmap(draws_path, dtype="<f8", mode="r", offset=24, shape=(C, N, D))
y = np.load(os.path.join(HERE, f"data_{args.size}", "y.npy"))
G, T = y.shape
assert D == 1 + G + T + G * T, (D, G, T)
pop = np.array(dr[:, :, 0])
beta = np.array(dr[:, :, 1:1 + G])
shared_total = np.array(dr[:, :, 1 + G:1 + G + T]).sum(axis=2)
innov_total = np.array(dr[:, :, 1 + G + T:]).reshape(C, N, G, T).sum(axis=3)
terminal = innov_total + shared_total[:, :, None]  # state[g, T] = sum_t shared[t] + innov[g, t]

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
base = os.path.join(HERE, "results", f"{args.variant}_{args.size}")
np.savez(base + "_draws.npz", pop=pop, beta=beta, terminal=terminal)
json.dump({
    "implementation": args.variant, "G": G, "T": T, "chains": C, "warmup": args.warmup, "draws": N, "thin": 1,
    "wall_seconds": sampling, "gradients": grads,
    "extra": {"seed": 7 if args.rust else args.seed, "sampler": "mint runtime NUTS"},
    "notes": ("hand-written AVX2 Rust log density (baselines/dynpois_max.rs), " if args.rust else "Mint-compiled log density, ")
             + f"NUTS (Mint runtime), 4 chains in parallel threads; wall_seconds is the sample() call; "
             f"process wall {wall:.2f} s incl. data load and summary; compile {compile_s:.2f} s; flags '{args.flags}'",
}, open(base + ".json", "w"), indent=2)
os.remove(draws_path)
print(f"wrote {base}.json and _draws.npz (sampling {sampling:.2f} s, {grads} gradients)")
