#!/usr/bin/env python3
"""Rust end to end: the hand-tuned gradient of baselines/dynpois_par.rs,
ported to a Rust program that samples with nuts-rs 0.19.0
(bench/shootout/rust_nuts).

usage: run_rust_nuts.py --seed S --adapt diag|lowrank
Compile time: an incremental release build of the crate after touching its
sources (dependencies such as nuts-rs and faer already built; a clean build
of everything is recorded once in the README).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import Phases  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--adapt", choices=["diag", "lowrank"], required=True)
a = ap.parse_args()
name = os.environ.get("SHOOTOUT_RUN", f"rust_nuts_{a.adapt}_s{a.seed}")
ph = Phases()
crate = os.path.join(common.HERE, "rust_nuts")
exe = os.path.join(common.ROOT, "build", "nutsrs", "target", "release", "rust_nuts")
with ph.time("compile"):
    for f in ("main.rs", "dynpois.rs", "team.rs"):
        os.utime(os.path.join(crate, "src", f))
    subprocess.run(["bash", os.path.join(crate, "build.sh"), "--offline"], check=True)
out_bin = os.path.join(common.ROOT, ".tmp", name + ".bin")
with ph.time("run"):
    r = subprocess.run([exe, "sample", os.path.join(common.DATA, "y.f64"), str(a.seed), a.adapt, out_bin],
                       cwd=common.ROOT, capture_output=True, text=True)
if r.returncode != 0:
    sys.exit(r.stdout + r.stderr)
rep = json.loads(r.stdout.strip().splitlines()[-1])
with ph.time("postprocess"):
    C, N, G = (int(x) for x in np.fromfile(out_bin, dtype="<u8", count=3))
    x = np.fromfile(out_bin, dtype="<f8", offset=24).reshape(C, N, 1 + 2 * G)
    common.save_draws(name, x[:, :, 0], x[:, :, 1:1 + G], x[:, :, 1 + G:])
    os.remove(out_bin)
ph.d["phases"]["sampling"] = rep["sampling_seconds"]
ph.set(implementation=f"rust_nuts_{a.adapt}", seed=a.seed, gradients=rep["gradients"],
       warmup_gradients=rep["warmup_gradients"],
       gradients_note="leapfrog steps over warmup and draws plus one per initial point, all chains",
       divergences=rep["divergences"], step_size=rep["step_size"], leapfrog_per_draw=rep["mean_steps"],
       threads_per_chain=rep["threads_per_chain"], chain_seconds=rep["chain_seconds"],
       logp_seconds_per_chain=rep.get("logp_seconds"), logp_calls_per_chain=rep.get("logp_calls"),
       sampling_note="the program's own timer around the 4 chain threads (warmup + draws)",
       settings=f"nuts-rs 0.19.0 {rep['adaptation']}, num_tune 1000, num_draws 1000, maxdepth 10; "
                f"4 chains on their own threads, gradient on {rep['threads_per_chain']} threads per chain")
ph.write()
