#!/usr/bin/env python3
"""Martin: examples/dynamic_poisson.mint on data_large, compiled by mintc and
run with the runtime's NUTS. Also runs the threaded Rust gradient
(baselines/dynpois_par.rs) under the same runtime with --rust.

usage: run_martin.py --seed S [--optin] [--rust]
  --optin  MINT_METRIC=lowrank MINT_WARMUP=fast ("opt-in sampler options")
  --rust   build and run baselines/dynpois_par.rs linked to Martin's runtime
           ("Rust gradient under Martin's sampler")
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import Phases  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ss_common", os.path.join(common.ROOT, "bench", "same_sampler", "common.py"))
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--optin", action="store_true")
ap.add_argument("--rust", action="store_true")
a = ap.parse_args()
name = os.environ.get("SHOOTOUT_RUN", f"martin_s{a.seed}")
ph = Phases()
os.makedirs(common.BUILD, exist_ok=True)
tmp = os.path.join(common.ROOT, ".tmp")
os.makedirs(tmp, exist_ok=True)

if a.rust:
    prog = os.path.join(common.BUILD, "rs_dynpois_par_run")
    rt = os.path.join(common.ROOT, "build", "mint_rt.o")
    with ph.time("compile"):
        for attempt in range(3):  # (compilers have crashed spuriously on this machine)
            r = subprocess.run(["rustc", "+nightly", "--edition", "2021", "-C", "opt-level=3", "-C", "target-cpu=native",
                                os.path.join(common.ROOT, "baselines", "dynpois_par.rs"), "-o", prog,
                                "-C", f"link-arg={rt}", "-C", "link-arg=-lomp", "-l", "m", "-l", "mvec"],
                               cwd=common.ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                break
        else:
            sys.exit(r.stderr)
    cmd = [prog, os.path.join(common.DATA, "y.f64"), str(a.seed)]
    env_extra = {}
else:
    src = open(os.path.join(common.ROOT, "examples", "dynamic_poisson.mint")).read()
    src = src.replace("bench/dynpois/data_small/y.f64", "bench/dynpois/data_large/y.f64")
    src, n = re.subn(r"draws = \d+, warmup = \d+, chains = \d+, seed = \d+",
                     f"draws = {common.DRAWS}, warmup = {common.WARMUP}, chains = {common.CHAINS}, seed = {a.seed}", src)
    assert n == 1
    prog = os.path.join(common.BUILD, f"martin_s{a.seed}")
    open(prog + ".mint", "w").write(src)
    with ph.time("compile"):
        for attempt in range(3):
            r = subprocess.run([ss.MINTC, "build", prog + ".mint", "-o", prog], cwd=common.ROOT,
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
        else:
            sys.exit(r.stderr)
    cmd = [prog]
    env_extra = {"MINT_METRIC": "lowrank", "MINT_WARMUP": "fast"} if a.optin else {}

draws_path = os.path.join(tmp, name + ".draws")
env = dict(os.environ, MINT_DRAWS=draws_path, **env_extra)
with ph.time("run"):
    r = subprocess.run(cmd, cwd=common.ROOT, env=env, capture_output=True, text=True)
out = r.stdout + r.stderr
if r.returncode != 0:
    sys.exit(out[-3000:])
print(out[-2500:])
rep = ss.parse_run(out)

with ph.time("postprocess"):
    C, N, D = (int(x) for x in np.fromfile(draws_path, dtype="<u8", count=3))
    y = common.load_y()
    G, T = y.shape
    assert (C, N, D) == (common.CHAINS, common.DRAWS, 1 + G + T + G * T)
    # read in blocks of 50 draws (about 15 MB), so this step adds little to the peak memory
    pop, beta, terminal = np.empty((C, N)), np.empty((C, N, G)), np.empty((C, N, G))
    with open(draws_path, "rb") as f:
        for c in range(C):
            for n0 in range(0, N, 50):
                k = min(50, N - n0)
                f.seek(24 + 8 * D * (c * N + n0))
                blk = np.fromfile(f, dtype="<f8", count=k * D).reshape(k, D)
                pop[c, n0:n0 + k] = blk[:, 0]
                beta[c, n0:n0 + k] = blk[:, 1:1 + G]
                terminal[c, n0:n0 + k] = (blk[:, 1 + G + T:].reshape(k, G, T).sum(axis=2)
                                          + blk[:, 1 + G:1 + G + T].sum(axis=1)[:, None])
    common.save_draws(name, pop, beta, terminal)
    os.remove(draws_path)

ph.d["phases"]["sampling"] = rep["sampling_seconds"]
ph.set(implementation="rust_under_martin" if a.rust else ("martin_optin" if a.optin else "martin"),
       seed=a.seed, gradients=rep["gradients"], warmup_gradients=rep["warmup_gradients"],
       gradients_note="all gradients, warmup and sampling (the runtime's count)",
       divergences=rep["divergences"], step_size=rep["step_size"], leapfrog_per_draw=rep["leapfrog_per_draw"],
       threads_per_chain=rep["threads_per_chain"], smallest_team=rep["smallest_team"],
       warmup_kind=rep["warmup_kind"], warmup_iterations=rep["warmup_iters"], prep_seconds=rep["prep_seconds"],
       sampling_note="the runtime's own 'sampling took' (warmup + draws, all chains)",
       settings=(f"MINT_METRIC=lowrank MINT_WARMUP=fast (the program asks for 1000 warmup iterations; the fast "
                 f"warmup runs {rep['warmup_iters']} of them) + 1000 draws" if a.optin
                 else "runtime defaults; 1000 warmup + 1000 draws")
       + "; 4 chains, max tree depth 10, target acceptance 0.8")
ph.write()
