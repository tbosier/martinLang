#!/usr/bin/env python3
"""Stan: CmdStan 2.40 through cmdstanpy, stanc --O1, g++ -O3 -march=native.

usage: run_cmdstan.py --seed S [--reduce-sum]
  default       bench/dynpois/dynpois.stan, 4 chain processes, one thread each
  --reduce-sum  models/dynpois_reduce_sum.stan, STAN_THREADS, threads_per_chain=3

Draws are written by CmdStan as CSV (default: warmup not saved). Only the
needed columns are read back. Gradients: CmdStan records n_leapfrog__ only for
saved iterations, so the count covers the 1000 sampling iterations per chain
(leapfrog steps plus one start-point gradient per transition); warmup
gradients are not recorded.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import Phases  # noqa: E402


def postprocess(csv_files, name, out_dir):
    """Reads the needed columns of CmdStan's CSV files, saves the draws and
    returns the sampler facts; then deletes the CSV files."""
    G = common.load_y().shape[0]
    # CmdStan's CSV header names array elements beta.1, ..., not beta[1]
    want = ["lp__", "n_leapfrog__", "divergent__", "stepsize__", "treedepth__", "pop"] \
        + [f"beta.{g + 1}" for g in range(G)] + [f"terminal.{g + 1}" for g in range(G)]
    frames = [pd.read_csv(p, comment="#", usecols=want, engine="c") for p in csv_files]
    assert all(len(f) == common.DRAWS for f in frames), [len(f) for f in frames]
    pop = np.stack([f["pop"].to_numpy() for f in frames])
    beta = np.stack([f[[f"beta.{g + 1}" for g in range(G)]].to_numpy() for f in frames])
    terminal = np.stack([f[[f"terminal.{g + 1}" for g in range(G)]].to_numpy() for f in frames])
    common.save_draws(name, pop, beta, terminal)
    leap = int(sum(f["n_leapfrog__"].sum() for f in frames))
    elapsed = []
    for p in csv_files:
        w = s = None
        for line in open(p):
            if "seconds (Warm-up)" in line:
                w = float(line.split(":")[1].split()[0])
            elif "seconds (Sampling)" in line:
                s = float(line.split("#")[1].split()[0])
        elapsed.append((w, s))
    csv_bytes = sum(os.path.getsize(p) for p in csv_files)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    return dict(
        gradients=leap + common.CHAINS * common.DRAWS, warmup_gradients=None,
        gradients_note="sampling iterations only: sum of n_leapfrog__ plus one per transition; warmup not recorded",
        divergences=int(sum(f["divergent__"].sum() for f in frames)),
        step_size=[float(f["stepsize__"].iloc[0]) for f in frames],
        leapfrog_per_draw=[float(f["n_leapfrog__"].mean()) for f in frames],
        treedepth_10_hits=int(sum((f["treedepth__"] >= 10).sum() for f in frames)),
        chain_elapsed_warmup_sampling=elapsed, csv_bytes=csv_bytes)


def describe(reduce_sum):
    return dict(
        implementation="cmdstan_reduce_sum" if reduce_sum else "cmdstan_plain",
        threads_per_chain=3 if reduce_sum else 1,
        sampling_note="the cmdstanpy sample() call: CmdStan processes including writing CSV",
        settings=("models/dynpois_reduce_sum.stan, STAN_THREADS, threads_per_chain=3, grainsize 1"
                  if reduce_sum else "bench/dynpois/dynpois.stan, one thread per chain")
        + "; stanc --O1, CXXFLAGS=-march=native (CmdStan's -O3); default NUTS (diag_e, adapt_delta 0.8, "
          "max_treedepth 10); 4 chains, parallel_chains=4, 1000 warmup + 1000 draws")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--reduce-sum", action="store_true")
    a = ap.parse_args()
    variant = "reduce_sum" if a.reduce_sum else "plain"
    name = os.environ.get("SHOOTOUT_RUN", f"cmdstan_{variant}_s{a.seed}")
    ph = Phases()
    os.chdir(common.ROOT)
    with ph.time("import"):
        sys.path.insert(0, os.path.join(common.HERE, "models"))
        import dynpois_cmdstan as m  # noqa: E402

    work = os.path.join(common.BUILD, f"cmdstan_{variant}")
    os.makedirs(work, exist_ok=True)
    src = os.path.join(common.HERE, "models", "dynpois_reduce_sum.stan") if a.reduce_sum \
        else os.path.join(common.ROOT, "bench", "dynpois", "dynpois.stan")
    stan_file = os.path.join(work, "dynpois.stan")
    shutil.copyfile(src, stan_file)
    out_dir = os.path.join(common.ROOT, ".tmp", name)
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    with ph.time("load_data"):
        data_file = os.path.join(out_dir, "data.json")
        json.dump(m.load_data(), open(data_file, "w"))
    with ph.time("compile"):
        model = m.compile_model(stan_file, threads=a.reduce_sum)
    with ph.time("sampling"):
        fit = m.sample(model, data_file, a.seed, 3 if a.reduce_sum else None, out_dir)

    with ph.time("postprocess"):
        facts = postprocess(fit.runset.csv_files, name, out_dir)
    ph.set(seed=a.seed, **facts, **describe(a.reduce_sum))
    ph.write()


if __name__ == "__main__":
    main()
