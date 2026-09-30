"""Fit dynpois.stan with CmdStan NUTS (default settings), 4 parallel chains.

Usage (rustmc venv, Python 3.12, cmdstanpy 1.3.0, CmdStan 2.40.0 in mint/build):
    python run_stan.py SIZE WARMUP DRAWS [--seed N] [--tag TAG]

No thinning (thin=1): gradient counts are summed over every saved warmup and
sampling iteration, which requires unthinned output.

Writes results/stan_<SIZE>.json and results/stan_<SIZE>_draws.npz. wall_seconds
times the sample() call only (CmdStan processes plus CSV writing); compilation is
excluded and reported in notes.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

from stan_common import BUILD, HERE, compile_model, load_data

CHAINS = 4
DEFAULT_SEED = 20260930


def csv_elapsed(path):
    """CmdStan's own per-chain timing comments (warmup, sampling seconds)."""
    warm = samp = None
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                continue
            m = re.search(r"Elapsed Time:\s*([\d.eE+-]+) seconds \(Warm-up\)", line)
            if m:
                warm = float(m.group(1))
            m = re.search(r"#\s*([\d.eE+-]+) seconds \(Sampling\)", line)
            if m:
                samp = float(m.group(1))
    return warm, samp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=["small", "large"])
    ap.add_argument("warmup", type=int)
    ap.add_argument("draws", type=int, help="retained draws per chain")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--tag", default="", help="suffix for the results file stem")
    a = ap.parse_args()
    size, warmup, draws, SEED = a.size, a.warmup, a.draws, a.seed
    thin = 1
    tag = f"_{a.tag}" if a.tag else ""
    y, data = load_data(size)
    G, T = y.shape

    model, compile_s = compile_model()
    out = os.path.join(BUILD, "stan_out", size + tag)
    os.makedirs(out, exist_ok=True)
    for old in glob.glob(os.path.join(out, "*")):
        os.remove(old)
    data_file = os.path.join(out, "data.json")
    with open(data_file, "w") as f:
        json.dump(data, f)

    t0 = time.perf_counter()
    fit = model.sample(data=data_file, chains=CHAINS, parallel_chains=CHAINS, seed=SEED,
                       iter_warmup=warmup, iter_sampling=draws, save_warmup=True,
                       output_dir=out, show_progress=False)
    wall = time.perf_counter() - t0

    cols = list(fit.column_names)
    arr = fit.draws(inc_warmup=True, concat_chains=False)  # (warmup+draws, chains, cols)
    assert arr.shape[:2] == (warmup + draws, CHAINS), arr.shape
    idx = {c: i for i, c in enumerate(cols)}
    leap = arr[:, :, idx["n_leapfrog__"]]
    leapfrog_total = int(leap.sum())
    transitions = int(leap.size)
    # Each NUTS transition also re-evaluates the gradient at its start point
    # (base_nuts.hpp: hamiltonian_.init), so the count is leapfrogs + transitions.
    # Step-size search evaluations (at start and each adaptation window) are not
    # in the CSV and are excluded.
    gradients = leapfrog_total + transitions
    post = arr[warmup:]  # (draws, chains, cols)

    def take(name):
        return np.moveaxis(post[:, :, idx[name]], 0, 1)  # (chains, draws)

    pop = take("pop")
    beta = np.stack([take(f"beta[{g + 1}]") for g in range(G)], axis=-1)
    terminal = np.stack([take(f"terminal[{g + 1}]") for g in range(G)], axis=-1)
    divergent = int(post[:, :, idx["divergent__"]].sum())
    treedepth = post[:, :, idx["treedepth__"]]
    stepsize = [float(post[0, c, idx["stepsize__"]]) for c in range(CHAINS)]
    max_depth_hits = int((treedepth >= 10).sum())
    elapsed = [csv_elapsed(p) for p in fit.runset.csv_files]

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    stem = os.path.join(HERE, "results", f"stan_{size}{tag}")
    np.savez(stem + "_draws.npz", pop=pop, beta=beta, terminal=terminal)
    D = 1 + G + T + G * T
    notes = (
        f"CmdStan 2.40.0 via cmdstanpy 1.3.0, default NUTS (diag_e, adapt_delta 0.8, max_treedepth 10), "
        f"chains={CHAINS} parallel_chains={CHAINS}, seed={SEED}, warmup={warmup}, draws={draws} retained, thin={thin}. "
        f"Model dynpois.stan (centred: pop, beta, shared, innov parameters; D={D}). "
        f"wall_seconds = sample() call (CmdStan runs + CSV output of ~{D + G} columns per draw incl. "
        f"warmup since save_warmup=True); compile seconds: "
        f"{'unknown' if compile_s is None else f'{compile_s:.1f}'} (excluded; measured when the executable was built). "
        f"Per-chain CmdStan (warmup, sampling) seconds: {elapsed}. "
        f"gradients = sum of n_leapfrog__ ({leapfrog_total}) plus one start-point gradient per "
        f"transition ({transitions}) over warmup+sampling, all chains; step-size-search "
        f"evaluations are not recorded and are excluded. "
        f"Post-warmup divergences {divergent}, treedepth-10 hits {max_depth_hits}, "
        f"step sizes {[round(s, 4) for s in stepsize]}, mean leapfrog/iter (sampling) "
        f"{float(post[:, :, idx['n_leapfrog__']].mean()):.1f}."
    )
    result = {
        "implementation": "stan",
        "G": G, "T": T, "chains": CHAINS, "warmup": warmup, "draws": draws, "thin": thin,
        "wall_seconds": wall, "gradients": gradients, "notes": notes,
        "extra": {"compile_seconds": compile_s, "chain_elapsed_warmup_sampling": elapsed,
                  "divergent": divergent, "treedepth_max_hits": max_depth_hits,
                  "stepsize": stepsize, "leapfrog_total": leapfrog_total,
                  "transitions": transitions, "seed": SEED, "tag": tag.lstrip("_")},
    }
    with open(stem + ".json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "extra"}, indent=1))


if __name__ == "__main__":
    main()
