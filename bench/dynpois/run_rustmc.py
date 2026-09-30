"""Fit rustmc's BayesianDynamicPoisson to the SPEC benchmark data.

Usage (rustmc_demo venv, Python 3.14, rustmc 0.13.0 from integrate/forecasting-toolbox):
    python run_rustmc.py SIZE WARMUP DRAWS THIN [--seed N] [--tag TAG]

SIZE is small or large. DRAWS is the number of retained draws per chain (rustmc
runs WARMUP + DRAWS * THIN sweeps per chain and keeps every THIN-th post-warmup
sweep). Writes results/rustmc_<SIZE>[_TAG].json and results/rustmc_<SIZE>[_TAG]_draws.npz.
"""
import argparse
import json
import os
import re
import resource
import sys
import time

import numpy as np
import rustmc as rmc

HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS = 4
DEFAULT_SEED = 20260930
CONFIG = dict(initial_mean=0.0, coefficient_sd=1.0, group_sd=0.4,
              process_sd=0.08, shared_process_sd=0.05)
CAP = 25_000_000  # rustmc_core::forecast_common::MAX_MATERIALIZED_VALUES


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def probe_latent_dimension(model, y):
    """Ask rustmc for a deliberately oversized fit; its allocation error reports
    chains * draws * latent_size, from which we read the latent size it uses."""
    draws = CAP  # chains * CAP * D > CAP for any D >= 1
    try:
        model.fit(y, chains=1, draws=draws, warmup=0, thin=1, seed=1)
    except Exception as e:  # noqa: BLE001 - we only parse the message
        msg = str(e)
        nums = [int(n.replace(",", "").replace("_", "")) for n in re.findall(r"\d[\d,_]*", msg)]
        cands = [n // draws for n in nums if n >= draws and n % draws == 0]
        return (cands[0] if cands else None), msg
    return None, "no error raised"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=["small", "large"])
    ap.add_argument("warmup", type=int)
    ap.add_argument("draws", type=int, help="retained draws per chain (after thinning)")
    ap.add_argument("thin", type=int)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--tag", default="", help="suffix for the results file stem")
    a = ap.parse_args()
    size, warmup, draws, thin, SEED = a.size, a.warmup, a.draws, a.thin, a.seed
    tag = f"_{a.tag}" if a.tag else ""
    y = np.load(os.path.join(HERE, f"data_{size}", "y.npy"))
    G, T = y.shape
    D = 1 + G + T + G * T
    retained = CHAINS * draws * D
    if retained > CAP:
        sys.exit(f"chains*draws*D = {retained} exceeds rustmc cap {CAP}; lower DRAWS")

    model = rmc.BayesianDynamicPoisson(**CONFIG)
    rustmc_dim, probe_msg = probe_latent_dimension(model, y)

    # y is passed as (G series, T times): rustmc reads y.len() as groups.
    c0, t0 = cpu_seconds(), time.perf_counter()
    fit = model.fit(y, chains=CHAINS, draws=draws, warmup=warmup, thin=thin, seed=SEED)
    wall = time.perf_counter() - t0
    cpu = cpu_seconds() - c0

    assert fit.groups == G, (fit.groups, G)
    assert fit.chains == CHAINS and fit.draws == draws
    s = fit.get_samples_2d()  # name -> (chains, draws)
    pop = np.asarray(s["population_beta[0,0]"])
    beta = np.stack([s[f"beta[0,{g},0]"] for g in range(G)], axis=-1)
    terminal = np.stack([s[f"terminal_state[0,{g}]"] for g in range(G)], axis=-1)
    states = np.asarray(fit.state_samples(0))  # (chains, draws, G, T)
    assert states.shape == (CHAINS, draws, G, T), states.shape
    # terminal_state[0,g] must be the last training state of group g (no beta).
    assert np.array_equal(terminal, states[:, :, :, -1])
    assert pop.shape == (CHAINS, draws) and beta.shape == (CHAINS, draws, G)

    stats = fit.sampler_stats
    evals = [int(v) for v in stats["likelihood_evaluations"]]
    fixed = dict(stats["fixed_parameters"])
    for k, v in CONFIG.items():
        if k in fixed:
            assert fixed[k] == v, (k, fixed[k], v)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    stem = os.path.join(HERE, "results", f"rustmc_{size}{tag}")
    np.savez(stem + "_draws.npz", pop=pop, beta=beta, terminal=terminal)
    notes = (
        f"rustmc 0.13.0 BayesianDynamicPoisson({', '.join(f'{k}={v}' for k, v in CONFIG.items())}); "
        f"fit(y[G,T], chains={CHAINS}, warmup={warmup}, draws={draws}, thin={thin}, seed={SEED}). "
        f"Sampler: block elliptical slice (no gradients). draws is retained per chain after "
        f"thinning; each chain ran {warmup + draws * thin} sweeps ({draws * thin} post-warmup). "
        f"rustmc's allocation guard checks chains*draws*D = {retained:,} <= {CAP:,} (the decoded "
        f"draws it actually keeps hold 1+G+G*T values each: population, group coefficients, states). "
        f"Latent size D per SPEC = {D}; the latent size in rustmc's allocation guard (read from its "
        f"error message for an oversized request) is {rustmc_dim}. "
        f"Likelihood evaluations per chain (incl. warmup, each is one group or all groups): {evals}. "
        f"Process CPU seconds during fit {cpu:.1f} vs wall {wall:.1f} "
        f"(ratio {cpu / wall:.2f}; chains run on a rayon pool)."
    )
    result = {
        "implementation": "rustmc",
        "G": G, "T": T, "chains": CHAINS, "warmup": warmup,
        "draws": draws, "thin": thin,
        "wall_seconds": wall, "gradients": None, "notes": notes,
        "extra": {
            "cpu_seconds": cpu, "likelihood_evaluations": evals,
            "latent_dim_spec": D, "latent_dim_rustmc_probe": rustmc_dim,
            "probe_message": probe_msg, "seed": SEED, "tag": tag.lstrip("_"),
        },
    }
    with open(stem + ".json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "extra"}, indent=1))


if __name__ == "__main__":
    main()
