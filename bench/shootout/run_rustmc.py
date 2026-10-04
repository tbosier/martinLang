#!/usr/bin/env python3
"""rustmc 0.13.0 BayesianDynamicPoisson (models/dynpois_rustmc.py): block
elliptical slice sampling, a different algorithm (no gradients).
Run with the rustmc_demo venv (Python 3.14):

usage: run_rustmc.py --seed S
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import Phases  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
name = os.environ.get("SHOOTOUT_RUN", f"rustmc_s{a.seed}")
ph = Phases()
os.chdir(common.ROOT)
with ph.time("import"):
    sys.path.insert(0, os.path.join(common.HERE, "models"))
    import dynpois_rustmc as m  # noqa: E402
    import numpy as np  # noqa: E402
with ph.time("load_data"):
    y = m.load_data()
G, T = y.shape
model = m.make_model()
with ph.time("sampling"):
    fit = m.sample(model, y, a.seed)
with ph.time("postprocess"):
    s = fit.get_samples_2d()
    pop = np.asarray(s["population_beta[0,0]"])
    beta = np.stack([s[f"beta[0,{g},0]"] for g in range(G)], axis=-1)
    terminal = np.stack([s[f"terminal_state[0,{g}]"] for g in range(G)], axis=-1)
    common.save_draws(name, pop, beta, terminal)
    stats = fit.sampler_stats
    evals = [int(v) for v in stats["likelihood_evaluations"]]
ph.d["phases"]["compile"] = 0.0
ph.set(implementation="rustmc", seed=a.seed, gradients=None, likelihood_evaluations=evals,
       gradients_note="no gradients (elliptical slice sampling); likelihood evaluations per chain recorded",
       divergences=None, kept_draws_per_chain=int(fit.draws), thin=8,
       sampling_note="the fit() call: 1000 warmup + 1000 sweeps per chain, every 8th kept",
       settings="rustmc 0.13.0 BayesianDynamicPoisson(initial_mean=0, coefficient_sd=1, group_sd=0.4, "
                "process_sd=0.08, shared_process_sd=0.05); fit(chains=4, warmup=1000, draws=125, thin=8): its "
                "documented example settings (4 chains, 1000 + 1000) thinned to fit its 25M stored-value cap")
ph.write()
