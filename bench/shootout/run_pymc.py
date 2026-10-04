#!/usr/bin/env python3
"""PyMC 5 model (models/dynpois_pymc.py) sampled by nutpie.

usage: run_pymc.py --seed S --backend numba|jax
  numba  nutpie.compile_pymc_model's default backend (PyTensor -> numba)
  jax    backend="jax", gradient_backend="pytensor" (float64; faster than "jax" in verify.py)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import Phases  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--backend", choices=["numba", "jax"], required=True)
a = ap.parse_args()
name = os.environ.get("SHOOTOUT_RUN", f"pymc_{a.backend}_s{a.seed}")
ph = Phases()
os.chdir(common.ROOT)
with ph.time("import"):
    if a.backend == "jax":
        import jax
        jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, os.path.join(common.HERE, "models"))
    import dynpois_pymc as m  # noqa: E402
    import numpy as np  # noqa: E402
with ph.time("load_data"):
    y = m.load_data()
with ph.time("build_model"):
    model = m.make_model(y)
with ph.time("compile"):
    compiled = m.compile_model(model, a.backend)
with ph.time("sampling"):
    tr = m.sample(compiled, a.seed)
with ph.time("postprocess"):
    post = tr.posterior
    pop = post["pop"].values
    beta = post["beta"].values
    terminal = post["shared"].values.sum(axis=-1)[:, :, None] + post["innov"].values.sum(axis=-1)
    common.save_draws(name, pop, beta, terminal)
    ss = tr.sample_stats
    ws = tr.warmup_sample_stats if hasattr(tr, "warmup_sample_stats") else None
    samp_steps = int(ss["n_steps"].values.sum())
    warm_steps = int(ws["n_steps"].values.sum()) if ws is not None else None
ph.set(implementation=f"pymc_{a.backend}", seed=a.seed,
       gradients=(samp_steps + warm_steps + common.CHAINS) if warm_steps is not None else None,
       warmup_gradients=warm_steps, sampling_gradients=samp_steps,
       gradients_note="sum of n_steps over warmup and draws plus one per chain for the initial point",
       divergences=int(ss["diverging"].values.sum()),
       step_size=[float(s) for s in ss["step_size"].values[:, -1]],
       leapfrog_per_draw=[float(v) for v in ss["n_steps"].values.mean(axis=1)],
       threads_per_chain=1,
       compile_note="build_model (PyMC graph) is reported separately; compile = nutpie.compile_pymc_model "
                    "(PyTensor rewrites and numba/JAX compilation)" + (
                        "; JAX also traces and compiles lazily on first call, which then falls in sampling"
                        if a.backend == "jax" else ""),
       sampling_note="the nutpie.sample() call (sampling and storing the trace in memory, warmup included)",
       settings=f"PyMC 5.28.5, nutpie 0.16.11 compile_pymc_model(backend='{a.backend}'"
                + (", gradient_backend='pytensor'" if a.backend == "jax" else "") +
                "), sample(tune 1000, draws 1000, chains 4, cores 4), defaults otherwise "
                "(adaptation 'diag', maxdepth 10, target accept 0.8, save_warmup True)")
ph.write()
