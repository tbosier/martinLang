#!/usr/bin/env python3
"""NumPyro (models/dynpois_numpyro.py): JAX on CPU, float64, default NUTS.

usage: run_numpyro.py --seed S --chain-method parallel|vectorized

JAX compiles inside mcmc.run. The compile time is the sum of JAX's own
reported tracing, lowering and XLA compilation durations
(jax.monitoring events /jax/core/compile/*), and the sampling time is the
mcmc.run wall time minus that sum. Gradients: NumPyro's extra fields are
collected for the kept draws only, so the count covers sampling (sum of
num_steps); warmup gradients are not recorded.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import Phases  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--chain-method", choices=["parallel", "vectorized"], required=True)
a = ap.parse_args()
name = os.environ.get("SHOOTOUT_RUN", f"numpyro_{a.chain_method}_s{a.seed}")
ph = Phases()
os.chdir(common.ROOT)
with ph.time("import"):
    sys.path.insert(0, os.path.join(common.HERE, "models"))
    import dynpois_numpyro as m  # noqa: E402  (sets 4 host devices and x64 before JAX starts)
    import jax  # noqa: E402
    import jax.monitoring  # noqa: E402
    import numpy as np  # noqa: E402

compile_events = {}


def listener(event, duration, **kw):
    if event.startswith("/jax/core/compile/"):
        compile_events[event] = compile_events.get(event, 0.0) + duration


jax.monitoring.register_event_duration_secs_listener(listener)
assert jax.config.jax_enable_x64
assert len(jax.devices()) == 4, jax.devices()
with ph.time("load_data"):
    y = m.load_data()
t = time.perf_counter()
mcmc = m.sample(y, a.seed, a.chain_method)
# JAX dispatches asynchronously: wait for the draws before stopping the clock
jax.block_until_ready(mcmc.get_samples(group_by_chain=True))
jax.block_until_ready(mcmc.get_extra_fields(group_by_chain=True))
run_s = time.perf_counter() - t
comp = sum(compile_events.values())
ph.d["phases"]["compile"] = comp
ph.d["phases"]["sampling"] = run_s - comp
with ph.time("postprocess"):
    s = mcmc.get_samples(group_by_chain=True)
    pop = np.asarray(s["pop"])
    beta = np.asarray(s["beta"])
    terminal = np.asarray(s["shared"]).sum(axis=-1)[:, :, None] + np.asarray(s["innov"]).sum(axis=-1)
    common.save_draws(name, pop, beta, terminal)
    ex = mcmc.get_extra_fields(group_by_chain=True)
    steps = np.asarray(ex["num_steps"])
    div = np.asarray(ex["diverging"])
    step_size = np.asarray(mcmc.last_state.adapt_state.step_size).ravel().tolist()
ph.set(implementation=f"numpyro_{a.chain_method}", seed=a.seed,
       gradients=int(steps.sum()), warmup_gradients=None,
       gradients_note="sampling iterations only (sum of num_steps over the kept draws); warmup not recorded",
       divergences=int(div.sum()), step_size=step_size,
       leapfrog_per_draw=[float(v) for v in steps.mean(axis=1)],
       compile_events=compile_events, mcmc_run_seconds=run_s, devices=[str(d) for d in jax.devices()],
       compile_note="sum of JAX's reported trace, lowering and backend compile durations inside mcmc.run",
       sampling_note="mcmc.run wall time, until the draws are ready, minus the compile durations",
       settings=f"NumPyro 0.22.0, JAX 0.11.2 CPU, x64; NUTS defaults (max_tree_depth 10, target_accept_prob 0.8, "
                f"diagonal mass matrix); 4 chains, chain_method='{a.chain_method}', 4 XLA host devices, "
                f"1000 warmup + 1000 draws")
ph.write()
