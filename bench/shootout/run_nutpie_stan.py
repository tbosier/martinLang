#!/usr/bin/env python3
"""nutpie with the Stan model (bench/dynpois/dynpois.stan), built by
BridgeStan 2.9.0 against CmdStan 2.40's Stan (stanc --O1, -O3 -march=native,
STAN_THREADS as nutpie requires).

usage: run_nutpie_stan.py --seed S --adapt diag|low_rank
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import Phases  # noqa: E402

os.environ.setdefault("BRIDGESTAN", os.path.join(common.ROOT, "build", "bs", "bridgestan-2.9.0"))
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--adapt", choices=["diag", "low_rank"], required=True)
a = ap.parse_args()
name = os.environ.get("SHOOTOUT_RUN", f"nutpie_stan_{a.adapt}_s{a.seed}")
ph = Phases()
os.chdir(common.ROOT)
with ph.time("import"):
    sys.path.insert(0, os.path.join(common.HERE, "models"))
    import dynpois_nutpie_stan as m  # noqa: E402
    import numpy as np  # noqa: E402
with ph.time("load_data"):
    data = m.load_data()
with ph.time("compile"):
    compiled = m.compile_model()
with ph.time("sampling"):
    tr = m.sample(compiled, data, a.seed, a.adapt)
with ph.time("postprocess"):
    post = tr.posterior
    pop = post["pop"].values
    beta = post["beta"].values
    terminal = post["terminal"].values
    common.save_draws(name, pop, beta, terminal)
    ss = tr.sample_stats
    ws = tr.warmup_sample_stats if hasattr(tr, "warmup_sample_stats") else None
    samp_steps = int(ss["n_steps"].values.sum())
    warm_steps = int(ws["n_steps"].values.sum()) if ws is not None else None
ph.set(implementation=f"nutpie_stan_{a.adapt}", seed=a.seed,
       gradients=(samp_steps + warm_steps + common.CHAINS) if warm_steps is not None else None,
       warmup_gradients=warm_steps, sampling_gradients=samp_steps,
       gradients_note="sum of n_steps over warmup and draws plus one per chain for the initial point",
       divergences=int(ss["diverging"].values.sum()),
       step_size=[float(s) for s in ss["step_size"].values[:, -1]],
       leapfrog_per_draw=[float(v) for v in ss["n_steps"].values.mean(axis=1)],
       threads_per_chain=1,
       sampling_note="the nutpie.sample() call (sampling and storing the trace in memory, warmup included)",
       settings=f"nutpie 0.16.11 adaptation='{a.adapt}', tune 1000, draws 1000, chains 4, cores 4, "
                f"save_warmup default (True), other settings default (maxdepth 10, target accept 0.8); "
                f"model built by BridgeStan 2.9.0 with stanc --O1, CXXFLAGS=-march=native, STAN_THREADS")
ph.write()
