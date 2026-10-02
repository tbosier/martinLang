#!/usr/bin/env python3
"""Sanity check of Mint's sampler against nutpie, on the same gradient code.

Both samplers run the same compiled Stan model (BridgeStan .so from
build.sh, STAN_THREADS) on the same data: Mint's runtime NUTS through
bs_driver, and nutpie's NUTS through its own BridgeStan binding. So the
gradient code is identical and what differs is the sampler: its overhead
per gradient and its adaptation (nutpie adapts the mass matrix from draws
and gradients; Mint ports Stan's windowed warmup). This is a sampler
comparison, not a language comparison.

Settings for both: 4 chains in parallel threads, 1000 warmup + 1000 draws,
the same seeds (the two samplers' random streams differ anyway), maximum
tree depth 10, target acceptance 0.8 (both defaults).

Measured per run:
  - wall seconds: Mint, the runtime's own sampling time; nutpie, the
    nutpie.sample() call (which also stores the trace in memory);
  - gradients: Mint, the runtime's count; nutpie, the sum of n_steps over
    warmup and kept draws (leapfrog steps, one gradient each), plus one per
    chain for the initial point;
  - microseconds per gradient per chain (wall x chains / gradients), and
    that minus the gradient alone (results/grad.json, same model, clean
    median), an estimate of the sampler's overhead per gradient;
  - bulk and tail ESS and split R-hat over every parameter, computed by
    the same ArviZ code on both samplers' constrained draws, and the lowest
    bulk ESS per 1000 gradients.

With --chains 1 --pin, each run is one chain pinned to the least busy core
(chosen afresh before each run): the cleaner measurement of the sampler's
overhead per gradient, since four chains on a loaded machine add contention
that has nothing to do with either sampler.

usage: python bench/same_sampler/nutpie_check.py --seeds 1 2 3
       python bench/same_sampler/nutpie_check.py --chains 1 --pin --seeds 1 2 3 4 5 --out results/nutpie_1chain.json
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--problems", default="dynpois_small,logistic,eight_schools")
ap.add_argument("--out", default="results/nutpie.json")
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--pin", action="store_true")
args = ap.parse_args()
CHAINS, WARMUP, DRAWS = args.chains, 1000, 1000
PIN = None  # the CPU of the current run (with --pin)


def diagnostics(x):
    """x: (chains, draws, D) constrained draws."""
    import arviz as az

    ds = az.convert_to_dataset({"x": x})
    bulk = az.ess(ds, method="bulk")["x"].values
    tail = az.ess(ds, method="tail")["x"].values
    rhat = az.rhat(ds)["x"].values
    return {"min_bulk_ess": float(np.min(bulk)), "min_tail_ess": float(np.min(tail)),
            "max_rhat": float(np.max(rhat)), "dim": int(x.shape[2])}


def run_mint(problem, seed):
    tmp = os.path.join(common.OUT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, f"nutpie_check_{problem}_{seed}.draws")
    argv, env = common.command(problem, "stan", DRAWS, WARMUP, CHAINS, seed)
    out, wall = common.run(argv, dict(env, MINT_DRAWS=path), pin=PIN)
    r = common.parse_run(out)
    hdr = np.fromfile(path, dtype="<u8", count=3)
    C, N, D = (int(v) for v in hdr)
    x = np.fromfile(path, dtype="<f8", offset=24).reshape(C, N, D)
    os.remove(path)
    return {"wall_seconds": r["sampling_seconds"], "gradients": r["gradients"],
            "warmup_gradients": r["warmup_gradients"], "divergences": r["divergences"],
            "step_size": r["step_size"], "leapfrog_per_draw": r["leapfrog_per_draw"], **diagnostics(x)}


def run_nutpie(problem, seed):
    import nutpie
    from nutpie import _lib
    from nutpie.compile_stan import CompiledStanModel

    p = common.PROBLEMS[problem]
    so = os.path.join(common.OUT, "stan", p["stan"]["stan"] + "_model.so")
    data = json.load(open(os.path.join(common.OUT, "data", p["json"] + ".json")))
    model = CompiledStanModel(code="", library=_lib.StanLibrary(so), dims=None, _coords=None,
                              model_name=problem, model=None, data=None)
    model = model.with_data(**{k: np.asarray(v) for k, v in data.items()})
    if PIN is not None:
        os.sched_setaffinity(0, {int(PIN)})
    t = time.perf_counter()
    tr = nutpie.sample(model, draws=DRAWS, tune=WARMUP, chains=CHAINS, cores=CHAINS, seed=seed,
                       progress_bar=False, save_warmup=True)
    wall = time.perf_counter() - t
    if PIN is not None:
        os.sched_setaffinity(0, set(range(os.cpu_count())))
    post = tr.posterior
    # every variable flattened in Stan's order (as BridgeStan's param_constrain writes it)
    parts = []
    for name in post.data_vars:
        v = post[name].values
        parts.append(v.reshape(v.shape[0], v.shape[1], -1))
    x = np.concatenate(parts, axis=2)
    ws, ss = tr.warmup_sample_stats, tr.sample_stats
    grads = int(ws["n_steps"].values.sum() + ss["n_steps"].values.sum()) + CHAINS
    return {"wall_seconds": wall, "gradients": grads, "warmup_gradients": int(ws["n_steps"].values.sum()) + CHAINS,
            "divergences": int(ss["diverging"].values.sum()),
            "step_size": [float(s) for s in ss["step_size"].values[:, -1]],
            "leapfrog_per_draw": [float(v) for v in ss["n_steps"].values.mean(axis=1)], **diagnostics(x)}


grad_ns = {}
gpath = os.path.join(common.RESULTS, "grad.json")
if os.path.exists(gpath):
    for row in json.load(open(gpath))["rows"]:
        if row["impl"] == "stan" and row["kernel_threads"] == 1:
            grad_ns[row["problem"]] = row["ns_median_clean"] or row["ns_median"]

out_path = os.path.join(common.ROOT, "bench", "same_sampler", args.out)
res = {"what": "Mint's NUTS vs nutpie's NUTS on the same BridgeStan gradient", "versions": {},
       "settings": {"chains": CHAINS, "warmup": WARMUP, "draws": DRAWS, "pinned": args.pin}, "runs": []}
import nutpie  # noqa: E402
import bridgestan  # noqa: E402
res["versions"] = {"nutpie": nutpie.__version__, "bridgestan": bridgestan.__version__}
rng = random.Random(7)
for seed in args.seeds:
    jobs = [(p, s) for p in args.problems.split(",") for s in ("mint", "nutpie")]
    rng.shuffle(jobs)
    for problem, sampler in jobs:
        before = {"loadavg": common.loadavg(), "cpu_busy_percent": common.cpu_busy(1.0)}
        if args.pin:
            PIN, _ = common.quiet_cpus(1, before["cpu_busy_percent"])
            before["pinned_cpu"] = PIN
        r = run_mint(problem, seed) if sampler == "mint" else run_nutpie(problem, seed)
        r.update({"problem": problem, "sampler": sampler, "seed": seed, "before": before,
                  "us_per_gradient_per_chain": 1e6 * r["wall_seconds"] * CHAINS / r["gradients"],
                  "min_bulk_ess_per_1k_gradients": 1e3 * r["min_bulk_ess"] / r["gradients"],
                  "min_bulk_ess_per_second": r["min_bulk_ess"] / r["wall_seconds"]})
        if problem in grad_ns:
            r["gradient_alone_us"] = grad_ns[problem] / 1e3
            r["overhead_us_per_gradient"] = r["us_per_gradient_per_chain"] - grad_ns[problem] / 1e3
        res["runs"].append(r)
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"seed {seed} {problem:14s} {sampler:6s} {r['wall_seconds']:7.2f} s {r['gradients']:8d} grads "
              f"{r['us_per_gradient_per_chain']:8.2f} us/grad/chain  bulk ESS {r['min_bulk_ess']:7.0f} "
              f"tail ESS {r['min_tail_ess']:7.0f} R-hat {r['max_rhat']:.3f}  "
              f"ESS/1k grads {r['min_bulk_ess_per_1k_gradients']:.2f}  load {before['loadavg'][0]:.1f}", flush=True)
print(f"wrote {out_path}")
