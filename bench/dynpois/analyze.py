"""Diagnostics and cross-implementation agreement for the dynamic Poisson bench.

Usage (rustmc venv, Python 3.12, ArviZ 0.23): python analyze.py

Loads every results/<stem>_draws.npz with a matching results/<stem>.json, and
applies the same ArviZ code to each: rank-normalised split R-hat, bulk ESS and
tail ESS for pop, beta[g] and terminal[g]. Writes a markdown report to stdout and
results/summary.json.

Agreement between two runs of the same size, per quantity q:
  sd_z   = |mean_a - mean_b| / sqrt((sd_a^2 + sd_b^2) / 2)   (posterior-sd units)
  mcse_z = |mean_a - mean_b| / sqrt(mcse_a^2 + mcse_b^2)     (Monte Carlo error units)
reported as the max over quantities. mcse_z is reported only when both runs
mixed (max R-hat <= 1.01, min bulk/tail ESS >= 400) and are independent (not the
same implementation and seed, which share chain trajectories); ArviZ's MCSE
understates the error of a stuck chain. Runs are paired by (G, T): each size has
one data set. Maxima are over 41 (small) or 501 (large) quantities.
Truth comparison (descriptive, one data set, not a calibration test):
z = (posterior mean - true value) / posterior sd, per quantity.
"""
import glob
import json
import os
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "build")
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(BUILD, "cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BUILD, "cache", "mpl"))
warnings.filterwarnings("ignore", category=FutureWarning)

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def flat(ds):
    """Dataset of pop (), beta (G), terminal (G) -> 1-D array in fixed order."""
    return np.concatenate([np.atleast_1d(ds["pop"].values),
                           ds["beta"].values.ravel(), ds["terminal"].values.ravel()])


def names(G):
    return ["pop"] + [f"beta[{g}]" for g in range(G)] + [f"terminal[{g}]" for g in range(G)]


def load_run(stem):
    meta = json.load(open(stem + ".json"))
    d = np.load(stem + "_draws.npz")
    pop, beta, terminal = d["pop"], d["beta"], d["terminal"]
    C, N = pop.shape
    G = beta.shape[2]
    assert beta.shape == terminal.shape == (C, N, G), (beta.shape, terminal.shape)
    assert G == meta["G"], (G, meta["G"])
    if C != meta["chains"] or N != meta["draws"]:
        print(f"warning: {stem}: npz shape ({C},{N}) differs from JSON chains/draws "
              f"({meta['chains']},{meta['draws']})")
    ds = xr.Dataset({
        "pop": (("chain", "draw"), pop),
        "beta": (("chain", "draw", "group"), beta),
        "terminal": (("chain", "draw", "group"), terminal),
    }, coords={"chain": np.arange(C), "draw": np.arange(N), "group": np.arange(G)})
    rhat = flat(az.rhat(ds, method="rank"))
    bulk = flat(az.ess(ds, method="bulk"))
    tail = flat(az.ess(ds, method="tail"))
    mcse = flat(az.mcse(ds, method="mean"))
    mean = flat(ds.mean(dim=("chain", "draw")))
    sd = flat(ds.std(dim=("chain", "draw"), ddof=1))
    return {"stem": os.path.basename(stem), "meta": meta, "G": G, "chains": C, "draws": N,
            "rhat": rhat, "bulk": bulk, "tail": tail, "mcse": mcse, "mean": mean, "sd": sd}


def size_of(G):
    return {20: "small", 250: "large"}.get(G, f"G{G}")


def truth_vector(size):
    path = os.path.join(HERE, f"data_{size}", "truth.json")
    if not os.path.exists(path):
        return None
    t = json.load(open(path))
    return np.concatenate([[t["pop"]], t["beta"], t["terminal"]])


MIX_RHAT, MIX_ESS = 1.01, 400  # Vehtari et al. (2021): R-hat < 1.01, ESS > 100 per chain


def nanworst_max(x):
    """Max that treats NaN as the worst value (reported as NaN)."""
    return float("nan") if np.isnan(x).any() else float(np.max(x))


def nanworst_min(x):
    return float("nan") if np.isnan(x).any() else float(np.min(x))


def clean(obj):
    """Replace non-finite floats by None so summary.json is standard JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def main():
    stems = sorted(p[:-len("_draws.npz")] for p in glob.glob(os.path.join(RESULTS, "*_draws.npz")))
    stems = [s for s in stems if os.path.exists(s + ".json")]
    runs = [load_run(s) for s in stems]
    rows, summary = [], {"runs": [], "agreement": [], "definitions": __doc__,
                         "mixed_criterion": f"max R-hat <= {MIX_RHAT} and min bulk and tail ESS >= {MIX_ESS}, no NaN"}
    for r in runs:
        m = r["meta"]
        size = size_of(r["G"])
        wall = float(m["wall_seconds"])
        nm = names(r["G"])
        truth = truth_vector(size) if m.get("T") == 150 else None
        z = None if truth is None else (r["mean"] - truth) / r["sd"]
        rhat_filled = np.where(np.isnan(r["rhat"]), np.inf, r["rhat"])
        bulk_filled = np.where(np.isnan(r["bulk"]), -np.inf, r["bulk"])
        max_rhat, min_bulk, min_tail = nanworst_max(r["rhat"]), nanworst_min(r["bulk"]), nanworst_min(r["tail"])
        mixed = bool(max_rhat <= MIX_RHAT and min_bulk >= MIX_ESS and min_tail >= MIX_ESS)
        r["mixed"] = mixed
        rec = {
            "stem": r["stem"], "implementation": m["implementation"], "size": size,
            "G": r["G"], "T": m.get("T"), "chains": r["chains"], "warmup": m["warmup"],
            "draws": r["draws"], "thin": m.get("thin", 1), "wall_seconds": wall,
            "gradients": m.get("gradients"), "n_quantities": len(nm), "mixed": mixed,
            "max_rhat": max_rhat, "max_rhat_at": nm[int(np.argmax(rhat_filled))],
            "share_rhat_gt_1.01": float(np.mean(~(r["rhat"] <= 1.01))),  # NaN counts as failing
            "min_bulk_ess": min_bulk, "min_bulk_ess_at": nm[int(np.argmin(bulk_filled))],
            "min_tail_ess": min_tail,
            "min_bulk_ess_per_second": min_bulk / wall,
            "pop_mean": float(r["mean"][0]), "pop_sd": float(r["sd"][0]),
            "truth_max_abs_z": None if z is None else float(np.max(np.abs(z))),
            "truth_share_abs_z_gt_2": None if z is None else float(np.mean(np.abs(z) > 2)),
            "truth_mean_z": None if z is None else float(np.mean(z)),
            "truth_pop_z": None if z is None else float(z[0]),
        }
        summary["runs"].append(rec)
        rows.append(rec)

    # Pair runs fitted to the same data shape (G, T); each size has one data set.
    by_shape = {}
    for r in runs:
        by_shape.setdefault((r["G"], r["meta"].get("T")), []).append(r)
    agree = []
    for (G, T), group in by_shape.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                seed_a = a["meta"].get("extra", {}).get("seed")
                seed_b = b["meta"].get("extra", {}).get("seed")
                same_chain = (a["meta"]["implementation"] == b["meta"]["implementation"]
                              and seed_a is not None and seed_a == seed_b)
                diff = np.abs(a["mean"] - b["mean"])
                sd_z = diff / np.sqrt((a["sd"] ** 2 + b["sd"] ** 2) / 2)
                mc_z = diff / np.sqrt(a["mcse"] ** 2 + b["mcse"] ** 2)
                usable = a["mixed"] and b["mixed"] and not same_chain
                nm = names(G)
                k = int(np.argmax(np.where(np.isnan(sd_z), np.inf, sd_z)))
                rec = {"size": size_of(G), "a": a["stem"], "b": b["stem"],
                       "both_mixed": a["mixed"] and b["mixed"],
                       "independent": not same_chain,
                       "max_sd_z": nanworst_max(sd_z), "max_sd_z_at": nm[k],
                       "median_sd_z": float(np.median(sd_z)),
                       "max_mcse_z": nanworst_max(mc_z) if usable else None,
                       "share_mcse_z_gt_3": float(np.mean(~(mc_z <= 3))) if usable else None}
                agree.append(rec)
    summary["agreement"] = agree

    def f(x, spec):
        return "NaN" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)

    print("## Diagnostics (pop, beta[g], terminal[g]; ArviZ rank R-hat, bulk/tail ESS)\n")
    print(f"'mixed' = max R-hat <= {MIX_RHAT} and min bulk and tail ESS >= {MIX_ESS}. ESS/s for a run "
          "that has not mixed is not a usable efficiency figure (its ESS estimate is unreliable).\n")
    print("| run | size | warmup | draws x thin | wall s | mixed | max R-hat (at) | share R-hat>1.01 "
          "| min bulk ESS (at) | min tail ESS | min bulk ESS/s | gradients |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['stem']} | {r['size']} | {r['warmup']} | {r['draws']} x {r['thin']} "
              f"| {r['wall_seconds']:.1f} | {'yes' if r['mixed'] else 'NO'} "
              f"| {f(r['max_rhat'], '.3f')} ({r['max_rhat_at']}) "
              f"| {r['share_rhat_gt_1.01']:.3f} | {f(r['min_bulk_ess'], '.0f')} ({r['min_bulk_ess_at']}) "
              f"| {f(r['min_tail_ess'], '.0f')} | {f(r['min_bulk_ess_per_second'], '.3g')}"
              f"{'' if r['mixed'] else ' (not mixed)'} "
              f"| {r['gradients'] if r['gradients'] is not None else 'n/a'} |")
    print("\n## Truth recovery (descriptive: z = (posterior mean - truth) / posterior sd, one data set)\n")
    print("| run | pop mean (sd) | pop z | mean z | max abs z | share abs z > 2 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r["truth_max_abs_z"] is None:
            continue
        print(f"| {r['stem']} | {r['pop_mean']:.3f} ({r['pop_sd']:.3f}) | {r['truth_pop_z']:.2f} "
              f"| {r['truth_mean_z']:.2f} | {r['truth_max_abs_z']:.2f} | {r['truth_share_abs_z_gt_2']:.3f} |")
    print("\n## Posterior-mean agreement between runs on the same data\n")
    print("MCSE-scaled differences are shown only when both runs mixed and are independent "
          "(same implementation with the same seed shares chain trajectories). Means only; "
          "agreement of means does not establish agreement of variances or tails.\n")
    print("| size | run a | run b | both mixed | independent | max abs diff / sd (at) | median "
          "| max abs diff / MCSE | share > 3 MCSE |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a in agree:
        print(f"| {a['size']} | {a['a']} | {a['b']} | {'yes' if a['both_mixed'] else 'NO'} "
              f"| {'yes' if a['independent'] else 'NO'} | {f(a['max_sd_z'], '.3f')} ({a['max_sd_z_at']}) "
              f"| {a['median_sd_z']:.3f} | {f(a['max_mcse_z'], '.2f') if a['max_mcse_z'] is not None else 'n/a'} "
              f"| {f(a['share_mcse_z_gt_3'], '.3f') if a['share_mcse_z_gt_3'] is not None else 'n/a'} |")
    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(clean(summary), fh, indent=1, allow_nan=False)


if __name__ == "__main__":
    main()
