#!/usr/bin/env python3
"""The one analysis script for every run of the shootout.

For each run (results/runs/NAME.json, draws in build/shootout/draws/NAME.npz)
it applies the same ArviZ code as bench/dynpois/analyze.py to pop, every
beta[g] and every terminal state (501 quantities): rank-normalised split
R-hat, bulk and tail ESS, MCSE of the mean, posterior mean and sd.

- mixed: max R-hat <= 1.01 and min bulk and tail ESS >= 400 (no NaN).
- ESS/s: lowest bulk ESS / sampling seconds, and / total wall seconds;
  reported for mixed runs only.
- agreement with the reference: per quantity
  sd_z = |mean - mean_ref| / sqrt((sd^2 + sd_ref^2) / 2); the maximum over the
  501 quantities is reported, and, when both runs mixed, the maximum of
  |mean - mean_ref| / sqrt(mcse^2 + mcse_ref^2) too. The reference is the
  CmdStan run (plain dynpois.stan first) with the highest lowest bulk ESS
  among those that mixed, else the best-mixing run of all.
- lines of code: non-blank, non-comment lines of the files listed per
  implementation in LOC_FILES.

Writes results/diagnostics.json (per run, so the draws need not be kept),
results/results.json and results.md (next to README.md).

usage: .venv/bin/python bench/shootout/analyze.py
"""
import glob
import json
import os
import re
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

os.environ.update(common.cache_env())
warnings.filterwarnings("ignore", category=FutureWarning)
import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

MIX_RHAT, MIX_ESS = 1.01, 400
LIMIT_SECONDS = 1200  # a run whose total wall time exceeds 20 minutes is not ranked
R = common.RESULTS

# implementation key -> (label, files counted as model code, note)
IMPLS = {
    "martin": ("Martin (default settings)", ["examples/dynamic_poisson.mint"], ""),
    "martin_optin": ("Martin, opt-in sampler options (MINT_METRIC=lowrank MINT_WARMUP=fast)",
                     ["examples/dynamic_poisson.mint"], "plus two environment variables"),
    "rust_nuts_diag": ("Rust end to end: tuned gradient + nuts-rs 0.19, diagonal adaptation",
                       ["bench/shootout/rust_nuts/src/main.rs", "bench/shootout/rust_nuts/src/dynpois.rs",
                        "bench/shootout/rust_nuts/src/team.rs", "bench/shootout/rust_nuts/Cargo.toml"],
                       "dynpois.rs is generated from baselines/dynpois_par.rs and includes its scalar "
                       "reference and self-test (about 270 lines)"),
    "rust_nuts_lowrank": ("Rust end to end: tuned gradient + nuts-rs 0.19, low-rank mass matrix",
                          ["bench/shootout/rust_nuts/src/main.rs", "bench/shootout/rust_nuts/src/dynpois.rs",
                           "bench/shootout/rust_nuts/src/team.rs", "bench/shootout/rust_nuts/Cargo.toml"],
                          "as above"),
    "rust_under_martin": ("Rust gradient under Martin's sampler (isolates the language)",
                          ["baselines/dynpois_par.rs", "baselines/common.rs", "baselines/simd.rs"],
                          "plus Martin's C runtime (sampler), not counted; includes a self-test"),
    "cmdstan_plain": ("Stan (CmdStan 2.40, cmdstanpy), dynpois.stan",
                      ["bench/dynpois/dynpois.stan", "bench/shootout/models/dynpois_cmdstan.py"],
                      "the .stan file includes 7 lines of generated quantities for the terminal states"),
    "cmdstan_reduce_sum": ("Stan (CmdStan 2.40, cmdstanpy), reduce_sum, 3 threads per chain",
                           ["bench/shootout/models/dynpois_reduce_sum.stan",
                            "bench/shootout/models/dynpois_cmdstan.py"], "as above"),
    "nutpie_stan_diag": ("nutpie 0.16 with the Stan model, diagonal adaptation",
                         ["bench/dynpois/dynpois.stan", "bench/shootout/models/dynpois_nutpie_stan.py"], ""),
    "nutpie_stan_low_rank": ("nutpie 0.16 with the Stan model, low-rank modified mass matrix",
                             ["bench/dynpois/dynpois.stan", "bench/shootout/models/dynpois_nutpie_stan.py"], ""),
    "pymc_numba": ("PyMC 5.28 + nutpie, numba backend (nutpie's default)",
                   ["bench/shootout/models/dynpois_pymc.py"], ""),
    "pymc_jax": ("PyMC 5.28 + nutpie, JAX backend", ["bench/shootout/models/dynpois_pymc.py"], ""),
    "numpyro_parallel": ("NumPyro 0.22 (JAX CPU, x64), chain_method=parallel",
                         ["bench/shootout/models/dynpois_numpyro.py"], ""),
    "numpyro_vectorized": ("NumPyro 0.22 (JAX CPU, x64), chain_method=vectorized",
                           ["bench/shootout/models/dynpois_numpyro.py"], ""),
    "rustmc": ("rustmc 0.13 BayesianDynamicPoisson (elliptical slice sampling)",
               ["bench/shootout/models/dynpois_rustmc.py"], "a library call; different algorithm"),
}
ORDER = list(IMPLS)


def loc(path):
    """Non-blank, non-comment lines. // and /* */ comments for .mint, .stan
    and .rs; # comments for .py and .toml (a line that is only a comment is
    dropped; code with a trailing comment counts)."""
    text = open(os.path.join(common.ROOT, path)).read()
    if path.endswith((".stan", ".rs", ".mint")):
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        mark = "//"
    else:
        mark = "#"
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith(mark))


def flat(ds):
    return np.concatenate([np.atleast_1d(ds["pop"].values), ds["beta"].values.ravel(),
                           ds["terminal"].values.ravel()])


def names(G):
    return ["pop"] + [f"beta[{g}]" for g in range(G)] + [f"terminal[{g}]" for g in range(G)]


def diagnose(npz):
    d = np.load(npz)
    pop, beta, terminal = d["pop"], d["beta"], d["terminal"]
    C, N = pop.shape
    G = beta.shape[2]
    ds = xr.Dataset({"pop": (("chain", "draw"), pop), "beta": (("chain", "draw", "group"), beta),
                     "terminal": (("chain", "draw", "group"), terminal)},
                    coords={"chain": np.arange(C), "draw": np.arange(N), "group": np.arange(G)})
    out = {"chains": C, "draws": N, "G": G,
           "rhat": flat(az.rhat(ds, method="rank")), "bulk": flat(az.ess(ds, method="bulk")),
           "tail": flat(az.ess(ds, method="tail")), "mcse": flat(az.mcse(ds, method="mean")),
           "mean": flat(ds.mean(dim=("chain", "draw"))), "sd": flat(ds.std(dim=("chain", "draw"), ddof=1))}
    return out


def worst_max(x):
    return float("nan") if np.isnan(x).any() else float(np.max(x))


def worst_min(x):
    return float("nan") if np.isnan(x).any() else float(np.min(x))


def clean(o):
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return clean(o.item())
    return o


def impl_of(rec):
    return rec.get("extra", {}).get("implementation") or re.sub(r"_s\d+$", "", rec["name"]).replace("martin_default", "martin")


def main():
    diag_path = os.path.join(R, "diagnostics.json")
    cache = json.load(open(diag_path)) if os.path.exists(diag_path) else {}
    runs, arrays = [], {}
    for p in sorted(glob.glob(os.path.join(R, "runs", "*.json"))):
        rec = json.load(open(p))
        rec["impl"] = impl_of(rec)
        npz = os.path.join(common.DRAWS_DIR, rec["name"] + ".npz")
        cached = [k for k in cache if k.split(":")[0] == rec["name"]]
        if rec["exit_status"] == 0 and (os.path.exists(npz) or cached):
            # the draws are not committed: without them, the cached diagnostics of the latest draws are used
            key = f"{rec['name']}:{os.path.getmtime(npz)}" if os.path.exists(npz) else sorted(cached)[-1]
            if key in cache:
                dg = {k: np.array(v, dtype=float) if isinstance(v, list) else v for k, v in cache[key].items()}
            else:
                dg = diagnose(npz)
                cache[key] = clean({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in dg.items()})
            arrays[rec["name"]] = dg
            nm = names(dg["G"])
            rh = np.where(np.isnan(dg["rhat"]), np.inf, dg["rhat"])
            bk = np.where(np.isnan(dg["bulk"]), -np.inf, dg["bulk"])
            mx, mb, mt = worst_max(dg["rhat"]), worst_min(dg["bulk"]), worst_min(dg["tail"])
            rec["diag"] = {"max_rhat": mx, "max_rhat_at": nm[int(np.argmax(rh))], "min_bulk_ess": mb,
                           "min_bulk_ess_at": nm[int(np.argmin(bk))], "min_tail_ess": mt,
                           "share_rhat_gt_1.01": float(np.mean(~(dg["rhat"] <= 1.01))),
                           "mixed": bool(mx <= MIX_RHAT and mb >= MIX_ESS and mt >= MIX_ESS),
                           "pop_mean": float(dg["mean"][0]), "pop_sd": float(dg["sd"][0]),
                           "chains": dg["chains"], "draws_per_chain": dg["draws"]}
        else:
            rec["diag"] = None
        runs.append(rec)
    json.dump(cache, open(diag_path, "w"))

    # reference
    def score(r):
        return (r["diag"]["mixed"], r["diag"]["min_bulk_ess"] if r["diag"]["mixed"] else
                -max(r["diag"]["max_rhat"], 0) if np.isfinite(r["diag"]["max_rhat"]) else -np.inf)

    ok = [r for r in runs if r["diag"]]
    stan_mixed = [r for r in ok if r["impl"].startswith("cmdstan") and r["diag"]["mixed"]]
    if stan_mixed:
        stan_mixed.sort(key=lambda r: (r["impl"] != "cmdstan_plain", -r["diag"]["min_bulk_ess"]))
        ref = stan_mixed[0]
        ref_why = "CmdStan run that mixed (plain dynpois.stan preferred, then highest lowest bulk ESS)"
    elif ok:
        ref = max(ok, key=lambda r: (r["diag"]["mixed"], r["diag"]["min_bulk_ess"] if r["diag"]["mixed"]
                                     else -r["diag"]["max_rhat"]))
        ref_why = "no CmdStan run mixed: the best-mixing run"
    else:
        ref, ref_why = None, "no runs"
    for r in runs:
        r["agreement"] = None
        if r["diag"] and ref is not None and r["name"] != ref["name"]:
            a, b = arrays[r["name"]], arrays[ref["name"]]
            diff = np.abs(a["mean"] - b["mean"])
            sdz = diff / np.sqrt((a["sd"] ** 2 + b["sd"] ** 2) / 2)
            mcz = diff / np.sqrt(a["mcse"] ** 2 + b["mcse"] ** 2)
            both = r["diag"]["mixed"] and ref["diag"]["mixed"]
            nm = names(a["G"])
            r["agreement"] = {"reference": ref["name"], "max_sd_z": worst_max(sdz),
                              "max_sd_z_at": nm[int(np.nanargmax(sdz))], "median_sd_z": float(np.nanmedian(sdz)),
                              "max_mcse_z": worst_max(mcz) if both else None,
                              "share_mcse_z_gt_3": float(np.mean(~(mcz <= 3))) if both else None}

    # pairs that must give the same posterior (variants of one framework)
    pairs = []
    by = {}
    for r in ok:
        by.setdefault(re.sub(r"_s\d+$", "", r["name"]), []).append(r)
    for a_key, b_key in [("cmdstan_plain", "cmdstan_reduce_sum"), ("pymc_numba", "pymc_jax"),
                         ("numpyro_parallel", "numpyro_vectorized"), ("nutpie_stan_diag", "nutpie_stan_low_rank"),
                         ("rust_nuts_diag", "rust_nuts_lowrank"), ("martin_default", "martin_optin"),
                         ("martin_default", "rust_under_martin"), ("martin_default", "cmdstan_plain")]:
        for ra in by.get(a_key, []):
            for rb in by.get(b_key, []):
                if ra["extra"].get("seed") != rb["extra"].get("seed"):
                    continue
                A, B = arrays[ra["name"]], arrays[rb["name"]]
                diff = np.abs(A["mean"] - B["mean"])
                sdz = diff / np.sqrt((A["sd"] ** 2 + B["sd"] ** 2) / 2)
                mcz = diff / np.sqrt(A["mcse"] ** 2 + B["mcse"] ** 2)
                both = ra["diag"]["mixed"] and rb["diag"]["mixed"]
                pairs.append({"a": ra["name"], "b": rb["name"], "both_mixed": both, "max_sd_z": worst_max(sdz),
                              "max_mcse_z": worst_max(mcz) if both else None,
                              "share_mcse_z_gt_3": float(np.mean(~(mcz <= 3))) if both else None})

    rows = []
    for r in runs:
        ph, ex = r.get("phases", {}), r.get("extra", {})
        samp = ph.get("sampling")
        comp = ph.get("compile")
        d = r["diag"]
        row = {
            "run": r["name"], "impl": r["impl"], "label": IMPLS.get(r["impl"], (r["impl"],))[0],
            "seed": ex.get("seed") or int(re.search(r"_s(\d+)$", r["name"]).group(1)), "status": "ok" if r["exit_status"] == 0 and not r["timed_out"] else
            ("timed out" if r["timed_out"] else f"failed (exit {r['exit_status']})"),
            "loc": sum(loc(f) for f in IMPLS[r["impl"]][1]) if r["impl"] in IMPLS else None,
            "loc_files": IMPLS.get(r["impl"], (None, []))[1],
            "compile_seconds": comp, "build_model_seconds": ph.get("build_model"),
            "sampling_seconds": samp, "total_wall_seconds": r["total_wall_seconds"],
            "peak_memory_bytes": r["peak_memory_bytes"], "peak_rss_tree_sum_bytes": r["peak_rss_tree_sum_bytes"],
            "peak_pss_tree_sum_bytes": r["peak_pss_tree_sum_bytes"],
            "maxrss_largest_process_bytes": r["maxrss_largest_process_bytes"],
            "cpu_seconds": r["cpu_seconds"], "avg_cores_busy": r["cpu_seconds"] / r["total_wall_seconds"],
            "peak_threads": r["peak_threads"], "loadavg_before": r["loadavg_before"],
            "gradients": ex.get("gradients"), "gradients_note": ex.get("gradients_note"),
            "divergences": ex.get("divergences"), "leapfrog_per_draw": ex.get("leapfrog_per_draw"),
            "step_size": ex.get("step_size"), "threads_per_chain": ex.get("threads_per_chain"),
            "settings": ex.get("settings"), "diag": d, "agreement": r["agreement"],
            "recovered": r.get("recovered"), "busy_cpus_outside_set": r.get("busy_cpus_outside_set"),
            "timeout_seconds": r.get("timeout_seconds"),
        }
        over = r["timed_out"] or r["total_wall_seconds"] > LIMIT_SECONDS
        if r["exit_status"] != 0 and not r["timed_out"]:
            row["verdict"] = "failed"
        elif over:
            row["verdict"] = "did not finish within 20 minutes"
        elif d and d["mixed"]:
            row["verdict"] = "ranked"
        else:
            row["verdict"] = "did not mix"
        row["timing_corrected"] = r.get("timing_corrected", False)
        if row["verdict"] == "ranked" and samp:
            row["ess_per_sampling_second"] = d["min_bulk_ess"] / samp
            row["ess_per_total_second"] = d["min_bulk_ess"] / r["total_wall_seconds"]
        else:
            row["ess_per_sampling_second"] = row["ess_per_total_second"] = None
        # only where the count covers warmup too (CmdStan and NumPyro count sampling-phase steps only)
        if d and samp and ex.get("gradients") and ex.get("warmup_gradients") is not None:
            row["us_per_gradient_per_chain"] = 1e6 * samp * common.CHAINS / ex["gradients"]
        rows.append(row)
    rows.sort(key=lambda x: (ORDER.index(x["impl"]) if x["impl"] in ORDER else 99, x["seed"] or 0))
    configs = []
    for impl in ORDER:
        rs = [x for x in rows if x["impl"] == impl]
        if not rs:
            continue

        done = [x for x in rs if x["status"] == "ok"]

        def med(key, sub=None):
            v = [x[key] if sub is None else (x[key] or {}).get(sub) for x in done]
            v = [float(z) for z in v if z is not None and np.isfinite(z)]
            return float(np.median(v)) if v else None

        all_ess = [x["diag"]["min_bulk_ess"] / x["sampling_seconds"] for x in done if x["diag"] and x["sampling_seconds"]]
        configs.append({
            "impl": impl, "label": IMPLS[impl][0], "seeds": [x["seed"] for x in rs],
            "verdicts": [x["verdict"] for x in rs], "n_ranked": sum(x["verdict"] == "ranked" for x in rs),
            "n_completed": len(done),
            "median_sampling_seconds": med("sampling_seconds"), "median_total_wall_seconds": med("total_wall_seconds"),
            "median_max_rhat": med("diag", "max_rhat"), "median_min_bulk_ess": med("diag", "min_bulk_ess"),
            "median_min_tail_ess": med("diag", "min_tail_ess"),
            "median_ess_per_sampling_second_all_runs": float(np.median(all_ess)) if all_ess else None,
            "median_peak_memory_bytes": med("peak_memory_bytes"), "median_cpu_seconds": med("cpu_seconds"),
        })
    out = {"reference": ref["name"] if ref else None, "reference_rule": ref_why,
           "mixed_criterion": f"max R-hat <= {MIX_RHAT} and min bulk and tail ESS >= {MIX_ESS} over pop, "
                              f"beta[250], terminal[250] (501 quantities), ArviZ rank-normalised",
           "time_limit_seconds": LIMIT_SECONDS, "rows": rows, "configurations": configs, "pairs": pairs}
    json.dump(clean(out), open(os.path.join(R, "results.json"), "w"), indent=1)
    write_md(out)


def fmt(x, spec, none="n/a"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return none
    return format(x, spec)


def write_md(out):
    L = []
    rows = out["rows"]
    L.append("# Shootout results: hierarchical dynamic Poisson panel, G = 250, T = 150 (37,901 parameters)\n")
    L.append("Generated by `bench/shootout/analyze.py` from `results/runs/*.json`. Rules, settings and "
             "caveats: [README.md](README.md). 4 chains, 1000 warmup + 1000 draws (Martin's opt-in fast warmup "
             "runs 200 of the 1000 warmup iterations; rustmc keeps every 8th of 1000 sweeps), every run pinned to "
             "the same 12 physical cores, one run at a time.\n")
    L.append("**Verdicts.** A run is ranked only if it finished within 20 minutes of total wall time "
             "(compilation included) and mixed. Mixed = " + out["mixed_criterion"] + ". A run over 20 minutes "
             "is \"did not finish within 20 minutes\" whatever its diagnostics; its real total time and "
             "whether it mixed are still shown. ESS/s = lowest bulk ESS / sampling seconds (and / total "
             "seconds), for ranked runs only.\n")
    L.append("**The 20-minute limit was chosen after the first results had been seen** (the seed-1 pass ran "
             "under a 2-hour limit; seeds 2 and 3 ran with the 20-minute limit enforced). \"Within 3 min?\" "
             "(total wall time at most 180 s) is informational only; the 20-minute rule is the verdict.\n")
    L.append(f"**Reference posterior** for the agreement column: `{out['reference']}`, the threaded Stan run, "
             "which mixed but took 40 minutes, so it serves as a reference posterior only and is not ranked. "
             "Agreement = largest difference of posterior means from it over the 501 quantities, in posterior "
             "sd (and in Monte Carlo standard errors when both runs mixed).\n")
    L.append("## Every run\n")
    L.append("| implementation | seed | lines of code | compile s | sampling s | total s | within 3 min? | peak memory GiB "
             "| CPU s | mixed? (max R-hat, min bulk / tail ESS) | verdict | ESS/s sampling (total) | agreement, sd (MCSE) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    notes = []
    for r in rows:
        d, ag = r["diag"], r["agreement"]
        mixed = (f"{'yes' if d['mixed'] else 'no'} ({fmt(d['max_rhat'], '.3f')}, "
                 f"{fmt(d['min_bulk_ess'], '.0f')} / {fmt(d['min_tail_ess'], '.0f')})") if d else "n/a (no draws)"
        ess = (f"{r['ess_per_sampling_second']:.2f} ({r['ess_per_total_second']:.2f})"
               if r["ess_per_sampling_second"] is not None else "not ranked")
        if not d:
            agr = "n/a"
        elif ag is None:
            agr = "reference"
        else:
            agr = f"{fmt(ag['max_sd_z'], '.3f')}" + (f" ({fmt(ag['max_mcse_z'], '.2f')})"
                                                    if ag["max_mcse_z"] is not None else "")
        comp = fmt(r["compile_seconds"], ".1f")
        if r["build_model_seconds"]:
            comp += f" (+{r['build_model_seconds']:.1f} model build)"
        lab = r["label"]
        if r["recovered"]:
            notes.append(f"`{r['run']}`: its post-processing crashed after sampling had finished (a CSV column-name "
                         "bug); the CSV files were post-processed afterwards by `recover_cmdstan.py`. Compile and "
                         "sampling seconds come from cmdstanpy's log (1 s resolution); the post-processing time was "
                         "added to the measured total.")
            lab += f" [note {len(notes)}]"
        if r["timing_corrected"]:
            notes.append(f"`{r['run']}`: JAX ran asynchronously and the script stopped its sampling clock too early; "
                         "the sampling time shown is an upper bound reconstructed from the measured phases (it "
                         "includes a second or two of post-processing). Total wall, CPU and memory are as measured.")
            lab += f" [note {len(notes)}]"
        if r["status"] == "timed out":
            lim = "2-hour limit in force when it ran" if r["timeout_seconds"] > 1200 else "20-minute limit"
            why = (" NumPyro reports no progress without its progress bar, so how far it got is unknown."
                   if r["impl"].startswith("numpyro") else "")
            notes.append(f"`{r['run']}`: stopped by the {lim}; no draws.{why}")
            lab += f" [note {len(notes)}]"
        verdict = r["verdict"] if r["verdict"] == "ranked" else "**" + r["verdict"] + "**"
        L.append(f"| {lab} | {r['seed']} | {r['loc']} | {comp} | {fmt(r['sampling_seconds'], '.1f')} | "
                 f"{r['total_wall_seconds']:.1f} | {'yes' if r['total_wall_seconds'] <= 180 else 'no'} | "
                 f"{r['peak_memory_bytes'] / 2**30:.2f} | {r['cpu_seconds']:.0f} | "
                 f"{mixed} | {verdict} | {ess} | {agr} |")
    if notes:
        L.append("")
        for i, n in enumerate(notes, 1):
            L.append(f"{i}. {n}")
    L.append("\n## Per configuration, over seeds\n")
    L.append("Medians over the seeds run. Seeds 2 and 3 were run only for configurations that finished within "
             "20 minutes with seed 1, except the Rust program with nuts-rs's low-rank adaptation. PyMC + nutpie with JAX was to be skipped at seed 3 (it had not mixed at "
             "seeds 1 and 2); its seed-3 run had already finished when that decision was made and is included. "
             "Most seed-2 and seed-3 runs ran while other work kept 1.5 to 4.7 logical CPUs busy, so their times (and the "
             "medians) are inflated against the quiet seed-1 pass; see README.md. \"Median ESS/s (all completed runs)\" "
             "includes runs that did not mix and only describes them; it is not a ranking. Medians are over the "
             "completed runs only: a run stopped by a time limit has no draws, so it contributes its verdict but "
             "no times, diagnostics or memory.\n")
    L.append("| implementation | seeds | verdict per seed | completed runs | median sampling s | median total s "
             "| median max R-hat | median min bulk / tail ESS | median ESS/s (all completed runs) | median peak memory GiB "
             "| median CPU s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in out["configurations"]:
        mem = c["median_peak_memory_bytes"]
        L.append(f"| {c['label']} | {', '.join(str(x) for x in c['seeds'])} | {'; '.join(c['verdicts'])} | "
                 f"{c['n_completed']} | {fmt(c['median_sampling_seconds'], '.1f')} | {fmt(c['median_total_wall_seconds'], '.1f')} | "
                 f"{fmt(c['median_max_rhat'], '.3f')} | {fmt(c['median_min_bulk_ess'], '.0f')} / "
                 f"{fmt(c['median_min_tail_ess'], '.0f')} | {fmt(c['median_ess_per_sampling_second_all_runs'], '.2f')} | "
                 f"{fmt(mem / 2**30 if mem else None, '.2f')} | {fmt(c['median_cpu_seconds'], '.0f')} |")
    L.append("\n## Ranking of the ranked runs (lowest bulk ESS per second of sampling)\n")
    ranked = sorted([r for r in rows if r["ess_per_sampling_second"] is not None],
                    key=lambda r: -r["ess_per_sampling_second"])
    if ranked:
        L.append("| rank | run | ESS/s sampling | ESS/s total |")
        L.append("|---|---|---|---|")
        for i, r in enumerate(ranked, 1):
            L.append(f"| {i} | {r['run']} | {r['ess_per_sampling_second']:.2f} | {r['ess_per_total_second']:.2f} |")
    L.append("\nEvery other run did not mix, failed, or did not finish within 20 minutes, and is not ranked. "
             "Differences of less than about 2x between ranked runs are within the variation between seeds "
             "seen here and are not findings.\n")
    L.append("\n## Variants of one framework: which is faster\n")
    L.append("Faster = higher ESS per second of sampling when both runs are ranked; otherwise only sampling "
             "time can be compared, which ignores how well each run mixed.\n")
    L.append("| seed | variant a | variant b | faster | basis |")
    L.append("|---|---|---|---|---|")
    by = {r["run"]: r for r in rows}
    for a_key, b_key in [("cmdstan_plain", "cmdstan_reduce_sum"), ("pymc_numba", "pymc_jax"),
                         ("numpyro_parallel", "numpyro_vectorized"), ("nutpie_stan_diag", "nutpie_stan_low_rank"),
                         ("rust_nuts_diag", "rust_nuts_lowrank"), ("martin_default", "martin_optin")]:
        for seed in sorted({r["seed"] for r in rows if r["seed"]}):
            ra, rb = by.get(f"{a_key}_s{seed}"), by.get(f"{b_key}_s{seed}")
            if not ra or not rb:
                continue
            if ra["ess_per_sampling_second"] and rb["ess_per_sampling_second"]:
                win = ra if ra["ess_per_sampling_second"] > rb["ess_per_sampling_second"] else rb
                basis = (f"ESS/s {ra['ess_per_sampling_second']:.2f} against {rb['ess_per_sampling_second']:.2f}")
            elif ra["status"] != "ok" or rb["status"] != "ok":
                win = ra if ra["status"] == "ok" else rb
                basis = (f"the other {'timed out' if 'timed' in (rb if win is ra else ra)['status'] else 'failed'}; "
                         f"total {ra['total_wall_seconds']:.0f} s against {rb['total_wall_seconds']:.0f} s")
            else:
                win = ra if ra["sampling_seconds"] < rb["sampling_seconds"] else rb
                basis = (f"sampling time only ({ra['sampling_seconds']:.0f} s against {rb['sampling_seconds']:.0f} s); "
                         f"verdicts: {ra['verdict']} / {rb['verdict']}")
            L.append(f"| {seed} | {ra['run']} | {rb['run']} | {win['run']} | {basis} |")
    L.append("\n## Sampler work and hardware use\n")
    L.append("| run | gradients | µs per gradient per chain (sampling s x 4 / gradients) | leapfrog steps per draw "
             "| divergences | threads per chain | peak threads | average cores busy (CPU s / wall s) | "
             "load before | other work, busy CPUs outside the 12 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lf = r["leapfrog_per_draw"]
        lf = ", ".join(f"{v:.0f}" for v in lf) if isinstance(lf, list) else fmt(lf, ".0f")
        g = r["gradients"]
        gs = "n/a" if g is None else (f"{g / 1e6:.2f} M" + ("" if r["gradients_note"] and
                                                            r["gradients_note"].startswith(("all", "leapfrog", "sum of n_steps over warmup"))
                                                            else " (sampling only)"))
        L.append(f"| {r['run']} | {gs} | {fmt(r.get('us_per_gradient_per_chain'), '.0f')} | {lf} | "
                 f"{fmt(r['divergences'], 'd')} | {fmt(r['threads_per_chain'], 'd')} | {r['peak_threads']} | "
                 f"{r['avg_cores_busy']:.1f} | {r['loadavg_before'][0]:.2f} | {fmt(r['busy_cpus_outside_set'], '.2f')} |")
    L.append("\n## Lines of code\n")
    L.append("| implementation | lines | files | note |")
    L.append("|---|---|---|---|")
    seen = set()
    for r in rows:
        if r["impl"] in seen or r["impl"] not in IMPLS:
            continue
        seen.add(r["impl"])
        files = IMPLS[r["impl"]][1]
        L.append(f"| {IMPLS[r['impl']][0]} | {r['loc']} | "
                 + ", ".join(f"`{f}` ({loc(f)})" for f in files) + f" | {IMPLS[r['impl']][2]} |")
    L.append("\n## Variants of one framework, same seed (do they give the same posterior?)\n")
    L.append("| run a | run b | both mixed | max difference of means, sd | in MCSE | share > 3 MCSE |")
    L.append("|---|---|---|---|---|---|")
    for p in out["pairs"]:
        L.append(f"| {p['a']} | {p['b']} | {'yes' if p['both_mixed'] else 'no'} | {fmt(p['max_sd_z'], '.3f')} | "
                 f"{fmt(p['max_mcse_z'], '.2f')} | {fmt(p['share_mcse_z_gt_3'], '.3f')} |")
    open(os.path.join(common.HERE, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
