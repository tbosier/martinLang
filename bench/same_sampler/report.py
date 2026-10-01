#!/usr/bin/env python3
"""Renders bench/same_sampler/results/*.json as results/results.md.

usage: python bench/same_sampler/report.py
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

R = common.RESULTS
NAMES = {
    "mint": "Mint",
    "rust_max": "Rust, max effort (dynpois_max.rs / logistic_bayes_max.rs)",
    "rust_par": "Rust, max effort v2 (dynpois_par.rs: table exp, threaded)",
    "rust_par[fused]": "dynpois_par.rs, exp fused into the forward pass",
    "rust_par[back]": "dynpois_par.rs, exp in the backward pass",
    "rust_par[glibc]": "dynpois_par.rs, glibc exp (= dynpois_max.rs + threads)",
    "rust": "Rust, straightforward (eight_schools.rs)",
    "stan": "Stan 2.40 via BridgeStan (stanc --O1)",
    "stan_glm": "Stan, bernoulli_logit_glm written by hand",
}


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def us(ns):
    v = ns / 1e3
    return f"{v:.3f}" if v < 1 else f"{v:.2f}" if v < 100 else f"{v:.0f}"


def rng(xs, f="{:.2f}"):
    return f"{f.format(min(xs))} to {f.format(max(xs))}"


def grad_section(g, out):
    out.append("## Gradient alone (MINT_BENCH_GRAD, the runtime's benchmark point)\n")
    loads = [x["loadavg"][0] for x in g["loadavg_per_round"]]
    out.append(f"{g['reps']} interleaved rounds of about {g['seconds_per_run']} s per run, pinned to the least busy "
               f"cores at the start of each round. 1-minute load average during the rounds: {min(loads):.1f} to "
               f"{max(loads):.1f} (24 hardware threads; other agents were running). A run is *clean* when the "
               "other hardware thread of its core was at most 10% busy and the run got at least 95% of its "
               "CPU time; the clean median is the figure to read, the all-run median and range show the noise.\n")
    out.append("| problem | implementation | kernel threads | clean median, µs | clean runs | all-run median, µs | range, µs | vs Mint (clean) |")
    out.append("|---|---|---|---|---|---|---|---|")
    base = {}
    for row in g["rows"]:
        if row["impl"] == "mint":
            base[(row["problem"], row["kernel_threads"])] = row["ns_median_clean"] or row["ns_median"]
    for row in g["rows"]:
        c = row["ns_median_clean"]
        b = base.get((row["problem"], row["kernel_threads"]))
        ratio = f"{(c or row['ns_median']) / b:.2f}x" if b else ""
        out.append(f"| {row['problem']} | {NAMES.get(row['impl'], row['impl'])} | {row['kernel_threads']} | "
                   f"{us(c) if c else 'none'} | {row['clean_runs']}/{len(row['runs'])} | {us(row['ns_median'])} | "
                   f"{us(row['ns_min'])} to {us(row['ns_max'])} | {ratio} |")
    out.append("")


def busy(r):
    """Hardware threads busy with other work, in the second before the run started."""
    return sum(r["before"]["cpu_busy_percent"].values()) / 100


def whole_section(w, title, out, note=""):
    out.append(f"## {title}\n")
    if note:
        out.append(note + "\n")
    s = w["settings"]
    runs = w["runs"]
    seeds = sorted({r["seed"] for r in runs})
    loads = [r["before"]["loadavg"][0] for r in runs]
    out.append(f"{s['chains']} chains in parallel, {s['warmup']} warmup + {s['draws']} draws, seeds "
               f"{', '.join(map(str, seeds))}, every implementation under the same runtime NUTS with the same "
               f"settings and seed; runs interleaved. Machine load before each run: 1-minute load average "
               f"{min(loads):.1f} to {max(loads):.1f} (it includes this harness's own previous run), and "
               f"{min(map(busy, runs)):.1f} to {max(map(busy, runs)):.1f} of the 24 hardware threads busy in the "
               "second before the run started (other work only). "
               "Medians over seeds, with the range. ESS and R-hat are the runtime's (rank-normalised bulk ESS, "
               "split R-hat), lowest and highest over all parameters.\n")
    out.append("| problem | implementation | runs | sampling s, median (range) | gradients, median | µs per gradient per chain, median (range) | lowest ESS, range | ESS per 1000 gradients, median | highest R-hat | divergences, total |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    keys = []
    for r in runs:
        if (r["problem"], r["impl"]) not in keys:
            keys.append((r["problem"], r["impl"]))
    order = {i: k for k, i in enumerate(["mint", "rust_max", "rust_par", "rust", "stan", "stan_glm"])}
    keys.sort(key=lambda k: (k[0], order.get(k[1], 99)))
    for p, impl in keys:
        rs = [r for r in runs if r["problem"] == p and r["impl"] == impl]
        sec = [r["sampling_seconds"] for r in rs]
        per = [r["us_per_gradient_per_chain"] for r in rs]
        ess = [r["min_ess"] for r in rs]
        e1k = [r["min_ess_per_1k_gradients"] for r in rs]
        out.append(f"| {p} | {NAMES.get(impl, impl)} | {len(rs)} | {statistics.median(sec):.2f} ({rng(sec)}) | "
                   f"{statistics.median([r['gradients'] for r in rs]):,.0f} | {statistics.median(per):.2f} ({rng(per)}) | "
                   f"{rng(ess, '{:.0f}')} | {statistics.median(e1k):.3g} | {max(r['max_rhat'] for r in rs):.3f} | "
                   f"{sum(r['divergences'] for r in rs)} |")
    out.append("")
    out.append("Per run (seed: sampling seconds, gradients, lowest ESS, 1-minute load average before, "
               "hardware threads busy with other work before):\n")
    for p, impl in keys:
        rs = sorted([r for r in runs if r["problem"] == p and r["impl"] == impl], key=lambda r: r["seed"])
        cells = "; ".join(f"{r['seed']}: {r['sampling_seconds']:.2f} s, {r['gradients']:,}, {r['min_ess']:.0f}, "
                          f"{r['before']['loadavg'][0]:.1f}, {busy(r):.1f}" for r in rs)
        out.append(f"- {p}, {impl}: {cells}")
    out.append("")


def nutpie_section(n, out, title):
    out.append(f"## {title}\n")
    loads = [r["before"]["loadavg"][0] for r in n["runs"]]
    st = n["settings"]
    how = (f"{st['chains']} chain{'s in threads' if st['chains'] > 1 else ''}"
           + (", pinned to the least busy core before each run" if st.get("pinned") else ""))
    out.append(f"nutpie {n['versions']['nutpie']}, BridgeStan {n['versions']['bridgestan']}; both run the same "
               f"compiled Stan model, {how}, 1000 warmup + 1000 draws. Bulk and tail ESS and R-hat "
               "come from the same ArviZ code over every parameter's constrained draws. nutpie's adaptation "
               "differs from Mint's (Stan's), so this compares samplers, not languages. 1-minute load average "
               f"before the runs: {min(loads):.1f} to {max(loads):.1f}.\n")
    out.append("| problem | sampler | seed | wall s | gradients | µs per gradient per chain | minus the gradient alone | lowest bulk ESS | lowest tail ESS | highest R-hat | bulk ESS per 1000 gradients | divergences |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(n["runs"], key=lambda r: (r["problem"], r["sampler"], r["seed"])):
        oh = r.get("overhead_us_per_gradient")
        out.append(f"| {r['problem']} | {r['sampler']} | {r['seed']} | {r['wall_seconds']:.2f} | {r['gradients']:,} | "
                   f"{r['us_per_gradient_per_chain']:.2f} | {'' if oh is None else f'{oh:.2f}'} | "
                   f"{r['min_bulk_ess']:.0f} | {r['min_tail_ess']:.0f} | {r['max_rhat']:.3f} | "
                   f"{r['min_bulk_ess_per_1k_gradients']:.3g} | {r['divergences']} |")
    out.append("")


def verify_section(v, out):
    out.append("## Verification (results/verify.json)\n")
    s = v["summary"]
    out.append(f"- At the benchmark point, through each program's own entry point: worst gradient difference from "
               f"Mint, per component relative to max(|g|, 1): {s['worst_grad_vs_mint']:.1e}. Log density minus "
               f"Mint's equals the constant that implementation drops to {s['worst_constant_error']:.1e} relative.")
    out.append(f"- Stan at 6 random points per model against numpy references: the log density differs by the "
               f"expected constant to within {s['worst_random_point_constant']:.1e} (absolute), gradients to "
               f"{s['worst_random_point_grad']:.1e}.")
    out.append(f"- Thread safety: shared model, one model per chain and a repeat gave byte-identical draws: "
               f"{s['thread_safety']}.")
    for t in v.get("rust_par_selftest", []):
        out.append(f"- dynpois_par.rs self-test, exp={t['exp']}, {t['data']} data, kernel threads "
                   f"{t['thread_counts'].replace('threads:', '')}{' (1, 2, 3, 4, 7)' if t['thread_counts'] == '1' else ''}: "
                   f"{'pass' if t['passed'] else 'FAIL'}; worst gradient against the scalar reference "
                   f"{t['worst_grad_vs_reference']:.1e}, worst finite-difference error {t['worst_fd_error']:.1e}.")
    out.append(f"- Overall: {'pass' if s['pass'] else 'FAIL'}.\n")


def main():
    out = ["# Same-sampler results\n",
           "Generated by `bench/same_sampler/report.py` from the JSON files in this directory. "
           "See `../README.md` for the rules and what each implementation does.\n"]
    v = load("verify.json")
    if v:
        verify_section(v, out)
    g = load("grad.json")
    if g:
        grad_section(g, out)
    for name, title in [("whole_small.json", "Whole runs: small dynamic Poisson, logistic regression, eight schools"),
                        ("whole_large.json", "Whole runs: large dynamic Poisson (37,901 parameters)")]:
        w = load(name)
        if w:
            whole_section(w, title, out)
    w = load("whole_small_loaded.json")
    if w:
        whole_section(w, "The same small runs on a loaded machine (an earlier pass)", out,
                      "The first pass of the small runs, while other agents kept 15 to 29 of the 24 hardware "
                      "threads busy. Same seeds, so the same gradient counts and ESS as the quiet pass above (the "
                      "draws are deterministic); only the times differ. Kept to show how much load moves wall "
                      "times: up to 3x.")
    n = load("nutpie.json")
    if n:
        nutpie_section(n, out, "Sampler check: Mint's NUTS against nutpie, same Stan gradient, 4 chains")
    n = load("nutpie_1chain.json")
    if n:
        nutpie_section(n, out, "Sampler overhead: one pinned chain, Mint's NUTS against nutpie, same Stan gradient")
    path = os.path.join(R, "results.md")
    open(path, "w").write("\n".join(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
