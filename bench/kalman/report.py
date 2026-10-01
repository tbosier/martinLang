#!/usr/bin/env python3
"""Summarises bench/kalman/results.json (written by bench.py) as Markdown.

usage: report.py [results.json]

Per size and variant, over the seeds: the dimension NUTS sees, gradients
per kept draw (warmup included), leapfrog steps per kept draw, sampling
time (warmup included; ranges, since other jobs share the machine), the
lowest ESS over the remaining parameters (pop, beta, sigma_w, sigma_y) and
its rate per second and per 1000 gradients, and sigma_w's (the slowest
parameter of full NUTS) separately. Ratios are collapsed over full, per
seed, as the range over seeds.
"""
import json
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
R = json.load(open(path))


def rng(vals, fmt="{:.3g}"):
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return lo if lo == hi else f"{lo} to {hi}"


def med(vals):
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


for size in R["sizes"]:
    G, T = size["G"], size["T"]
    runs = size["runs"]
    seeds = sorted({r["seed"] for r in runs})
    print(f"\n### G = {G}, T = {T} ({G * T:,} latent scalars), seeds {', '.join(map(str, seeds))}\n")
    print("| | NUTS dimension | gradients per draw | leapfrog per draw | divergences | sampling time (s) "
          "| lowest ESS, remaining params | per s | per 1000 gradients | sigma_w ESS per s | highest R-hat of pop, sigma_w, sigma_y | threads per chain | load average |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    by = {}
    for v in ("collapsed", "full", "full_nc"):
        rs = [r for r in runs if r["variant"] == v]
        if not rs:
            continue
        by[v] = {r["seed"]: r for r in rs}
        gpd = [r["gradients"] / r["draws"] for r in rs]
        lpd = [sum(r["leapfrog_per_draw"]) / len(r["leapfrog_per_draw"]) for r in rs]
        ess = [r["remaining_min_ess"] for r in rs]
        eps = [r["remaining_min_ess"] / r["sampling_s"] for r in rs]
        epg = [1000 * r["remaining_min_ess"] / r["gradients"] for r in rs]
        sw = [r["sigma_w"]["ess"] / r["sampling_s"] for r in rs]
        load = [x for r in rs for x in (r["load_before"][0], r["load_after"][0])]
        div = [r["divergences"] for r in rs]
        name = {"collapsed": "collapsed (Kalman)", "full": "full NUTS, centred", "full_nc": "full NUTS, non-centred"}[v]
        print(f"| {name} | {rs[0]['nuts_dim']:,} | {rng(gpd, '{:.0f}')} | {rng(lpd, '{:.1f}')} | {rng(div, '{}')} "
              f"| {rng([r['sampling_s'] for r in rs])} | {rng(ess, '{:.0f}')} | {rng(eps)} | {rng(epg)} | {rng(sw)} "
              f"| {rng([r['remaining_rhat_max'] for r in rs], '{:.3f}')} | {rng([r['threads_per_chain'] for r in rs], '{}')} "
              f"| {rng(load, '{:.1f}')} |")
    if "gradient_ns" in size:
        gt = size["gradient_ns"]
        print("\none gradient, one thread (MINT_BENCH_GRAD, 7 alternated repetitions): "
              + "; ".join(f"{v} {min(t) / 1000:.1f} us (median {med(t) / 1000:.1f})" for v, t in gt.items())
              + f"; load average after {size['gradient_load'][0]:.1f}")
    if "collapsed" in by:
        for f in ("full", "full_nc"):
            if f not in by:
                continue
            common = [s for s in seeds if s in by["collapsed"] and s in by[f]]
            c, o = by["collapsed"], by[f]
            r1 = [(c[s]["remaining_min_ess"] / c[s]["sampling_s"]) / (o[s]["remaining_min_ess"] / o[s]["sampling_s"]) for s in common]
            r2 = [(c[s]["remaining_min_ess"] / c[s]["gradients"]) / (o[s]["remaining_min_ess"] / o[s]["gradients"]) for s in common]
            r3 = [(c[s]["sigma_w"]["ess"] / c[s]["sampling_s"]) / (o[s]["sigma_w"]["ess"] / o[s]["sampling_s"]) for s in common]
            print(f"\ncollapsed / {f}, per seed: lowest remaining ESS per second {rng(r1)} (median {med(r1):.3g}); "
                  f"per gradient {rng(r2)}; sigma_w ESS per second {rng(r3)}")
