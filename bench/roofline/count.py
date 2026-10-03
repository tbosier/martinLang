#!/usr/bin/env python3
"""Retired floating-point operations and cycles per gradient (or per Newton
fit), from the hardware counters, for the roofline estimate in
docs/roofline.md. Gradient programs run at K and 2K evaluations and the
difference is divided by K, so start-up work cancels. Pinned to core 2.
perf counts user mode only here (events show as :u without extra rights).

usage (from the repository root, after building the programs as
bench/mint_vs_ref.py does and the Rust baselines as tests/run.sh does):
  python3 bench/roofline/count.py
Needs perf with the AMD Zen event fp_ret_sse_avx_ops.all (an FMA counts as
two operations).
"""
import os
import subprocess


def stat(cmd, env):
    e = dict(os.environ)
    e.update(env)
    r = subprocess.run(["perf", "stat", "-x,", "-e", "fp_ret_sse_avx_ops.all,cycles,instructions",
                        "taskset", "-c", "2"] + cmd, capture_output=True, text=True, env=e)
    d = {}
    for line in r.stderr.splitlines():
        p = line.split(",")
        if len(p) > 3 and p[0].replace(".", "").isdigit():
            d[p[2].split(":")[0]] = float(p[0])
    return d


cases = [
    ("time series 3,171, Martin", ["build/cmp/cur_dynpois_small"], 20000),
    ("time series 3,171, Rust", ["build/rs_dynpois_max", "bench/dynpois/data_small/y.f64"], 20000),
    ("time series 37,901, Martin", ["build/cmp/cur_dynpois_large"], 2000),
    ("time series 37,901, Rust", ["build/rs_dynpois_max", "bench/dynpois/data_large/y.f64"], 2000),
    ("logistic, Martin", ["build/cmp/cur_logistic"], 10000),
    ("logistic, Rust", ["build/rs_logistic_bayes_max"], 10000),
]
for name, cmd, k in cases:
    a, b = stat(cmd, {"MINT_BENCH_GRAD": str(k)}), stat(cmd, {"MINT_BENCH_GRAD": str(2 * k)})
    f = (b["fp_ret_sse_avx_ops.all"] - a["fp_ret_sse_avx_ops.all"]) / k
    c = (b["cycles"] - a["cycles"]) / k
    i = (b["instructions"] - a["instructions"]) / k
    print(f"{name:28} flops/gradient {f:10.0f}  cycles/gradient {c:9.0f}  flops/cycle {f / c:5.2f}  IPC {i / c:4.2f}")
for name, cmd in [("Newton, Martin", ["build/cmp/cur_newton"]), ("Newton, Rust", ["build/rs_logistic_newton_max"])]:
    a = stat(cmd, {})
    print(f"{name:28} flops {a['fp_ret_sse_avx_ops.all']:.3g}  cycles {a['cycles']:.3g}  "
          f"flops/cycle {a['fp_ret_sse_avx_ops.all'] / a['cycles']:.2f} (whole program, including reading the data)")
