#!/usr/bin/env python3
"""Renders bench/results.json as the markdown tables used in docs/benchmark.md.

Ratios are medians divided by medians. With 7 runs per cell, a ratio is
reported only as "faster" or "slower" when the min-max ranges do not overlap;
otherwise the table says the ordering is not established.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = json.load(open(os.path.join(ROOT, "bench", "results.json")))
res = r["results"]


def fmt(v, unit):
    if unit == "ns":
        return f"{v / 1000:.1f} µs" if v >= 1000 else f"{v:.0f} ns"
    if v < 0.1:
        return f"{v * 1000:.2f} ms"
    return f"{v:.3f} s" if v < 10 else f"{v:.1f} s"


def table(group, unit, ref="mint", note=None):
    items = res[group]
    base = items[ref]
    out = [f"| implementation | median | range (min to max) | relative to {ref} |", "|---|---|---|---|"]
    for k, v in items.items():
        if k == ref:
            rel = "1.00x"
        else:
            ratio = v["median"] / base["median"]
            overlap = not (v["min"] > base["max"] or v["max"] < base["min"])
            rel = f"{ratio:.2f}x" + (" (ranges overlap: ordering not established)" if overlap else "")
        extra = ""
        if "gradients" in v:
            extra = f"; {v['gradients'][0]:,} gradients"
        out.append(f"| {k} | {fmt(v['median'], unit)} | {fmt(v['min'], unit)} to {fmt(v['max'], unit)}{extra} | {rel} |")
    if note:
        out.append("")
        out.append(note)
    return "\n".join(out)


def main():
    m = r["meta"]
    print(f"Machine: {m['cpu']}, Linux {m['kernel']}. {m['rustc']} (LLVM 20); {m['clang']}.")
    print(f"Each cell is {m['reps']} runs, interleaved, pinned to core {m['pinned_core']}; "
          f"load average at start {', '.join(m['loadavg_at_start'])}.\n")
    sections = [
        ("logistic_grad_ns", "ns", "Logistic regression gradient, n=5000, p=20 (time per gradient; lower is better)"),
        ("linear_grad_ns", "ns", "Linear regression gradient, n=50000, p=20"),
        ("newton_fit_s", "s", "Newton's method, n=200000, p=50, 10 iterations (fit time, excluding file reading)"),
        ("logistic_sampling_s", "s", "Logistic regression, NUTS, 1 chain, 1000 warmup + 1000 draws (sampler wall time plus model preparation)"),
        ("linear_sampling_s", "s", "Linear regression, NUTS, 1 chain, 1000 warmup + 1000 draws (sampler wall time plus model preparation)"),
    ]
    for g, unit, title in sections:
        print(f"#### {title}\n")
        print(table(g, unit))
        print()
    sweeps = [g for g in res if g.startswith("sweep_")]
    if sweeps:
        print("#### Logistic gradient across problem shapes (median time per gradient)\n")
        print("| n | p | mint | rust straightforward | rust tuned | rust max effort | max effort / mint |")
        print("|---|---|---|---|---|---|---|")
        for g in sweeps:
            shape = g.split("n=")[1]
            n, p = shape.split(" p=")
            it = res[g]
            mi, rs, rt, rm = (it[k]["median"] for k in ("mint", "rust straightforward", "rust tuned", "rust max effort"))
            print(f"| {n} | {p} | {fmt(mi, 'ns')} | {fmt(rs, 'ns')} | {fmt(rt, 'ns')} | {fmt(rm, 'ns')} | {rm / mi:.2f}x |")
        print()
    print("#### Lines of code (non-blank, non-comment; whole file)\n")
    print("| example | mint | rust straightforward |")
    print("|---|---|---|")
    for k, v in r["lines_of_code"].items():
        print(f"| {k} | {v['mint']} | {v['rust']} |")
    print("\n#### Compile time (seconds, one run each)\n")
    print("| program | seconds |")
    print("|---|---|")
    for k, v in r["compile_seconds"].items():
        print(f"| {k} | {v:.2f} |")


if __name__ == "__main__":
    main()
