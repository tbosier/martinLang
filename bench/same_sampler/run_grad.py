#!/usr/bin/env python3
"""Gradient time of every implementation at the runtime's benchmark point
(MINT_BENCH_GRAD: the same point, the same timing loop, through each
program's own logp entry point; no sampler).

Runs are interleaved (every configuration once, in a shuffled order, then
again), each pinned with taskset: single-threaded configurations to one
hardware thread, the three-thread kernel configurations (Mint's parallel
fused kernel and rust_par with MINT_KERNEL_THREADS=3: what each chain uses
during a large run) to three physical cores of one L3. Each run takes about
--seconds; the repetition count is calibrated once per configuration.

Other work shares the machine, so at the start of every round the CPUs are
chosen afresh: the physical cores whose hardware threads were least busy over
the last second. Every run records two contention measures: how busy the
other hardware threads of its cores were while it ran (sibling_busy, %), and
its CPU time over wall time (cpu_share; below 0.95 x threads means it was
preempted). A run counts as clean when sibling_busy <= 10 and cpu_share >=
0.95 x threads. Reported: median and range over all runs, and the median over
clean runs with their count.

The new Rust baseline's exp variants (DYNPOIS_EXP, see baselines/dynpois_par.rs)
are measured as separate configurations, rust_par[fused] etc.; plain rust_par
is its default (table). Mint is also built without the column-major layout
and without narrow data (mint[no-scan-layout] etc., mintc flags), the two
things the Rust baseline cannot do under the shared parameter order. Before each round the busy fraction of every CPU is
recorded (other work shares the machine).

usage: python bench/same_sampler/run_grad.py [--reps 15] [--seconds 0.25]
"""
import argparse
import json
import os
import random
import resource
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--seconds", type=float, default=0.25)
ap.add_argument("--problems", default=",".join(common.PROBLEMS))
args = ap.parse_args()

configs = []  # (problem, impl, kernel threads); impl "rust_par[MODE]" sets DYNPOIS_EXP=MODE
for p in args.problems.split(","):
    for impl in common.implementations(p):
        configs.append((p, impl, 1))
    if p.startswith("dynpois"):
        configs += [(p, f"rust_par[{m}]", 1) for m in ("fused", "back", "glibc")]
        # Mint without the two things the Rust cannot do: the column-major
        # layout of innov and the int8 copy of the counts
        configs += [(p, f"mint[{m}]", 1) for m in ("no-scan-layout", "no-narrow-data",
                                                    "no-scan-layout+no-narrow-data")]
    if p == "dynpois_large":
        configs += [(p, "mint", 3), (p, "rust_par", 3), (p, "rust_par[glibc]", 3)]


pins = {1: "9", 3: "9,10,11"}  # replaced every round


def siblings_of(cpus):
    sib = set()
    for c in cpus:
        sib |= set(common._read_list(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"))
    return sorted(sib - set(cpus))


def bench(cfg, k):
    p, impl, nt = cfg
    mode = None
    if "[" in impl:
        impl, mode = impl[:-1].split("[")
    flags = ["--" + f for f in mode.split("+")] if mode and impl == "mint" else []
    argv, env = common.command(p, impl, mint_flags=flags)
    env = dict(env, MINT_BENCH_GRAD=str(k))
    if mode and impl == "rust_par":
        env["DYNPOIS_EXP"] = mode
    if nt > 1:
        env["MINT_KERNEL_THREADS"] = str(nt)
    pin = pins[nt]
    cpus = [int(c) for c in pin.split(",")]
    sib = siblings_of(cpus)
    t0 = common._cpu_times()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    out, wall = common.run(argv, env, pin=pin)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t1 = common._cpu_times()
    res = common.parse_grad_bench(out)
    busy = [100 * (1 - (t1[c][1] - t0[c][1]) / max(1, t1[c][0] - t0[c][0])) for c in sib]
    res["sibling_busy"] = round(max(busy) if busy else 0.0, 1)
    res["cpu_share"] = round((r1.ru_utime - r0.ru_utime + r1.ru_stime - r0.ru_stime) / wall, 3)
    res["cpus"] = pin
    res["clean"] = res["sibling_busy"] <= 10 and res["cpu_share"] >= 0.95 * nt
    return res


load_start = common.loadavg()
reps_for = {}
pins[1], _ = common.quiet_cpus(1, common.cpu_busy(1.0))
pins[3], _ = common.quiet_cpus(3, common.cpu_busy(1.0))
for cfg in configs:
    t = bench(cfg, 20)["ns"] * 1e-9
    reps_for[cfg] = max(20, int(args.seconds / t))
    print(f"calibrated {cfg}: {t * 1e6:.1f} us, {reps_for[cfg]} evaluations per run", flush=True)

samples = {cfg: [] for cfg in configs}
loads = []
rng = random.Random(1)
for r in range(args.reps):
    order = configs[:]
    rng.shuffle(order)
    busy = common.cpu_busy(1.0)
    pins[1], s1 = common.quiet_cpus(1, busy)
    pins[3], s3 = common.quiet_cpus(3, busy)
    loads.append({"loadavg": common.loadavg(), "cpu_busy_percent": busy, "pins": dict(pins)})
    for cfg in order:
        samples[cfg].append(bench(cfg, reps_for[cfg]))
    print(f"round {r + 1}/{args.reps} done, load {loads[-1]['loadavg']}", flush=True)
load_end = common.loadavg()

rows = []
for cfg in configs:
    ns = [s["ns"] for s in samples[cfg]]
    clean = [s["ns"] for s in samples[cfg] if s["clean"]]
    rows.append({"problem": cfg[0], "impl": cfg[1], "kernel_threads": cfg[2], "evals_per_run": reps_for[cfg],
                 "ns_median": statistics.median(ns), "ns_min": min(ns), "ns_max": max(ns),
                 "clean_runs": len(clean), "ns_median_clean": statistics.median(clean) if clean else None,
                 "runs": samples[cfg], "logp": samples[cfg][0]["logp"], "grad_norm": samples[cfg][0]["grad_norm"]})
res = {"what": "gradient time at the runtime's benchmark point (MINT_BENCH_GRAD), interleaved, pinned",
       "reps": args.reps, "seconds_per_run": args.seconds, "loadavg_start": load_start,
       "loadavg_per_round": loads, "loadavg_end": load_end, "rows": rows}
os.makedirs(common.RESULTS, exist_ok=True)
path = os.path.join(common.RESULTS, "grad.json")
json.dump(res, open(path, "w"), indent=1)
for row in rows:
    mc = row["ns_median_clean"]
    print(f"{row['problem']:14s} {row['impl']:16s} nt={row['kernel_threads']} median {row['ns_median'] / 1e3:9.2f} us "
          f"range {row['ns_min'] / 1e3:.2f} to {row['ns_max'] / 1e3:.2f}; clean {row['clean_runs']}: "
          + (f"{mc / 1e3:.2f} us" if mc else "none"))
print(f"wrote {path}")
