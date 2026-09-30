#!/usr/bin/env python3
"""Builds every example and baseline, runs them interleaved, and writes
bench/results.json plus a markdown summary to stdout.

Standard library only. Run from the repository root:  python3 bench/bench.py [REPS]
"""
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
MINTC = os.path.join(ROOT, "compiler", "target", "release", "mintc")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
CORE = os.environ.get("BENCH_CORE", "5")


def sh(cmd, env=None, cwd=ROOT):
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def timed(cmd):
    t = time.perf_counter()
    sh(cmd)
    return time.perf_counter() - t


def build():
    os.makedirs(BUILD, exist_ok=True)
    sh(["cargo", "build", "--release", "-q"], cwd=os.path.join(ROOT, "compiler"))
    compile_s = {}
    sh(["clang", "-O3", "-march=native", "-c", "runtime/mint_rt.c", "-o", "build/mint_rt.o"])
    mint = {
        "logistic_newton": [],
        "logistic_bayes": [],
        "linear_bayes": [],
        "linear_bayes_nss": ["--no-suffstats"],
        "logistic_newton_strict": ["--strict-fp"],
        "logistic_newton_noblock": ["--no-gram-blocking"],
        "logistic_bayes_strict": ["--strict-fp"],
        "linear_bayes_nss_strict": ["--no-suffstats", "--strict-fp"],
        "logistic_bayes_nofission": ["--no-fission"],
        "logistic_bayes_novecmath": ["--no-vecmath"],
        "logistic_bayes_nofission_strict": ["--no-fission", "--strict-fp"],
    }
    for name, flags in mint.items():
        src = re.sub(r"_(nss|strict|nofission|novecmath|noblock).*$", "", name)
        compile_s["mint_" + name] = timed([MINTC, "build", f"examples/{src}.mint", "-o", f"build/{name}"] + flags)
    for b in ["logistic_newton", "logistic_newton_tuned", "logistic_bayes", "logistic_bayes_tuned",
              "linear_bayes", "linear_bayes_suffstats"]:
        compile_s["rust_" + b] = timed(["rustc", "--edition", "2021", "-C", "opt-level=3", "-C", "target-cpu=native",
                                        f"baselines/{b}.rs", "-o", f"build/rs_{b}",
                                        "-C", f"link-arg={BUILD}/mint_rt.o", "-l", "m"])
    # max-effort baselines: nightly Rust (LLVM 21), AVX2 intrinsics, glibc vector exp/log
    for b in ["logistic_newton_max", "logistic_bayes_max"]:
        compile_s["rust_" + b] = timed(["rustc", "+nightly", "--edition", "2021", "-C", "opt-level=3", "-C",
                                        "target-cpu=native", f"baselines/{b}.rs", "-o", f"build/rs_{b}",
                                        "-C", f"link-arg={BUILD}/mint_rt.o", "-l", "m", "-l", "mvec"])
    return compile_s


def run(binary, env=None, cwd=ROOT):
    return sh(["taskset", "-c", CORE, os.path.join(BUILD, binary)], env=env, cwd=cwd)


def fit_seconds(out):
    return float(re.search(r"fit_seconds (\S+)", out).group(1))


def grad_ns(out):
    return float(re.search(r"ns_per_eval=(\S+)", out).group(1))


def sampling_s(out):
    """Sampler wall time plus model preparation (e.g. sufficient statistics)."""
    m = re.search(r"sampling took (\S+) s .*preparation took (\S+) s", out)
    return float(m.group(1)) + float(m.group(2))


def gradients(out):
    return int(re.search(r"gradients=(\d+)", out).group(1))


def summarize(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "runs": xs}


def bench(groups):
    """groups: {group: {label: (binary, env, extractor)}}; runs interleaved."""
    raw = {g: {k: [] for k in v} for g, v in groups.items()}
    extra = {g: {k: [] for k in v} for g, v in groups.items()}
    for rep in range(REPS):
        for g, items in groups.items():
            for label, spec in items.items():
                binary, env, fn = spec[:3]
                out = run(binary, env, spec[3] if len(spec) > 3 else ROOT)
                raw[g][label].append(fn(out))
                if fn is sampling_s:
                    extra[g][label].append(gradients(out))
        print(f"  rep {rep + 1}/{REPS} done", file=sys.stderr)
    res = {}
    for g in groups:
        res[g] = {}
        for k in groups[g]:
            res[g][k] = summarize(raw[g][k])
            if extra[g][k]:
                res[g][k]["gradients"] = extra[g][k]
    return res


def code_lines(path):
    """Non-blank lines that are not // comments."""
    n = 0
    for line in open(os.path.join(ROOT, path)):
        t = line.strip()
        if t and not t.startswith("//"):
            n += 1
    return n


def line_counts():
    pairs = {
        "newton": ("examples/logistic_newton.mint", "baselines/logistic_newton.rs"),
        "logistic_bayes": ("examples/logistic_bayes.mint", "baselines/logistic_bayes.rs"),
        "linear_bayes": ("examples/linear_bayes.mint", "baselines/linear_bayes.rs"),
    }
    return {k: {"mint": code_lines(a), "rust": code_lines(b)} for k, (a, b) in pairs.items()}


def main():
    print("building...", file=sys.stderr)
    compile_s = build()
    G = {"MINT_BENCH_GRAD": "3000"}
    groups = {
        "newton_fit_s": {
            "mint": ("logistic_newton", None, fit_seconds),
            "mint --strict-fp": ("logistic_newton_strict", None, fit_seconds),
            "mint --no-gram-blocking": ("logistic_newton_noblock", None, fit_seconds),
            "rust straightforward": ("rs_logistic_newton", None, fit_seconds),
            "rust tuned": ("rs_logistic_newton_tuned", None, fit_seconds),
            "rust max effort": ("rs_logistic_newton_max", None, fit_seconds),
        },
        "logistic_grad_ns": {
            "mint": ("logistic_bayes", G, grad_ns),
            "mint --strict-fp": ("logistic_bayes_strict", G, grad_ns),
            "mint --no-fission": ("logistic_bayes_nofission", G, grad_ns),
            "mint --no-vecmath": ("logistic_bayes_novecmath", G, grad_ns),
            "mint --no-fission --strict-fp": ("logistic_bayes_nofission_strict", G, grad_ns),
            "rust straightforward": ("rs_logistic_bayes", G, grad_ns),
            "rust tuned": ("rs_logistic_bayes_tuned", G, grad_ns),
            "rust max effort": ("rs_logistic_bayes_max", G, grad_ns),
        },
        "linear_grad_ns": {
            "mint": ("linear_bayes", {"MINT_BENCH_GRAD": "200000"}, grad_ns),
            "mint --no-suffstats": ("linear_bayes_nss", G, grad_ns),
            "mint --no-suffstats --strict-fp": ("linear_bayes_nss_strict", G, grad_ns),
            "rust straightforward": ("rs_linear_bayes", G, grad_ns),
            "rust sufficient statistics": ("rs_linear_bayes_suffstats", {"MINT_BENCH_GRAD": "200000"}, grad_ns),
        },
        "logistic_sampling_s": {
            "mint": ("logistic_bayes", None, sampling_s),
            "rust straightforward": ("rs_logistic_bayes", None, sampling_s),
            "rust tuned": ("rs_logistic_bayes_tuned", None, sampling_s),
            "rust max effort": ("rs_logistic_bayes_max", None, sampling_s),
        },
        "linear_sampling_s": {
            "mint": ("linear_bayes", None, sampling_s),
            "mint --no-suffstats": ("linear_bayes_nss", None, sampling_s),
            "rust straightforward": ("rs_linear_bayes", None, sampling_s),
            "rust sufficient statistics": ("rs_linear_bayes_suffstats", None, sampling_s),
        },
    }
    # Shape sweep for the logistic gradient: same binaries, different data.
    gen = os.path.join(BUILD, "gen_data")
    sh(["rustc", "-O", "bench/gen_data.rs", "-o", gen])
    for n, p in [(20000, 5), (5000, 20), (2000, 100), (100000, 20)]:
        d = os.path.join(BUILD, "sweep", f"n{n}_p{p}")
        os.makedirs(os.path.join(d, "data"), exist_ok=True)
        sh([gen, "logistic", str(n), str(p), "21", os.path.join(d, "data", "logit"), "0.3"])
        reps = str(max(200, int(3e8 / (n * (p + 10)))))
        env = {"MINT_BENCH_GRAD": reps}
        groups[f"sweep_logistic_grad_ns n={n} p={p}"] = {
            "mint": ("logistic_bayes", env, grad_ns, d),
            "rust straightforward": ("rs_logistic_bayes", env, grad_ns, d),
            "rust tuned": ("rs_logistic_bayes_tuned", env, grad_ns, d),
            "rust max effort": ("rs_logistic_bayes_max", env, grad_ns, d),
        }
    print("running...", file=sys.stderr)
    res = bench(groups)
    meta = {
        "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?"),
        "kernel": platform.release(),
        "rustc": sh(["rustc", "--version"]).strip(),
        "clang": sh(["clang", "--version"]).splitlines()[0],
        "reps": REPS,
        "pinned_core": CORE,
        "loadavg_at_start": open("/proc/loadavg").read().split()[:3],
    }
    out = {"meta": meta, "compile_seconds": compile_s, "lines_of_code": line_counts(), "results": res}
    with open(os.path.join(ROOT, "bench", "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    for g, items in res.items():
        print(f"\n### {g}\n")
        print("| implementation | median | min | max |")
        print("|---|---|---|---|")
        for k, v in items.items():
            print(f"| {k} | {v['median']:.4g} | {v['min']:.4g} | {v['max']:.4g} |")
    print("\n### lines of code (non-blank, non-comment)\n")
    for k, v in out["lines_of_code"].items():
        print(f"- {k}: mint {v['mint']}, rust {v['rust']}")
    print("\n### compile seconds\n")
    for k, v in compile_s.items():
        print(f"- {k}: {v:.2f}")


if __name__ == "__main__":
    main()
