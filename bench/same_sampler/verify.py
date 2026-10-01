#!/usr/bin/env python3
"""Checks that every implementation computes the same log density (up to the
constant each one drops) and the same gradient as Mint, and that the Stan
models are safe to call from the chains' threads.

1. At the runtime's benchmark point, through each program's own entry point
   (MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1, so the code path is the one the
   sampler calls, including Mint's layout conversion): the gradient of every
   implementation against Mint's, component by component, and the log
   density difference against the constant that implementation is known to
   drop (computed here from the model's scales).
2. At random points, through BridgeStan's Python interface on the same
   compiled models: Stan's log density minus a numpy reference formula must
   be that same constant at every point, and the gradients must agree.
3. Thread safety: a short 4-chain run of the Stan model with one shared model
   object and with one model per chain (BS_MODEL_PER_CHAIN=1), and a repeat of
   the shared run: the draws must be byte-identical.
4. The new Rust baseline (baselines/dynpois_par.rs) against its scalar
   reference implementation, for each exp variant and several thread counts.

usage: python bench/same_sampler/verify.py   (writes results/verify.json)
"""
import json
import math
import re
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

sys.path.insert(0, os.path.join(common.ROOT, "bench", "dynpois"))
import spec_logdensity  # noqa: E402

S8 = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=float)
Y8 = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=float)


def read_f64(path):
    raw = open(os.path.join(common.ROOT, path), "rb").read()
    r, c = np.frombuffer(raw[:16], dtype="<u8")
    return np.frombuffer(raw[16:], dtype="<f8").reshape(int(r), int(c))


def dynpois_y(problem):
    return read_f64(f"bench/dynpois/data_{problem.split('_')[1]}/y.f64")


def stan_constant(problem):
    """log density kept by Mint minus Stan's (propto = true drops every term
    that does not depend on the parameters: here each Normal's -log(scale))."""
    if problem.startswith("dynpois"):
        G, T = dynpois_y(problem).shape
        return -(G * math.log(0.4) + T * math.log(0.05) + G * T * math.log(0.08))
    if problem == "logistic":
        return -math.log(2.5)
    if problem == "eight_schools":
        return -2 * math.log(5.0) - float(np.sum(np.log(S8)))
    raise KeyError(problem)


# numpy references (Mint's convention: every Normal keeps -log(scale))
def ref_logistic(theta, X, y):
    a, b = theta[0], theta[1:]
    eta = a + X @ b
    lp = -0.5 * (a / 2.5) ** 2 - math.log(2.5) - 0.5 * b @ b + np.sum(y * eta - np.logaddexp(0, eta))
    r = y - 1 / (1 + np.exp(-eta))
    return lp, np.concatenate([[-a / 6.25 + r.sum()], -b + X.T @ r])


def ref_eight(theta):
    mu, lt, eta = theta[0], theta[1], theta[2:]
    tau = math.exp(lt)
    z = (Y8 - mu - tau * eta) / S8
    lp = (-0.5 * (mu / 5) ** 2 - 0.5 * (tau / 5) ** 2 + lt - 2 * math.log(5) - 0.5 * eta @ eta
          - 0.5 * z @ z - np.sum(np.log(S8)))
    dm = z / S8
    return lp, np.concatenate([[-mu / 25 + dm.sum(), (-tau / 25 + dm @ eta) * tau + 1], -eta + dm * tau])


def bench_point_checks(problem, report):
    res = {}
    for impl in common.implementations(problem):
        argv, env = common.command(problem, impl, draws=1000, warmup=1000, chains=4, seed=1)
        out, _ = common.run(argv, dict(env, MINT_BENCH_GRAD="1", MINT_PRINT_GRAD="1"))
        res[impl] = common.parse_printed_grad(out)
    lp_m, g_m = res["mint"]
    g_m = np.array(g_m)
    rows = {}
    for impl, (lp, g) in res.items():
        g = np.array(g)
        const = stan_constant(problem) if impl.startswith("stan") else 0.0
        rows[impl] = {
            "logp": lp,
            "logp_mint_minus_this": lp_m - lp,
            "expected_constant": const,
            "constant_error": abs((lp_m - lp) - const) / max(1.0, abs(lp_m)),
            "grad_max_rel_diff_vs_mint": float(np.max(np.abs(g - g_m) / np.maximum(1.0, np.abs(g_m)))),
            "grad_norm_rel_diff_vs_mint": float(np.linalg.norm(g - g_m) / np.linalg.norm(g_m)),
            "dim": len(g),
        }
        print(f"  {problem:14s} {impl:9s} D={len(g):6d} lp={lp:.12e} mint-lp={lp_m - lp:.10f} "
              f"(expected {const:.10f}) grad max rel diff vs Mint {rows[impl]['grad_max_rel_diff_vs_mint']:.2e}")
    report["bench_point"][problem] = rows


def random_point_checks(problem, report):
    import bridgestan as bs

    stan = common.PROBLEMS[problem]["stan"]
    rng = np.random.default_rng(2024)
    for impl, name in stan.items():
        m = bs.StanModel(os.path.join(common.OUT, "stan", name + "_model.so"),
                         data=os.path.join(common.OUT, "data", common.PROBLEMS[problem]["json"] + ".json"))
        D = m.param_unc_num()
        consts, gerr = [], []
        for k in range(6):
            if problem.startswith("dynpois"):
                y = dynpois_y(problem)
                G, T = y.shape
                th = np.concatenate([[1.5 + 0.2 * rng.normal()], 1.5 + 0.4 * rng.normal(size=G),
                                     0.05 * rng.normal(size=T), 0.08 * rng.normal(size=G * T)])
                lp_r, g_r = spec_logdensity.log_density(th, y), spec_logdensity.gradient(th, y)
            elif problem == "logistic":
                X, y = read_f64("data/logit_X.f64"), read_f64("data/logit_y.f64").ravel()
                th = 0.5 * rng.normal(size=D)
                lp_r, g_r = ref_logistic(th, X, y)
            else:
                th = rng.normal(size=D)
                lp_r, g_r = ref_eight(th)
            lp_s, g_s = m.log_density_gradient(th, propto=True, jacobian=True)
            consts.append(lp_r - lp_s)
            gerr.append(float(np.max(np.abs(g_s - g_r) / np.maximum(1.0, np.abs(g_r)))))
        c = stan_constant(problem)
        spread = max(abs(x - c) for x in consts)
        report["random_points"][f"{problem}/{impl}"] = {
            "points": len(consts), "ref_minus_stan": consts, "expected_constant": c,
            "max_constant_error": spread, "max_grad_rel_diff": max(gerr)}
        print(f"  {problem:14s} {impl:9s} {len(consts)} points: ref-stan - constant within {spread:.2e}; "
              f"grad max rel diff {max(gerr):.2e}")


def thread_safety(report):
    tmp = os.path.join(common.OUT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    blobs = {}
    for label, extra in [("shared", {}), ("per_chain", {"BS_MODEL_PER_CHAIN": "1"}), ("shared_again", {})]:
        path = os.path.join(tmp, f"ts_{label}.draws")
        argv, env = common.command("dynpois_small", "stan", draws=200, warmup=200, chains=4, seed=5)
        common.run(argv, dict(env, MINT_DRAWS=path, **extra))
        blobs[label] = open(path, "rb").read()
        os.remove(path)
    same = blobs["shared"] == blobs["per_chain"] == blobs["shared_again"]
    report["thread_safety"] = {
        "run": "dynpois_small, Stan via BridgeStan, 4 chains in threads, 200 + 200, seed 5",
        "draws_bytes": len(blobs["shared"]),
        "shared_model_equals_model_per_chain_and_repeat": same}
    print(f"  thread safety: shared model, one model per chain and a repeat give identical draws: {same}")
    return same


def rust_par_selftests(report):
    """baselines/dynpois_par.rs against its scalar reference (random shapes and
    points, finite differences), for every exp variant and several kernel
    thread counts."""
    runs = [(m, "small", "1") for m in ("table", "fused", "back", "glibc")]
    runs += [("table", "large", "threads:1,3")]
    ok = True
    report["rust_par_selftest"] = []
    for mode, size, which in runs:
        out, _ = common.run([os.path.join(common.OUT, "rs_dynpois_par"), f"bench/dynpois/data_{size}/y.f64"],
                            {"DYNPOIS_EXP": mode, "DYNPOIS_SELFTEST": which})
        passes = out.count("selftest: PASS")
        nts = out.count("selftest: kernel threads =")
        worst = [float(x) for x in re.findall(r"worst per-component \|g - ref\|/max\(\|ref\|,1\) = (\S+);", out)]
        fd = [float(x) for x in re.findall(r"worst relative error (\S+)", out)]
        good = passes == nts and nts > 0 and "FAIL" not in out
        ok = ok and good
        report["rust_par_selftest"].append({"exp": mode, "data": size, "thread_counts": which, "passed": good,
                                            "worst_grad_vs_reference": max(worst), "worst_fd_error": max(fd)})
        print(f"  rust_par exp={mode:6s} data={size:5s} threads={which}: {passes}/{nts} pass, "
              f"worst gradient vs reference {max(worst):.2e}, worst finite-difference error {max(fd):.2e}")
    return ok


def main():
    report = {"bench_point": {}, "random_points": {}, "loadavg": common.loadavg()}
    print("1. benchmark point, through each program's own entry point")
    for p in common.PROBLEMS:
        bench_point_checks(p, report)
    print("2. random points, Stan (BridgeStan Python) against numpy references")
    for p in common.PROBLEMS:
        random_point_checks(p, report)
    print("3. thread safety")
    ok = thread_safety(report)
    print("4. the new Rust baseline against its scalar reference")
    ok = rust_par_selftests(report) and ok
    worst_g = max(r["grad_max_rel_diff_vs_mint"] for rows in report["bench_point"].values() for r in rows.values())
    worst_c = max(r["constant_error"] for rows in report["bench_point"].values() for r in rows.values())
    worst_rg = max(r["max_grad_rel_diff"] for r in report["random_points"].values())
    worst_rc = max(r["max_constant_error"] for r in report["random_points"].values())
    report["summary"] = {"worst_grad_vs_mint": worst_g, "worst_constant_error": worst_c,
                         "worst_random_point_grad": worst_rg, "worst_random_point_constant": worst_rc,
                         "thread_safety": ok}
    ok = ok and worst_g < 1e-9 and worst_c < 1e-12 and worst_rg < 1e-9 and worst_rc < 1e-6
    report["summary"]["pass"] = ok
    os.makedirs(common.RESULTS, exist_ok=True)
    json.dump(report, open(os.path.join(common.RESULTS, "verify.json"), "w"), indent=1)
    print(f"summary: {report['summary']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
