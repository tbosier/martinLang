#!/usr/bin/env python3
"""Kalman collapse against full NUTS on the random-walk panel.

usage: bench.py [--sizes 20x150,250x150] [--seeds 1,2,3] [--out bench/kalman/results.json]

For each size, simulates one data set from the model (make_data.py, data
seed 1 for the small size and 2 for the large one), then for each sampler
seed runs, interleaved:

  collapsed     examples/random_walk_panel.mint, default build: the G x T
                innovations integrated out by a Kalman filter, NUTS on G + 3
  full          the same file built with --no-collapse: NUTS on G + 3 + G T
  full_nc       the non-centred form (innov ~ Normal(0, 1), sigma_w inside
                the running sum), --no-collapse: the usual fix for the
                centred form's funnel

each with 4 chains x 1000 draws after 1000 warmup iterations (Stan's warmup,
the runtime's defaults otherwise, including its threads per chain). Records
the runtime's own report: sampling time (warmup included), gradients,
leapfrog steps per draw, step sizes, divergences, and the ESS of pop,
sigma_w, sigma_y and the lowest over beta, plus the load average around each
run (other jobs share the machine, so wall times are noisy; gradients per
effective draw are not affected by load). Then times one gradient of each
build (MINT_BENCH_GRAD, one thread, 7 alternated repetitions). Writes JSON;
report.py prints the tables.
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MINTC = os.path.join(ROOT, "compiler/target/release/mintc")
PY = sys.executable


def arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


SIZES = [tuple(int(v) for v in s.split("x")) for s in arg("--sizes", "20x150,250x150").split(",")]
SEEDS = [int(s) for s in arg("--seeds", "1,2,3").split(",")]
OUT = arg("--out", os.path.join(ROOT, "bench/kalman/results.json"))
BUILD = os.path.join(ROOT, "build/kalman_bench")


def model_source(G, T, data, seed, noncentred):
    src = open(os.path.join(ROOT, "examples/random_walk_panel.mint")).read()
    src = src.replace("bench/kalman/data_small", data).replace("seed = 5", f"seed = {seed}")
    if noncentred:
        src = src.replace("innov   ~ Normal(0, sigma_w)", "innov   ~ Normal(0, 1)")
        src = src.replace("cumsum(innov, T)", "cumsum(sigma_w * innov, T)")
        assert "sigma_w * innov" in src and "Normal(0, 1)\n\n    y" in src
    return src


def build(src, out, flags):
    path = out + ".mint"
    with open(path, "w") as f:
        f.write(src)
    t0 = time.time()
    r = subprocess.run([MINTC, "build", path, "-o", out] + flags, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"build failed: {out}\n{r.stderr}")
    return time.time() - t0, r.stderr.strip()


def ess_of(out, name):
    m = re.search(rf"^{re.escape(name)}\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)$", out, re.M)
    return None if not m else {"mean": float(m.group(1)), "sd": float(m.group(2)), "ess": float(m.group(3)),
                               "rhat": float(m.group(4))}


def block_min_ess(out, name, G):
    vals = [ess_of(out, f"{name}[{i}]") for i in range(1, min(G, 12) + 1)]
    vals = [v["ess"] for v in vals if v]
    m = re.search(rf"^{name}\s+\d+ more entries: lowest ess (\S+), highest rhat (\S+)", out, re.M)
    if m:
        vals.append(float(m.group(1)))
    return min(vals)


def run(binary, G):
    la0 = os.getloadavg()
    t0 = time.time()
    r = subprocess.run([binary], capture_output=True, text=True)
    wall = time.time() - t0
    la1 = os.getloadavg()
    if r.returncode != 0:
        sys.exit(f"{binary} failed\n{r.stderr}")
    out, err = r.stdout, r.stderr
    g = re.search(r"chains=(\d+) draws/chain=(\d+) gradients=(\d+) divergences=(\d+) step size=(\S+) leapfrog/draw=(\S+)", out)
    chains, draws, grads, div = (int(g.group(i)) for i in range(1, 5))
    step = [float(v) for v in g.group(5).split(",")]
    leap = [float(v) for v in g.group(6).split(",")]
    s = re.search(r"sampling took (\S+) s", err)
    w = re.search(r"gradients: warmup=(\d+) sampling=(\d+)", err)
    tpc = re.search(r"threads per chain=(\d+)", err)
    allp = re.search(r"all (\d+) parameters: highest rhat (\S+) \((\S+)\), lowest ess (\S+) \((\S+)\)", out)
    col = re.search(r"NUTS sampled (\d+) of the (\d+) parameters", err)
    rec = {
        "wall_s": wall,
        "sampling_s": float(s.group(1)),
        "gradients": grads,
        "warmup_gradients": int(w.group(1)),
        "sampling_gradients": int(w.group(2)),
        "divergences": div,
        "step_size": step,
        "leapfrog_per_draw": leap,
        "threads_per_chain": int(tpc.group(1)),
        "draws": chains * draws,
        "nuts_dim": int(col.group(1)) if col else int(allp.group(1)),
        "all_params": int(allp.group(1)),
        "highest_rhat_all": float(allp.group(2)),
        "highest_rhat_at": allp.group(3),
        "lowest_ess_all": float(allp.group(4)),
        "lowest_ess_at": allp.group(5),
        "load_before": la0,
        "load_after": la1,
    }
    for p in ("pop", "sigma_w", "sigma_y"):
        rec[p] = ess_of(out, p)
    rec["beta_min_ess"] = block_min_ess(out, "beta", G)
    rem = [rec["pop"]["ess"], rec["sigma_w"]["ess"], rec["sigma_y"]["ess"], rec["beta_min_ess"]]
    rec["remaining_min_ess"] = min(rem)
    rec["remaining_rhat_max"] = max(rec[p]["rhat"] for p in ("pop", "sigma_w", "sigma_y"))
    return rec


def grad_ns(binary, reps):
    env = dict(os.environ, MINT_BENCH_GRAD=str(reps))
    r = subprocess.run([binary], env=env, capture_output=True, text=True)
    return float(re.search(r"ns_per_eval=(\S+)", r.stdout).group(1))


def main():
    os.makedirs(BUILD, exist_ok=True)
    results = {"sizes": [], "cpus": os.cpu_count()}
    for (G, T) in SIZES:
        data = os.path.join(BUILD, f"data_{G}x{T}")
        dseed = 1 if G <= 20 else 2
        subprocess.run([PY, os.path.join(ROOT, "bench/kalman/make_data.py"), str(G), str(T), str(dseed), data], check=True)
        size = {"G": G, "T": T, "data_seed": dseed, "runs": []}
        for seed in SEEDS:
            for variant, flags, nc in (("collapsed", [], False), ("full", ["--no-collapse"], False),
                                       ("full_nc", ["--no-collapse"], True)):
                out = os.path.join(BUILD, f"{variant}_{G}x{T}_s{seed}")
                bt, msg = build(model_source(G, T, data, seed, nc), out, flags)
                rec = run(out, G)
                rec.update(variant=variant, seed=seed, build_s=bt, mintc_says=msg)
                size["runs"].append(rec)
                print(f"G={G} T={T} seed={seed} {variant:9s} dim={rec['nuts_dim']:6d} sampling={rec['sampling_s']:8.2f}s "
                      f"grads={rec['gradients']:9d} leap/draw={sum(rec['leapfrog_per_draw'])/4:6.1f} "
                      f"ESS pop={rec['pop']['ess']:6.0f} sw={rec['sigma_w']['ess']:6.0f} sy={rec['sigma_y']['ess']:6.0f} "
                      f"beta>={rec['beta_min_ess']:6.0f} div={rec['divergences']} load={rec['load_before'][0]:.1f}",
                      flush=True)
                with open(OUT, "w") as f:
                    json.dump(results | {"sizes": results["sizes"] + [size]}, f, indent=1)
        # one gradient on one thread, the three builds of seed 1 alternated
        reps = 20000 if G <= 20 else 2000
        times = {v: [] for v in ("collapsed", "full", "full_nc")}
        for _ in range(7):
            for v in times:
                times[v].append(grad_ns(os.path.join(BUILD, f"{v}_{G}x{T}_s{SEEDS[0]}"), reps))
        size["gradient_ns"] = times
        size["gradient_load"] = os.getloadavg()
        print(f"G={G} T={T} gradient ns (min of 7): " + ", ".join(f"{v} {min(t):.0f}" for v, t in times.items()), flush=True)
        results["sizes"].append(size)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
