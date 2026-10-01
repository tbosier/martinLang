#!/usr/bin/env python3
"""The Kalman collapse samples the same posterior as full NUTS.

For each test model (noncentred, shared, series, mixed in tests/kalman),
on one small
simulated panel, runs the collapsed build and the `--no-collapse` build with
several seeds each (4 chains x DRAWS draws, raw draws via MINT_DRAWS) and
compares, for every quantity in the draws (the remaining parameters and
every innovation, which the collapsed build draws by forward filtering,
backward sampling), the pooled posterior mean and standard deviation:

  z_mean = (mean_c - mean_f) / sqrt(mcse_c^2 + mcse_f^2),  mcse = sd / sqrt(ESS)
  z_sd   = (sd_c - sd_f) / sqrt(se_c^2 + se_f^2),          se = sd(dev^2) / sqrt(ESS(dev^2)) / (2 sd)

with ESS from Geyer's initial monotone sequence over all chains of all
seeds. With about two hundred quantities per model the largest |z| of
independent standard normals is usually below 3.7; the test fails above
4.5 (means or sds), when the collapsed build has any divergent transition
or the full one more than 0.5% of its draws (full NUTS on a random walk
is the hard case the collapse removes; the test data are chosen so that
it mostly copes), or when an R-hat is above 1.05. The collapsed run must
report its sampled dimension.

usage: compare_posterior.py MINTC BUILD_DIR [DRAWS]
"""
import os
import struct
import subprocess
import sys

import numpy as np

MINTC, BUILD = sys.argv[1], sys.argv[2]
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
SEEDS = (1, 2, 3)
HERE = os.path.dirname(os.path.abspath(__file__))
fail = False


def report(ok, msg):
    global fail
    print(("PASS  " if ok else "FAIL  ") + msg)
    fail |= not ok


def write_f64(path, a):
    a = np.asarray(a, dtype="<f8")
    rows, cols = (a.shape[0], 1) if a.ndim == 1 else a.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", rows, cols))
        f.write(np.ascontiguousarray(a).tobytes())


def read_draws(path):
    with open(path, "rb") as f:
        c, n, d = struct.unpack("<QQQ", f.read(24))
        return np.frombuffer(f.read(), dtype="<f8").reshape(c, n, d)


def ess(x):
    """x: chains x draws. Geyer's initial monotone sequence (Stan's method)."""
    m, n = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(xc, 2 * n, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    w = ac[:, 0].mean() * n / (n - 1)
    var = w * (n - 1) / n + x.mean(axis=1).var(ddof=1) if m > 1 else w
    rho = 1 - (w - ac.mean(axis=0)) / var
    rho[0] = 1
    t, s, prev = 0, 0.0, np.inf
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        p = min(p, prev)
        s += p
        prev = p
        t += 2
    tau = -1 + 2 * s
    return m * n / max(tau, 1.0 / np.log10(m * n))


def rhat(x):
    m, n = x.shape
    h = n // 2
    sp = np.concatenate([x[:, :h], x[:, h:2 * h]])
    w = sp.var(axis=1, ddof=1).mean()
    b = h * sp.mean(axis=1).var(ddof=1)
    return np.sqrt(((h - 1) / h * w + b / h) / w) if w > 0 else 1.0


def run(binary, draws_path):
    # a higher acceptance target than Stan's 0.8 (smaller steps), for both
    # builds, so that neither has divergent transitions on these weakly
    # identified scales
    env = dict(os.environ, MINT_DRAWS=draws_path, MINT_TARGET_ACCEPT="0.95")
    r = subprocess.run([binary], env=env, capture_output=True, text=True)
    return r


def divergences(out):
    for line in out.splitlines():
        if "divergences=" in line:
            return int(line.split("divergences=")[1].split()[0])
    return -1


# (model, sd of the simulated innovations, sd of the noise): the centred
# models get data that pin each innovation down, where centred full NUTS
# samples well; the non-centred one gets weakly informative data, where it
# does (the collapsed density is the same for either form)
# (single.mint is left out: on one series full NUTS of the centred form
# has hundreds of divergent transitions and R-hat 1.1, so it is no reference)
MODELS = (("noncentred", 0.25, 0.3), ("shared", 0.25, 0.3), ("series", 0.25, 0.3), ("mixed", 0.3, 0.3))
rng = np.random.default_rng(7)
G, T = 6, 30
for name, sd_w, sd_y in MODELS:
    d = os.path.join(BUILD, f"kalpost_{name}")
    os.makedirs(d, exist_ok=True)
    x = rng.normal(0, 1, T)
    beta = rng.normal(0.3, 0.4, G)
    y = beta[:, None] + np.cumsum(rng.normal(0, sd_w, (G, T)), axis=1) + rng.normal(0, sd_y, (G, T))
    write_f64(f"{d}/y.f64", y)
    write_f64(f"{d}/x.f64", x)
    write_f64(f"{d}/n.f64", rng.poisson(2.0, (G, T)).astype(float))
    src = open(os.path.join(HERE, name + ".mint")).read()
    pooled, divs = {}, {}
    ok = True
    for variant, flags in (("collapsed", []), ("full", ["--no-collapse"])):
        chains = []
        for seed in SEEDS:
            prog = src.replace("DATA", d).replace("DRAWS", str(DRAWS)).replace("WARMUP", "1000").replace("SEED", str(seed))
            with open(f"{d}/{variant}_{seed}.mint", "w") as f:
                f.write(prog)
            b = subprocess.run([MINTC, "build", f"{d}/{variant}_{seed}.mint", "-o", f"{d}/{variant}_{seed}"] + flags,
                               capture_output=True, text=True)
            if b.returncode != 0:
                report(False, f"posterior {name}: build {variant} failed\n{b.stderr}")
                ok = False
                break
            if (("collapsed innov" in b.stderr) != (variant == "collapsed")):
                report(False, f"posterior {name}: {variant} build collapse message wrong: {b.stderr!r}")
                ok = False
            r = run(f"{d}/{variant}_{seed}", f"{d}/{variant}_{seed}.draws")
            if r.returncode != 0:
                report(False, f"posterior {name}: {variant} seed {seed} failed\n{r.stderr}")
                ok = False
                break
            div = divergences(r.stdout)
            divs[variant] = divs.get(variant, 0) + max(div, 0)
            if div < 0:
                report(False, f"posterior {name}: {variant} seed {seed}: no divergence count in the output")
                ok = False
            if variant == "collapsed" and "collapsed: NUTS sampled" not in r.stderr:
                report(False, f"posterior {name}: the collapsed run does not report its sampled dimension")
                ok = False
            chains.append(read_draws(f"{d}/{variant}_{seed}.draws"))
        if not ok:
            break
        pooled[variant] = np.concatenate(chains)  # (seeds x chains) x draws x D
    if not ok:
        continue
    c, f = pooled["collapsed"], pooled["full"]
    D = c.shape[2]
    if f.shape[2] != D:
        report(False, f"posterior {name}: draws have {D} and {f.shape[2]} columns")
        continue
    zm, zs, worst_r = [], [], 1.0
    for j in range(D):
        a, b = c[:, :, j], f[:, :, j]
        ea, eb = ess(a), ess(b)
        worst_r = max(worst_r, rhat(a), rhat(b))
        ma, mb, sa, sb = a.mean(), b.mean(), a.std(ddof=1), b.std(ddof=1)
        zm.append((ma - mb) / np.sqrt(sa * sa / ea + sb * sb / eb))
        # the sd's standard error from the ESS of the squared deviations
        # (often far below the ESS of the draws themselves)
        se = []
        for x, m, s in ((a, ma, sa), (b, mb, sb)):
            v = (x - m) ** 2
            se.append(v.std(ddof=1) / np.sqrt(ess(v)) / (2 * s))
        zs.append((sa - sb) / np.hypot(*se))
    zm, zs = np.abs(zm), np.abs(zs)
    n_all = c.shape[0] * c.shape[1]  # seeds x chains x draws
    report(divs["collapsed"] == 0 and divs["full"] <= 0.005 * n_all,
           f"posterior {name}: divergent transitions: collapsed {divs['collapsed']}, full {divs['full']} of {n_all} draws each")
    print(f"      worst mean: column {zm.argmax()} ({c[:, :, zm.argmax()].mean():.4g} vs {f[:, :, zm.argmax()].mean():.4g}); "
          f"worst sd: column {zs.argmax()} ({c[:, :, zs.argmax()].std():.4g} vs {f[:, :, zs.argmax()].std():.4g})")
    report(worst_r < 1.05, f"posterior {name}: highest split R-hat {worst_r:.3f} over both builds")
    report(zm.max() < 4.5 and zs.max() < 4.5,
           f"posterior {name}: collapsed (Kalman + FFBS) and full NUTS agree on all {D} quantities, "
           f"{len(SEEDS)} seeds each: largest |z| of the means {zm.max():.2f} (median {np.median(zm):.2f}), "
           f"of the sds {zs.max():.2f}")
sys.exit(1 if fail else 0)
