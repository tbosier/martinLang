#!/usr/bin/env python3
"""Exact check of the draws of a collapsed walk (forward filtering, backward
sampling), joint distribution included.

Each model below has fixed scales and one unrelated parameter (`dummy`), so
the collapsed walk's posterior is a known Gaussian: per series, with prior
X ~ N(m_w, s_w^2 I) and y = a + c L (B + k X) + noise (L the running sum),
its covariance is S = (I / s_w^2 + A'A / r)^-1 and its mean
S (m_w / s_w^2 + A'(y - a - c L B) / r), A = c L diag(k). Each kept draw's
X is an independent draw from it (the sampler's draws of `dummy` do not
enter), so with N draws every mean and every covariance entry (all pairs of
times within a series) has a known standard error: sqrt(S_ii / N) and
sqrt((S_ii S_jj + S_ij^2) / N). The test fails if any |z| exceeds 4.5 (the
threshold of compare_posterior.py; with a few hundred comparisons the
largest |z| of standard normals is usually below 3.7) or any draw is not
finite. Covered: a walk with an element-wise coefficient that is 0 at some
elements (their X must come from the prior) and c = -0.5 written as a
negated literal (the general kernel); a single Vector[T] walk (the shared
kernel); and two collapses in one model.

usage: check_ffbs.py MINTC BUILD_DIR
"""
import os
import struct
import subprocess
import sys

import numpy as np

MINTC, BUILD = sys.argv[1], sys.argv[2]
N_DRAWS = 2500  # per chain, 4 chains
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
        return np.frombuffer(f.read(), dtype="<f8").reshape(c * n, d)


def posterior(y, a, c, B, k, mw, sw, r):
    """Exact posterior mean and covariance of one series' X."""
    T = len(y)
    L = np.tril(np.ones((T, T)))
    A = c * L @ np.diag(k)
    prec = np.eye(T) / sw ** 2 + A.T @ A / r
    S = np.linalg.inv(prec)
    mean = S @ (np.full(T, mw) / sw ** 2 + A.T @ (y - a - c * L @ B) / r)
    return mean, S


def compare(name, X, mean, S):
    """X: draws x T for one series."""
    n = X.shape[0]
    zm = (X.mean(0) - mean) / np.sqrt(np.diag(S) / n)
    C = np.cov(X, rowvar=False)
    se = np.sqrt((np.outer(np.diag(S), np.diag(S)) + S ** 2) / n)
    zc = ((C - S) / se)[np.triu_indices(len(mean))]
    return np.abs(zm), np.abs(zc)


MODELS = {
    "matrix": """model FM {
    data y: Matrix[G, T]
    data m: Matrix[G, T]
    param dummy: Real
    param innov: Matrix[G, T]
    dummy ~ Normal(0, 1)
    innov ~ Normal(0.1, 0.4)
    y ~ Normal(-0.5 * cumsum(0.2 + m .* innov, T) + 1, 0.3)
}
fn main() {
    let y: Matrix[G, T] = read("DATA/y.f64")
    let m: Matrix[G, T] = read("DATA/m.f64")
    print(sample(FM(y, m), draws = NDRAWS, warmup = 200, chains = 4, seed = 9))
}
""",
    "vector": """model FV {
    data y: Vector[T]
    param dummy: Real
    param innov: Vector[T]
    dummy ~ Normal(0, 1)
    innov ~ Normal(0, 0.3)
    y ~ Normal(0.5 + cumsum(innov), 0.2)
}
fn main() {
    let y: Vector[T] = read("DATA/yv.f64")
    print(sample(FV(y), draws = NDRAWS, warmup = 200, chains = 4, seed = 9))
}
""",
    "two": """model F2 {
    data y: Matrix[G, T]
    data z: Vector[S]
    param dummy: Real
    param innov: Matrix[G, T]
    param walk: Vector[S]
    dummy ~ Normal(0, 1)
    innov ~ Normal(0, 0.3)
    walk  ~ Normal(0.05, 0.2)
    y ~ Normal(cumsum(innov, T), 0.4)
    z ~ Normal(2 * cumsum(walk), 0.3)
}
fn main() {
    let y: Matrix[G, T] = read("DATA/y.f64")
    let z: Vector[S] = read("DATA/z.f64")
    print(sample(F2(y, z), draws = NDRAWS, warmup = 200, chains = 4, seed = 9))
}
""",
}

rng = np.random.default_rng(11)
G, T, S = 3, 8, 6
y = 1 + np.cumsum(rng.normal(0, 0.4, (G, T)), axis=1) + rng.normal(0, 0.3, (G, T))
m = rng.choice([1.0, 2.0], (G, T))
m[0, 3] = m[2, 0] = m[1, T - 1] = 0.0  # no effect on y: these X follow their prior
yv = y[0]
z = np.cumsum(rng.normal(0.1, 0.4, S)) + rng.normal(0, 0.3, S)
d = os.path.join(BUILD, "kalffbs")
os.makedirs(d, exist_ok=True)
for nm, a in (("y", y), ("m", m), ("yv", yv), ("z", z)):
    write_f64(f"{d}/{nm}.f64", a)

for name, src in MODELS.items():
    path = f"{d}/{name}"
    with open(path + ".mint", "w") as f:
        f.write(src.replace("DATA", d).replace("NDRAWS", str(N_DRAWS)))
    b = subprocess.run([MINTC, "build", path + ".mint", "-o", path], capture_output=True, text=True)
    if b.returncode != 0 or b.stderr.count("collapsed ") != (2 if name == "two" else 1):
        report(False, f"ffbs {name}: build or collapse report\n{b.stderr}")
        continue
    r = subprocess.run([path], env=dict(os.environ, MINT_DRAWS=path + ".draws"), capture_output=True, text=True)
    if r.returncode != 0:
        report(False, f"ffbs {name}: run failed\n{r.stderr}")
        continue
    D = read_draws(path + ".draws")
    if not np.all(np.isfinite(D)):
        report(False, f"ffbs {name}: {np.sum(~np.isfinite(D))} draws are not finite")
        continue
    # (series draws x T, exact mean, exact covariance) for every walk
    cases = []
    if name == "matrix":
        X = D[:, 1:1 + G * T].reshape(-1, G, T)
        for g in range(G):
            cases.append((X[:, g], *posterior(y[g], 1.0, -0.5, np.full(T, 0.2), m[g], 0.1, 0.4, 0.09)))
    elif name == "vector":
        cases.append((D[:, 1:1 + T], *posterior(yv, 0.5, 1.0, np.zeros(T), np.ones(T), 0.0, 0.3, 0.04)))
    else:
        X = D[:, 1:1 + G * T].reshape(-1, G, T)
        for g in range(G):
            cases.append((X[:, g], *posterior(y[g], 0.0, 1.0, np.zeros(T), np.ones(T), 0.0, 0.3, 0.16)))
        cases.append((D[:, 1 + G * T:1 + G * T + S], *posterior(z, 0.0, 2.0, np.zeros(S), np.ones(S), 0.05, 0.2, 0.09)))
    zm, zc = [], []
    for Xs, mean, Sig in cases:
        a, b2 = compare(name, Xs, mean, Sig)
        zm.extend(a)
        zc.extend(b2)
    zm, zc = np.array(zm), np.array(zc)
    report(zm.max() < 4.5 and zc.max() < 4.5,
           f"ffbs {name}: {D.shape[0]} draws of the walk match its exact Gaussian posterior: largest |z| of "
           f"{len(zm)} means {zm.max():.2f}, of {len(zc)} covariance entries {zc.max():.2f} "
           f"(median {np.median(np.concatenate([zm, zc])):.2f})")
sys.exit(1 if fail else 0)
