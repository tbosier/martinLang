#!/usr/bin/env python3
"""Exact small-instance checks of the Kalman collapse.

For each test model in tests/kalman and a few small panels (G x T = 2 x 5,
3 x 7 and 9 x 4; single.mint uses the first series), builds the model with mintc (which must report that it
collapsed `innov`), evaluates the compiled log density and gradient at
random points (MINT_THETA) and compares them with a dense Gaussian
computation in numpy: innov integrated out analytically, each series
y_g ~ N(a_g + c L d_g, c^2 L diag(q_g) L' + diag(r_g)) with L the
lower-triangular matrix of ones (the running sum), plus the remaining
priors and Jacobians, all without the -0.5 log(2 pi) per observation that
Mint drops. The reference gradient is a 5-point central difference of the
numpy function. Also runs the compiled finite-difference check
(MINT_GRADCHECK) at each point.

usage: check_marginal.py MINTC BUILD_DIR
Prints PASS/FAIL lines; exits 1 on any failure."""
import os
import struct
import subprocess
import sys

import numpy as np

MINTC, BUILD = sys.argv[1], sys.argv[2]
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


def norm_lp(x, m, s):
    """Mint's Normal log density: no -0.5 log(2 pi)."""
    return np.sum(-0.5 * ((x - m) / s) ** 2 - np.log(s))


def marginal(y, a, c, d, q, r):
    G, T = y.shape
    L = np.tril(np.ones((T, T)))
    lp = 0.0
    for g in range(G):
        mean = a[g] + c * L @ d[g]
        cov = c * c * L @ np.diag(q[g]) @ L.T + np.diag(r[g])
        res = y[g] - mean
        _, logdet = np.linalg.slogdet(cov)
        lp += -0.5 * logdet - 0.5 * res @ np.linalg.solve(cov, res)
    return lp


def full(G, T):
    return lambda v: np.broadcast_to(v, (G, T)).astype(float)


# Each variant: number of parameters NUTS sees, and theta -> log density.
def basic(th, y, x):
    G, T = y.shape
    F = full(G, T)
    pop, beta, uw, uy = th[0], th[1:1 + G], th[1 + G], th[2 + G]
    sw, sy = np.exp(uw), np.exp(uy)
    lp = norm_lp(pop, 0, 1) + norm_lp(beta, pop, 0.4) + norm_lp(sw, 0, 0.5) + uw + norm_lp(sy, 0, 1) + uy
    return lp + marginal(y, F(beta[:, None]), 1.0, F(0.0), F(sw ** 2), F(sy ** 2))


def shared(th, y, x):
    G, T = y.shape
    F = full(G, T)
    pop, beta, sh = th[0], th[1:1 + G], th[1 + G:1 + G + T]
    uw, uy = th[1 + G + T], th[2 + G + T]
    sw, sy = np.exp(uw), np.exp(uy)
    lp = (norm_lp(pop, 0, 1) + norm_lp(beta, pop, 0.4) + norm_lp(sh, 0, 0.05) + norm_lp(sw, 0, 0.5) + uw
          + norm_lp(sy, 0, 1) + uy)
    return lp + marginal(y, F(beta[:, None]), 1.0, F(sh[None, :]), F(sw ** 2), F(sy ** 2))


def general(th, y, x):
    G, T = y.shape
    F = full(G, T)
    beta, gamma, mu = th[0:G], th[G], th[G + 1]
    uw, uy, ut = th[G + 2:2 * G + 2], th[2 * G + 2:3 * G + 2], th[3 * G + 2]
    sw, sy, tau = np.exp(uw), np.exp(uy), np.exp(ut)
    lp = (norm_lp(beta, 0, 1) + norm_lp(gamma, 0, 1) + norm_lp(mu, 0, 0.1) + norm_lp(sw, 0, 0.5) + uw.sum()
          + norm_lp(sy, 0, 1) + uy.sum() + norm_lp(tau, 0, 1) + ut)
    a = F(beta[:, None] + gamma * x[None, :])
    d = F(0.1 * x[None, :] + 2 * tau * mu)
    q = F((2 * tau * sw[:, None]) ** 2)
    r = F(sy[:, None] ** 2)
    return lp + marginal(y, a, -0.5, d, q, r)


def series(th, y, x):
    G, T = y.shape
    F = full(G, T)
    beta, gamma, mu = th[0:G], th[G], th[G + 1]
    uw, uy = th[G + 2:2 * G + 2], th[2 * G + 2]
    sw, sy = np.exp(uw), np.exp(uy)
    lp = (norm_lp(beta, 0, 1) + norm_lp(gamma, 0, 1) + norm_lp(mu, 0, 0.1) + norm_lp(sw, 0, 0.5) + uw.sum()
          + norm_lp(sy, 0, 1) + uy)
    a = F(beta[:, None] + gamma * x[None, :])
    d = F(0.1 * x[None, :] + sw[:, None] * mu)
    return lp + marginal(y, a, -0.5, d, F(sw[:, None] ** 2), F(sy ** 2))


def mixed(th, y, x, n):
    G, T = y.shape
    F = full(G, T)
    beta, uy, other = th[0:G], th[G], th[G + 1:].reshape(G, T)
    sy = np.exp(uy)
    eta = beta[:, None] + np.cumsum(other, axis=1)
    lp = (norm_lp(beta, 0, 1) + norm_lp(sy, 0, 1) + uy + norm_lp(other, 0, 0.1)
          + np.sum(n * eta - np.exp(eta)))
    return lp + marginal(y, F(beta[:, None]), 1.0, F(0.0), F(0.09), F(sy ** 2))


def single(th, y, x):
    T = y.shape[1]
    l0, uw, uy = th
    sw, sy = np.exp(uw), np.exp(uy)
    lp = norm_lp(l0, 0, 1) + norm_lp(sw, 0, 0.5) + uw + norm_lp(sy, 0, 1) + uy
    one = lambda v: np.full((1, T), v)
    return lp + marginal(y[:1], one(l0), 1.0, one(0.0), one(sw ** 2), one(sy ** 2))


def two(th, y, x, z):
    G, T = y.shape
    S = z.shape[1]
    beta, uy, uz = th[0:G], th[G], th[G + 1]
    sy, sz = np.exp(uy), np.exp(uz)
    lp = norm_lp(beta, 0, 1) + norm_lp(sy, 0, 1) + uy + norm_lp(sz, 0, 1) + uz
    F, H = full(G, T), full(G, S)
    lp += marginal(y, F(beta[:, None]), 1.0, F(0.0), F(0.09), F(sy ** 2))
    return lp + marginal(z, H(beta[:, None]), 2.0, H(0.1), H(sz ** 2), H(0.25))


VARIANTS = {
    "basic": (lambda G, T: G + 3, basic),
    "noncentred": (lambda G, T: G + 3, basic),
    "shared": (lambda G, T: G + T + 3, shared),
    "general": (lambda G, T: 3 * G + 3, general),
    "series": (lambda G, T: 2 * G + 3, series),
    "mixed": (lambda G, T: G + 1 + G * T, mixed),
    "single": (lambda G, T: 3, single),
    "two": (lambda G, T: G + 2, two),
}


def fd_grad(f, th, h=1e-4):
    g = np.zeros_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th)
        e[i] = h
        g[i] = (-f(th + 2 * e) + 8 * f(th + e) - 8 * f(th - e) + f(th - 2 * e)) / (12 * h)
    return g


def run(binary, theta_path, extra=None):
    env = dict(os.environ, MINT_THETA=theta_path, MINT_BENCH_GRAD="1", MINT_PRINT_GRAD="1")
    env.update(extra or {})
    return subprocess.run([binary], env=env, capture_output=True, text=True)


rng = np.random.default_rng(2024)
for name, (dim, ref) in VARIANTS.items():
    src = open(os.path.join(HERE, name + ".mint")).read()
    for G, T in ((2, 5), (3, 7), (9, 4)):
        d = os.path.join(BUILD, f"kalman_{name}_{G}x{T}")
        os.makedirs(d, exist_ok=True)
        x = rng.normal(0, 1, T)
        y = 0.3 + np.cumsum(rng.normal(0, 0.3, (G, T)), axis=1) + rng.normal(0, 0.5, (G, T))
        write_f64(f"{d}/y.f64", y)
        write_f64(f"{d}/x.f64", x)
        write_f64(f"{d}/yv.f64", y[0])  # the one series of single.mint
        n = rng.poisson(2.0, (G, T)).astype(float)
        write_f64(f"{d}/n.f64", n)
        z = np.cumsum(rng.normal(0, 0.3, (G, T + 2)), axis=1) + rng.normal(0, 0.5, (G, T + 2))
        write_f64(f"{d}/z.f64", z)
        prog = src.replace("DATA", d).replace("DRAWS", "4").replace("WARMUP", "0").replace("SEED", "1")
        with open(f"{d}/m.mint", "w") as f:
            f.write(prog)
        b = subprocess.run([MINTC, "build", f"{d}/m.mint", "-o", f"{d}/m"], capture_output=True, text=True)
        if b.returncode != 0:
            report(False, f"kalman {name} {G}x{T}: build failed\n{b.stderr}")
            continue
        shape = "T" if name == "single" else "G x T"
        said = f"collapsed innov ({shape} latent scalars) by a Kalman filter" in b.stderr
        if name == "two":
            said &= "collapsed walk (G x S latent scalars) by a Kalman filter" in b.stderr
        report(said, f"kalman {name} {G}x{T}: mintc reports the collapse")
        worst_lp = worst_g = worst_fd = 0.0
        ok = True
        for k in range(3):
            th = rng.normal(0, 0.7 if name != "mixed" else 0.2, dim(G, T))
            write_f64(f"{d}/theta{k}.f64", th)
            out = run(f"{d}/m", f"{d}/theta{k}.f64")
            if out.returncode != 0 or "logp=" not in out.stdout:
                ok = False
                print(out.stdout, out.stderr)
                continue
            lp = float(out.stdout.split("logp=")[1].split()[0])
            lp = float(out.stdout.split("exact log density:")[1].split()[0])
            g = np.array([float(v) for v in out.stdout.split("grad:")[1].split()])
            extra = {"mixed": (n,), "two": (z,)}.get(name, ())
            f = lambda t: ref(t, y, x, *extra)
            lref, gref = f(th), fd_grad(f, th)
            worst_lp = max(worst_lp, abs(lp - lref) / max(1.0, abs(lref)))
            worst_g = max(worst_g, np.max(np.abs(g - gref) / np.maximum(1.0, np.abs(gref))) if len(g) == len(gref) else np.inf)
            gc = run(f"{d}/m", f"{d}/theta{k}.f64", {"MINT_GRADCHECK": "1"})
            errs = [float(l.split("error=")[1]) for l in gc.stdout.splitlines() + gc.stderr.splitlines() if "gradcheck" in l]
            worst_fd = max(worst_fd, errs[0] if errs else np.inf)
        report(ok and worst_lp < 1e-11 and worst_g < 1e-7,
               f"kalman {name} {G}x{T}: log density and gradient equal the dense Gaussian marginal "
               f"(rel. error {worst_lp:.1e}, gradient {worst_g:.1e})")
        report(worst_fd < 1e-5, f"kalman {name} {G}x{T}: compiled gradient passes finite differences ({worst_fd:.1e})")
sys.exit(1 if fail else 0)
