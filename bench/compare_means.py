#!/usr/bin/env python3
"""Compares posterior means between two groups of runs written with MINT_DRAWS.

usage: python3 bench/compare_means.py MODEL "A1.draws A2.draws ..." "B1.draws ..."
  MODEL: dynpois (needs G and T: quantities pop, beta[g] and the terminal state
  of each series), eight_schools (mu, tau) or all (every coordinate).

Each group's runs are combined with equal weights: per quantity, the group
mean is the average of the runs' means, and its Monte Carlo standard error is
the larger of (a) the runs' own MCSEs combined, sqrt(sum mcse_i^2) / runs,
each from an autocorrelation ESS (Geyer's initial monotone sequence over the
run's chains), and (b) the spread of the run means, sd(run means) /
sqrt(runs), when there are at least 3 runs. (b) guards against runs whose
chains are stuck together and so report a small MCSE. Prints, per quantity,
z = |mean_A - mean_B| / sqrt(mcse_A^2 + mcse_B^2) and the difference in
posterior sd units, with the maximum over quantities. With 41 quantities and correct samplers, max z above about 3.2
would be unusual.
"""
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        C, N, D = struct.unpack("<3Q", f.read(24))
        x = np.frombuffer(f.read(), dtype="<f8").reshape(C, N, D)
    return x


def ess(x):
    """x: chains x draws for one quantity."""
    C, N = x.shape
    cm = x.mean(1)
    cv = x.var(1, ddof=1)
    W = cv.mean()
    B = N * cm.var(ddof=1) if C > 1 else 0.0
    vp = (N - 1) / N * W + B / N
    xc = x - cm[:, None]
    nfft = 1 << (2 * N - 1).bit_length()
    f = np.fft.rfft(xc, nfft, axis=1)
    acov = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :N] / N
    acov = acov.mean(0)
    rho = 1.0 - (W - acov) / vp
    rho[0] = 1.0
    tau, prev = 0.0, np.inf
    for t in range(0, N - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair = min(pair, prev)
        prev = pair
        tau += 2 * pair
    tau -= 1
    return C * N / max(tau, 1.0 / np.log10(C * N))


def quantities(x, model):
    C, N, D = x.shape
    if model == "eight_schools":
        return {"mu": x[:, :, 0], "tau": x[:, :, 1]}
    if model == "all":
        return {f"x{j}": x[:, :, j] for j in range(D)}
    # dynpois: D = 1 + G + T + G T
    T = 150
    G = (D - 1 - T) // (T + 1)
    assert 1 + G + T + G * T == D, D
    q = {"pop": x[:, :, 0]}
    for g in range(G):
        q[f"beta[{g}]"] = x[:, :, 1 + g]
    shared = x[:, :, 1 + G:1 + G + T].sum(2)
    innov = x[:, :, 1 + G + T:].reshape(C, N, G, T)
    term = shared[:, :, None] + innov.sum(3)
    for g in range(G):
        q[f"terminal[{g}]"] = term[:, :, g]
    return q


def group(paths, model):
    per = [quantities(load(p), model) for p in paths]
    out = {}
    for k in per[0]:
        ms, vs, sds = [], [], []
        for q in per:
            v = q[k]
            e = ess(v)
            sd = v.std(ddof=1)
            ms.append(v.mean())
            vs.append(sd * sd / e)
            sds.append(sd)
        r = len(ms)
        within = np.sqrt(np.sum(vs)) / r
        between = np.std(ms, ddof=1) / np.sqrt(r) if r >= 3 else 0.0
        out[k] = (float(np.mean(ms)), float(max(within, between)), float(np.mean(sds)))
    return out


def main():
    model, a, b = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
    A, B = group(a, model), group(b, model)
    rows = []
    for k in A:
        ma, ea, sa = A[k]
        mb, eb, sb = B[k]
        z = abs(ma - mb) / np.hypot(ea, eb)
        rows.append((z, k, ma, mb, abs(ma - mb) / np.sqrt((sa * sa + sb * sb) / 2)))
    rows.sort(reverse=True)
    for z, k, ma, mb, sdz in rows[:5]:
        print(f"{k:14s} A {ma:10.5f}  B {mb:10.5f}  |diff| = {z:.2f} MCSE = {sdz:.3f} sd")
    print(f"max over {len(rows)} quantities: {rows[0][0]:.2f} MCSE, {max(r[4] for r in rows):.3f} sd")


if __name__ == "__main__":
    main()
