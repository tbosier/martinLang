"""Pools several runs of tests/metric/gauss.mint (different seeds) as extra
chains and prints the whitened first and second moment errors in Monte Carlo
standard errors (batch means), which are about five times smaller than in a
single run. Not part of tests/run.sh; needs numpy.

usage: python3 tests/metric/pool_gauss.py build/gauss_exact.txt DRAWS...
  (each DRAWS file written with MINT_DRAWS=... by a build of gauss.mint with
  its own seed)
"""
import struct, sys
import numpy as np
lines = open(sys.argv[1]).read().split("\n")
mu = np.array([float(v) for v in lines[0].split()]); D = len(mu)
L = np.array([[float(v) for v in lines[1 + i].split()] for i in range(D)])
Zs = []
per = []
for f in sys.argv[2:]:
    h = open(f, "rb").read()
    C, N, _ = struct.unpack("<QQQ", h[:24])
    x = np.frombuffer(h[24:24 + 8 * C * N * D], dtype="<f8").reshape(C, N, D)
    Z = (x - mu) @ L
    Zs.append(Z)
Z = np.concatenate(Zs)  # chains x N x D
C, N, _ = Z.shape
for nb in (20, 10):
    bs = N // nb
    def zs(v):
        b = v[:, :nb * bs].reshape(C, nb, bs, -1).mean(2).reshape(C * nb, -1)
        return b.mean(0) / (b.std(0, ddof=1) / np.sqrt(C * nb))
    z1, z2 = zs(Z), zs(Z * Z - 1)
    print(f"{len(sys.argv)-2} runs, {C} chains, batches of {bs}: first max|z| {np.abs(z1).max():.2f} mean z^2 {np.mean(z1**2):.2f}; "
          f"second max|z| {np.abs(z2).max():.2f} mean z^2 {np.mean(z2**2):.2f} mean z {z2.mean():+.2f}")
# per-run worst first moment
for f, Zr in zip(sys.argv[2:], Zs):
    m = np.abs(Zr.reshape(-1, D).mean(0)).max()
    print(f.split('/')[-1], f"max |mean z| {m:.3f}", end="; ")
print()
