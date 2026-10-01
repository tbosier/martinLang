"""Checks draws of tests/metric/gauss.mint against the exact posterior.

usage: python3 tests/metric/check_gauss.py DRAWS_FILE

Whitens each draw, z = L'(x - mu) with P = L L', so that z ~ N(0, I) exactly.
It tests every first moment E[z_i] = 0, second moment E[z_i^2] = 1 and cross
moment E[z_i z_j] = 0 (i < j), each against its Monte Carlo standard error
estimated by batch means (20 batches per chain): within 4 standard errors
for the first and second moments, and within 5 for the cross moments, of
which there are many more (2,415 for d = 70). A wrong momentum distribution,
kinetic energy or position update for the metric biases the stationary
distribution and shows up here, along the narrow directions in particular.
"""
import struct
import sys

h = open(sys.argv[1], "rb").read()
C, N, D = struct.unpack("<QQQ", h[:24])
x = struct.unpack(f"<{C * N * D}d", h[24:24 + 8 * C * N * D])
lines = open("build/gauss_exact.txt").read().split("\n")
mu = [float(v) for v in lines[0].split()]
L = [[float(v) for v in lines[1 + i].split()] for i in range(D)]
assert len(mu) == D

# whitened draws, one list per draw
Z = []
for t in range(C * N):
    base = t * D
    d = [x[base + i] - mu[i] for i in range(D)]
    Z.append([sum(L[i][j] * d[i] for i in range(j, D)) for j in range(D)])

nb = 20
bs = N // nb


def worst_error(values):
    """values: one number per draw, in chain order; returns |mean| / mcse"""
    batches = []
    for c in range(C):
        for k in range(nb):
            seg = values[c * N + k * bs:c * N + (k + 1) * bs]
            batches.append(sum(seg) / bs)
    m = sum(batches) / len(batches)
    var = sum((v - m) ** 2 for v in batches) / (len(batches) - 1)
    return abs(m) / (var / len(batches)) ** 0.5


w1 = max(worst_error([z[j] for z in Z]) for j in range(D))
w2 = max(worst_error([z[j] * z[j] - 1.0 for z in Z]) for j in range(D))
wc = max(worst_error([z[i] * z[j] for z in Z]) for i in range(D) for j in range(i + 1, D))
print(f"      worst whitened moment errors in Monte Carlo standard errors: first {w1:.2f}, "
      f"second {w2:.2f} ({D} each), cross {wc:.2f} ({D * (D - 1) // 2})")
sys.exit(0 if w1 < 4 and w2 < 4 and wc < 5 else 1)
