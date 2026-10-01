"""Data and exact posterior for tests/metric/gauss.mint.

x ~ Normal(0, I) in d = 70 dimensions and b ~ Normal(A x, 1) with 8 rows of A
in random dense directions with large norms, so the posterior is a Gaussian
that is 4 to 40 times narrower than the prior along 8 directions that are not
aligned with the axes: a diagonal metric cannot absorb them, a low-rank one
can. The posterior precision is P = I + A'A and the mean P^-1 A'b. 70 is
at least 64 (so the sampler's eight-tile kernels run) and not a multiple of 8
(so the partial last tile is exercised too).

Writes build/gauss_A.f64, build/gauss_b.f64 (rows, cols as little-endian u64,
then row-major f64) and build/gauss_exact.txt (the mean on the first line,
then the rows of the Cholesky factor L of P, P = L L').
"""
import random
import struct

random.seed(11)
d = 70
scales = [40, 30, 20, 15, 10, 8, 6, 4]
A = []
for s in scales:
    v = [random.gauss(0, 1) for _ in range(d)]
    n = sum(x * x for x in v) ** 0.5
    A.append([s * x / n for x in v])
m = len(A)
b = [random.gauss(0, 1) * s / 4 for s in scales]

P = [[float(i == j) + sum(A[r][i] * A[r][j] for r in range(m)) for j in range(d)] for i in range(d)]
L = [[0.0] * d for _ in range(d)]
for j in range(d):
    for i in range(j, d):
        s = P[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
        L[i][j] = s ** 0.5 if i == j else s / L[j][j]
rhs = [sum(A[r][i] * b[r] for r in range(m)) for i in range(d)]
# solve L L' mu = rhs
y = [0.0] * d
for i in range(d):
    y[i] = (rhs[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
mu = [0.0] * d
for i in reversed(range(d)):
    mu[i] = (y[i] - sum(L[k][i] * mu[k] for k in range(i + 1, d))) / L[i][i]

with open("build/gauss_A.f64", "wb") as f:
    f.write(struct.pack("<QQ", m, d) + struct.pack(f"<{m * d}d", *[v for r in A for v in r]))
with open("build/gauss_b.f64", "wb") as f:
    f.write(struct.pack("<QQ", m, 1) + struct.pack(f"<{m}d", *b))
with open("build/gauss_exact.txt", "w") as f:
    f.write(" ".join(repr(v) for v in mu) + "\n")
    for r in L:
        f.write(" ".join(repr(v) for v in r) + "\n")
