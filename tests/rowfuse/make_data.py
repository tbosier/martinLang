"""Writes build/rowfuse_X.f64 (n x p, default 1003 x 13) and build/rowfuse_y.f64
for the row-fusion tests: rows, cols as little-endian u64, then row-major f64.
Usage: make_data.py [n p]"""
import math
import random
import struct
import sys

n, p = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (1003, 13)
random.seed(3 if (n, p) == (1003, 13) else n * 1000 + p)
X = [[random.gauss(0, 1) for _ in range(p)] for _ in range(n)]
beta = [random.gauss(0, 0.5) for _ in range(p)]
y = [float(random.random() < 1 / (1 + math.exp(-sum(a * b for a, b in zip(r, beta))))) for r in X]
with open("build/rowfuse_X.f64", "wb") as f:
    f.write(struct.pack("<QQ", n, p) + struct.pack(f"<{n * p}d", *[v for r in X for v in r]))
with open("build/rowfuse_y.f64", "wb") as f:
    f.write(struct.pack("<QQ", n, 1) + struct.pack(f"<{n}d", *y))
