"""Writes the data of the fission-kernel differential tests to build/:
n = 1003 rows (not a multiple of the kernel's chunk or of 4) and p = 13
columns (not a multiple of 4). fis_Xbig.f64 has two rows with one huge entry,
so that at the benchmark point |eta| > 708 there and the kernel's exp takes
its full-range path. Format: rows, cols as little-endian u64, then
row-major f64."""
import random
import struct
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "build"
n, p = 1003, 13
random.seed(17)


def write(name, rows, cols, vals):
    with open(f"{out}/{name}.f64", "wb") as f:
        f.write(struct.pack("<QQ", rows, cols))
        f.write(struct.pack(f"<{len(vals)}d", *vals))


X = [random.gauss(0, 1) for _ in range(n * p)]
write("fis_X", n, p, X)
big = list(X)
for i in (5, 700):
    for k in range(p):
        big[i * p + k] = 0.0
    big[i * p] = 1e5  # beta_0 is -0.01 at the benchmark point: eta = -1000
write("fis_Xbig", n, p, big)
write("fis_y", n, 1, [float(random.random() < 0.4) for _ in range(n)])
write("fis_c", n, 1, [float(random.randint(0, 5)) for _ in range(n)])
write("fis_z", n, 1, [random.gauss(0, 1.5) for _ in range(n)])
write("fis_w", n, 1, [random.expovariate(1.0) for _ in range(n)])
write("fis_off", n, 1, [random.uniform(-1, 1) for _ in range(n)])
write("fis_w0", p, 1, [random.gauss(0, 0.3) for _ in range(p)])
