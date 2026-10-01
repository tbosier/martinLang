"""Writes the data of the narrow-data tests to build/, from the fission and
scan test data (run tests/fission/make_data.py and tests/scan/make_data.py
first):

  *_f32.f64       every value rounded to float, so that real-valued data
                  can be read through a float copy;
  fis_c_i16.f64   counts with some above 127 (int16, not int8);
  fis_c_big.f64   counts with one of 70000 (float, not int16);
  fis_c_negz.f64  counts with one -0.0 (float: an integer copy would give +0.0);
  fis_z_edge.f64  float-exact values plus -0.0, the smallest float subnormal,
                  a float subnormal and 1e30 (float);
  fis_z_wide.f64  float-exact values plus one 0.1 (stays double);
  fis_X_i8.f64, fis_X_i16.f64
                  an integer design matrix in int8 range, and with some
                  entries up to 30000 (int16);
  scan_count_61_{i16,big,negz,huge}.f64
                  a count panel with 128 and 32767 (int16), 32768 (float),
                  -0.0 (float) and 2^24 + 1 (stays double).

Format: rows, cols as little-endian u64, then row-major f64."""
import glob
import os
import random
import struct
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "build"


def read(path):
    with open(path, "rb") as f:
        b = f.read()
    r, c = struct.unpack("<QQ", b[:16])
    return r, c, list(struct.unpack(f"<{r * c}d", b[16:]))


def write(name, r, c, vals):
    with open(f"{out}/{name}.f64", "wb") as f:
        f.write(struct.pack("<QQ", r, c))
        f.write(struct.pack(f"<{len(vals)}d", *vals))


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


srcs = sorted(glob.glob(f"{out}/fis_*.f64") + glob.glob(f"{out}/scan_*.f64"))
for p in srcs:
    name = os.path.basename(p)[:-4]
    if name.endswith(("_f32", "_i8", "_i16", "_big", "_negz", "_huge", "_edge", "_wide")):
        continue
    r, c, v = read(p)
    write(name + "_f32", r, c, [f32(x) for x in v])

random.seed(23)
r, c, cnt = read(f"{out}/fis_c.f64")
i16 = list(cnt)
for i in range(0, len(i16), 50):
    i16[i] = float(random.randint(128, 3000))
write("fis_c_i16", r, c, i16)
big = list(cnt)
big[17] = 70000.0
write("fis_c_big", r, c, big)
negz = list(cnt)
negz[3] = -0.0
write("fis_c_negz", r, c, negz)

# an integer-valued design matrix: int8, and int16 (p = 13, so the kernels'
# masked column tails load the integer copies)
r, c, X = read(f"{out}/fis_X.f64")
xi8 = [float(round(v * 30)) + 0.0 for v in X]  # + 0.0: no -0.0
xi8 = [min(127.0, max(-128.0, v)) for v in xi8]
write("fis_X_i8", r, c, xi8)
xi16 = list(xi8)
for i in range(0, len(xi16), 97):
    xi16[i] = float(random.choice([-1, 1]) * random.randint(200, 30000))
write("fis_X_i16", r, c, xi16)

# the same boundaries for a count panel that is a model's only candidate
# (all three types available)
r, c, cnt = read(f"{out}/scan_count_61.f64")
i16 = list(cnt)
i16[5] = 128.0
i16[300] = 32767.0
write("scan_count_61_i16", r, c, i16)
big = list(cnt)
big[7] = 32768.0
write("scan_count_61_big", r, c, big)
negz = list(cnt)
negz[9] = -0.0
write("scan_count_61_negz", r, c, negz)
huge = list(cnt)
huge[11] = 2.0**24 + 1
write("scan_count_61_huge", r, c, huge)

r, c, z = read(f"{out}/fis_z.f64")
z = [f32(x) for x in z]
edge = list(z)
edge[1] = -0.0
edge[2] = 2.0**-149
edge[3] = -(2.0**-140)
edge[4] = 1e30 if f32(1e30) == 1e30 else f32(1e30)
write("fis_z_edge", r, c, edge)
wide = list(z)
wide[500] = 0.1
write("fis_z_wide", r, c, wide)
