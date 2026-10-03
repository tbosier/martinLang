"""Generates the table of Martin's vector log (LOG_TAB in compiler/src/ir.rs).

x = 2^k z with z in [0x3FE5F00000000000, 0x3FF5F00000000000) as bit patterns
(about [0.6875, 1.375)); that range is split into 128 equal steps of the bit
pattern, and step j has a point c_j with 1/c_j = invc_j (a double) and
logc_j = -log(invc_j) rounded to double. The step containing 1.0 has c = 1
exactly (invc 1, logc 0), so log x near 1 has no cancellation. Elsewhere
invc is the double nearest 1/center moved by up to 2^32 ulps (a relative
change of at most 2^-20, which barely changes the range of r = z invc - 1):
of 4096 pseudo-random candidates, the one whose -log(invc) is closest to a
double (so logc is nearly exact). Small moves alone do not work for round
centers such as 255/256, where one ulp of invc moves logc by an exact number
of its own ulps.

usage: python3 tests/log/make_table.py   (prints the Rust table)
"""
import random
import struct
from decimal import Decimal, getcontext

getcontext().prec = 50
OFF = 0x3FE5F00000000000
N = 128
STEP = 1 << 45
T = 4096
rng = random.Random(5)


def d(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def ulp(x):
    b = bits(abs(x))
    return d(b + 1) - d(b)


worst = 0
rows = []
for j in range(N):
    a, b = d(OFF + j * STEP), d(OFF + (j + 1) * STEP)
    if a <= 1.0 < b:
        rows.append((1.0, 0.0))
        continue
    m = (Decimal(a) + Decimal(b)) / 2
    base = bits(float(1 / m))
    best = None
    for t in [0] + [rng.randint(-(1 << 32), 1 << 32) for _ in range(T)]:
        inv = d(base + t)
        lc = -Decimal(inv).ln()
        f = float(lc)
        err = abs(Decimal(f) - lc) / Decimal(ulp(f))
        if best is None or err < best[0]:
            best = (err, inv, f)
    worst = max(worst, best[0])
    rows.append((best[1], best[2]))

print(f"// worst rounding error of logc: {float(worst):.2e} ulp")
print("const LOG_TAB: [(u64, u64); 128] = [")
for inv, lc in rows:
    print(f"    (0x{bits(inv):016X}, 0x{bits(lc):016X}),")
print("];")
