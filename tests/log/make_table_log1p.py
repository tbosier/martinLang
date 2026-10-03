"""Generates LOG1P_TAB in compiler/src/ir.rs: -log(1 - m/256) for m = 0..128,
each correctly rounded to double (50-digit decimal arithmetic), for Martin's
log1p on [0, 1] (BernoulliLogit's softplus). Entries 129..255 are 0 (a NaN
input can index them; its result is NaN whatever the entry).

usage: python3 tests/log/make_table_log1p.py   (prints the Rust table)
"""
import struct
from decimal import Decimal, getcontext

getcontext().prec = 50


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


vals = [float(-(1 - Decimal(m) / 256).ln()) if m <= 128 else 0.0 for m in range(256)]
print("const LOG1P_TAB: [u64; 256] = [")
for i in range(0, 256, 4):
    print("    " + " ".join(f"0x{bits(v):016X}," for v in vals[i:i + 4]))
print("];")
