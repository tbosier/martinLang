"""Writes small panels for the scan-kernel differential tests:
build/scan_{kind}_{G}.f64 for kind in normal, binary, count and G in 7, 13, 20
(T = 11). Format: rows, cols as little-endian u64, then row-major f64."""
import random
import struct
import sys

T = 11
random.seed(5)
out = sys.argv[1] if len(sys.argv) > 1 else "build"
for G in (7, 13, 20):
    for kind in ("normal", "binary", "count"):
        vals = []
        for g in range(G):
            for t in range(T):
                if kind == "normal":
                    vals.append(random.gauss(0.2 * t, 1.0))
                elif kind == "binary":
                    vals.append(float(random.random() < 0.4))
                else:
                    vals.append(float(random.randint(0, 6)))
        with open(f"{out}/scan_{kind}_{G}.f64", "wb") as f:
            f.write(struct.pack("<QQ", G, T))
            f.write(struct.pack(f"<{len(vals)}d", *vals))
