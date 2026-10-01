"""Compares the log density and gradient printed by two builds
(MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 output files): the log densities must
agree to 1e-12 relative and every gradient component to 1e-12 of the
largest. usage: python3 tests/fission/compare.py A.out B.out"""
import math
import sys


def rd(p):
    o = open(p).read()
    return float(o.split("logp=")[1].split()[0]), [float(x) for x in o.split("grad:")[1].split()]


(la, ga), (lb, gb) = rd(sys.argv[1]), rd(sys.argv[2])
if not all(map(math.isfinite, ga + gb + [la, lb])):
    print("      non-finite log density or gradient")
    sys.exit(1)
err = max(abs(a - b) for a, b in zip(ga, gb)) / max(abs(x) for x in gb)
print(f"      logp {la:.15g} vs {lb:.15g}; max grad diff / max |grad| {err:.1e}")
sys.exit(0 if len(ga) == len(gb) and abs(la - lb) <= 1e-12 * abs(lb) and err < 1e-12 else 1)
