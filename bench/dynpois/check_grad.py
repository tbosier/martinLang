#!/usr/bin/env python3
"""Checks a dynamic Poisson log density and gradient (printed by a program run
with MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1) against the exact formula in SPEC.md,
at the runtime's fixed benchmark point. usage: check_grad.py DATA_DIR < program_output"""
import sys

import numpy as np

y = np.load(sys.argv[1] + "/y.npy")
G, T = y.shape
D = 1 + G + T + G * T
i = np.arange(D)
th = 0.05 * (((i * 37) % 11) - 5) / 5.0
pop, beta, sh = th[0], th[1:1 + G], th[1 + G:1 + G + T]
inn = th[1 + G + T:].reshape(G, T)
eta = beta[:, None] + np.cumsum(sh[None, :] + inn, axis=1)
e = np.exp(eta)
lp = (-0.5 * pop ** 2 + np.sum(-0.5 * ((beta - pop) / 0.4) ** 2 - np.log(0.4))
      + np.sum(-0.5 * (sh / 0.05) ** 2 - np.log(0.05)) + np.sum(-0.5 * (inn / 0.08) ** 2 - np.log(0.08))
      + np.sum(y * eta - e))
d = y - e
rc = np.cumsum(d[:, ::-1], axis=1)[:, ::-1]
ref = np.concatenate([[-pop + np.sum((beta - pop) / 0.16)], d.sum(1) - (beta - pop) / 0.16,
                      rc.sum(0) - sh / 0.0025, (rc - inn / 0.0064).ravel()])
out = sys.stdin.read()
got_lp = float(out.split("logp=")[1].split()[0])
got = np.array([float(x) for x in out.split("grad:")[1].split()])
ok = len(got) == D and abs(got_lp - lp) <= 1e-9 * abs(lp) and np.max(np.abs(got - ref) / np.maximum(1, np.abs(ref))) < 1e-9
print(f"lp {got_lp} vs {lp}; max grad rel diff {np.max(np.abs(got - ref) / np.maximum(1, np.abs(ref))):.2e}")
sys.exit(0 if ok else 1)
