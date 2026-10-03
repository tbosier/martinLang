#!/usr/bin/env python3
"""Simulates a Gaussian random-walk panel from the model in
examples/random_walk_panel.mint and writes it as Martin .f64 files.

usage: make_data.py G T SEED OUTDIR [--shared]

  pop ~ N(0, 1), beta[g] ~ N(pop, 0.4), sigma_w = 0.1, sigma_y = 0.3 (fixed
  true values inside the priors' range), innov[g, t] ~ N(0, sigma_w),
  y[g, t] ~ N(beta[g] + sum_{s <= t} innov[g, s], sigma_y).

With --shared, a common drift shared[t] ~ N(0, 0.05) is added inside the
running sum (as in tests/kalman/shared.mint).

Writes OUTDIR/y.f64 (G x T) and OUTDIR/truth.npz.
"""
import struct
import sys
from pathlib import Path

import numpy as np


def write_f64(path, a):
    a = np.asarray(a, dtype="<f8")
    rows, cols = (a.shape[0], 1) if a.ndim == 1 else a.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", rows, cols))
        f.write(np.ascontiguousarray(a).tobytes())


def main():
    G, T, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    shared_on = "--shared" in sys.argv[5:]
    rng = np.random.default_rng(seed)
    pop = rng.normal(0, 1)
    beta = rng.normal(pop, 0.4, size=G)
    sigma_w, sigma_y = 0.1, 0.3
    innov = rng.normal(0, sigma_w, size=(G, T))
    shared = rng.normal(0, 0.05, size=T) if shared_on else np.zeros(T)
    state = np.cumsum(shared[None, :] + innov, axis=1)
    y = beta[:, None] + state + rng.normal(0, sigma_y, size=(G, T))
    out.mkdir(parents=True, exist_ok=True)
    write_f64(out / "y.f64", y)
    np.savez(out / "truth.npz", pop=pop, beta=beta, sigma_w=sigma_w, sigma_y=sigma_y, innov=innov, shared=shared, y=y)


if __name__ == "__main__":
    main()
