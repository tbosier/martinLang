"""Generate the hierarchical dynamic Poisson benchmark data (see SPEC.md).

Usage: python make_data.py G T SEED OUTDIR

Writes OUTDIR/y.f64 (Martin format: u64 rows, u64 cols, little-endian, then
row-major little-endian f64), OUTDIR/y.npy (same array, shape (G, T)) and
OUTDIR/truth.json (pop, beta[G], terminal state[G] = state[g, T]).

Random draws are taken from numpy default_rng(SEED) in this fixed order:
beta deviations (G), shared innovations (T), group innovations (G, T),
Poisson counts (G, T).
"""
import json
import os
import sys

import numpy as np

POP = 1.5
GROUP_SD = 0.4
SHARED_SD = 0.05
PROCESS_SD = 0.08


def write_f64(path, a):
    a = np.ascontiguousarray(a, dtype="<f8")
    rows, cols = a.shape
    with open(path, "wb") as f:
        f.write(np.array([rows, cols], dtype="<u8").tobytes())
        f.write(a.tobytes(order="C"))


def main():
    if len(sys.argv) != 5:
        sys.exit("usage: make_data.py G T SEED OUTDIR")
    G, T, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    outdir = sys.argv[4]
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(seed)
    beta = POP + rng.normal(0.0, GROUP_SD, G)
    shared = rng.normal(0.0, SHARED_SD, T)
    innov = rng.normal(0.0, PROCESS_SD, (G, T))
    # state[g, t] = sum_{s <= t} (shared[s] + innov[g, s]); innovation at t included.
    state = np.cumsum(shared[None, :] + innov, axis=1)
    y = rng.poisson(np.exp(beta[:, None] + state)).astype(np.float64)

    write_f64(os.path.join(outdir, "y.f64"), y)
    np.save(os.path.join(outdir, "y.npy"), y)
    truth = {
        "G": G,
        "T": T,
        "seed": seed,
        "pop": POP,
        "beta": beta.tolist(),
        "terminal": state[:, -1].tolist(),
    }
    with open(os.path.join(outdir, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    print(f"wrote {outdir}: y shape {y.shape}, total count {y.sum():.0f}, "
          f"max {y.max():.0f}, zeros {(y == 0).mean():.3f}")


if __name__ == "__main__":
    main()
