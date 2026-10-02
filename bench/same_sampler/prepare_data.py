#!/usr/bin/env python3
"""Writes the Stan (JSON) copies of the benchmark data, from the same files
Mint and the Rust baselines read, to build/same_sampler/data/.

usage: python bench/same_sampler/prepare_data.py   (from the repository root)
"""
import json
import os

import numpy as np

OUT = os.path.join("build", "same_sampler", "data")


def read_f64(path):
    raw = open(path, "rb").read()
    rows, cols = np.frombuffer(raw[:16], dtype="<u8")
    return np.frombuffer(raw[16:], dtype="<f8").reshape(int(rows), int(cols))


def main():
    os.makedirs(OUT, exist_ok=True)
    for size in ("small", "large"):
        y = read_f64(f"bench/dynpois/data_{size}/y.f64")
        assert np.all(y == np.round(y)) and np.all(y >= 0)
        G, T = y.shape
        json.dump({"G": G, "T": T, "y": y.astype(int).tolist()}, open(f"{OUT}/dynpois_{size}.json", "w"))
    X = read_f64("data/logit_X.f64")
    y = read_f64("data/logit_y.f64").ravel()
    assert set(np.unique(y)) <= {0.0, 1.0}
    n, p = X.shape
    # repr() of a Python float round-trips exactly, so Stan reads the same doubles
    json.dump({"n": n, "p": p, "X": X.tolist(), "y": y.astype(int).tolist()}, open(f"{OUT}/logistic.json", "w"))
    # examples/eight_schools.mint
    json.dump({"J": 8, "y": [28, 8, -3, 7, -1, 1, 18, 12], "s": [15, 10, 16, 11, 9, 11, 10, 18]},
              open(f"{OUT}/eight_schools.json", "w"))
    print(f"wrote {OUT}/dynpois_small.json, dynpois_large.json, logistic.json, eight_schools.json")


if __name__ == "__main__":
    main()
