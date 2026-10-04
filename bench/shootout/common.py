"""Shared paths and helpers for the shootout run scripts (instrumentation
only: none of this is counted as model code)."""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "bench", "dynpois", "data_large")
BUILD = os.path.join(ROOT, "build", "shootout")
DRAWS_DIR = os.path.join(BUILD, "draws")
RESULTS = os.path.join(HERE, "results")
CHAINS, WARMUP, DRAWS = 4, 1000, 1000


def cache_env():
    """Caches and temporary files inside the worktree (the sandbox this was
    run in has a read-only home directory)."""
    cache = os.path.join(ROOT, "build", "cache")
    env = {
        "XDG_CACHE_HOME": cache,
        "MPLCONFIGDIR": os.path.join(cache, "mpl"),
        "NUMBA_CACHE_DIR": os.path.join(cache, "numba"),
        "PYTENSOR_FLAGS": f"base_compiledir={os.path.join(cache, 'pytensor')}",
        "TMPDIR": os.path.join(ROOT, ".tmp"),
    }
    for k in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "TMPDIR"):
        os.makedirs(env[k], exist_ok=True)
    return env


def load_y():
    return np.load(os.path.join(DATA, "y.npy"))


class Phases:
    """Records named phase durations (wall seconds) and extra facts, and
    writes them to $SHOOTOUT_PHASES for measure.py to merge."""

    def __init__(self):
        self.d = {"phases": {}, "extra": {}}

    def time(self, name):
        ph = self

        class _T:
            def __enter__(self):
                self.t = time.perf_counter()
                return self

            def __exit__(self, *a):
                ph.d["phases"][name] = ph.d["phases"].get(name, 0.0) + time.perf_counter() - self.t

        return _T()

    def set(self, **kw):
        self.d["extra"].update(kw)

    def write(self):
        path = os.environ.get("SHOOTOUT_PHASES")
        if path:
            with open(path, "w") as f:
                json.dump(self.d, f, indent=1)
        print(json.dumps(self.d, indent=1, default=str))


def save_draws(name, pop, beta, terminal):
    """pop (chains, draws); beta, terminal (chains, draws, G)."""
    os.makedirs(DRAWS_DIR, exist_ok=True)
    pop, beta, terminal = (np.asarray(a, dtype=float) for a in (pop, beta, terminal))
    G = load_y().shape[0]
    assert pop.shape == (CHAINS, pop.shape[1]) and beta.shape == terminal.shape == (CHAINS, pop.shape[1], G), \
        (pop.shape, beta.shape, terminal.shape)
    path = os.path.join(DRAWS_DIR, name + ".npz")
    np.savez(path, pop=pop, beta=beta, terminal=terminal)
    return path
