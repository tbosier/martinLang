"""Shared Stan setup: CmdStan location, model compilation, data loading."""
import os
import shutil
import time

BUILD = "/home/taylo/projects/testRandomShiz/mint/build"
os.environ.setdefault("TMPDIR", os.path.join(BUILD, "tmp"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(BUILD, "cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BUILD, "cache", "mpl"))
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import cmdstanpy  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CMDSTAN = os.path.join(BUILD, "cmdstan", "cmdstan-2.40.0")
STAN_DIR = os.path.join(BUILD, "stan")
cmdstanpy.set_cmdstan_path(CMDSTAN)


def compile_model(force=False):
    """Copy dynpois.stan to the build dir (so the executable lands there) and
    compile it. Returns (model, seconds the (last) compilation took)."""
    os.makedirs(STAN_DIR, exist_ok=True)
    src = os.path.join(HERE, "dynpois.stan")
    dst = os.path.join(STAN_DIR, "dynpois.stan")
    with open(src) as f:
        new = f.read()
    changed = not os.path.exists(dst) or open(dst).read() != new
    if changed:
        shutil.copyfile(src, dst)
    exe = os.path.join(STAN_DIR, "dynpois")
    need = force or changed or not os.path.exists(exe)
    t0 = time.perf_counter()
    model = cmdstanpy.CmdStanModel(stan_file=dst, force_compile=need)
    record = os.path.join(STAN_DIR, "compile_seconds.txt")
    if need:
        seconds = time.perf_counter() - t0
        with open(record, "w") as f:
            f.write(f"{seconds:.3f}\n")
        return model, seconds
    # Cached executable: report the time its last compilation took.
    return model, float(open(record).read()) if os.path.exists(record) else None


def load_data(size):
    y = np.load(os.path.join(HERE, f"data_{size}", "y.npy"))
    G, T = y.shape
    return y, {"G": G, "T": T, "y": y.astype(int).tolist()}
