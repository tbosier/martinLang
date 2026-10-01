"""Randomised check of narrow data: models whose linear predictors mix data
and parameters in several ways (so that LLVM has several multiplies it could
fuse into an add), on random data exact in float or int8, each run with the
narrow copies (MINT_NARROW unset) and without them (MINT_NARROW=0). The
exact log density, every gradient component at the benchmark point and,
the raw draws of a short sampling run (which amplify any difference in
the last bit anywhere along the trajectory)
must be identical, and the report must show that a narrow copy was used.

usage: fuzz.py MINTC DATASETS_PER_MODEL SEED   (run from the repository root;
writes to build/). Exits 1 on any difference."""
import os
import random
import struct
import subprocess
import sys

mintc, nsets, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = random.Random(seed)
B = "build"

# (name, source); data files are build/fz_<name>_<data>.f64
VEC = {
    "rv": """model M {
    data y: Vector[n]
    data X: Matrix[n, p]
    param beta: Vector[p]
    param a: Real
    param b: Real
    y ~ Normal(a * X * beta + b * y, exp(X * beta))
}""",
    "rv2": """model M {
    data y: Vector[n]
    data X: Matrix[n, p]
    data z: Vector[n]
    param beta: Vector[p]
    param s: Vector[p]
    param a: Real
    param b: Real
    a ~ Normal(0, 1)
    y ~ Normal(a * z + X * beta * b + z .* z * 0.3, exp(X * s) + 0.5)
}""",
    "bl": """model M {
    data y: Vector[n]
    data X: Matrix[n, p]
    data z: Vector[n]
    data w0: Vector[p]
    param a: Real
    param c: Real
    param b: Vector[p]
    b ~ Normal(0, 1)
    y ~ BernoulliLogit(a * (X * w0) + X * b + c * z - z .* z)
}""",
    "po": """model M {
    data y: Vector[n]
    data X: Matrix[n, p]
    data z: Vector[n]
    param a: Real
    param b: Vector[p]
    y ~ PoissonLog(a + X * b + 0.5 * z * a)
}""",
    "ex": """model M {
    data y: Positive[n]
    data X: Matrix[n, p]
    data z: Vector[n]
    param a: Real
    param b: Vector[p]
    y ~ Exponential(exp(X * b + z * a))
}""",
}
SCAN = {
    "sn": """model M {
    data y: Matrix[G, T]
    data Z: Matrix[G, T]
    param beta: Vector[G]
    param c: Real
    param s: Positive
    param innov: Matrix[G, T]
    innov ~ Normal(0, 0.3)
    y ~ Normal(beta + cumsum(innov, T) + c * Z, s)
}""",
    "sp": """model M {
    data y: Matrix[G, T]
    data Z: Matrix[G, T]
    param beta: Vector[G]
    param shared: Vector[T]
    param d: Real
    param innov: Matrix[G, T]
    innov ~ Normal(Z * 0.1, 0.2)
    y ~ PoissonLog(beta + cumsum(shared + innov + d * Z, T))
}""",
}


def write(name, r, c, vals):
    with open(f"{B}/{name}.f64", "wb") as f:
        f.write(struct.pack("<QQ", r, c))
        f.write(struct.pack(f"<{len(vals)}d", *vals))


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def real(k, scale=1.0):
    # float-exact reals, or small integers, chosen per buffer
    if rng.random() < 0.25:
        return [float(rng.randint(-3, 3)) for _ in range(k)]
    return [f32(rng.gauss(0, scale)) for _ in range(k)]


def main_fn(model, reads, ctor):
    return model + "\n\nfn main() {\n" + reads + f"    let post = sample(M({ctor}), draws = 10, warmup = 10, chains = 1, seed = 3)\n    print(post)\n}}\n"


def run(binary, env):
    e = dict(os.environ, MINT_BENCH_GRAD="1", MINT_PRINT_GRAD="1", MINT_NARROW_REPORT="1", **env)
    p = subprocess.run([f"./{B}/{binary}"], env=e, capture_output=True, text=True)
    out = [l for l in p.stdout.splitlines() if l.startswith(("exact log density", "grad:"))]
    rep = [l for l in p.stderr.splitlines() if l.startswith("narrow:")]
    return p.returncode, out, rep


def draws(binary, env):
    path = f"{B}/{binary}.draws"
    if os.path.exists(path):
        os.remove(path)
    e = dict(os.environ, MINT_DRAWS=path, **env)
    p = subprocess.run([f"./{B}/{binary}"], env=e, capture_output=True)
    if p.returncode != 0 or not os.path.exists(path):
        return None
    return open(path, "rb").read()


bad = 0
checked = 0
for name, model in list(VEC.items()) + list(SCAN.items()):
    scan = name in SCAN
    pre = f"fz_{name}"
    if scan:
        reads = (f'    let y: Matrix[G, T] = read("{B}/{pre}_y.f64")\n'
                 f'    let Z: Matrix[G, T] = read("{B}/{pre}_Z.f64")\n')
        ctor = "y, Z"
    else:
        names = [d for d in ("y", "X", "z", "w0") if f"data {d}:" in model]
        ty = {"y": "Positive[n]" if "Positive[n]" in model else "Vector[n]", "X": "Matrix[n, p]", "z": "Vector[n]", "w0": "Vector[p]"}
        reads = "".join(f'    let {d}: {ty[d]} = read("{B}/{pre}_{d}.f64")\n' for d in names)
        ctor = ", ".join(names)
    src = f"{B}/{pre}.mint"
    open(src, "w").write(main_fn(model, reads, ctor))
    # shapes are fixed per binary only through the data, so one build serves every data set
    if subprocess.run([mintc, "build", src, "-o", f"{B}/{pre}"], capture_output=True).returncode != 0:
        print(f"FAIL  fuzz {name}: build failed")
        bad += 1
        continue
    for k in range(nsets):
        if scan:
            G, T = rng.choice([5, 8, 13, 17]), rng.choice([3, 7, 11])
            if name == "sp":
                write(f"{pre}_y", G, T, [float(rng.randint(0, 9)) for _ in range(G * T)])
            else:
                write(f"{pre}_y", G, T, real(G * T))
            write(f"{pre}_Z", G, T, real(G * T, 0.5))
        else:
            n, p = rng.choice([4, 5, 8, 9, 13, 37]), rng.randint(1, 6)
            if name == "bl":
                y = [float(rng.random() < 0.4) for _ in range(n)]
            elif name == "po":
                y = [float(rng.randint(0, 9)) for _ in range(n)]
            elif name == "ex":
                y = [f32(rng.expovariate(1.0)) + 2.0**-20 for _ in range(n)]
            else:
                y = real(n)
            write(f"{pre}_y", n, 1, y)
            write(f"{pre}_X", n, p, real(n * p))
            write(f"{pre}_z", n, 1, real(n, 0.5))
            write(f"{pre}_w0", p, 1, real(p, 0.3))
        rc1, a, rep = run(pre, {})
        rc2, b, _ = run(pre, {"MINT_NARROW": "0"})
        checked += 1
        if rc1 or rc2 or len(a) != 2 or a != b:
            bad += 1
            print(f"FAIL  fuzz {name} data set {k}: narrow and wide differ (or failed)")
            print("      " + " | ".join(a))
            print("      " + " | ".join(b))
        elif not any(not l.endswith(": double") for l in rep):
            bad += 1
            print(f"FAIL  fuzz {name} data set {k}: no narrow copy used ({rep})")
        da, db = draws(pre, {}), draws(pre, {"MINT_NARROW": "0"})
        if da is None or da != db:
            bad += 1
            print(f"FAIL  fuzz {name} data set {k}: raw draws differ (or the run failed)")
print(f"fuzz: {checked} data sets over {len(VEC) + len(SCAN)} models, {bad} failures")
sys.exit(1 if bad else 0)
