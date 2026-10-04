#!/usr/bin/env python3
"""Checks that every implementation computes the same model before anything
is timed: the log density and its gradient at two points, against the exact
formula of bench/dynpois/SPEC.md (bench/dynpois/spec_logdensity.py), whose
agreement with Martin's compiled model is itself checked here.

Points (unconstrained, SPEC's theta order; every parameter here is
unconstrained, so constrained = unconstrained):
  P1  the runtime's benchmark point, theta_i = 0.05 * ((37 i mod 11) - 5) / 5;
  P2  a point near the posterior: pop = 1.5 + 0.2 z, beta = pop + 0.4 z,
      shared = 0.05 z, innov = 0.08 z (numpy default_rng(2026)).

Frameworks drop different constants (Stan drops all of them, PyMC and NumPyro
keep the Poisson's log(y!) and the Normals' log(2 pi)), so log densities are
compared through lp(P1) - lp(P2), and gradients component by component
(relative error |g - ref| / max(|ref|, 1)). rustmc exposes no log density;
it is checked by posterior means only (analyze.py).

Also times one gradient per framework at P2 (single-threaded unless stated),
for context only, and compares the two gradient backends of nutpie's JAX path.

usage: .venv/bin/python bench/shootout/verify.py   (writes results/verify.json)
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

os.environ.update(common.cache_env())
os.environ.setdefault("BRIDGESTAN", os.path.join(common.ROOT, "build", "bs", "bridgestan-2.9.0"))
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(common.ROOT, "bench", "dynpois"))
import spec_logdensity as spec  # noqa: E402

os.chdir(common.ROOT)
V = os.path.join(common.BUILD, "verify")
os.makedirs(V, exist_ok=True)
y = common.load_y()
G, T = y.shape
D = 1 + G + T + G * T
i = np.arange(D)
P1 = 0.05 * (((i * 37) % 11) - 5) / 5.0
rng = np.random.default_rng(2026)
pop = 1.5 + 0.2 * rng.standard_normal()
P2 = np.concatenate([[pop], pop + 0.4 * rng.standard_normal(G), 0.05 * rng.standard_normal(T),
                     0.08 * rng.standard_normal(G * T)])
POINTS = {"P1": P1, "P2": P2}
for k, th in POINTS.items():
    with open(os.path.join(V, k + ".f64"), "wb") as f:
        np.array([D, 1], dtype="<u8").tofile(f)
        th.astype("<f8").tofile(f)
ref = {k: (spec.log_density(th, y), spec.gradient(th, y)) for k, th in POINTS.items()}
results = {"points": {"P1": "runtime benchmark point", "P2": "near-posterior point, default_rng(2026)"},
           "reference": {k: v[0] for k, v in ref.items()}, "frameworks": {}}


def record(name, vals, note="", us=None):
    """vals: {point: (lp, grad in SPEC order)}"""
    d_ref = ref["P1"][0] - ref["P2"][0]
    d = vals["P1"][0] - vals["P2"][0]
    gerr = max(float(np.max(np.abs(vals[k][1] - ref[k][1]) / np.maximum(1, np.abs(ref[k][1])))) for k in POINTS)
    rec = {"lp_P1": vals["P1"][0], "lp_P2": vals["P2"][0], "constant_offset_P1": vals["P1"][0] - ref["P1"][0],
           "constant_offset_P2": vals["P2"][0] - ref["P2"][0],
           "lp_difference_rel_error": abs(d - d_ref) / abs(d_ref), "max_grad_rel_error": gerr,
           "us_per_gradient": us, "note": note}
    rec["ok"] = bool(rec["lp_difference_rel_error"] < 1e-9 and gerr < 1e-9)
    results["frameworks"][name] = rec
    print(f"{name:28s} lp diff rel err {rec['lp_difference_rel_error']:.2e}  grad rel err {gerr:.2e}  "
          f"offset {rec['constant_offset_P1']:.6g} / {rec['constant_offset_P2']:.6g}  "
          f"{'' if us is None else f'{us:.1f} us/grad'}  {'OK' if rec['ok'] else 'MISMATCH'}", flush=True)


def runtime_grad(cmd, k, env_extra=None):
    env = dict(os.environ, MINT_BENCH_GRAD="200", MINT_PRINT_GRAD="1", MINT_THETA=os.path.join(V, k + ".f64"),
               MINT_KERNEL_THREADS="1", **(env_extra or {}))
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
    lp = float(out.split("exact log density:")[1].split()[0])
    g = np.array([float(x) for x in out.split("grad:")[1].split("\n")[0].split()])
    ns = float(out.split("ns_per_eval=")[1].split()[0])
    return lp, g, ns / 1000


def safe(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        results["frameworks"][name] = {"ok": False, "error": repr(e)}
        print(f"{name}: FAILED {e!r}", flush=True)


# --- Martin
def martin():
    src = open("examples/dynamic_poisson.mint").read().replace("data_small", "data_large")
    prog = os.path.join(V, "martin")
    open(prog + ".mint", "w").write(src)
    subprocess.run([os.path.join(common.ROOT, "compiler/target/release/mintc"), "build", prog + ".mint", "-o", prog],
                   check=True)
    vals, us = {}, None
    for k in POINTS:
        lp, g, us = runtime_grad([prog], k)
        vals[k] = (lp, g)
    record("martin", vals, "Martin-compiled model, MINT_BENCH_GRAD at MINT_THETA, 1 kernel thread", us)


# --- Rust gradient under Martin's runtime
def rust_martin():
    prog = os.path.join(V, "rs_dynpois_par")
    subprocess.run(["rustc", "+nightly", "--edition", "2021", "-C", "opt-level=3", "-C", "target-cpu=native",
                    "baselines/dynpois_par.rs", "-o", prog, "-C", f"link-arg={common.ROOT}/build/mint_rt.o",
                    "-C", "link-arg=-lomp", "-l", "m", "-l", "mvec"], check=True)
    vals, us = {}, None
    for k in POINTS:
        lp, g, us = runtime_grad([prog, os.path.join(common.DATA, "y.f64")], k)
        vals[k] = (lp, g)
    record("rust_under_martin", vals, "baselines/dynpois_par.rs, 1 kernel thread", us)


# --- Rust with nuts-rs
def rust_nuts():
    subprocess.run(["bash", "bench/shootout/rust_nuts/build.sh", "--offline"], check=True)
    exe = os.path.join(common.ROOT, "build/nutsrs/target/release/rust_nuts")
    for nt in (1, 3):
        vals, us = {}, None
        for k in POINTS:
            out = subprocess.run([exe, "grad", os.path.join(common.DATA, "y.f64"), os.path.join(V, k + ".f64")],
                                 env=dict(os.environ, DYNPOIS_THREADS=str(nt)), capture_output=True, text=True,
                                 check=True).stdout
            lp = float(out.split("logp")[1].split()[0])
            g = np.array([float(x) for x in out.split("grad:")[1].split()])
            us = float(out.split("us_per_gradient")[1].split()[0])
            vals[k] = (lp, g)
        record(f"rust_nuts_{nt}thread", vals, f"bench/shootout/rust_nuts, gradient on {nt} thread(s)", us)


def stan_params(th):
    return {"pop": float(th[0]), "beta": th[1:1 + G].tolist(), "shared": th[1 + G:1 + G + T].tolist(),
            "innov": th[1 + G + T:].reshape(G, T).tolist()}


# --- CmdStan (both programs)
def cmdstan():
    sys.path.insert(0, os.path.join(common.HERE, "models"))
    import dynpois_cmdstan as m
    import shutil
    data_file = os.path.join(V, "data.json")
    json.dump(m.load_data(), open(data_file, "w"))
    for variant, src, threads in [("plain", "bench/dynpois/dynpois.stan", False),
                                  ("reduce_sum", "bench/shootout/models/dynpois_reduce_sum.stan", True)]:
        d = os.path.join(V, "cmdstan_" + variant)
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(src, os.path.join(d, "dynpois.stan"))
        model = m.compile_model(os.path.join(d, "dynpois.stan"), threads)
        vals = {}
        for k, th in POINTS.items():
            pf = os.path.join(V, f"params_{k}.json")
            json.dump(stan_params(th), open(pf, "w"))
            df = model.log_prob(pf, data_file, jacobian=True, sig_figs=18)
            vals[k] = (float(df["lp__"].iloc[0]), df.iloc[0, 1:].to_numpy(dtype=float))
        record(f"cmdstan_{variant}", vals, "cmdstanpy log_prob (CmdStan log_prob method, jacobian, propto)")


# --- BridgeStan library as nutpie builds it
def bridgestan():
    import bridgestan as bs
    import shutil
    d = os.path.join(V, "bridgestan")
    os.makedirs(d, exist_ok=True)
    shutil.copyfile("bench/dynpois/dynpois.stan", os.path.join(d, "dynpois.stan"))
    so = bs.compile_model(os.path.join(d, "dynpois.stan"), stanc_args=["--O1"],
                          make_args=["STAN_THREADS=true", "CXXFLAGS=-march=native"])
    data = json.dumps({"G": G, "T": T, "y": y.astype(int).tolist()})
    model = bs.StanModel(str(so), data)
    vals = {}
    for k, th in POINTS.items():
        lp, g = model.log_density_gradient(th, propto=True, jacobian=True)
        vals[k] = (float(lp), np.array(g))
    t = time.perf_counter()
    for _ in range(200):
        model.log_density_gradient(P2, propto=True, jacobian=True)
    us = 1e6 * (time.perf_counter() - t) / 200
    record("nutpie_stan (BridgeStan)", vals, "BridgeStan 2.9.0 library built with nutpie's flags "
           "(stanc --O1, STAN_THREADS, CXXFLAGS=-march=native); log_density_gradient(propto, jacobian)", us)


# --- PyMC (numba and JAX compiled log density and gradient)
def pymc():
    import jax
    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, os.path.join(common.HERE, "models"))
    import dynpois_pymc as m
    import pytensor
    model = m.make_model(m.load_data())
    names = [v.name for v in model.value_vars]
    assert names == ["pop", "beta", "shared", "innov"], names

    def split(th):
        return {"pop": th[0], "beta": th[1:1 + G], "shared": th[1 + G:1 + G + T],
                "innov": th[1 + G + T:].reshape(G, T)}

    for mode in ("NUMBA", "JAX"):
        logp = model.compile_logp(jacobian=True, mode=mode)
        dlogp = model.compile_dlogp(jacobian=True, mode=mode)
        vals = {}
        for k, th in POINTS.items():
            vals[k] = (float(logp(split(th))), np.asarray(dlogp(split(th)), dtype=float))
        pt = split(P2)
        dlogp(pt)
        t = time.perf_counter()
        for _ in range(200):
            dlogp(pt)
        us = 1e6 * (time.perf_counter() - t) / 200
        record(f"pymc_{mode.lower()}", vals, f"PyMC model.compile_logp/compile_dlogp(jacobian=True, mode='{mode}'); "
               "the timing includes PyTensor's Python call overhead", us)
    # nutpie's JAX path: gradient by PyTensor (compiled to JAX) or by jax.grad of the JAX log density
    from pytensor.link.jax.dispatch import jax_funcify  # noqa: F401
    import jax.numpy as jnp
    vars_ = model.value_vars
    lp_graph = model.logp(jacobian=True)
    grads = pytensor.grad(lp_graph, vars_)
    f_lp = pytensor.function(vars_, lp_graph, mode="JAX")
    f_both = pytensor.function(vars_, [lp_graph] + grads, mode="JAX")
    jlp = f_lp.vm.jit_fn if hasattr(f_lp.vm, "jit_fn") else None
    pt = split(P2)
    args = [pt[n] for n in names]
    f_both(*args)
    t = time.perf_counter()
    for _ in range(200):
        f_both(*args)
    us_pt = 1e6 * (time.perf_counter() - t) / 200
    results["pymc_jax_gradient_backends"] = {"pytensor_us_per_gradient": us_pt,
                                             "note": "PyTensor-derived gradient compiled to JAX, called through "
                                                     "pytensor.function; jax.grad variant timed below if available"}
    if jlp is not None:
        g = jax.jit(jax.value_and_grad(lambda *a: jlp(*a)[0], argnums=(0, 1, 2, 3)))
        ja = [jnp.asarray(a) for a in args]
        jax.block_until_ready(g(*ja))
        t = time.perf_counter()
        for _ in range(200):
            jax.block_until_ready(g(*ja))
        results["pymc_jax_gradient_backends"]["jax_grad_us_per_gradient"] = 1e6 * (time.perf_counter() - t) / 200
    print("pymc jax gradient backends:", results["pymc_jax_gradient_backends"], flush=True)


# --- NumPyro
def numpyro_():
    sys.path.insert(0, os.path.join(common.HERE, "models"))
    import dynpois_numpyro as m
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_density
    yy = jnp.asarray(m.load_data())

    def lp(params):
        return log_density(m.model, (yy,), {}, params)[0]

    vg = jax.jit(jax.value_and_grad(lp))
    vals = {}
    for k, th in POINTS.items():
        p = {"pop": jnp.asarray(th[0]), "beta": jnp.asarray(th[1:1 + G]), "shared": jnp.asarray(th[1 + G:1 + G + T]),
             "innov": jnp.asarray(th[1 + G + T:].reshape(G, T))}
        v, g = vg(p)
        vals[k] = (float(v), np.concatenate([np.atleast_1d(np.asarray(g[n])).ravel()
                                             for n in ("pop", "beta", "shared", "innov")]))
    jax.block_until_ready(vg(p))
    t = time.perf_counter()
    for _ in range(200):
        jax.block_until_ready(vg(p))
    us = 1e6 * (time.perf_counter() - t) / 200
    record("numpyro", vals, "numpyro.infer.util.log_density + jax.value_and_grad, jit, x64 (all XLA threads)", us)


if __name__ == "__main__":
    which = sys.argv[1:] or ["martin", "rust_martin", "rust_nuts", "cmdstan", "bridgestan", "pymc", "numpyro"]
    out = os.path.join(common.RESULTS, "verify.json")
    if os.path.exists(out):
        old = json.load(open(out))
        results["frameworks"] = {**old.get("frameworks", {})}
        for k in ("pymc_jax_gradient_backends",):
            if k in old:
                results[k] = old[k]
    fns = {"martin": martin, "rust_martin": rust_martin, "rust_nuts": rust_nuts, "cmdstan": cmdstan,
           "bridgestan": bridgestan, "pymc": pymc, "numpyro": numpyro_}
    for w in which:
        safe(w, fns[w])
    os.makedirs(common.RESULTS, exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out)
