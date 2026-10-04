# Hierarchical dynamic Poisson panel (bench/dynpois/SPEC.md) in PyMC,
# sampled by nutpie. Centred form, the same parameters as dynpois.stan.
import numpy as np
import nutpie
import pymc as pm


def load_data():
    return np.load("bench/dynpois/data_large/y.npy").astype("int64")


def make_model(y):
    G, T = y.shape
    with pm.Model() as model:
        pop = pm.Normal("pop", 0, 1)
        beta = pm.Normal("beta", pop, 0.4, shape=G)
        shared = pm.Normal("shared", 0, 0.05, shape=T)
        innov = pm.Normal("innov", 0, 0.08, shape=(G, T))
        state = pm.math.cumsum(shared + innov, axis=1)
        pm.Poisson("y", mu=pm.math.exp(beta[:, None] + state), observed=y)
    return model


def compile_model(model, backend):
    if backend == "jax":
        return nutpie.compile_pymc_model(model, backend="jax", gradient_backend="pytensor")
    return nutpie.compile_pymc_model(model, backend="numba")


def sample(compiled, seed):
    return nutpie.sample(compiled, draws=1000, tune=1000, chains=4, cores=4, seed=seed, progress_bar=False)


if __name__ == "__main__":
    trace = sample(compile_model(make_model(load_data()), "numba"), seed=1)
    print(trace.posterior["pop"].mean())
