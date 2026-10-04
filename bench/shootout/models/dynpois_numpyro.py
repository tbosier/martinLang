# Hierarchical dynamic Poisson panel (bench/dynpois/SPEC.md) in NumPyro,
# JAX on CPU with float64. Centred form, the same parameters as dynpois.stan.
import numpyro

numpyro.set_host_device_count(4)  # 4 XLA CPU devices, for chain_method="parallel"
numpyro.enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402


def load_data():
    return np.load("bench/dynpois/data_large/y.npy")


def model(y):
    G, T = y.shape
    pop = numpyro.sample("pop", dist.Normal(0.0, 1.0))
    beta = numpyro.sample("beta", dist.Normal(pop, 0.4).expand([G]).to_event(1))
    shared = numpyro.sample("shared", dist.Normal(0.0, 0.05).expand([T]).to_event(1))
    innov = numpyro.sample("innov", dist.Normal(0.0, 0.08).expand([G, T]).to_event(2))
    state = jnp.cumsum(shared + innov, axis=1)
    numpyro.sample("y", dist.Poisson(jnp.exp(beta[:, None] + state)).to_event(2), obs=y)


def sample(y, seed, chain_method):
    mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=1000, num_chains=4,
                chain_method=chain_method, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), y, extra_fields=("num_steps", "diverging"))
    return mcmc


if __name__ == "__main__":
    sample(load_data(), seed=1, chain_method="parallel").print_summary()
