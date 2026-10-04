# Fit the panel with rustmc's BayesianDynamicPoisson (block elliptical slice
# sampling), with the fixed scales of bench/dynpois/SPEC.md.
import numpy as np
import rustmc as rmc


def load_data():
    return np.load("bench/dynpois/data_large/y.npy")


def make_model():
    return rmc.BayesianDynamicPoisson(initial_mean=0.0, coefficient_sd=1.0, group_sd=0.4,
                                      process_sd=0.08, shared_process_sd=0.05)


def sample(model, y, seed):
    # rustmc's documented example uses chains=4, warmup=1000, draws=1000; its
    # 25M stored-value cap allows 125 kept draws per chain at this size, so
    # the 1000 post-warmup sweeps are thinned by 8.
    return model.fit(y, chains=4, warmup=1000, draws=125, thin=8, seed=seed)


if __name__ == "__main__":
    fit = sample(make_model(), load_data(), 1)
    print(np.mean(fit.get_samples_2d()["population_beta[0,0]"]))
