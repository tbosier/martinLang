# Fit dynpois.stan with nutpie (BridgeStan builds the model library).
import nutpie
import numpy as np


def load_data():
    y = np.load("bench/dynpois/data_large/y.npy").astype(np.int64)
    return {"G": y.shape[0], "T": y.shape[1], "y": y}


def compile_model():
    code = open("bench/dynpois/dynpois.stan").read()
    return nutpie.compile_stan_model(code=code, extra_stanc_args=["--O1"],
                                     extra_compile_args=["CXXFLAGS=-march=native"])


def sample(compiled, data, seed, adaptation):
    return nutpie.sample(compiled.with_data(**data), draws=1000, tune=1000, chains=4, cores=4, seed=seed,
                         adaptation=adaptation, progress_bar=False)


if __name__ == "__main__":
    trace = sample(compile_model(), load_data(), 1, "diag")
    print(trace.posterior["pop"].mean())
