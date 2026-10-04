# Fit dynpois.stan (or its reduce_sum version) with CmdStan through cmdstanpy.
import json

import cmdstanpy
import numpy as np

cmdstanpy.set_cmdstan_path("build/cmdstan/cmdstan-2.40.0")


def load_data():
    y = np.load("bench/dynpois/data_large/y.npy").astype(int)
    return {"G": y.shape[0], "T": y.shape[1], "y": y.tolist()}


def compile_model(stan_file, threads):
    cpp = {"CXXFLAGS": "-march=native", "STAN_THREADS": "true"} if threads else {"CXXFLAGS": "-march=native"}
    return cmdstanpy.CmdStanModel(stan_file=stan_file, stanc_options={"O1": True}, cpp_options=cpp,
                                  force_compile=True)


def sample(model, data_file, seed, threads_per_chain, output_dir):
    return model.sample(data=data_file, chains=4, parallel_chains=4, threads_per_chain=threads_per_chain,
                        iter_warmup=1000, iter_sampling=1000, seed=seed, output_dir=output_dir,
                        show_progress=False)


if __name__ == "__main__":
    json.dump(load_data(), open("data.json", "w"))
    fit = sample(compile_model("bench/dynpois/dynpois.stan", False), "data.json", 1, None, ".")
    print(fit.stan_variable("pop").mean())
