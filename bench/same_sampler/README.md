# One sampler, many gradient backends

This harness compares Mint, hand-written Rust and Stan with identical
sampling algorithms. Everything except the code that computes the model's log
density and gradient is shared:

- the NUTS implementation (the Mint runtime's `mint_sample` in
  `runtime/mint_rt.c`, a port of Stan's `base_nuts`);
- the warmup and metric adaptation (Stan's windowed warmup: step-size dual
  averaging and a diagonal metric);
- the random number stream (xoshiro256++, seeded per chain from the run's
  seed), so the initial points and momenta are the same random numbers;
- the settings: 4 chains on parallel threads, 1000 warmup + 1000 draws, the
  same seeds, and the same threads per chain for the sampler's own passes
  (1 for the small models, 3 for the large one, the runtime's default);
- the diagnostics (the runtime's summary for the whole runs, ArviZ for the
  nutpie check).

Each implementation hands the runtime a function `logp(theta, grad)` over
the same unconstrained parameter vector, in the same order.

## The implementations

| name | log density and gradient |
|---|---|
| `mint` | Mint's compiled model (`examples/*.mint`, settings rewritten per run) |
| `rust_max` | `baselines/dynpois_max.rs`, `baselines/logistic_bayes_max.rs`: nightly Rust, AVX2 intrinsics, glibc's vector `exp`/`log`, gradients by hand |
| `rust_par` | `baselines/dynpois_par.rs`: the dynamic Poisson baseline brought up to Mint's algorithmic level (below) |
| `rust` | `baselines/eight_schools.rs`: plain Rust (eight schools has 10 parameters; there is nothing to vectorise) |
| `stan` | the Stan program in `stan/`, compiled by BridgeStan 2.9.0 against the installed CmdStan 2.40 (its Stan, Stan Math and stanc), `stanc --O1`, `g++ -O3 -march=native`, `STAN_THREADS`, and run by `bs_driver` |
| `stan_glm` | logistic regression written with `bernoulli_logit_glm` (it turned out that `stanc --O1` already makes this rewrite: the two produce identical draws) |

### Stan under Mint's sampler (`bs_driver.c`)

`bs_driver MODEL_model.so DATA.json DRAWS WARMUP CHAINS SEED` loads a
BridgeStan model with `dlopen` and calls `mint_sample` with:

- `logp` = `bs_log_density_gradient(model, propto = true, jacobian = true, ...)`.
  Stan then drops every constant term, while Mint keeps each Normal's
  `-log(scale)`. The two log densities therefore differ by a constant, which
  the sampler never sees, and `verify.py` checks that the constant is exactly
  the one expected. Both add the log Jacobian of their constraining transforms
  (only eight schools' `tau` is constrained).
- `constrain` = `bs_param_constrain` (no transformed parameters or generated
  quantities).
- A Stan exception (for example an overflowing Poisson rate early in warmup)
  becomes a log density of `-inf`. The sampler treats that as a divergence,
  as Stan's own sampler does with a rejection.

**Threads.** The runtime calls `logp` from each chain's thread concurrently.
BridgeStan documents that a model built with `STAN_THREADS=true` may be
called from several threads at once: the autodiff tape is thread-local, and
the model object is read-only. By default the chains share one model object.
`BS_MODEL_PER_CHAIN=1` gives each calling thread its own instead.
`verify.py` checks that a shared model, one model per chain and a repeat
produce byte-identical draws.

**Parameter order.** The Stan programs declare their parameters in Mint's
order. `bench/dynpois/dynpois.stan`'s `array[G] vector[T] innov`
unconstrains row by row, which is Mint's user order for `innov: Matrix[G, T]`
(a Stan `matrix[G, T]` would have been column-major and the wrong order).
The copy in `stan/dynpois.stan` has the same model block and drops the
generated quantities. The gradients agree component by component.

### The new Rust baseline (`baselines/dynpois_par.rs`)

`dynpois_max.rs` is kept unchanged. The new file adds what Mint's compiler
does and the shared sampler allows:

- **Mint's table-driven exp**, with the same 256-entry table (copied bit for
  bit), the same degree-4 polynomial and the same AVX2 gather, inline. Where
  to put it was measured. `DYNPOIS_EXP` selects:
  - `table` (the default): a separate tight pass over each block's `eta`,
    replacing `dynpois_max.rs`'s glibc calls. This is the fastest.
  - `fused`: inside the forward pass, as Mint does. It measured 20 to 25%
    slower here; Mint's column-major layout, which this file cannot use, may
    be what makes fusing pay there.
  - `back`: inside the backward pass.
  - `glibc`: `dynpois_max.rs`'s kernel with only the threading added.
- **The gradient split across the chain's threads**, like Mint's parallel
  fused scan kernel. Blocks of eight series go to the runtime's
  `mint_par_groups`, with the thread count `mint_par_threads()` reports for
  the calling chain. That count is 3 per chain during a large run, and 1 (or
  `MINT_KERNEL_THREADS`) in `MINT_BENCH_GRAD`. Each thread writes its shared
  gradient partial sums and its log density to its own slot, and the caller
  adds them in thread order, so the result is deterministic for a given
  thread count.

What it cannot do: **Mint's column-major storage of `innov`**. Mint's
compiled program stores the matrix by columns internally and converts at the
sampler's boundary (`mint_set_layout`), so four series at one time step are
one contiguous load. The Rust baseline must keep the user's row-major order,
because the parameter vector's order is part of the shared interface. It
brings rows into column form with half-width loads and unpacks, and writes
the gradient back with 4x4 transposes.

A consequence for the "same random numbers" rule: Mint's initial points and
momenta for `innov` are drawn in its internal (column-major) order. On the
dynamic Poisson model, Mint's chains therefore start from a permutation of
the point the Rust and Stan chains start from. That is statistically
equivalent, but not the same coordinates. On the other models all
implementations start from the same point.

### Which tricks each implementation uses (dynamic Poisson)

| | Mint | rust_max | rust_par | Stan |
|---|---|---|---|---|
| gradient | compile-time AD | by hand | by hand | reverse-mode autodiff at run time |
| vectorised across series (4 per AVX2 vector) | yes | yes | yes | no (Eigen within a series) |
| column-major `innov` | yes | no (fixed order) | no (fixed order) | no |
| exp | own table exp, fused into the pass | glibc vector exp, separate pass | own table exp (Mint's), separate pass | scalar libm inside autodiff |
| narrow data (counts read as int8) | yes | no | no | no |
| gradient split across the chain's threads | yes (3 per chain on the large model) | no | yes | no |

## Rules for anyone extending this

1. Only the `logp` function may differ. Never change sampler settings for one
   implementation. A new backend goes through `mint_sample`, or it is
   reported as a sampler comparison (like nutpie), not a language comparison.
2. Keep the parameter order. Verify the gradient against Mint's at the
   benchmark point and the log density difference against the expected
   constant (`verify.py`) before timing anything.
3. Machine load is part of every number. Record the load average and CPU
   busy fractions (the scripts do), interleave the configurations, and say
   which comparisons the noise allows.

## Running

```sh
uv venv .venv -p 3.12 && uv pip install -p .venv bridgestan nutpie numpy arviz
bench/same_sampler/build.sh                 # runtime, mintc, Rust, BridgeStan models, bs_driver, JSON data
.venv/bin/python bench/same_sampler/verify.py
.venv/bin/python bench/same_sampler/run_grad.py
.venv/bin/python bench/same_sampler/run_whole.py --out results/whole_small.json --problems dynpois_small,logistic,eight_schools --seeds 1 2 3 4 5
.venv/bin/python bench/same_sampler/run_whole.py --out results/whole_large.json --problems dynpois_large --seeds 1 2 3
.venv/bin/python bench/same_sampler/nutpie_check.py --seeds 1 2 3
.venv/bin/python bench/same_sampler/report.py   # results/results.md
```

`build.sh` expects CmdStan 2.40 at the path in `bench/dynpois/stan_common.py`
(override with `CMDSTAN=`). It downloads BridgeStan's source release into
`build/bs/` if it is missing.

Results: `results/*.json` holds every run, and `results/results.md` has the
tables, generated by `report.py`. The files ending in `_loaded` are earlier
passes taken while other work kept 15 to 29 of the 24 hardware threads busy.
They have the same seeds, gradient counts and ESS, and slower times.

## Findings (run of 2026-10-01, at de58040 plus this harness)

Machine: Ryzen 9 5900X, shared with other agents. The figures below come
from passes taken while under one hardware thread was busy with other work,
except where noted.

**Gradient alone.** Median µs, 15 interleaved pinned rounds, all clean:

| problem | Mint | Rust max effort | Rust v2 (`dynpois_par.rs`) | Stan via BridgeStan |
|---|---|---|---|---|
| dynamic Poisson, D = 3,171 | 3.57 | 4.10 | 3.74 | 50.5 |
| dynamic Poisson, D = 37,901 | 44.2 | 52.8 | 46.7 | 654 |
| same, 3 kernel threads | 16.9 | | 18.3 | |
| logistic, n = 5000, p = 20 | 23.7 | 38.2 | | 42.6 |
| eight schools | 0.024 | 0.024 (plain Rust) | | 0.49 |

**Whole runs.** 4 chains, 1000 + 1000. Medians over 5 seeds (3 for the
large model):

| problem | Mint | Rust max effort | Rust v2 | Stan |
|---|---|---|---|---|
| dynamic Poisson, small | 3.32 s | 3.76 s | 3.50 s | 28.3 s |
| dynamic Poisson, large | 62.4 s | 98.6 s | 65.2 s | 741 s (two quiet seeds 737 and 741 s; one loaded seed 3887 s) |
| logistic | 0.38 s | 0.59 s | | 0.64 s |
| eight schools | 0.01 s | 0.01 s | | 0.02 s |

- The sampler is shared. At the same seed, gradient counts agree to within
  1.6% on the time-series and logistic models. On eight schools they agree
  to within 3 to 17%, presumably because its trajectories part ways sooner
  once rounding differs (not checked). ESS per gradient agrees within the
  spread between seeds.
- The new Rust baseline is within 5% of Mint's gradient. On the large model
  it is within 5% of Mint's whole run, where `dynpois_max.rs` (not threaded)
  takes 1.6x as long. The gap that remains is in what the fixed parameter
  order rules out (column-major `innov`), plus Mint's narrow data; neither
  was isolated.
- Stan's gradient is 14 to 15x Mint's on the time-series model, 1.8x on
  logistic regression and about 20x on eight schools. On eight schools
  Mint's gradient takes 24 ns, so the timing loop's own overhead is a large
  part of that ratio. `stanc --O1` already rewrites the logistic likelihood
  to `bernoulli_logit_glm`.
- On the large model, subtracting the gradient (the 3-thread figure where
  the gradient is threaded) from the median per-gradient cost leaves about
  51 to 55 µs per gradient per chain for Mint and both Rust versions. For
  Stan it leaves about 150 µs, over the two quiet seeds. Why Stan's is larger was not
  established.
- **nutpie, same Stan gradient** (a sampler comparison):
  - **Overhead.** On one pinned chain, nutpie's overhead per gradient was
    7.5 to 9.4 µs on the small time series, against Mint's 4.2 to 5.3. On
    eight schools it was 2.0 to 2.5 µs against 0.2. Its figure includes
    setup and trace storage.
  - **Efficiency.** nutpie got more ESS per gradient on logistic (104 to 114
    against 91 to 97 per 1000) and on eight schools (37 to 42 against 28 to
    30). It got less on the small time series (0.73 to 0.76 against 0.91 to
    1.09), with half-length trajectories.
  - **A reproducible failure.** With seed 3 on the small time series, one
    nutpie chain's step size collapsed to 7e-189 and the chain never moved
    (R-hat 1.55, lowest ESS 7), in both runs of that seed.

**Caveats.**

- Initial points differ on the time-series model. Mint draws its initial
  point and momenta in its internal column-major order, so its chains start
  from a permutation of the Rust and Stan point.
- Each implementation uses its own compiler: clang 22, LLVM 21 and g++ 16.
- Stan was built with `--O1` and `-march=native`, not CmdStan's defaults.
- The ESS in the whole-run tables is the runtime's lowest over all
  parameters, and it varies by more than 1.5x between seeds. ESS-per-second
  ratios between implementations are not findings.
- One machine, one dataset per problem.
