# One sampler, many gradient backends

This harness compares Martin, hand-written Rust and Stan with identical
sampling algorithms. Everything except the code that computes the model's log
density and gradient is shared:

- the NUTS implementation (the Martin runtime's `mint_sample` in
  `runtime/mint_rt.c`, a port of Stan's `base_nuts`);
- the warmup and metric adaptation (Stan's windowed warmup: step-size dual
  averaging and a diagonal metric);
- the random number generator and its seeding (xoshiro256++, seeded per
  chain from the run's seed). The initial point is the same draw (but see
  the layout note below). After that the streams stay in step only while
  the trajectories do: accept/reject and tree-selection draws depend on log
  density values, so once rounding differences change a branch, the chains
  use the same generator from different positions;
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
| `mint` | Martin's compiled model (`examples/*.mint`, settings rewritten per run) |
| `rust_max` | `baselines/dynpois_max.rs`, `baselines/logistic_bayes_max.rs`: nightly Rust, AVX2 intrinsics, glibc's vector `exp`/`log`, gradients by hand |
| `rust_par` | `baselines/dynpois_par.rs`: the dynamic Poisson baseline brought up to Martin's algorithmic level (below) |
| `rust` | `baselines/eight_schools.rs`: plain Rust (eight schools has 10 parameters; there is nothing to vectorise) |
| `stan` | the Stan program in `stan/`, compiled by BridgeStan 2.9.0 against the installed CmdStan 2.40 (its Stan, Stan Math and stanc), `stanc --O1`, `g++ -O3 -march=native`, `STAN_THREADS`, and run by `bs_driver` |
| `stan_glm` | logistic regression written with `bernoulli_logit_glm` (it turned out that `stanc --O1` already makes this rewrite: the two produce identical draws) |

### Stan under Martin's sampler (`bs_driver.c`)

`bs_driver MODEL_model.so DATA.json DRAWS WARMUP CHAINS SEED` loads a
BridgeStan model with `dlopen` and calls `mint_sample` with:

- `logp` = `bs_log_density_gradient(model, propto = true, jacobian = true, ...)`.
  Stan then drops every constant term, while Martin keeps each Normal's
  `-log(scale)`. The two log densities therefore differ by a constant, which
  the sampler never sees. `verify.py` checks that the difference is the
  expected constant, to 1e-13 relative (3.4e-9 absolute on the large model's
  constant of 95,393). Both add the log Jacobian of their constraining transforms
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

**Parameter order.** The Stan programs declare their parameters in Martin's
order. `bench/dynpois/dynpois.stan`'s `array[G] vector[T] innov`
unconstrains row by row, which is Martin's user order for `innov: Matrix[G, T]`
(a Stan `matrix[G, T]` would have been column-major and the wrong order).
The copy in `stan/dynpois.stan` has the same model block and drops the
generated quantities. The gradients agree component by component.

### The new Rust baseline (`baselines/dynpois_par.rs`)

`dynpois_max.rs` is kept unchanged. The new file adds what Martin's compiler
does and the shared sampler allows:

- **Martin's table-driven exp**, with the same 256-entry table (copied bit for
  bit), the same degree-4 polynomial and the same AVX2 gather, inline. Where
  to put it was measured. `DYNPOIS_EXP` selects:
  - `table` (the default): a separate tight pass over each block's `eta`,
    replacing `dynpois_max.rs`'s glibc calls. This is the fastest.
  - `fused`: inside the forward pass, as Martin does. It measured about 20%
    slower than `table` here (1.19 to 1.24x over the passes). Why fusing
    pays in Martin's code and not here was not established.
  - `back`: inside the backward pass.
  - `glibc`: `dynpois_max.rs`'s kernel with only the threading added.
- **The gradient split across the chain's threads**, like Martin's parallel
  fused scan kernel. Blocks of eight series go to the runtime's
  `mint_par_groups`, with the thread count `mint_par_threads()` reports for
  the calling chain. That count is 3 per chain during a large run, and 1 (or
  `MINT_KERNEL_THREADS`) in `MINT_BENCH_GRAD`. Each thread writes its shared
  gradient partial sums and its log density to its own slot, and the caller
  adds them in thread order, so the result is deterministic for a given
  thread count.

What it cannot do: **Martin's column-major storage of `innov`**. Martin's
compiled program stores the matrix by columns internally and converts at the
sampler's boundary (`mint_set_layout`), so four series at one time step are
one contiguous load. The Rust baseline must keep the user's row-major order,
because the parameter vector's order is part of the shared interface. It
brings rows into column form with half-width loads and unpacks, and writes
the gradient back with 4x4 transposes.

A consequence for the shared random numbers: Martin's initial points and
momenta for `innov` are drawn in its internal (column-major) order. On the
dynamic Poisson model, Martin's chains therefore start from a permutation of
the point the Rust and Stan chains start from. That is statistically
equivalent, but not the same coordinates. On the other models all
implementations start from the same point.

### Which tricks each implementation uses (dynamic Poisson)

| | Martin | rust_max | rust_par | Stan |
|---|---|---|---|---|
| gradient | compile-time AD | by hand | by hand | reverse-mode autodiff at run time |
| vectorised across series (4 per AVX2 vector) | yes | yes | yes | no (Eigen expressions within a series) |
| column-major `innov` | yes | no (fixed order) | no (fixed order) | no |
| exp | own table exp, fused into the pass | glibc vector exp, separate pass | own table exp (Martin's), separate pass | Eigen's vectorised exp on the values; `poisson_log_lpmf` forms its partials analytically |
| narrow data (counts read as int8) | yes | no | no | no |
| gradient split across the chain's threads | yes (3 per chain on the large model) | no | yes | no: the Stan program has no `reduce_sum`; a version with it was not written |

Martin's two layout-dependent tricks were measured by turning them off
(`run_grad.py`, `mint[no-scan-layout]`, `mint[no-narrow-data]`). Without the
column-major layout, Martin's gradient takes 2.1 to 2.2x as long: 7.35 and
96.2 µs, slower than both Rust versions. So Martin's code generator depends on
that layout, but the layout alone does not explain the remaining 5% between
Martin and `dynpois_par.rs`, which gets close without it by transposing in
registers. Reading the counts as int8 made no measurable difference here:
3.49 against 3.56 µs, and 44.6 against 44.4 µs. What the remaining 5% is
was not established.

## Rules for anyone extending this

1. Only the `logp` function may differ. Never change sampler settings for one
   implementation. A new backend goes through `mint_sample`, or it is
   reported as a sampler comparison (like nutpie), not a language comparison.
2. Keep the parameter order. Verify the gradient against Martin's at the
   benchmark point and the log density difference against the expected
   constant (`verify.py`) before timing anything.
3. Machine load is part of every number. Record the load average and CPU
   busy fractions (the scripts do; for whole runs only in the second before
   each run), interleave the configurations, and say which comparisons the
   noise allows.

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
passes taken while other agents loaded the machine: a 1-minute load average
of 15 to 29, with 6 to 24 of the 24 hardware threads busy. They have the
same seeds, gradient counts and ESS, and slower times.

## Findings (run of 2026-10-01, at de58040 plus this harness)

Machine: Ryzen 9 5900X, shared with other agents. Unless noted, the figures
come from passes in which at most 1.6 of the 24 hardware threads were busy
with other work in the second before each run. Load during a whole run was
not recorded. `results/results.md` has every row, with ranges.

**Gradient alone.** Median µs over 15 interleaved, pinned rounds. In the
latest pass 12 to 15 of the 15 rounds were clean for each configuration;
the clean medians are shown.

| problem | Martin | Rust max effort | Rust v2 (`dynpois_par.rs`) | Stan via BridgeStan |
|---|---|---|---|---|
| dynamic Poisson, D = 3,171 | 3.56 | 4.10 (1.15x) | 3.73 (1.05x) | 51.0 (14.3x) |
| dynamic Poisson, D = 37,901 | 44.4 | 52.9 (1.19x) | 46.6 (1.05x) | 654 (14.7x) |
| same, 3 kernel threads | 17.0 | not threaded | 18.4 (1.08x) | not threaded |
| logistic, n = 5000, p = 20 | 23.6 | 38.0 (1.61x) | | 42.7 (1.81x) |
| eight schools | 0.025 | 0.025 (plain Rust) | | 0.50 (about 20x) |

Two earlier passes agree with these clean medians to within 4%. One was
taken on a loaded machine and is kept as `grad_loaded.json`. The other was
quiet and was overwritten by this one.

**Whole runs.** 4 chains, 1000 + 1000. Medians over 5 seeds, 3 for the
large model:

| problem | Martin | Rust max effort | Rust v2 | Stan |
|---|---|---|---|---|
| dynamic Poisson, small | 3.32 s | 3.76 s | 3.50 s | 28.3 s |
| dynamic Poisson, large | 62.4 s | 98.6 s | 65.2 s | 737 and 741 s (two quiet seeds); 3887 s for a third seed run while about 16 threads were busy |
| logistic | 0.38 s | 0.59 s | | 0.64 s |
| eight schools | 5.5 ms | 5.7 ms (plain Rust) | | 16.6 ms |

- **Same sampler, same work.** All runs did the same amount of sampler work:
  trajectories almost always hit a fixed length (255 leapfrog steps per
  draw on the small time series, 511 on the large one, 7 on logistic), so
  gradient counts agree to within 1.6% at a given seed. That follows from
  the shared sampler and similar step sizes; it is not by itself evidence
  that the posteriors match (`bench/dynpois` compares posterior means). On
  eight schools, where tree depth varies, counts at a seed differ by 3 to
  17%. ESS per gradient agrees within the spread between seeds.
- **The new Rust baseline.** Its gradient is 5% slower than Martin's on one
  thread and 8% slower on three, where `dynpois_max.rs` is 15 to 19% slower.
  On the large model its whole run is within 5% of Martin's (65.2 against
  62.4 s); `dynpois_max.rs`, which does not split its gradient across the
  chain's threads, takes 1.6x as long. What the remaining 5 to 8% is was not
  established. Martin without its column-major layout is 2.1x slower than
  Martin, so its own code depends on that layout. Martin without narrow data is
  no slower.
- **Stan's gradient is 14 to 15x Martin's** on the time-series model, 1.8x on
  logistic regression, and about 20x on eight schools. At 25 ns per Martin
  gradient, the timing loop's own overhead is a large part of the eight
  schools ratio. Two things inflate Stan's gradient time and were not
  measured. One is BridgeStan's per-call cost: it copies the parameters and
  catches exceptions. The other is `STAN_THREADS`, which concurrent chains
  in one process require. On the large model the comparison also pits
  Stan's single gradient thread against three for Martin and Rust v2, because
  the Stan program has no `reduce_sum`. `stanc --O1` already rewrites the
  logistic likelihood to `bernoulli_logit_glm`.
- **Whole-run cost per gradient is not gradient plus a fixed sampler cost.**
  Subtracting the pinned gradient time leaves about 51 to 55 µs per
  gradient per chain for Martin and both Rust versions on the large model,
  and about 150 µs for Stan. A constant of about 48 µs plus 16% of the
  gradient time fits all four. That suggests every gradient runs about 16%
  slower inside a 12-thread run than pinned alone, which would make Stan's
  extra residual the same effect rather than sampler overhead. This was not
  measured. The per-gradient figure is also wall time to the slowest chain,
  so it includes chain imbalance.
- **nutpie, same Stan gradient.** This is a sampler comparison, with 3 seeds
  per cell.
  - **Time per gradient.** nutpie's time per gradient was higher on all
    three problems. On one pinned chain it was 58 to 60 µs against Martin's
    54.7 to 55.8 µs on the small time series, and 2.5 to 3.0 µs against
    0.70 µs on eight schools. nutpie's time covers the whole `sample()`
    call, including setup and storing 2000 draws per chain. Martin's covers
    the sampling loop only. So the gap is an upper bound on any difference
    in per-gradient sampler overhead, and on eight schools (13,500
    gradients, 30 to 40 ms) it is mostly fixed cost.
  - **ESS per gradient.** On the small time series nutpie's trajectories
    are half as long (127 steps), but its lowest bulk ESS per 1000
    gradients was lower in every seed: 0.73 to 0.76 against 0.91 to 1.09
    with 4 chains, and 0.47 to 0.66 against 0.86 to 1.27 with one chain.
    On eight schools nutpie was higher in 3 of 3 seeds (37 to 42 against 28
    to 30). On logistic it was higher in 3 of 3 seeds (104 to 114 against
    91 to 97), but Martin's own range over 5 seeds of the same estimator
    reaches 105, so logistic is not a finding.
  - **A reproducible failure.** With seed 3 on the small time series, one
    of nutpie's chains had its step size collapse to 7e-189. It never moved
    (R-hat 1.55, lowest ESS 7), and the same happened in both runs of that
    seed.

**Caveats.**

- **Initial points.** These differ on the time-series model: Martin draws its
  initial point and momenta in its internal column-major order, so its
  chains start from a permutation of the Rust and Stan point. After the
  first rounding-dependent branch, every implementation's chains use the
  random stream differently anyway.
- **Compilers.** Each implementation uses its own: clang 22, LLVM 21 and
  g++ 16. Stan was built with `--O1` and `-march=native`, not CmdStan's
  defaults.
- **ESS.** The runtime's ESS (Geyer, on raw draws) reads higher than ArviZ's
  rank-normalised bulk ESS, by up to 38% on eight schools. The whole-run
  ESS is the lowest over all parameters and varies by more than 1.5x
  between seeds, so ESS-per-second ratios between implementations are not
  findings.
- **Order of the large runs.** Stan's large runs were not interleaved with
  the others: seed 1 ran first, under load, and seeds 2 and 3 ran last.
- **Thread safety.** This was checked by byte-identical draws on short runs
  of both time-series sizes. That is evidence, not proof.
- **Scope.** One machine and one dataset per problem.
