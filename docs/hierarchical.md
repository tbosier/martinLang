# Hierarchical time series: Mint against rustmc, Stan and hand-written Rust

## The problem

A panel of G series observed at T = 150 times: counts with a series-specific
intercept around a population mean, and a latent random walk per series driven
by a shared innovation plus the series' own. This is the model rustmc fits with
`BayesianDynamicPoisson` (its dynamic GLM), with the scales fixed because
rustmc cannot estimate them. The exact specification, data generator and
parameter layout are in [bench/dynpois/SPEC.md](../bench/dynpois/SPEC.md).

Two sizes:

| size | series G | latent dimensions D |
|---|---|---|
| small | 20 | 3,171 (the size rustmc's own review used) |
| large | 250 | 37,901 |

In Mint the whole model is:

```
model DynamicPoisson {
    data y: Matrix[G, T]

    param pop: Real
    param beta: Vector[G]
    param shared: Vector[T]
    param innov: Matrix[G, T]

    pop    ~ Normal(0, 1)
    beta   ~ Normal(pop, 0.4)
    shared ~ Normal(0, 0.05)
    innov  ~ Normal(0, 0.08)

    let state = cumsum(shared + innov, T)
    y ~ PoissonLog(beta + state)
}
```

The dimension names do the broadcasting: `shared` (a `Vector[T]`) is added to
every row of `innov` (a `Matrix[G, T]`), and `beta` (a `Vector[G]`) to every
column of `state`. The compiler derives the gradient, including the reverse
running sum that `cumsum` needs.

| implementation | lines (non-blank, non-comment) | sampler |
|---|---|---|
| Mint | 18 (model and main) | Mint runtime NUTS |
| Stan | 29 (data, parameters, model, generated quantities) | Stan NUTS |
| hand-written Rust | 591 (AVX2 intrinsics, glibc vector exp, gradient by hand) | Mint runtime NUTS |
| rustmc | a library call | block elliptical slice sampling |

## Correctness checks

- Mint's log density and every gradient component match the exact formula in
  the spec (numpy) to about 1e-14 on both sizes (`tests/run.sh`,
  `bench/dynpois/check_grad.py`).
- The hand-written Rust gives the same log density and gradient norm at the
  test point, and matches a scalar reference to about 1e-13 on 453 random cases.
- Stan's log density differs from the spec only by the constants Stan drops;
  differences between points agree to about 1e-11.
- Posterior means of pop, every beta and every terminal state agree across
  Mint, Rust and Stan (table below). This compares means only, not variances,
  tails or the joint distribution. rustmc's means do not agree, and rustmc did
  not converge.

## Results

4 chains in parallel for every implementation; 1000 warmup + 1000 draws unless
noted. Diagnostics are computed by one script (`bench/dynpois/analyze.py`,
ArviZ rank-normalised split R-hat, bulk and tail ESS) over pop, every beta[g]
and every terminal state. A run counts as mixed when max R-hat ≤ 1.01 and min
bulk and tail ESS ≥ 400.

**Small (D = 3,171).**

| implementation | wall time | converged | max R-hat | min bulk ESS | min bulk ESS per second | gradients |
|---|---|---|---|---|---|---|
| Mint | 8.3 s | yes | 1.002 | 2301 | 276 | 2.00 M |
| hand-written Rust | 6.6 s | yes | 1.003 | 1983 | 299 | 2.02 M |
| Stan, stanc `--O1` | 48.5 s | yes | 1.002 | 1860 | 38.3 | 2.01 M |
| Stan, stanc default (`--O0`) | 67.7 s | yes | 1.002 | 1860 | 27.5 | 2.01 M |
| rustmc | 7.0 s | **no** | 1.953 | 6 | not usable | n/a |

**Large (D = 37,901).**

| implementation | draws | wall time | max R-hat | min bulk ESS | gradients | wall time per gradient, per chain, including the sampler |
|---|---|---|---|---|---|---|
| Mint | 1000 + 1000 | 278 s | 1.009 | 463 | 3.66 M | 0.30 ms |
| hand-written Rust | 1000 + 1000 | 220 s | 1.009 | 449 | 3.67 M | 0.24 ms |
| Stan (stanc default `--O0`) | 300 + 300 | 3322 s | 1.023 | 125 | 1.19 M | 11.2 ms |
| rustmc | 1000 + 125 × 8 | 109 s | 2.565 | 5 | n/a | n/a |

Wall times vary between runs of the same binary, even with identical draws.
Repeats gave 278, 281 and 325 s for Mint and 220, 220 and 245 s for Rust; the
table shows the run whose draws were analysed. Both used 3 sampler threads per
chain, and every OpenMP team ran at full size (recorded in the result files).

Mint and Rust share a sampler and sit right at the mixing bar together: these
runs clear it (pop has ESS 449 to 463), but an earlier pair with the same
settings fell just short (ESS 363 to 370). Splitting the sampler's sums across
threads changes rounding and therefore the chains' paths, so treat the large
model as borderline. A longer run, or better mass-matrix adaptation, is the
fix. Stan's shorter run does not meet the bar. rustmc is far from it: its four
chains disagree about pop (1.51, 1.31, 1.54, 1.44).

**Agreement of posterior means** (max difference over pop, beta and terminal
states, in posterior standard deviations; in Monte Carlo standard errors where
both runs mixed). Means only: this does not show that variances or tails
agree.

| size | pair | max difference |
|---|---|---|
| small | Mint vs Stan | 0.028 sd (1.45 MCSE) |
| small | Mint vs Rust | 0.048 sd (2.27 MCSE) |
| large | Mint vs Stan | 0.11 sd |
| large | Mint vs Rust | 0.054 sd (2.50 MCSE) |
| small / large | rustmc vs Stan | 0.86 / 0.93 sd |

## Reading the numbers

- **Mint against Stan: 7.2x the effective samples per second on the small
  model.** Both run NUTS and need the same number of gradients (2.0 million),
  so the difference is the cost of each gradient plus the sampler around it:
  - Mint: 17 µs per gradient per chain;
  - Stan with `--O1`: 97 µs;
  - Stan with the default `--O0`: 135 µs.

  These are whole-run figures (wall time × chains / gradients), not isolated
  gradient timings. On the large model the ratio is 0.30 ms against 11.2 ms,
  but that Stan run used the default `--O0`, a shorter run and one CPU chiplet.
  Mint's sampler also splits its passes across 3 threads per chain there, which
  Stan does not. `--O1` made Stan 1.4x faster on the small model; it was not
  measured on the large one.
- **Mint against rustmc: rustmc is faster and wrong.** Its elliptical slice
  sampler finishes the small model in 7 s. With 1000 warmup and 1000 kept
  sweeps it did not converge (R-hat 1.95, ESS 6), consistent with its own
  review (R-hat 2.1 to 2.3 at this size). Mint's posterior means agree with
  Stan's; rustmc's do not. How long rustmc would need to converge was not
  measured.
- **Mint against the best hand-written Rust: Rust is faster.** For the
  gradient alone on the small model, it is 1.8x faster (19k against 33k CPU
  cycles, measured one gradient at a time). End to end, with the same sampler,
  it is 1.1 to 1.3x faster on both sizes, depending on the run. It gets there
  with 591 lines of intrinsics against Mint's 18 lines of model.

## What changed in Mint to get here

- Matrix-shaped parameters, `Positive[n]` parameters, `PoissonLog`, `cumsum`
  along a named dimension, and broadcasting by dimension name.
- Matrix-shaped `~` statements, with register accumulators for row-indexed
  gradients. This made the gradient 2.4x faster, because it let the Poisson
  loop vectorise, including `exp`.
- **The sampler.** At 37,901 dimensions the original sampler spent over 80%
  of its time copying state vectors and only about 6% in the model. Four
  changes brought the large model from 1448 s to 278 to 325 s, 4.5 to 5.2x in
  all:
  - States are immutable and shared by reference, and the loops over them are
    fused (2.1x).
  - The scaled momenta are recomputed instead of stored, which removes memory
    traffic.
  - For models with at least 8,192 parameters, each chain splits its sampler
    passes across threads (by default, about physical cores ÷ chains).
  - Below that size the sampler runs serially and its draws are bit-identical
    to the original implementation. The small model runs 1.8x faster.
- **Tried and not adopted: gradient-informed metric adaptation.** nutpie's
  `sqrt(var(draws) / var(gradients))` diagonal metric (`MINT_METRIC=grad`)
  halved the trajectory length on the small model (127 leapfrog steps per
  draw instead of 255). But the lowest ESS fell from about 2,080 to about 380,
  so the effective draws per gradient fell about 3.5x. Starting it from the
  identity instead of the initial gradient (`MINT_METRIC_INIT=0`) did not
  change that. On eight schools and logistic regression the two metrics'
  ranges over 5 seeds overlap, so no difference was found there. Stan's
  metric stays the default. Why the gradient metric does badly here was not
  established. The full comparison is `bench/metric_experiment.py`, with
  results in `bench/metric_results.json`.

## Caveats

- Stan was compiled with stanc's default optimisation level (`--O0`) except for
  the one small-model `--O1` run in the table. Stan recommends `--O1`.
- Stan's large run used 300 warmup + 300 draws, because 1000 + 1000 would take
  3 to 4 hours. It was pinned to one CPU chiplet (its own 32 MB L3) while other
  work ran on the other. The Mint and Rust runs used the whole idle machine. The
  per-gradient figures are the fairer comparison, and even those favour Mint
  somewhat, because Stan's four chains shared one L3 cache.
- Stan's times include writing its CSV output (38,000 columns per draw on the
  large model); CmdStan's own per-chain times are within about 3% of the wall
  times on the small model.
- rustmc keeps at most 25 million stored values, so its large run kept 125
  thinned draws per chain after the same 1000 post-warmup sweeps.
- rustmc was run through its Python bindings from the integrate/forecasting
  branch (0.13.0). Its sampler is the one its authors ship for this model. It
  fixes the scales; the comparison uses the same fixed scales everywhere.
- One data set per size, one seed per implementation, one machine (Ryzen 9
  5900X).
