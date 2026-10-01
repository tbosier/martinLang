# Flagship demo: hierarchical time-series forecasting

## The goal

The long-term aim is the fastest system there is for numerical
computation: models, optimisation and statistical machine learning, with
every layer below the source language open to specialisation. It starts
with one domain, because depth in one domain is what makes the speed
believable.

One benchmark, chosen deliberately: hierarchical Bayesian forecasting
across thousands of related time series. The aim is a compiler that
understands this model family well enough to remove most of the work a
generic probabilistic programming language does, exactly where the model's
structure allows it and approximately (with a stated correction and
diagnostics) where it does not, and to run what remains as specialised
machine code.

Two separate claims have to be kept apart, and the benchmark is designed to
measure them separately:

- **algorithm:** collapsing latent states (Kalman, Laplace) changes how much
  work the inference needs. Other systems can do this too (Stan's
  `gaussian_dlm_obs`, bssm, R-INLA, TMB), usually when the user writes the
  collapsed model by hand or picks the method;
- **compiler:** Mint detects the structure from the plain model, chooses the
  method, and generates faster code for whatever algorithm is chosen.

## Where Mint is today

Measured on one machine (Ryzen 9 5900X); see
[compiler-round.md](compiler-round.md), [hierarchical.md](hierarchical.md)
and [results/seed_runs_d51f859.md](../bench/dynpois/results/seed_runs_d51f859.md).

- A Poisson panel of random walks with fixed scales (up to 37,901
  parameters) in 18 lines. The compiler stores the scanned matrix
  column-major, runs the running sums as a fused kernel vectorised across
  series, uses its own `exp`, and splits the gradient across each chain's
  threads.
- At commit d51f859, a whole run of the 37,901-parameter model took 61 to
  65 s (three runs) against 93 s for a hand-written AVX2 Rust gradient on the
  same sampler runtime (one run). Run-to-run spread on this machine is about
  20%, so the Rust side needs repeats before this is a result.
- Every parameter is still sampled by NUTS. Nothing is collapsed yet, and
  the existing benchmark fixes the scales, so it says little about the harder
  model below.

## The flagship model

Written out here so that every system implements the same thing. For series
s = 1..S and day t = 1..T:

- level and slope: l[s,t] = l[s,t-1] + b[s,t-1] + e_l[s,t],
  b[s,t] = b[s,t-1] + e_b[s,t], with e_l ~ N(0, sl[s]^2), e_b ~ N(0, sb[s]^2);
- weekly seasonality: six seasonal states per series with the usual
  sum-to-zero dummy form (its transition is a deterministic shift plus one
  noisy component, so its covariance is singular);
- observation: y[s,t] = l[s,t] + seasonal[s,t] + e_y[s,t], e_y ~ N(0, sy[s]^2)
  in the Gaussian variant; Poisson with log link and Student-t variants
  later;
- hierarchy: log sl[s], log sb[s], log sy[s] each normal around
  population means with population scales; priors on those hyperparameters
  and on the initial states stated in the benchmark specification;
- size ladder: S = 100, 1,000, 10,000; T = 730 (two years daily), with the
  last 28 days held out.

With the Gaussian observation, the latent states (8 per series and day:
level, slope, six seasonal) can be integrated out by a Kalman filter given
the scales. The series-specific scales are still unknown, so NUTS still runs
over 3S + a few parameters: about 30,000 at S = 10,000, against about 58
million latent states without collapsing. That is the reduction to aim at;
"a handful of hyperparameters" would require the scales to be shared.

## The pipeline

What exists is marked as such; everything else is planned.

1. **Source.** Mint models: types for shapes, positivity and SPD matrices;
   data and parameters declared separately. (Exists.)
2. **Structure detection** in the typed model:
   - Kalman eligibility: transitions linear in the states with Gaussian
     noise whose covariance does not depend on the states, and observations
     whose mean is affine in the states and whose noise is Gaussian with
     covariance independent of the states (y ~ N(exp(x), s) does not
     qualify). Degenerate (singular) state covariances, as in seasonal
     models, are allowed: the filter is used, not an assumed SPD precision;
   - Laplace eligibility: a latent Gaussian prior with a non-Gaussian
     observation density;
   - conditional independence across series (already used for batching);
   - centred hierarchies that would sample better non-centred, and the
     reverse.
3. **Inference plans**, each a Bayesian computation over the remaining
   parameters:
   - NUTS on all parameters (exists);
   - NUTS on the remaining parameters with the latent states collapsed
     exactly by a Kalman filter;
   - NUTS on the remaining parameters using a Laplace approximation of the
     marginal likelihood (non-Gaussian observations). This targets an
     approximate posterior. It is corrected the way bssm does: an unbiased
     estimate of the exact marginal likelihood per kept draw (importance
     sampling from the Laplace approximation, or a particle filter) gives
     importance weights on the draws, with Pareto-k diagnostics of the
     weights, the weighted Monte Carlo error reported, and a fallback to
     full NUTS (or a refusal) when the weights are unreliable. Across 10,000
     series the per-series weights multiply, so weight degeneracy is the
     main risk and is measured, not assumed away;
   - Student-t observations are handled by direct Laplace, not by one
     auxiliary scale per observation (that would add about 7.3 million
     parameters at the full size).

   Point estimation (maximising the Laplace marginal likelihood, TMB's usual
   workflow) is not one of the plans: it does not integrate over the
   hyperparameters, so it is reported separately and never compared on
   effective draws.
4. **Planner.** Estimates cost per effective draw for each eligible plan,
   subject to a common accuracy requirement, and checks the estimates with a
   short pilot run. Pilot cost is counted in every reported time. A short
   pilot can miss slow mixing or unstable weights, so the main run's
   diagnostics can overrule the choice.
5. **Specialisation.** Recompile once the data is loaded, with the real
   dimensions as constants (exact loop counts, fixed-size state kernels such
   as the 8 x 8 Kalman updates written out as FMAs), exact narrow data types,
   and autotuned variants (vector width, unroll, tile and chunk sizes) chosen
   by timing them on the machine and cached by CPU model, model and shape.
6. **Kernels.** Fused, allocation-free loops that read the data once,
   vectorised across series; the sampler's per-step work fused with the
   gradient. LLVM stays the backend unless a measured kernel shows it is the
   bottleneck.

The Laplace plan needs machinery that does not exist yet: a conditional mode
solver (Newton with line search, convergence and curvature checks: a
Student-t likelihood has negative curvature away from the data, so the mode
may be non-unique or the Hessian indefinite), the log determinant of the
sparse Hessian, and gradients through the mode and the determinant
(implicit differentiation; third derivatives of the density, as TMB uses).
Leaving out the determinant would give a profile likelihood, not a Laplace
marginal.

## The demo

- **`mint explain model.mint`** prints what the compiler found and did:
  the detected structure, which latent scalars were integrated out and how
  (exactly or approximately), which parameters NUTS still samples, the
  reparameterisations applied, the memory layout and vector width, and the
  kernels generated. Every number it prints is computed from the model and
  data, not estimated.
- **Baselines, two groups:**
  - same algorithm as Mint's chosen plan, to isolate the compiler: collapsed
    Stan (`gaussian_dlm_obs` where the model fits it), collapsed JAX/NumPyro,
    bssm (Laplace with importance correction), and a hand-optimised Rust
    implementation of the same collapsed algorithm;
  - what a user would otherwise run, to show the end-to-end gain: full-state
    Stan (`--O1`), PyMC and NumPyro (CPU, and GPU where available), and
    R-INLA for the Gaussian and Poisson variants (deterministic: compared on
    accuracy and time, not effective draws). TMB is reported as a point
    estimate with its approximate uncertainty, separately.
- **Full-state NUTS at full size is not feasible as written**: storing every
  draw of 58 million parameters for 4 chains and 1000 draws needs about
  1.9 TB. Full-state baselines run on the size ladder until they time out
  or run out of memory (limits stated in advance and reported as results),
  and store only the reported quantities, not every latent state.
- **Measure:** effective draws per second (bulk and tail ESS of the reported
  quantities, R-hat), wall time, peak memory, CPU and GPU utilisation, and
  agreement with the reference. Forecasting quality on the held-out 28 days:
  log predictive score and CRPS, empirical coverage of 50% and 90% intervals,
  rolling-origin evaluation, joint checks on aggregates (totals across series
  and across days), and simple forecasting baselines (seasonal naive,
  exponential smoothing) so that "useful" is measured, not assumed.

## Correctness and reference

- **Exact checks on small instances:** for a few series and days, the
  Kalman marginal log density and its gradient against a dense Gaussian
  computation; the Laplace marginal against numerical integration in low
  dimension.
- **Reference posterior:** long runs of full-state NUTS on the smaller
  sizes, with stated targets (R-hat below 1.01, bulk and tail ESS above a
  stated minimum for every reported quantity, no divergences) and a stated
  rule for agreement across many quantities (largest standardised difference
  compared with what independent reference runs show among themselves).
- **Calibration:** simulation-based calibration of each plan on data
  simulated from the model, so that transformations, Jacobians and
  approximations are tested across data sets, not on one.

## Timing rules

- Headline times include everything a user waits for: compilation,
  data-specific recompilation, autotuning, pilot runs, warmup, sampling,
  importance correction and forecast generation. Cached compilation is
  reported separately, never as the headline.
- The same thread budget for every system, stated per run; GPU times
  synchronised on completion; outputs of equivalent size written by every
  system.
- Several seeds and data sets, ranges reported, runs interleaved on a quiet
  machine, tuning effort per system reported, and everything needed to rerun
  it in the repository.
- Ablations: each compiler stage switched off in turn, so each speedup is
  attributed.

## Milestones

Each has a completion test fixed in advance.

0. **Streaming draws.** The runtime keeps every draw of every parameter in
   memory today (chains x draws x parameters). It will instead keep running
   means and variances (and ESS and R-hat inputs) for everything, and full
   draws only for the quantities a program asks for (hyperparameters,
   forecasts). Completion test: memory independent of the number of latent
   parameters at fixed reported output, and the same summaries as the
   stored-draws path on the existing benchmarks.

   Status: implemented in the runtime (see the Runtime section of
   [architecture.md](architecture.md)). Full draws are kept for the rows
   `print` shows and for parameters named in `MINT_KEEP_DRAWS`; everything
   else is summarised as it is drawn, and `MINT_DRAWS` streams to its file.
   On the same draws, the summaries printed for both regressions and both
   dynamic Poisson sizes are identical to the stored-draws path's (eight
   schools prints every parameter, so it keeps every draw). Peak resident
   memory of the 37,901-parameter model (4 chains x 1000 draws, 3 threads
   per chain) was 224 to 266 MiB in 4 runs of the final runtime, against
   1,210 to 1,242 MiB with every draw stored (10 runs of the previous
   runtime). Wall time could not be measured cleanly: the machine was
   shared with other jobs throughout (load average 13 to 39 on 24 CPUs).
   In three interleaved pairs with the final runtime, the streaming run
   took 212 s against 211 s for the previous runtime when it ran second
   (load 25), 140 s against 141 s when it ran first (load 17 to 22), and
   660 s against 349 s when it ran second at a load of 27 to 33. Earlier
   pairs with intermediate versions, always with the previous runtime
   first while the load rose, had the streaming run 0.4 to 33% slower.
   Measured inside the runs, constraining, keeping and summarising the
   draws took 3.6 to 6.1 s summed over the 4 chains, under 1% of the
   chains' time, so we do not attribute the slow pairs to it; a quiet
   machine is needed to settle this. What still grows with the number
   of latent parameters is the sampler's own state and the per-parameter
   running statistics (120 doubles' worth per parameter and chain at 1000
   draws, against 1000 doubles of stored draws before), because the
   summary still reports the lowest ESS and highest R-hat over every
   parameter. Memory that does not grow with the number of latent
   parameters at all needs a summary that stops reporting diagnostics for
   every latent scalar; that is not done.
1. **Kalman collapse, Gaussian observations.** Exact small-instance checks
   pass; on S = 100, the posterior of the scales and the held-out forecasts
   agree with the reference within the stated rule; simulation-based
   calibration passes.
2. **`mint explain`.** Output checked against hand counts on three models.
3. **Laplace collapse with correction** for Poisson and Student-t
   observations: mode-solver and determinant checks, gradient checks of the
   marginal, Pareto-k reported per run, and agreement with the reference on
   S = 100 after correction.
4. **Data-time specialisation and autotuning.** Measured speedup over the
   generic build on the flagship model, autotuning cost included, and no
   regression on the other benchmarks.
5. **Planner.** On data sets and sizes not used to develop it, chooses a plan
   that meets the accuracy requirement and is within a stated factor of the
   fastest such plan, pilot cost included.
6. **Benchmark suite** against the systems above on the size ladder, and on
   models from posteriordb that the language can express.

## Not in scope for this demo

General-purpose language features, deep-network training, and a custom
replacement for LLVM. Reduced precision inside the sampler's state is left
out for now: HMC's acceptance step corrects integration error, but lower
precision can break the integrator's reversibility and the accuracy of the
log density that acceptance relies on, and checking that is a project of its
own. Exact narrowing of data (values that are representable without loss)
is in.
