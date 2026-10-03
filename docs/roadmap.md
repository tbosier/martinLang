# Roadmap: further optimisations

Ideas for making Martin faster, from the profiles and reviews of the last
rounds. None of this is done; expected gains are hypotheses until measured.

What Martin can offer over hand-written code is automation: the compiler
reads the model and decides, for every model a user writes, things an expert
would otherwise work out and code by hand for one model. Rust (or C++) can
express anything Martin generates, and several of the ideas below already
exist in other tools; where they do, the prior art is named. Items that change
the inference algorithm or the sampler are not language comparisons under
the same-sampler rules ([bench/same_sampler](../bench/same_sampler/README.md))
and would be reported separately.

## Changes to how inference is done

1. **Laplace collapse for Poisson and Bernoulli observations.** Integrate
   the latent walks of the Poisson time-series model out with a Laplace
   approximation: Newton steps for the conditional mode (in state
   coordinates, the running sums, where each series' Hessian is tridiagonal;
   in the model's `innov` coordinates it is dense), its log determinant, and
   gradients through both, which need third derivatives of the likelihood
   (cheap and elementwise for these links). An importance-sampling
   correction makes the estimates asymptotically exact if the weights
   behave; with one weight per kept draw built from all series, weight
   degeneracy is the main risk, so Pareto-k diagnostics and weighted ESS are
   part of the work. Prior art: TMB and R-INLA (automatic Laplace), bssm
   (Laplace with importance correction for exactly this kind of model). The
   Kalman collapse gained 100 to 146 times per gradient on Gaussian walks
   with unknown scales ([kalman.md](kalman.md)), much of it by removing the
   scale's funnel; the benchmark model fixes its scales, so its gain is an
   open question, and each Laplace gradient costs several ordinary ones.
2. **Kalman collapse for forecasting models.** Local linear trend (2 states)
   and weekly seasonality (8 states with the trend), several states per
   series, the small state updates written out as FMAs and vectorised across
   series. The flagship demo ([flagship-demo.md](flagship-demo.md)). Prior
   art: Stan's `gaussian_dlm_obs` when the user writes the collapsed model.
3. **Automatic reparameterisation.** Choose centred or non-centred form per
   group from the model and, with a pilot run, the data. In the Kalman
   benchmarks the centred form did not reach R-hat 1.01 in some seeds where
   the non-centred one did. Prior art: Gorinova, Moore and Hoffman (2020);
   NumPyro's `LocScaleReparam`.
4. **Gibbs updates for conjugate blocks.** Exact conditional distributions
   where the model has them, alternated with NUTS on the rest. Prior art:
   JAGS and NIMBLE assign conjugate samplers automatically.
5. **Batched chains.** (The [roofline estimate](roofline.md) suggests the
   logistic and time-series kernels are latency-bound, which interleaving
   chains would address.) Several chains through one gradient call, reading the
   data once for all of them. The repository's own measurement found these
   kernels were not limited by shared memory bandwidth with four processes
   ([compiler-round.md](compiler-round.md)), so the gain would come per core
   and trades against running chains on separate cores. Needs chains that
   stay in step: ChEES-style trajectories (one jittered length shared by all
   chains, an adaptation designed for many chains) or masked batched NUTS.
   Prior art: TensorFlow Probability and BlackJAX.
6. **A sampler generated with the model.** Dimension known at compile time,
   one layout chosen for the gradient, the metric and the sampler's state,
   the low-rank metric's projections inside the gradient's passes, no
   function-pointer boundary. The bolted-on fused leapfrog did not help
   (compiler-round.md); this would be the integrated version. Prior art:
   JAX-based samplers compile model and sampler into one program.
7. **Hessians from the compiler.** Hessian-vector products or Hessian blocks
   generated with the gradient, sharing intermediate values: part of what
   item 1 needs (with the third derivatives above), and useful for a
   Newton-based warmup.

## Specialising to the data

8. **Aggregate repeated rows automatically.** When every per-row input to a
   likelihood other than the outcome is data (covariates, offsets, exposures,
   known scales, group membership), no parameter is indexed by row, and the
   outcome is used nowhere else, the likelihood can be summed over unique
   input rows with counts: Bernoulli rows become s·η − n·log(1 + e^η),
   Poisson rows (Σy)·η − n·e^η, Normal rows centred sufficient statistics
   (Martin already does this for the affine Normal case). Per-row outputs
   such as predictions would be expanded back. With five categorical
   predictors of four levels there are at most 1,024 distinct rows. This is
   routine practice when done by hand, and a fixed-model program can do it
   at load time; what the compiler adds is deciding validity and the
   aggregated form from any model. No gain on the current benchmarks: the
   logistic data has continuous covariates, and the time-series model has a
   latent value per observation.
9. **Data-dependent code generation.** Binary covariates as selects, constant
   columns dropped, sparse design matrices with their pattern built into the
   code, exact loop counts from the real dimensions. Today Martin compiles
   ahead of time, before the data exists, so this needs a compile step after
   the data is read (or more precompiled variants, as narrow data does).
   Sorted group segments would also need indexing, which the language lacks.
10. **Autotuning per machine.** Time a few variants (tile and chunk sizes,
    group width, unroll) on first run and cache the winner per CPU, model and
    shape. These were tuned by hand on one CPU.
11. **Precision analysis.** Find data-only computations that tolerate float32
    within a stated error bound, extending the exact narrow-data copies.

## Tune-ups to what exists

12. **Martin's own thread pool instead of OpenMP.** An early profile of a
    large run (not recorded in the repository, before the fused leaf passes)
    showed much of the time in OpenMP's fork, join and spinning. Profile the
    current build first; if it holds, a persistent pool with spin barriers
    sized for microsecond tasks, pinned per L3 cache.
13. **Cache-line alignment in the blocked scan layout.** Each group's 8 series
    at one time can straddle cache lines. Padding inside the sampler's vector
    would add dimensions NUTS samples (zero gradient, constant momentum),
    which changes the sampler; alignment would have to come from the layout's
    offsets or from a copy that is not part of the sampler's state.
14. **Read `innov` once per gradient.** It is read in two passes.
15. **Common subexpressions across statements.** A `let` used by two `~`
    statements is computed twice.
16. **Newton's Gram kernel.** It still misses L1 several times as often as
    the Rust's (compiler-round.md); a packed upper-triangle layout or smaller
    chunks.
17. **Narrow data.** No measurable gain in the later measurements, but it
    multiplies build time (2.5 times on the two benchmark models): keep only
    int8 for counts, or make it opt-in.
18. **Make the low-rank metric and fast warmup the defaults** once they are
    tried on funnels, heavy tails and posteriordb models.
19. **The streaming ESS fallback.** It read up to about a third low on one
    hierarchical model; a longer default lag window would send fewer
    parameters to it.
20. **Build time.** Compile model variants in parallel; cache LLVM output per
    model hash.

## Later

21. **Beyond sampling.** Optimisation as something a user asks for (today it
    runs only inside the fast warmup's starting point), and models with
    ordinary differential equations, with the same compiler-derived
    gradients (sensitivities).

## Order

1. The Laplace collapse (item 1) and the forecasting Kalman collapse
   (item 2): the flagship demo, with bssm, TMB and a hand-collapsed Stan
   model as baselines.
2. Profile the current sampler, then the thread pool and a generated sampler
   (items 12 and 6).
3. Data-dependent compilation and autotuning (items 9 and 10), which item 8
   then builds on.
4. Reparameterisation and the new sampler defaults (items 3 and 18).
