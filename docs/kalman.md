# Kalman collapse: integrating out a latent random walk

This is the first case where Mint's compiler removes most of an inference
problem instead of making it faster. In an unmodified model it finds a latent
Gaussian random walk observed with Gaussian noise, integrates it out exactly
with a Kalman filter, and lets NUTS sample only what is left. On the panel
below that takes NUTS from 3,023 dimensions to 23 (G = 20, T = 150) and from
37,753 to 253 (G = 250, T = 150).

```
model RandomWalkPanel {
    data y: Matrix[G, T]

    param pop: Real
    param beta: Vector[G]
    param sigma_w: Positive
    param sigma_y: Positive
    param innov: Matrix[G, T]

    pop     ~ Normal(0, 1)
    beta    ~ Normal(pop, 0.4)
    sigma_w ~ Normal(0, 0.5)
    sigma_y ~ Normal(0, 1)
    innov   ~ Normal(0, sigma_w)

    y ~ Normal(beta + cumsum(innov, T), sigma_y)
}
```

`mintc build examples/random_walk_panel.mint` prints

```
mintc: model RandomWalkPanel: collapsed innov (G x T latent scalars) by a Kalman filter (variances shared by every series: one recursion per time step); NUTS samples pop, beta[G], sigma_w, sigma_y (G + 3 parameters)
```

and the program, when it samples, prints the numbers it found:

```
collapsed: NUTS sampled 23 of the 3023 parameters; 3000 latent scalars were integrated out by a Kalman filter and drawn for each kept draw by forward filtering, backward sampling
```

The posterior still has every parameter, `innov` included, so `print(post)`
and the raw draws (`MINT_DRAWS`) have the same columns as without the
collapse. The gradient tools (`MINT_GRADCHECK`, `MINT_BENCH_GRAD`, and
`MINT_THETA`, which sets the point they evaluate) work on the parameters NUTS
samples, so on the reduced model. `mintc build --no-collapse` samples all of
them with NUTS.

## The rule

A parameter `X` (a `Matrix[G, T]`, one walk per row, or a `Vector[T]`, one
walk) is integrated out when the model contains exactly two statements that
mention it:

```
X ~ Normal(m_w, s_w)
y ~ Normal(a + c * cumsum(B + k * X, T), s_y)
```

where

- `m_w`, `s_w`, `a`, `B`, `k` and `s_y` do not mention `X` and contain no
  running sum; they may use any other parameter or data, indexed by element,
  by series (`Vector[G]`) or by time (`Vector[T]`);
- `c` is a nonzero literal (`a - 0.5 * cumsum(...)`, `-0.5 * cumsum(...)`
  and `cumsum(...) / 4` are fine; a parameter there is not recognised);
- `y` is data: no parameter and no running sum in it;
- inside the running sum `X` enters linearly (`B + k * X`, sums, differences,
  negation, multiplication or division by expressions without `X`);
- the running sum is over `X`'s own shape, and the observation has that shape.

The mean may be written in any order (`cumsum(innov, T) + beta`, `beta -
cumsum(...)`), and `let`s are inlined first, so `let state = cumsum(shared +
innov, T)` qualifies. If two candidates would depend on each other, only one
is collapsed, and nothing is collapsed when NUTS would be left with no
parameter at all (a walk with fixed scales and nothing else unknown). When a
matrix or vector parameter appears in a running sum but the rule fails,
mintc says why and samples the model as written, for example

```
mintc: model DynamicPoisson: innov was not integrated out: its observation is PoissonLog, not Normal (that needs a Laplace approximation, which is not implemented)
```

A drift shared by every series inside the running sum (the `shared:
Vector[T]` of `examples/dynamic_poisson.mint`) makes the series dependent a
priori. It is not integrated out (that needs one filter over all series
together); it stays a NUTS parameter, and given it the series are
independent, so each is still filtered on its own (`tests/kalman/shared.mint`).

## Why it is exact

Given every parameter except `X`, write `x[t] = cumsum(B + k X)[t]` for one
series. Then `x[t] = x[t-1] + e[t]` with `x[-1] = 0` and independent
increments `e[t] = B[t] + k[t] X[t] ~ N(d[t], q[t])`, `d = B + k m_w`,
`q = (k s_w)^2`; the observation is `y[t] ~ N(a[t] + c x[t], r[t])`,
`r = s_y^2`. That is a linear Gaussian state-space model with a scalar state
(a local-level model), and the series are independent. Integrating `X` out
of the joint density gives, per series, the multivariate normal density of
`y` with mean `a + c L d` and covariance `c^2 L diag(q) L' + diag(r)` (`L`
the lower-triangular matrix of ones), which the Kalman filter evaluates in
O(T) instead of O(T^3). The change of variables from `X` to the increments
needs no Jacobian, because only the density of `y` is kept.

Mint drops normalising constants (each `Normal` term lacks -0.5 log 2 pi),
and the collapsed term does too: it is the Gaussian log density of `y`
without them. So the collapsed log density equals log of the integral of the
full one over `X`, up to the constant G T log(2 pi) / 2, which sampling
ignores.

## The filter and its gradient

Forward, per series (`m`, `P`: filtered mean and variance, starting at 0):

```
mp = m + d[t]          Pp = P + q[t]          F = c^2 Pp + r[t]
v  = y[t] - a[t] - c mp
ll += -0.5 log F - 0.5 v^2 / F
K  = c Pp / F          m' = mp + K v          P' = Pp r[t] / F
```

`P' = Pp r / F` is the update `Pp - K c Pp` written so that it stays
positive. Compile-time differentiation in Mint is per observation, and this
recursion couples the observations, so the adjoint is written out by hand
and runs as the filter's reverse pass, from t = T - 1 down, carrying the
adjoints `mb`, `Pb` of `m'` and `P'` (both 0 after the last step):

```
Kb  = mb v                      vb  = mb K - v / F
Ppb = Pb r / F + Kb c / F       rb  = Pb Pp / F
Fb  = -Pb Pp r / F^2 - Kb K / F + 0.5 (v^2 / F - 1) / F
ab  = -vb                       mpb = mb - c vb
Ppb += c^2 Fb                   rb += Fb
a'[t] = ab   d'[t] = mpb   q'[t] = Ppb   r'[t] = rb
mb <- mpb    Pb <- Ppb            (the adjoints of m and P one step back)
```

The runtime kernel (`mint_kalman_ll` in `runtime/mint_rt.c`) takes `a`, `d`,
`q` and `r` time-major (element (g, t) at t G + g), so the loop over series
is contiguous and vectorises (4 series per AVX2 vector), and returns the log
density with these adjoints written over its inputs. The forward pass stores
`Pp`, `1/F` and `v`, so the reverse pass divides by nothing. `sum log F` is
the log of a running product per series whose binary exponent is moved to an
integer counter at every step, so the loop calls no `log`; if any `F` falls
outside [2^-1022, 2^1022] the sum is recomputed with `log`. That fallback is
not exercised by any test, and it does not rescue an `F` below about 5e-309
(a variance that has underflowed), where `1/F` is infinite and the log
density comes out NaN, which the sampler treats as a divergence.

The compiler generates the rest: a loop that evaluates `a`, `d = B + k m_w`,
`q = (k s_w)^2` and `r = s_y^2` for every element from the model's own
expressions, the kernel call, and a loop that pushes the kernel's adjoints
through those expressions with the ordinary per-element reverse sweep
(`k` gets `d' m_w + 2 q' k s_w^2`, `m_w` gets `d' k`, `s_w` gets
`2 q' k^2 s_w`, `s_y` gets `2 r' s_y`). The observations are copied
time-major once per `sample()`, in `init`.

**Shared variances.** When `s_w`, `k` and `s_y` have no element- or
series-indexed leaf (scalars, time-indexed vectors, constants; checked on the
expression tree), `q` and `r` are the same for every series, and so are `Pp`,
`F`, `K` and `P`, which do not depend on `y`. The compiler then calls
`mint_kalman_ll_shared`, which runs that recursion once per time step and
filters only the means per series (a few multiply-adds per element, no
division); the variance recursion's adjoints are summed over series.

## Draws of the integrated-out parameter

For each kept draw the generated `collapse` function maps NUTS's position to
the constrained draw in the user's layout and draws `X` from its conditional
posterior by forward filtering, backward sampling (`mint_kalman_ffbs`):
filter forward storing `m[t]`, `P[t]`; draw `x[T-1] ~ N(m, P)`; then for
t = T-2 down, `x[t] | x[t+1] ~ N(m[t] + J (x[t+1] - m[t] - d[t+1]),
P[t] q[t+1] / Pp[t+1])` with `J = P[t] / Pp[t+1]`; then
`X[t] = (x[t] - x[t-1] - B[t]) / k[t]`. Where `k[t]` is exactly 0 (a data
mask, say) that element does not reach `y`, and it is drawn from its prior
`N(m_w, s_w)` instead. It uses its own random stream per chain, so the NUTS
draws do not depend on it, and the G T normal variates come first from a
ziggurat generator, so the backward pass vectorises. The draws are exact
given each kept draw of the other parameters.

Its cost is not negligible. In a timing harness around the runtime kernels
(load average about 27), one FFBS draw took 21 us at G = 20 and 277 us at
G = 250 (T = 150), about 2.4 and 5.6 gradients' worth, before counting the
generated loops around it. With 19 and 45 gradients per kept draw that is
roughly a tenth to an eighth of the collapsed run. (The benchmark below ran
an earlier FFBS, with the polar method's normals drawn one at a time and an
integer division per element, which the independent review measured at 11
to 21 gradients' worth; its collapsed times are correspondingly
pessimistic.) The sampling times reported include it; the per-gradient
ratios do not.

## Where the code is

- `compiler/src/model.rs`, section "Kalman collapse": `detect_kalman` (the
  rule, on the lowered expression trees), `kal_inputs`, `gen_kalman_logp`,
  `gen_collapse_fn`; `gen_model` builds the reduced model that NUTS samples
  (the same model without `X` and its two statements), and `gen_logp`,
  `gen_init` and `gen_sample_fn` take the plans.
- `runtime/mint_rt.c`, section "Kalman": the two filter kernels, FFBS, and
  `mint_set_collapsed`, through which the sampler stores draws with more
  values than it samples.
- `--no-collapse` (`compiler/src/main.rs`).

## Correctness

All in `tests/run.sh` (section "Kalman collapse"):

- **Exact, small instances** (`tests/kalman/check_marginal.py`): for eight
  models (centred and non-centred walks, a shared drift, per-series scales,
  every filter input depending on parameters with `c = -0.5`, a collapse
  beside a Poisson walk in the scan layout, one series as a `Vector[T]`, and
  two collapses in one model) on panels of 2 x 5, 3 x 7 and 9 x 4, at three
  random points each, the compiled log density equals a dense Gaussian
  computation in numpy (`X` integrated out analytically, as above) to at most
  8e-16 relative, and every gradient component agrees with a 5-point finite
  difference of the numpy density to at most 8e-10 (relative to the
  component, or absolute when it is below 1). The compiled
  gradient also passes the runtime's own finite-difference check (worst 5e-9).
  The runtime gained `MINT_THETA=file` to evaluate at a given point.
- **Draws of the walk, exact** (`tests/kalman/check_ffbs.py`): three models
  with fixed scales and one unrelated parameter, so that the walk's posterior
  is a known Gaussian (an element-wise coefficient that is 0 at three
  elements and `c = -0.5` written as a negated literal, on the general
  kernel; one `Vector[T]` walk; two collapses in one model). 10,000 draws:
  every mean and every covariance entry between two times of a series (24
  to 30 means and 36 to 129 covariances per model) is within 2.8 standard
  errors of the exact value; the test fails above 4.5. This checks the joint
  distribution, which the per-column posterior comparison below does not.
  With the backward step's conditioning on `x[t+1]` removed (each `x[t]`
  drawn from its filtered marginal) it fails with |z| up to 195. The normal
  generator is checked on 2e7 variates (`tests/kalman/zig_test.c`: first four
  moments and 42 binned probabilities).
- **Posterior** (`tests/kalman/compare_posterior.py`): on four models
  (G = 6, T = 30), 3 seeds x 4 chains x 1000 draws of the collapsed build and
  of the `--no-collapse` build, every quantity in the draws (the remaining
  parameters and every innovation, the latter from FFBS) has the same mean
  and standard deviation within Monte Carlo error: largest |z| over 189 to 367
  quantities per model between 2.4 and 3.7 for means and 2.3 and 3.7 for
  standard deviations, median |z| 0.56 to 0.86 (0.67 is what calibrated
  standard errors give). The threshold is 4.5. Standard errors come from the
  ESS of the draws (means) and of the squared deviations (standard
  deviations). With FFBS's backward variance broken on purpose the test
  failed with |z| of 28 to 73 on the standard deviations. Caveats: it
  compares each column on its own, not the joint distribution (check_ffbs.py
  does that, with fixed scales); full NUTS is only a usable reference where
  it copes, so the data were chosen for that (weakly informative, non-centred
  forms, target acceptance 0.95; one series, `single.mint`, is left out
  because full NUTS had hundreds of divergences and R-hat 1.1 there); and the
  4.5 threshold was set after a first run had shown a largest |z| of 3.5, so
  it is not a pre-registered test.
- **Interplay**: a collapse whose walk shares a running sum with a NUTS
  parameter next to a fused scan kernel, built with `--fused-leapfrog`
  (`tests/scan/twoowned.mint`): one fused leaf matches the runtime's, and the
  gradient passes finite differences. (This found a bug: the scan kernel
  claimed sole ownership of that parameter's gradient, though the filter adds
  to it afterwards; fixed.)
- **Eligibility**: nine models that must not be collapsed build, print the
  reason, and run to the end sampled as written.
- `--strict-fp` gives the example's log density and gradient to 1e-12, and
  the collapsed example sampled with 3 threads per chain gives finite draws
  and the same posterior mean of sigma_w.
- The earlier scan-kernel, parallel-kernel, fused-leapfrog and narrow-data
  tests now build with `--no-collapse`: several of their models are Gaussian
  random walks, and they exist to test the code that samples them as written.

Not tested: simulation-based calibration (milestone 1 asks for it); the
flagship's level-plus-slope and seasonal states (below); the collapse under
the randomised narrow-data check (it runs with `--no-collapse`; the filter's
code reads doubles only) and with the fused scan kernel on several threads
other than `twoowned`.

## Measurements

`bench/kalman/bench.py` simulates one panel per size from the model
(`make_data.py`: sigma_w = 0.1, sigma_y = 0.3, intercepts around a pooled
mean) and runs `examples/random_walk_panel.mint`, unmodified, three ways with
sampler seeds 1, 2 and 3, interleaved: collapsed (the default build), full
NUTS on the same file (`--no-collapse`), and full NUTS on the non-centred
form (`innov ~ Normal(0, 1)`, `cumsum(sigma_w * innov, T)`; the usual fix
for the centred form's funnel, and the better of the two here). 4 chains x
1000 draws after 1000 warmup iterations, Stan's warmup, the runtime's default
threads per chain (1 for the collapsed model, which has 23 or 253
parameters; 3 for the full model at G = 250, which has more than 8,192). The
"remaining parameters" are pop, every beta, sigma_w and sigma_y: what both
samplers estimate. ESS is the runtime's (Geyer's initial monotone sequence);
the lowest over beta comes from its printout. The full draws were the same in
both runs below (the sampler is deterministic for a seed and thread count);
only the times differ.

Other agents' jobs shared the 12-core, 24-thread Ryzen 9 5900X throughout:
the load average was 15.6 to 23.5 in the first run and 23.5 to 32.0 in the
second, so wall times, and so ESS per second, are noisy (the same full run
took 67 s in the first and 470 s in the second). Gradients per effective
draw do not depend on load and are the steadier comparison.
`bench/kalman/results.json` (second run) and `results_run1.json` (first)
hold everything; `report.py` prints the tables.

**G = 20, T = 150** (3,000 latent scalars).

| | NUTS dimension | gradients per kept draw | leapfrog steps per draw | lowest ESS, remaining parameters | per 1000 gradients | sampling time, s (run 1 / run 2) | lowest ESS per second (run 1 / run 2) | highest R-hat of pop, sigma_w, sigma_y |
|---|---|---|---|---|---|---|---|---|
| collapsed | 23 | 19 | 7.0 | 5,503 to 6,857 | 74 to 91 | 0.38 to 0.41 / 0.41 to 0.49 | 13,900 to 16,700 / 12,500 to 15,600 | 1.000 |
| full NUTS, centred | 3,023 | 416 to 466 | 208 to 255 | 201 to 214 (sigma_w) | 0.12 | 4.9 to 6.2 / 4.6 to 6.6 | 33 to 44 / 31 to 44 | 1.007 to **1.034** |
| full NUTS, non-centred | 3,023 | 505 to 510 | 255 | 1,261 to 1,463 (sigma_w) | 0.62 to 0.72 | 5.6 to 6.8 / 5.2 to 7.0 | 185 to 262 / 210 to 243 | 1.001 to 1.002 |

**G = 250, T = 150** (37,500 latent scalars).

| | NUTS dimension | gradients per kept draw | leapfrog steps per draw | lowest ESS, remaining parameters | per 1000 gradients | sampling time, s (run 1 / run 2) | lowest ESS per second (run 1 / run 2) | highest R-hat of pop, sigma_w, sigma_y |
|---|---|---|---|---|---|---|---|---|
| collapsed | 253 | 45 to 46 | 15.0 | 6,459 to 7,498 | 35 to 42 | 2.8 to 4.8 / 4.0 to 7.3 | 1,350 to 2,440 / 890 to 1,640 | 1.000 |
| full NUTS, centred | 37,753 | 815 to 838 | 423 to 447 | 210 to 292 (sigma_w) | 0.063 to 0.089 | 67 to 185 / 173 to 470 | 1.1 to 4.4 / 0.45 to 1.7 | 1.008 to 1.013 |
| full NUTS, non-centred | 37,753 | 840 to 966 | 383 to 471 | 931 to 1,307 (sigma_w) | 0.24 to 0.37 | 68 to 208 / 209 to 500 | 4.7 to 18 / 2.0 to 4.5 | 1.000 to 1.005 |

Ratios, collapsed over full NUTS, per seed (range over the three seeds):

| | lowest remaining ESS per 1000 gradients | lowest remaining ESS per second, run 1 / run 2 |
|---|---|---|
| G = 20, against centred | 640 to 790 | 320 to 500 / 300 to 450 |
| G = 20, against non-centred | 100 to 140 | 53 to 90 / 60 to 64 |
| G = 250, against centred | 400 to 670 | 550 to 2,150 / 970 to 2,350 |
| G = 250, against non-centred | 113 to 146 | 140 to 290 / 280 to 450 |

How to read these:

- **Where the gain comes from.** NUTS on 23 or 253 well-scaled parameters
  takes 7 or 15 leapfrog steps per draw instead of 200 to 470, and its draws
  of sigma_w are nearly independent (ESS 5,900 to 8,900 of 4,000 draws)
  instead of 200 to 1,500. Full NUTS's slowest quantity is always sigma_w,
  the scale of the walk it has to explore jointly with the 3,000 or 37,500
  innovations. The centred full model at G = 20 did not reach R-hat 1.01 on
  sigma_w in two of three seeds.
- **Cost per gradient is higher, not lower.** One gradient on one thread
  (`MINT_BENCH_GRAD`, fastest of 7 alternated repetitions, load about 27):
  8.7 us collapsed against 2.5 us full at G = 20, and 49 us against 28 us at
  G = 250. The filter does more per element than the fused scan kernel's
  running sum, and the generated loops around it are not fused with it. The
  collapse wins on gradients per effective draw, by a factor of 100 to 800,
  not on the gradient. Those ratios count gradients only: FFBS, in the form
  the benchmark ran, cost the equivalent of another 11 to 21 gradients per
  kept draw (see above), which would shrink them by about 1.3 to 1.7x; with
  the current FFBS, by about 1.1x.
- **Threads.** At G = 250 the full model ran 3 threads per chain (12 in
  all) and the collapsed one 1 per chain (4 in all), so the full model had
  the larger budget; on a loaded machine that also exposes it to more
  contention, which is part of why its wall times vary so much. Gradients per
  effective draw are unaffected.
- **Included in the collapsed times:** drawing all 3,000 or 37,500
  innovations for each of the 4,000 kept draws by FFBS. Not included in any
  time: compilation, 0.1 to 0.8 s for the collapsed build and 0.7 to 2.9 s
  for the full ones (whose scan kernel gets narrow-data variants), under
  this load.
- The innovations' own draws: lowest ESS 2,810 to 3,084 over all of them in
  the collapsed runs, highest R-hat 1.004.

## Limitations

- **Only the local-level model.** One scalar state per series: a random walk
  of the innovations. The flagship model's level plus slope and weekly
  seasonality need a vector state (8 per series) and its filter; the
  detection would have to recognise a running sum of a running sum and a
  sum-to-zero seasonal pattern. Not done.
- **Gaussian observations only.** Poisson and Student-t observations need the
  Laplace collapse (milestone 3).
- **A walk shared across series** (a `Vector[T]` added to every series inside
  the running sum) is sampled by NUTS, not integrated out.
- **Literal coefficient on the running sum.** `c` must be a number; a
  parameter there is not recognised.
- **Underflowed variances.** An observation or innovation variance below
  about 5e-309 gives a NaN log density (see the filter section).
- **Cost per gradient.** The filter costs more per element than the fused
  scan kernel's running sum, and the generated input and adjoint loops are
  plain loops around a runtime call rather than one fused kernel: see the
  gradient timings above.
- **Memory.** Draws of the collapsed parameter are still stored for every
  kept draw (milestone 0, streaming draws, is not done).
- **Threads.** The filter runs on the chain's own thread; it is not split
  across the chain's threads as the scan kernel is.
