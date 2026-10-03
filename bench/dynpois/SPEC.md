# Benchmark spec: hierarchical dynamic Poisson panel

Target: rustmc's `BayesianDynamicPoisson` (integrate/forecasting-toolbox,
`rust_core/src/dynamic_glm.rs`, `decode` lines ~200-240) with intercept only
(K = 0 covariates), exposure 1, and these fixed scales (rustmc cannot estimate
them, so every implementation fixes them):

| rustmc config field | value |
|---|---|
| initial_mean | 0.0 |
| coefficient_sd | 1.0 |
| group_sd | 0.4 |
| process_sd | 0.08 |
| shared_process_sd | 0.05 |

## Model (identical posterior in every implementation)

For groups g = 1..G and times t = 1..T:

```
pop            ~ Normal(0, 1)
beta[g]        ~ Normal(pop, 0.4)
shared[t]      ~ Normal(0, 0.05)
innov[g, t]    ~ Normal(0, 0.08)
state[g, t]    = sum_{s <= t} (shared[s] + innov[g, s])      (the innovation at t is included)
y[g, t]        ~ Poisson(exp(beta[g] + state[g, t]))
```

This is rustmc's non-centred model written in the natural (centred) form:
rustmc's `population = 0 + 1 * z0`, `beta_g = population + 0.4 * z_g`,
`level += 0.05 * z_shared[t] + 0.08 * z_g[t]` gives exactly this distribution
over (pop, beta, state).

## Unconstrained parameter layout (Martin and the hand-written Rust baseline)

`theta = [pop, beta[0..G], shared[0..T], innov[0..G*T] (row-major: g*T + t)]`,
dimension D = 1 + G + T + G*T.

## Log density to implement exactly (so implementations agree to 1e-9)

Constants dropped: 0.5*log(2*pi) per Normal term and log(y!) per Poisson term.
Every Normal term keeps its `-log(scale)`.

```
lp  = -0.5*pop^2 - log(1)
    + sum_g  [ -0.5*((beta[g] - pop)/0.4)^2  - log(0.4)  ]
    + sum_t  [ -0.5*(shared[t]/0.05)^2       - log(0.05) ]
    + sum_gt [ -0.5*(innov[g,t]/0.08)^2      - log(0.08) ]
    + sum_gt [ y[g,t]*eta[g,t] - exp(eta[g,t]) ],   eta[g,t] = beta[g] + state[g,t]
```

## Data

`bench/dynpois/make_data.py G T SEED OUTDIR` writes:

- `OUTDIR/y.f64`: Martin .f64 format (two little-endian u64 rows=G, cols=T, then
  row-major little-endian f64), counts.
- `OUTDIR/y.npy`: same array.
- `OUTDIR/truth.json`: true pop, beta[G], terminal state[G] (= state[g, T]).

Data generating process (seeded numpy `default_rng(SEED)`): pop = 1.5,
beta[g] = pop + N(0, 0.4), shared innovations N(0, 0.05), group innovations
N(0, 0.08), state as above, y ~ Poisson(exp(beta + state)).

Sizes: `small` G=20, T=150 (D = 3,171; the size rustmc's own review used) and
`large` G=250, T=150 (D = 37,901).

## Runs and reporting

4 chains run in parallel, seeds fixed, for every implementation. Each runner
writes `bench/dynpois/results/<impl>_<size>.json`:

```
{ "implementation": str, "G": int, "T": int, "chains": int, "warmup": int,
  "draws": int (retained per chain), "thin": int,
  "wall_seconds": float (whole fit call, excluding data loading),
  "gradients": int or null, "notes": str }
```

and `bench/dynpois/results/<impl>_<size>_draws.npz` with arrays of shape
(chains, draws): `pop`, and (chains, draws, G): `beta`, `terminal`
(terminal = state[g, T], excluding beta). A single script,
`bench/dynpois/analyze.py`, computes rank-normalised split R-hat and bulk/tail
ESS (ArviZ) for pop, beta[g], terminal[g] from every npz, plus posterior means,
agreement between implementations, and ESS per second. The same diagnostic code
is applied to every implementation.
