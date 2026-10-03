<img src="docs/assets/tony/src-logo.png" width="150" alt="Tony, Martin's bald eagle mascot">

# Martin

Martin is a prototype language for Bayesian inference and small numerical
programs. You write the model as mathematics, with its intent stated in the
types: shapes, positivity, symmetric positive definiteness, which quantities
are data and which are unknown. The compiler uses those facts to reject wrong
programs, to derive gradients, to lay out memory, and to decide what work the
inference does not need at all.

Martin has its own compiler, `mintc`, written in Rust: a parser, type checker,
compile-time differentiation, layout and fusion passes, and an LLVM IR
emitter. LLVM turns that IR into machine code; nothing interprets a Martin
program while it runs. A small runtime library, written in C, supplies file
I/O, printing, a Cholesky solve and the NUTS sampler that calls the compiled
model. (The tools still carry the earlier name: the compiler is `mintc`,
programs end in `.mint`, and the runtime's variables start with `MINT_`.)

It is a research prototype: one machine (Ryzen 9 5900X, Linux), a handful of
models, and every number below comes from files in `bench/`.

## Speed

**Same sampler, different languages.** Martin, hand-written Rust and Stan's
C++ (through BridgeStan) were run under the same NUTS implementation, with the
same warmup, settings and seeds; only the code computing the log density and
its gradient differs, and all of them agree on the gradient to about 3e-14
([rules and full results](bench/same_sampler/README.md)).

Time per gradient, µs:

| model | Martin | Rust, max effort | Rust with Martin's tricks | Stan |
|---|---|---|---|---|
| time series, 3,171 parameters | **3.56** | 4.10 | 3.73 | 51.0 |
| time series, 37,901 parameters | **44.4** | 52.9 | 46.6 | 654 |
| same, 3 threads | **17.0** | single-threaded | 18.4 | single-threaded |
| logistic regression (n = 5000, p = 20) | **23.6** | 38.0 | | 42.7 |
| eight schools | 0.025 | 0.025 | | 0.50 |

Whole runs, 4 chains, 1000 warmup + 1000 draws (medians over 5 seeds, 3 for
the large model):

| model | Martin | Rust, max effort | Rust with Martin's tricks | Stan |
|---|---|---|---|---|
| time series, 3,171 parameters | **3.32 s** | 3.76 s | 3.50 s | 28.3 s |
| time series, 37,901 parameters | **62.4 s** | 98.6 s | 65.2 s | 737 and 741 s |
| logistic regression | **0.38 s** | 0.59 s | | 0.64 s |
| eight schools | 5.5 ms | 5.7 ms | | 16.6 ms |

- "Rust, max effort" is hand-written with AVX2 intrinsics and glibc's vector
  math. "Rust with Martin's tricks" also has Martin's table-driven `exp` and
  splits its gradient across threads; against it Martin's lead is 5 to 8%.
  The Rust keeps the parameter order the shared sampler fixes, so it cannot
  use the column-major layout Martin's compiler chooses; nobody has
  established what the remaining gap comes from.
- Stan's gradient on the large model ran on one thread (no `reduce_sum`
  version was written), and its large runs were not interleaved with the
  others.
- Newton's method for logistic regression (200,000 rows, 50 features, 10
  iterations) takes 0.114 s against 0.209 s for the max-effort Rust
  ([merged build](bench/merged_vs_ref.md); the Rust figure was measured once
  and frozen).
- The Martin programs are 15 to 18 lines; the Rust is 92 to 105 lines plus
  shared SIMD helpers, and 591 lines for the time series.

## Work the compiler removes

**A latent random walk integrated out** ([details](docs/kalman.md)). In
`examples/random_walk_panel.mint`, a panel of Gaussian random walks observed
with Gaussian noise and unknown scales, `mintc` finds the walk, integrates it
out exactly with a Kalman filter per series, and NUTS samples the 23 (G = 20)
or 253 (G = 250) remaining parameters instead of 3,023 or 37,753; the walk is
drawn back afterwards. Over three seeds the lowest effective sample size of
the remaining parameters per gradient was 100 to 146 times that of full NUTS
on the non-centred form, and 400 to 790 times on the centred form as written,
though each collapsed gradient costs 1.8 to 3.5 times as much. The log density
matches a dense Gaussian computation to about 1e-15 on small panels, and the
posterior matches full NUTS within Monte Carlo error. Only a local-level walk
with Gaussian observations is recognised so far.

**Sampler options** ([details](docs/architecture.md)). Two opt-in changes to
the runtime's NUTS cut the gradients needed per effective draw:
`MINT_WARMUP=fast` (an L-BFGS starting point, a shorter warmup, chains pooling
their adaptation) and `MINT_METRIC=lowrank` (Stan's diagonal metric plus up to
24 directions estimated from warmup gradients). On the small time-series
model, one seed, effective draws per second went from 517 with the defaults
to 850 with the fast warmup and 1,470 with the low-rank metric
([data](bench/wave_benchmarks.json)). They are not the defaults yet: four
models is too few to rule out posteriors where they do worse. They are part
of the shared runtime, so they speed up the Rust baselines too.

## Two examples

Newton's method for L2-regularised logistic regression. The checker proves `H`
is symmetric positive definite, so `solve` compiles to a Cholesky solve;
`diag(...)` is never built as an n-by-n matrix, and only one triangle of the
Gram product is computed.

```
fn fit(X: Matrix[n, p], y: Vector[n], lambda: Positive) -> Vector[p] {
    let mut w = zeros(p)
    repeat 10 {
        let mu = sigmoid(X * w)
        let g  = X' * (mu - y) + lambda * w
        let H  = X' * diag(mu .* (1 - mu)) * X + lambda * I(p)
        w = w - solve(H, g)
    }
    w
}
```

Replace `H` with `X' * X` and the program no longer compiles:

```
error: solve needs a matrix the compiler can prove is SPD, but it is only known to be PSD, which allows it to be singular
help: SPD comes from I(p), from PSD + (Positive * I(p)), from sums and positive multiples of SPD matrices, or from assume_spd(A), which is checked at run time
```

Bayesian logistic regression, sampled with NUTS. The gradient is derived by
the compiler:

```
model Logistic {
    data X: Matrix[n, p]
    data y: Vector[n]

    param alpha: Real
    param beta: Vector[p]

    alpha ~ Normal(0, 2.5)
    beta  ~ Normal(0, 1)
    y     ~ BernoulliLogit(alpha + X * beta)
}

fn main() {
    let X: Matrix[n, p] = read("data/logit_X.f64")
    let y: Vector[n]    = read("data/logit_y.f64")
    let post = sample(Logistic(X, y), draws = 1000, warmup = 1000, chains = 4, seed = 7)
    print(post)
}
```

The straightforward Rust equivalent of the model block is a hand-written
`logp` function with the gradient worked out on paper: priors and their
derivatives, the stable softplus, the residuals `y - sigmoid(eta)`, and the
`X' r` accumulation. See `baselines/logistic_bayes.rs`. Declare a scale
parameter `Real` instead of `Positive` and Martin refuses it:

```
error: the scale of Normal must be Positive, but `sigma` is Real
help: declare the parameter as `param sigma: Positive`; Martin then samples log(sigma) and adds the Jacobian itself
```

All examples are in `examples/`: `logistic_newton.mint`, `logistic_bayes.mint`,
`linear_bayes.mint`, `eight_schools.mint`, `dynamic_poisson.mint` and
`random_walk_panel.mint`. Programs that must fail to compile
are in `examples/errors/`.

## Quick start

Requirements: Rust (cargo), clang with its OpenMP runtime (`libomp`), and
glibc's vector math library (`libmvec`). It has been tested only on Linux x86-64 with rustc 1.89,
clang 22.1 and glibc 2.44; other versions are untested.

```sh
./bench/setup.sh            # builds mintc and writes the example datasets to data/
./compiler/target/release/mintc build examples/logistic_bayes.mint -o build/logistic_bayes
./build/logistic_bayes
```

`mintc check FILE` type-checks only; `mintc emit FILE` writes the LLVM IR;
`mintc explain FILE` reports what the compiler found and did (below).

The full benchmark (it builds everything, generates data and runs about 6
minutes; the max-effort baselines need nightly Rust) and the test suite:

```sh
python3 bench/bench.py 7    # writes bench/results.json
./tests/run.sh              # compile errors, runtime checks, gradients, posteriors
(cd compiler && cargo test --release)
```

Environment variables for compiled programs (threads, draws, the sampler's
metric and warmup) and the compiler switches that turn each optimisation
off are listed in [docs/options.md](docs/options.md).

## What the compiler did: `mintc explain`

`mintc explain FILE [flags]` compiles the program as `build` does, with the
same flags (a test checks that `explain -o` writes the same IR as `emit`),
and prints what the compiler decided: per model, the parameters and their
transforms, how each `~` statement was compiled (including which latent
random walk the Kalman collapse integrated out, with which filter, and what
NUTS samples instead, or why a candidate was refused), the data layouts and
the data that may be read narrow; per function, the row fusion groups, the
kernels and the type facts behind `solve`. The decisions and their reasons
are recorded by the code generators where they make them; the descriptions
of what a chosen kernel does are fixed text kept next to its code. Sizes that
depend on the data are symbolic. An excerpt for
`examples/dynamic_poisson.mint` (`...` marks omitted lines):

```
model DynamicPoisson
  ...
  parameters: NUTS samples G x T + G + T + 1 unconstrained values
  ...
    line 20: y ~ PoissonLog(beta + state)
      with state = cumsum(shared + innov, T)
      ...
      fused scan kernel over Matrix[G, T]: 1 running sum along T, each carried in a register along its row
      vector code: 4 lanes (<4 x double>), one row per lane, in groups of 8 rows (2 vectors); ...
      ...
      absorbed: line 17 (innov ~ Normal(0, 0.08)), in the reverse loop
      owned gradient: innov; ...
```

and for the Newton example, why `solve` may use Cholesky:

```
      solve(H, g): Cholesky solve (mint_chol_solve), allowed because H is proved SPD:
        H = X' * diag(mu .* (1 - mu)) * X + lambda * I(p) is SPD: PSD + SPD
          X' * diag(mu .* (1 - mu)) * X is PSD: a Gram product with weights >= 0 (the weights are Prob)
            mu .* (1 - mu) is Prob: Prob .* Prob
            ...
          lambda * I(p) is SPD: Positive * SPD
```

Choices that depend on the data values, such as narrow copies, are reported
as the checks that `sample()` makes when it starts.

## The language in one page

**Types.**

- Scalars: `Real`, `Positive`, `Prob` (the open interval 0 to 1), `Int`.
- Vectors: `Vector[n]`, and vectors whose entries lie in a domain:
  `Positive[n]`, `Prob[n]`.
- Matrices: `Matrix[m, n]`, `PSD[n]`, `SPD[n]`.

Dimension names such as `n` are bound by function parameters or by an annotated
`read`, and are checked everywhere after. A vector combined with a matrix is
matched to the dimension with the same name: a `Vector[G]` plus a
`Matrix[G, T]` repeats across `T`. If both dimensions have the same name, that
is a compile error, because it is ambiguous.

**Expressions.**

- `+ - * /` and `^`. The elementwise operators are `.*` and `./`, and `A'` is
  the transpose.
- `*` means the mathematical product. Between a scalar and anything it scales;
  `A * v` is a matrix-vector product and `v' * w` is a dot product. Between two
  vectors it is an error, because it is ambiguous.
- Functions: `exp`, `log`, `log1p`, `sqrt`, `sigmoid` and `abs` apply
  elementwise; `sum`, `dot` and `norm` reduce.
- `cumsum(v)` and `cumsum(M, T)` are running sums along the last dimension.
- Constructors: `zeros(p)`, `ones(p)`, `I(p)`, `diag(w)` (inside a product
  only), and literals like `[1, 2, 3]`.
- `solve(H, g)`, which needs a proved-SPD `H`.
- `assume_spd(A)`, which checks symmetry and positive definiteness at run
  time.

**Statements.**

- `let x = ...`, `let mut x = ...`, `x = ...`
- `repeat N { ... }`
- `print(...)`
- `let X: Matrix[n, p] = read("file.f64")`
- `clock()`

**Models.**

- `data` and `param` declarations. Parameters are `Real`, `Positive`,
  `Vector[n]`, `Positive[n]` or `Matrix[m, n]`. Positive parameters are
  sampled on the log scale, with the Jacobian added for you.
- `let` definitions and `x ~ Distribution(...)` statements, with `Normal`,
  `BernoulliLogit`, `PoissonLog` and `Exponential`. The two sides of a `~` can
  be scalars, vectors or matrices.
- `sample(Model(data...), draws =, warmup =, chains =, seed =)` runs NUTS with
  Stan's warmup and returns a posterior; `print` summarises it (mean, sd,
  quantiles, ESS, split R-hat).

**Data files** (`.f64`) hold two little-endian u64 values (rows, cols),
followed by row-major little-endian f64.

## Documents

- [Same-sampler benchmark](bench/same_sampler/README.md): the rules, and
  Martin, Rust and Stan under one NUTS
- [Hierarchical time series against rustmc and Stan](docs/hierarchical.md)
- [Benchmark report](docs/benchmark.md) and
  [compiler rounds](docs/compiler-round.md)
- [Compiler architecture](docs/architecture.md)
- [Kalman collapse](docs/kalman.md): the compiler integrates a latent
  Gaussian random walk out of a model
- [Runtime and compiler options](docs/options.md)
- [Flagship demo plan](docs/flagship-demo.md): where the project is going
- [Next milestone](docs/next-milestone.md)
- [Tony](docs/mascot.md): the mascot and its artwork
