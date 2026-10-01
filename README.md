# Mint

Mint is a prototype language for small numerical programs. You write the
mathematics and its intent: shapes, positivity, symmetric positive
definiteness, which quantities are data and which are unknown. The compiler
uses those facts to reject wrong programs and to choose faster code.

It has its own compiler (`mintc`: parser, type checker, IR, compile-time
automatic differentiation, loop fusion and fission, a sufficient-statistics
rewrite, and an LLVM IR emitter). LLVM turns that IR into machine code. Like
most languages, Mint also has a small runtime library, written in C and
compiled once. It supplies file I/O, printing, a Cholesky solve and the NUTS
sampler that repeatedly calls your compiled model. Building a program takes
60 to 160 ms.

## What the prototype shows

**Against Stan and rustmc, on a hierarchical time-series model with up to
37,901 parameters** ([details](docs/hierarchical.md)). A panel of Poisson count
series has a random walk per series and pooled intercepts. It is 18 lines of
Mint, and the compiler derives the gradient.

| 3,171 parameters, 4 chains × 1000 draws | wall time | converged? | effective draws per second |
|---|---|---|---|
| Mint | 6.6 s | yes (R-hat 1.003) | 387 |
| Stan (stanc `--O1`) | 48.5 s | yes (R-hat 1.002) | 38.3 |
| rustmc (elliptical slice) | 7.0 s | **no** (R-hat 1.95) | not usable |
| hand-written SIMD Rust, same sampler as Mint | 6.6 s | yes | 300 |

- Mint and Stan both converged; their posterior means agree (details in the
  report). That is means only, not variances or tails.
- The effective-draws figure moves with the seed: across three seeds Mint's
  lowest ESS ranged from 1,600 to 2,600. The steadier comparison is the cost
  per gradient including the sampler: 13 µs for Mint against 97 µs for Stan.
- rustmc's chains did not converge in these runs (1000 warmup + 1000 sweeps),
  and its means are off by up to 0.9 posterior standard deviations.
- At 37,901 parameters Mint's run took 226 to 238 s (two runs) and was right
  at the mixing bar; Stan's shorter run did not reach it. See the report for
  why the per-gradient figures there are not a like-for-like comparison.

**Against hand-written Rust** ([details](docs/benchmark.md),
[this round](docs/compiler-round.md)):

| problem | Mint | straightforward Rust | tuned Rust | max-effort Rust |
|---|---|---|---|---|
| logistic gradient (n=5000, p=20) | **33 µs** | 178 µs | 139 µs | 37 µs |
| logistic, full NUTS run | **0.49 s** | 2.57 s | 1.92 s | 0.56 s |
| Newton's method (n=200000, p=50) | **0.16 s** | 2.33 s | 0.40 s | 0.21 s |
| time-series gradient, 3,171 / 37,901 parameters | 4.22 / 53.4 µs | | | **4.07 / 52.2 µs** |
| lines of code | 15 to 18 | 59 to 78 | 66 to 97 | 92 to 105 plus SIMD helpers; 591 for the time series |

The original question for this prototype was whether Mint can be clearer than
Rust and within 2x of straightforward Rust. The answer is yes: Mint is 5 to
14x faster than straightforward Rust on these problems. Against expert Rust
written with SIMD intrinsics and glibc's vector math, Mint is now 1.15x faster
on the logistic gradient and 1.3x faster on Newton, and 2 to 4% slower on the
time-series gradient (it was 1.8x slower). The Rust takes roughly 10 to
33 times as much code (counting its shared SIMD helpers), including
hand-derived gradients.

That is not a claim that Rust cannot be as fast: every technique Mint's
compiler uses could be written by hand in Rust (see the caveats in
[compiler-round.md](docs/compiler-round.md)). The case for Mint is that its
compiler produces this from a few lines of mathematics, with gradients derived
for you and with shape, positivity and SPD errors caught before anything runs.

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
parameter `Real` instead of `Positive` and Mint refuses it:

```
error: the scale of Normal must be Positive, but `sigma` is Real
help: declare the parameter as `param sigma: Positive`; Mint then samples log(sigma) and adds the Jacobian itself
```

All examples are in `examples/`: `logistic_newton.mint`, `logistic_bayes.mint`,
`linear_bayes.mint`, `eight_schools.mint` and `dynamic_poisson.mint`. Programs that must fail to compile
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

`mintc check FILE` type-checks only; `mintc emit FILE` writes the LLVM IR.

The full benchmark (it builds everything, generates data and runs about 6
minutes; the max-effort baselines need nightly Rust) and the test suite:

```sh
python3 bench/bench.py 7    # writes bench/results.json
./tests/run.sh              # compile errors, runtime checks, gradients, posteriors
(cd compiler && cargo test --release)
```

Useful environment variables for compiled programs:

- `MINT_GRADCHECK=1` compares the compiled gradient with finite differences.
- `MINT_BENCH_GRAD=K` times K gradient evaluations and exits.
- `MINT_THREADS_PER_CHAIN=N` sets how many threads each chain's sampler
  passes use. The default is 1 below 8,192 parameters. The same threads
  also share the fused scan kernel of the model gradient.
- `MINT_KERNEL_THREADS=N` overrides the number of threads the fused scan
  kernel uses: during sampling (default: the threads per chain) and in
  `MINT_BENCH_GRAD` and `MINT_GRADCHECK` (default 1).
- `MINT_CHAIN_AFFINITY=0` stops the runtime from keeping each threaded
  chain's threads on the CPUs of one L3 cache.
- `MINT_METRIC=grad` switches to the experimental gradient-based metric
  adaptation.
- `MINT_METRIC=lowrank` adds to Stan's diagonal metric up to 8, 16 or 24
  directions, depending on the model's size (`MINT_LOWRANK_K` sets another
  number), estimated from the gradients of the warmup draws; see
  [hierarchical.md](docs/hierarchical.md) for what it gains and costs.

Compiler switches, each turning one optimisation off (for measuring it):
`--no-suffstats`, `--no-fission`, `--no-vecmath`, `--no-gram-blocking`,
`--no-scan-layout`, `--no-scan-fusion`, `--no-inline-exp`, `--no-row-fusion`,
`--no-fission-kernel`, `--no-inline-log`, `--no-parallel-kernel`, and `--strict-fp`
(strict IEEE evaluation order, no vector math).

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

- [Hierarchical time series against rustmc and Stan](docs/hierarchical.md)
- [Benchmark report](docs/benchmark.md)
- [Compiler architecture](docs/architecture.md)
- [Next milestone](docs/next-milestone.md): the minimum work needed to test
  whether first-class mathematical types enable useful optimisations in
  general, not only on examples I chose.
