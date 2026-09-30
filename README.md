# Mint

Mint is a prototype language for small numerical programs. You write the
mathematics and its intent: shapes, positivity, symmetric positive
definiteness, which quantities are data and which are unknown. The compiler
uses those facts to reject wrong programs and to choose faster code.

It has its own compiler (`mintc`: parser, type checker, IR, compile-time
automatic differentiation, loop fusion and fission, a sufficient-statistics
rewrite, and an LLVM IR emitter). LLVM turns that IR into machine code. A
small C runtime supplies I/O, a Cholesky solve and a NUTS sampler.

## The gate, answered

The question set for this prototype had two parts. Can this syntax express a
realistic small numerical problem more clearly than equivalent Rust? And does
it run within about 2x of a straightforward Rust implementation?

**Yes on both counts, for the three examples tested.** Measured on one machine
against Rust baselines I wrote myself (see [the benchmark report](docs/benchmark.md)
for the method, the full tables and the caveats):

| problem | Mint | straightforward Rust | hand-tuned Rust |
|---|---|---|---|
| logistic regression, time per gradient (n=5000, p=20) | 46 µs | 178 µs | 139 µs |
| logistic regression, full NUTS run, same sampler | 0.68 s | 2.56 s | 1.91 s |
| Newton's method, logistic regression (n=200000, p=50) | 0.35 s | 2.35 s | 0.40 s |
| linear regression, time per gradient (n=50000, p=20) | 100 ns | 897 µs | 258 ns (with the same rewrite by hand) |
| linear regression, full NUTS run including preparation | 8.9 ms | 16.7 s | 14.9 ms (same rewrite) |
| lines of code, three examples | 15 to 18 | 59 to 78 | |

These are medians of 7 interleaved runs, and no min-max range overlaps its
neighbour in this table. Across four logistic problem shapes, Mint's gradient
is 2.2x to 3.1x faster than the tuned Rust.

Where that comes from:

- **Logistic regression.** Loop fission lets the density and gradient pass
  vectorise, including `exp` and `log`. Dot products are allowed to
  reassociate.
- **Newton.** A Gram kernel chosen because the compiler knows it is computing
  `X' diag(w) X`.
- **Linear regression.** An algebraic rewrite to sufficient statistics that the
  compiler applies because it knows which quantities are data.

The honest reading: nothing here is beyond what a determined Rust programmer
could write by hand. My tuned baselines stay in plain scalar Rust and are still
1.1x to 3x slower. Rust using SIMD intrinsics, a vector math crate and the same
loop structure could close that gap. The difference is that Mint gets there
from source that reads like the mathematics, with no hand-derived gradients,
and that its type checker rejects several classes of mistake before anything
runs.

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
`linear_bayes.mint` and `eight_schools.mint`. Programs that must fail to compile
are in `examples/errors/`.

## Quick start

Requirements: Rust (cargo), clang, and glibc's vector math library
(`libmvec`). It has been tested only on Linux x86-64 with rustc 1.89,
clang 22.1 and glibc 2.44; other versions are untested.

```sh
./bench/setup.sh            # builds mintc and writes the example datasets to data/
./compiler/target/release/mintc build examples/logistic_bayes.mint -o build/logistic_bayes
./build/logistic_bayes
```

`mintc check FILE` type-checks only; `mintc emit FILE` writes the LLVM IR.

The full benchmark (it builds everything, generates data and runs about 12
minutes) and the test suite:

```sh
python3 bench/bench.py 7    # writes bench/results.json
./tests/run.sh              # compile errors, runtime checks, gradients, posteriors
(cd compiler && cargo test --release)
```

Useful environment variables for compiled programs:

- `MINT_GRADCHECK=1` compares the compiled gradient with finite differences.
- `MINT_BENCH_GRAD=K` times K gradient evaluations and exits.

## The language in one page

**Types.**

- Scalars: `Real`, `Positive`, `Prob` (the open interval 0 to 1), `Int`.
- Vectors: `Vector[n]`, and vectors whose entries lie in a domain:
  `Positive[n]`, `Prob[n]`.
- Matrices: `Matrix[m, n]`, `PSD[n]`, `SPD[n]`.

Dimension names such as `n` are bound by function parameters or by an annotated
`read`, and are checked everywhere after.

**Expressions.**

- `+ - * /` and `^`. The elementwise operators are `.*` and `./`, and `A'` is
  the transpose.
- `*` means the mathematical product. Between a scalar and anything it scales;
  `A * v` is a matrix-vector product and `v' * w` is a dot product. Between two
  vectors it is an error, because it is ambiguous.
- Functions: `exp`, `log`, `log1p`, `sqrt`, `sigmoid` and `abs` apply
  elementwise; `sum`, `dot` and `norm` reduce.
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

- `data` and `param` declarations. Parameters are `Real`, `Positive` or
  `Vector[n]`.
- `let` definitions and `x ~ Distribution(...)` statements, with `Normal`,
  `BernoulliLogit` and `Exponential`.
- `sample(Model(data...), draws =, warmup =, chains =, seed =)` runs NUTS with
  Stan's warmup and returns a posterior; `print` summarises it (mean, sd,
  quantiles, ESS, split R-hat).

**Data files** (`.f64`) hold two little-endian u64 values (rows, cols),
followed by row-major little-endian f64.

## Documents

- [Benchmark report](docs/benchmark.md)
- [Compiler architecture](docs/architecture.md)
- [Next milestone](docs/next-milestone.md): the minimum work needed to test
  whether first-class mathematical types enable useful optimisations in
  general, not only on examples I chose.
