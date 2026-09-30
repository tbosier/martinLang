# Benchmark report

## Verdict

The gate asked whether Mint can express a realistic small numerical problem
materially more clearly than equivalent Rust, while running within about 2x of
a straightforward Rust implementation. On the three problems tested, both
parts hold, and Mint is faster rather than slower:

- **Logistic regression gradient** (the core of Bayesian inference for this
  model): 46 µs in Mint, against 178 µs for straightforward Rust and 139 µs for
  hand-tuned Rust. Across four problem shapes, Mint is 3.2x to 4.7x faster than
  straightforward Rust and 2.2x to 3.1x faster than tuned Rust.
- **Logistic regression, full NUTS run** on the identical sampler: 0.68 s,
  against 2.56 s and 1.91 s.
- **Newton's method**: 0.35 s, against 2.35 s straightforward and 0.40 s tuned.
- **Linear regression**: 100 ns per gradient with the sufficient-statistics
  rewrite, against 897 µs for straightforward Rust and 258 ns for Rust that uses
  the same rewrite by hand. For a full NUTS run, including the time spent
  computing the statistics, Mint takes 8.9 ms against 16.7 s and 14.9 ms.
- **Clarity**: 15 to 18 lines of Mint against 59 to 78 lines of Rust. In the
  Bayesian examples, none of Mint's lines are derivatives; the Rust versions
  carry hand-derived gradients and a hand-written Jacobian.

So the project continues, per the brief. The minimum next milestone is in
[next-milestone.md](next-milestone.md).

## What is and is not being compared

- **Same sampler on both sides.** The Rust baselines call the Mint runtime's
  NUTS through FFI, standing in for a sampler crate. The only thing that
  differs is the model code: Mint's compiled log density and gradient, against
  Rust written by hand. The gradient-time tables measure exactly that, at one
  fixed parameter point shared by every implementation (the runtime's
  `MINT_BENCH_GRAD`). The sampling tables add sampler overhead and model
  preparation (for sufficient statistics, computing Z'Z, Z'y and y'y).
  Trajectories diverge slightly between implementations because the arithmetic
  differs in the last bits, so gradient counts differ by up to 4%. The counts
  are listed.
- **Same answers.** At the test point, the log density and every gradient
  component of each benchmarked configuration agree with the Rust baselines to
  1e-9 relative (`tests/run.sh`). Newton's coefficients agree to 1e-8. Posterior
  means from Mint and from the straightforward Rust logistic baseline agree
  within one Monte Carlo standard error.
- **Two Rust baselines per problem.**
  - "Straightforward" is what a competent Rust programmer writes first: flat
    row-major `Vec<f64>`, iterator sums, full matrices, allocation per call.
  - "Tuned" (and the linear "sufficient statistics" baseline) is my best hand
    optimisation that stays in plain scalar Rust: fused passes,
    four-accumulator dot products, symmetry, no allocation per gradient.
  - Neither uses SIMD intrinsics, a vector math crate, BLAS or nalgebra.
- **Different LLVM versions.** Mint's IR goes through clang 22 (LLVM 22), and
  rustc 1.89 uses LLVM 20. This is a confound I could not remove on this
  machine; see the ablation below for how much it could account for.
- **One machine, one author.** I wrote the compiler, the examples and both
  sets of baselines. The next milestone replaces my baselines with external
  models and Stan.

## Method

`python3 bench/bench.py 7` builds everything, then runs every configuration 7
times. Runs are interleaved (every configuration once, then repeat), so drift
affects all of them alike, and each run is pinned to one core with `taskset`.
The machine had other load. The tables report the median and the full range.
A ratio is labelled as not establishing an ordering when the two ranges
overlap. Newton times are measured inside the program and exclude file
reading. Sampling times are recorded to the microsecond. Raw numbers are in
`bench/results.json`, and `bench/report.py` renders these tables from that
file.

## Results

Machine: AMD Ryzen 9 5900X 12-Core Processor, Linux 6.18.53-1-lts. rustc 1.89.0 (29483883e 2025-08-04) (LLVM 20); clang version 22.1.8.
Each cell is 7 runs, interleaved, pinned to core 5; load average at start 3.92, 2.14, 2.16.

#### Logistic regression gradient, n=5000, p=20 (time per gradient; lower is better)

| implementation | median | range (min to max) | relative to mint |
|---|---|---|---|
| mint | 46.1 µs | 45.5 µs to 46.6 µs | 1.00x |
| mint --strict-fp | 108.1 µs | 107.0 µs to 109.7 µs | 2.35x |
| mint --no-fission | 101.6 µs | 100.7 µs to 102.3 µs | 2.20x |
| mint --no-vecmath | 66.4 µs | 65.8 µs to 66.9 µs | 1.44x |
| mint --no-fission --strict-fp | 123.0 µs | 122.7 µs to 124.8 µs | 2.67x |
| rust straightforward | 177.7 µs | 176.9 µs to 178.3 µs | 3.86x |
| rust tuned | 139.4 µs | 138.8 µs to 139.5 µs | 3.03x |

#### Linear regression gradient, n=50000, p=20

| implementation | median | range (min to max) | relative to mint |
|---|---|---|---|
| mint | 100 ns | 99 ns to 101 ns | 1.00x |
| mint --no-suffstats | 330.3 µs | 328.5 µs to 332.2 µs | 3286.11x |
| mint --no-suffstats --strict-fp | 474.3 µs | 471.3 µs to 478.8 µs | 4719.12x |
| rust straightforward | 897.1 µs | 889.1 µs to 902.4 µs | 8926.29x |
| rust sufficient statistics | 258 ns | 257 ns to 274 ns | 2.57x |

#### Newton's method, n=200000, p=50, 10 iterations (fit time, excluding file reading)

| implementation | median | range (min to max) | relative to mint |
|---|---|---|---|
| mint | 0.354 s | 0.337 s to 0.375 s | 1.00x |
| mint --strict-fp | 0.366 s | 0.356 s to 0.493 s | 1.04x (ranges overlap: ordering not established) |
| mint --no-gram-blocking | 0.458 s | 0.445 s to 0.490 s | 1.29x |
| rust straightforward | 2.346 s | 2.342 s to 2.457 s | 6.63x |
| rust tuned | 0.402 s | 0.399 s to 0.419 s | 1.14x |

#### Logistic regression, NUTS, 1 chain, 1000 warmup + 1000 draws (sampler wall time plus model preparation)

| implementation | median | range (min to max) | relative to mint |
|---|---|---|---|
| mint | 0.679 s | 0.657 s to 0.747 s; 14,472 gradients | 1.00x |
| rust straightforward | 2.561 s | 2.542 s to 3.344 s; 14,438 gradients | 3.77x |
| rust tuned | 1.906 s | 1.899 s to 1.920 s; 14,288 gradients | 2.81x |

#### Linear regression, NUTS, 1 chain, 1000 warmup + 1000 draws (sampler wall time plus model preparation)

| implementation | median | range (min to max) | relative to mint |
|---|---|---|---|
| mint | 8.91 ms | 8.79 ms to 8.93 ms; 18,917 gradients | 1.00x |
| mint --no-suffstats | 6.275 s | 6.253 s to 7.602 s; 18,994 gradients | 704.01x |
| rust straightforward | 16.7 s | 16.2 s to 17.6 s; 18,159 gradients | 1876.38x |
| rust sufficient statistics | 14.94 ms | 14.86 ms to 15.23 ms; 18,946 gradients | 1.68x |

#### Logistic gradient across problem shapes (median time per gradient)

| n | p | mint | rust straightforward | rust tuned | straightforward / mint | tuned / mint |
|---|---|---|---|---|---|---|
| 20000 | 5 | 152.4 µs | 490.3 µs | 417.8 µs | 3.22x | 2.74x |
| 5000 | 20 | 45.7 µs | 178.0 µs | 139.5 µs | 3.89x | 3.05x |
| 2000 | 100 | 48.7 µs | 227.2 µs | 106.5 µs | 4.66x | 2.19x |
| 100000 | 20 | 926.2 µs | 3620.9 µs | 2792.7 µs | 3.91x | 3.02x |

#### Lines of code (non-blank, non-comment; whole file)

| example | mint | rust straightforward |
|---|---|---|
| newton | 18 | 78 |
| logistic_bayes | 15 | 61 |
| linear_bayes | 17 | 59 |

#### Compile time (seconds, one run each)

| program | seconds |
|---|---|
| mint_logistic_newton | 0.58 |
| mint_logistic_bayes | 0.51 |
| mint_linear_bayes | 0.54 |
| mint_linear_bayes_nss | 0.50 |
| mint_logistic_newton_strict | 0.57 |
| mint_logistic_newton_noblock | 0.54 |
| mint_logistic_bayes_strict | 0.49 |
| mint_linear_bayes_nss_strict | 0.48 |
| mint_logistic_bayes_nofission | 0.51 |
| mint_logistic_bayes_novecmath | 0.50 |
| mint_logistic_bayes_nofission_strict | 0.50 |
| rust_logistic_newton | 0.22 |
| rust_logistic_newton_tuned | 0.26 |
| rust_logistic_bayes | 0.23 |
| rust_logistic_bayes_tuned | 0.19 |
| rust_linear_bayes | 0.20 |
| rust_linear_bayes_suffstats | 0.20 |

## Where the speed comes from

Most of Mint's own choices have a switch, so they can be measured. For the
logistic gradient (n=5000, p=20; Mint 46 µs, tuned Rust 139 µs):

- **Loop fission.** `--no-fission` gives 102 µs. The row dot products, the
  elementwise density pass and the row gradient updates run as separate loops,
  so the middle loop has no inner loop and LLVM vectorises it.
- **Vector math.** `--no-vecmath` gives 66 µs. In the vectorised loop, `exp`
  and `log` go through glibc's 4-lane versions (within 4 ulp).
- **Strict floating point.** `--strict-fp` gives 108 µs. It removes
  reassociation, FMA contraction, vector math and the `log(1 + e)` substitution
  for `log1p`.
- **All of these off.** `--no-fission --strict-fp` gives 123 µs, and its range
  does not overlap tuned Rust's 139 µs.

That remaining 12% is not isolated. It could come from the different LLVM
versions or from small differences in the generated loops; I did not separate
them. It is small next to the 3x from the switchable choices.

Other choices:

- **Newton.** The gain over tuned Rust comes from register blocking in the
  Gram kernel. The compiler emits that kernel because it knows it is computing
  `X' diag(w) X`: it processes four rows per pass over the destination and
  computes one triangle.
  - With `--no-gram-blocking`, Mint takes 0.458 s, slower than tuned Rust's
    0.402 s.
  - With blocking, Mint takes 0.354 s.
  - `--strict-fp` changes little here (the ranges overlap).
  - The straightforward Rust is 6.6x slower, mostly because it computes the
    full symmetric Hessian with a bounds-checked inner loop.
- **Linear regression.** The sufficient-statistics rewrite changes the
  algorithm, not the code quality: each gradient becomes O(p²) instead of
  O(np). Without it, Mint is still 2.7x faster than straightforward Rust
  (330 µs against 897 µs) from single-pass fusion and vectorised reductions.
  Against Rust written with the same rewrite by hand, allocation-free and with
  four-accumulator dot products, Mint is 2.6x faster per gradient (100 ns
  against 258 ns). I have not isolated that constant-factor gap. At this size
  both are dominated by the sampler: 8.9 ms against 14.9 ms for a whole run.

## Clarity

Line counts are a crude measure, so here is what the extra Rust lines are.

- **Bayesian examples.** The Rust log density is written with its gradient,
  and each derivative is a place to make a silent mistake:
  - the prior terms;
  - the stable softplus;
  - `y - sigmoid(eta)`;
  - the `X' r` accumulation;
  - the `log sigma` transform, with its Jacobian term and the chain rule
    through `exp`;
  - the packing of `alpha`, `beta` and `sigma` into one parameter vector.

  A wrong derivative does not crash; the sampler just explores the wrong
  distribution. In Mint the model block states the model, and the compiler
  derives the rest (checked against finite differences by `MINT_GRADCHECK=1`).
- **Newton.** The Mint version is the textbook update. The Rust version is
  index loops over the Hessian and a hand-written Cholesky.
- **Checked before running.** Mint rejects several things that Rust accepts
  and runs, each a compile error with a hint (`examples/errors/`):
  - a shape mismatch between `X` and `y`;
  - solving with `X' X`, which is only PSD;
  - a `Real` Normal scale;
  - `v * w` on two vectors;
  - an `SPD` annotation the compiler cannot prove.

A Rust programmer using nalgebra or ndarray would write a shorter Newton than
my baseline. Nightly Rust's Enzyme-based `#[autodiff]` could remove the
hand-written gradients. Neither combines positivity, SPD-ness and
shape-by-name checking at compile time with these kernels.

## Caveats

- Three problems, chosen by me. The next milestone exists to test models I did
  not choose.
- The linear-regression speedup is algorithmic, and it applies narrowly: only
  to Normal likelihoods whose mean is affine in the parameters, with literal or
  data-vector coefficients. A scalar `data` coefficient is not recognised.
  Computing the residual sum of squares as y'y − 2θ'Z'y + θ'Z'Zθ loses relative
  precision when the residuals are tiny compared with y.
- Fast-math is limited to four things:
  - FMA contraction;
  - reassociation of reductions;
  - the vector math library;
  - `log(1 + e)` in place of `log1p(e)` for BernoulliLogit.

  Results differ from strict evaluation in the last bits, and `--strict-fp`
  restores strict evaluation.
- Domain types such as Positive are facts about real numbers. Plain arithmetic
  that underflows (`exp(-1000)` is 0) is not checked at run time; see
  [architecture.md](architecture.md).
- Mint pays about 0.5 s to compile each program, mostly in clang. That is about
  twice rustc's time for these single-file baselines.
- Kernels are single-threaded; chains run in parallel. A BLAS `dsyrk` would
  beat the Gram kernel for large p.
