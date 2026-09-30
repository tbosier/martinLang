# Benchmark report

## Verdict

The original gate asked two things. Can Mint express a realistic small
numerical problem materially more clearly than equivalent Rust? Does it run
within about 2x of a straightforward Rust implementation? It does both, with
room to spare. The honest boundary is expert, hand-vectorised Rust, which ties
Mint on the logistic gradient and beats it on Newton's method.

| problem | Mint | straightforward Rust | tuned Rust | max-effort Rust |
|---|---|---|---|---|
| logistic gradient (n=5000, p=20) | 35 µs | 178 µs | 139 µs | 37 µs (tie: ranges overlap) |
| logistic, full NUTS run | 0.51 s | 2.56 s | 1.93 s | 0.55 s |
| Newton, n=200000, p=50 | 0.32 s | 2.41 s | 0.41 s | **0.21 s** |
| linear gradient (sufficient statistics) | 102 ns | 896 µs | 258 ns (same rewrite by hand) | not written |
| lines of code | 15 to 18 | 59 to 78 | 66 to 97 | 92 to 105, plus 100 of shared SIMD helpers |

The Rust baselines come in three levels of effort:

- **Straightforward.** What a competent Rust programmer writes first.
- **Tuned.** My best plain scalar Rust.
- **Max effort.** Nightly Rust (LLVM 21) with AVX2 intrinsics and the same
  glibc vector `exp` and `log` Mint uses. Its logistic version came from trying
  three structures and keeping the fastest.

Two of the tricks the max-effort Rust used first are now in Mint's compiler:
computing four row dot products that share loads of the parameter vector, and
four-row gradient updates. Before that, the Rust was 18% faster on the logistic
gradient.

For the hierarchical time-series model against rustmc and Stan, see
[hierarchical.md](hierarchical.md).

## What is and is not being compared

- **Same sampler on both sides.** The Rust baselines call the Mint runtime's
  NUTS through FFI, standing in for a sampler crate. The only difference is the
  model code: Mint's compiled log density and gradient against Rust written by
  hand. The gradient tables measure exactly that, at one fixed parameter point
  shared by every implementation (the runtime's `MINT_BENCH_GRAD`). The
  sampling tables add sampler overhead and model preparation (for sufficient
  statistics, computing Z'Z, Z'y and y'y). Trajectories diverge slightly
  because the arithmetic differs in the last bits, so gradient counts differ by
  up to 4%. The counts are listed.
- **Same answers.** `tests/run.sh` checks this at a fixed test point for every
  benchmarked logistic and linear configuration, including each Mint flag
  variant and the max-effort Rust: the log density and every gradient
  component agree with the Rust baselines to 1e-9 relative. Every Newton
  variant, Mint's and the max-effort Rust's, produces the same coefficients to
  1e-8.
- **LLVM versions.** Mint's IR goes through clang 22 (LLVM 22). Stable rustc
  1.89 uses LLVM 20 and nightly uses LLVM 21, so the max-effort baselines are
  within one LLVM version of Mint.
- **One machine, one author.** I wrote the compiler, the examples and all the
  Rust baselines, apart from the hierarchical time-series comparison, where
  Stan and rustmc are external.

## Method

`python3 bench/bench.py 7` builds everything, then runs every configuration 7
times. Runs are interleaved (every configuration once, then repeat), so drift
affects all of them alike, and each run is pinned to one core with `taskset`.
The tables report the median and the full range. A ratio is labelled as not
establishing an ordering when the two ranges overlap. Newton times are
measured inside the program and exclude file reading. Sampling times are
recorded to the microsecond and include model preparation. Raw numbers are in
`bench/results.json`, and `bench/report.py` renders these tables from that
file.

## Results

Machine: AMD Ryzen 9 5900X 12-Core Processor, Linux 6.18.53-1-lts. rustc 1.89.0 (29483883e 2025-08-04) (LLVM 20); clang version 22.1.8.
Each cell is 7 runs, interleaved, pinned to core 5; load average at end 1.27, 2.97, 4.56 (at start about 4.5, from the run log).

#### Logistic regression gradient, n=5000, p=20 (time per gradient; lower is better)

| implementation | median | range (min to max) | relative to mint |
|---|---|---|---|
| mint | 35.3 µs | 33.2 µs to 44.3 µs | 1.00x |
| mint --strict-fp | 89.6 µs | 88.7 µs to 90.3 µs | 2.53x |
| mint --no-fission | 101.7 µs | 101.2 µs to 102.6 µs | 2.88x |
| mint --no-vecmath | 56.1 µs | 54.4 µs to 57.1 µs | 1.59x |
| mint --no-fission --strict-fp | 123.2 µs | 122.8 µs to 125.4 µs | 3.49x |
| rust straightforward | 178.4 µs | 176.0 µs to 194.1 µs | 5.05x |
| rust tuned | 139.3 µs | 138.3 µs to 141.0 µs | 3.94x |
| rust max effort | 37.3 µs | 35.8 µs to 40.1 µs | 1.06x (ranges overlap: ordering not established) |

#### Linear regression gradient, n=50000, p=20

| implementation | median | range (min to max) | relative to mint |
|---|---|---|---|
| mint | 102 ns | 100 ns to 105 ns | 1.00x |
| mint --no-suffstats | 330.2 µs | 328.3 µs to 337.7 µs | 3233.85x |
| mint --no-suffstats --strict-fp | 473.5 µs | 472.7 µs to 499.9 µs | 4637.45x |
| rust straightforward | 896.0 µs | 890.1 µs to 913.7 µs | 8776.01x |
| rust sufficient statistics | 258 ns | 258 ns to 265 ns | 2.53x |

#### Newton's method, n=200000, p=50, 10 iterations (fit time, excluding file reading)

| implementation | median | range (min to max) | relative to mint |
|---|---|---|---|
| mint | 0.322 s | 0.311 s to 0.391 s | 1.00x |
| mint --strict-fp | 0.346 s | 0.336 s to 0.392 s | 1.07x (ranges overlap: ordering not established) |
| mint --no-gram-blocking | 0.431 s | 0.425 s to 0.478 s | 1.34x |
| rust straightforward | 2.405 s | 2.336 s to 2.453 s | 7.46x |
| rust tuned | 0.405 s | 0.399 s to 0.428 s | 1.26x |
| rust max effort | 0.208 s | 0.206 s to 0.237 s | 0.65x |

#### Logistic regression, NUTS, 1 chain, 1000 warmup + 1000 draws (sampler wall time plus model preparation)

| implementation | median | range (min to max) | relative to mint |
|---|---|---|---|
| mint | 0.507 s | 0.498 s to 0.546 s; 14,323 gradients | 1.00x |
| rust straightforward | 2.557 s | 2.532 s to 2.591 s; 14,438 gradients | 5.05x |
| rust tuned | 1.926 s | 1.912 s to 1.934 s; 14,288 gradients | 3.80x |
| rust max effort | 0.554 s | 0.552 s to 0.558 s; 14,408 gradients | 1.09x |

#### Linear regression, NUTS, 1 chain, 1000 warmup + 1000 draws (sampler wall time plus model preparation)

| implementation | median | range (min to max) | relative to mint |
|---|---|---|---|
| mint | 7.79 ms | 7.72 ms to 7.87 ms; 18,917 gradients | 1.00x |
| mint --no-suffstats | 6.264 s | 6.226 s to 6.419 s; 18,994 gradients | 803.83x |
| rust straightforward | 16.5 s | 16.2 s to 18.4 s; 18,159 gradients | 2118.12x |
| rust sufficient statistics | 13.94 ms | 13.84 ms to 14.15 ms; 18,946 gradients | 1.79x |

#### Logistic gradient across problem shapes (median time per gradient)

| n | p | mint | rust straightforward | rust tuned | rust max effort | max effort / mint |
|---|---|---|---|---|---|---|
| 20000 | 5 | 96.4 µs | 490.4 µs | 415.9 µs | 113.3 µs | 1.18x |
| 5000 | 20 | 35.2 µs | 177.8 µs | 139.4 µs | 37.4 µs | 1.06x |
| 2000 | 100 | 48.3 µs | 227.1 µs | 106.3 µs | 51.6 µs | 1.07x |
| 100000 | 20 | 675.5 µs | 3615.5 µs | 2784.5 µs | 742.9 µs | 1.10x |

#### Lines of code (non-blank, non-comment; whole file)

| example | mint | rust straightforward |
|---|---|---|
| newton | 18 | 78 |
| logistic_bayes | 15 | 61 |
| linear_bayes | 17 | 59 |

#### Compile time (seconds, one run each)

| program | seconds |
|---|---|
| mint_logistic_newton | 0.16 |
| mint_logistic_bayes | 0.08 |
| mint_linear_bayes | 0.10 |
| mint_linear_bayes_nss | 0.07 |
| mint_logistic_newton_strict | 0.16 |
| mint_logistic_newton_noblock | 0.13 |
| mint_logistic_bayes_strict | 0.09 |
| mint_linear_bayes_nss_strict | 0.06 |
| mint_logistic_bayes_nofission | 0.06 |
| mint_logistic_bayes_novecmath | 0.09 |
| mint_logistic_bayes_nofission_strict | 0.06 |
| rust_logistic_newton | 0.23 |
| rust_logistic_newton_tuned | 0.25 |
| rust_logistic_bayes | 0.23 |
| rust_logistic_bayes_tuned | 0.19 |
| rust_linear_bayes | 0.20 |
| rust_linear_bayes_suffstats | 0.19 |
| rust_logistic_newton_max | 0.27 |
| rust_logistic_bayes_max | 0.19 |

## Where the speed comes from

Most of Mint's own choices have a switch, so they can be measured. For the
logistic gradient (n=5000, p=20; Mint 35 µs, tuned Rust 139 µs):

- **Loop fission.** `--no-fission` gives 102 µs. The row dot products, the
  elementwise density pass and the row gradient updates run as separate loops,
  so the middle loop vectorises.
- **Vector math.** `--no-vecmath` gives 56 µs. In the vectorised loop, `exp`
  and `log` go through glibc's 4-lane versions (within 4 ulp).
- **Strict floating point.** `--strict-fp` gives 90 µs. It removes
  reassociation, FMA contraction, vector math and the `log(1 + e)`
  substitution.
- **All of these off.** `--no-fission --strict-fp` gives 123 µs, still ahead
  of tuned Rust's 139 µs with the ranges separate. That remaining 12% is not
  isolated; it may be the different LLVM versions.

Other choices:

- **Newton.** Against tuned Rust (0.32 s against 0.41 s), Mint's gain comes
  from register blocking in the Gram kernel; `--no-gram-blocking` gives 0.43 s.
  The max-effort Rust is 1.55x faster than Mint. It makes one pass over X per
  iteration, where Mint makes three, and it vectorises the sigmoid. Fusing
  kernels that stream the same matrix is listed in
  [next-milestone.md](next-milestone.md).
- **Linear regression.** The sufficient-statistics rewrite changes the
  algorithm, not the code quality: each gradient is O(p²) instead of O(np).
  Without it, Mint is still 2.7x faster than straightforward Rust (330 µs
  against 896 µs). Against allocation-free Rust using the same rewrite with
  four-accumulator dot products, Mint is 2.5x faster per gradient (102 ns
  against 258 ns); I have not isolated why. For a whole run the gap is 1.8x
  (7.8 ms against 13.9 ms), because the sampler dominates at this size.
- **The sampler.** The runtime's NUTS was rewritten to share immutable states
  by reference instead of copying them, and to fuse its passes over the
  parameter vector. Its draws are bit-identical to before. That rewrite made a
  whole run 1.9x faster on the 3,171-dimension model and 2.1x faster on the
  37,901-dimension model; the later changes in
  [hierarchical.md](hierarchical.md) (recomputed momenta, threads within a
  chain) take the large model to 4.5 to 5.2x. The logistic full-run numbers above
  were measured with the first rewrite, for Mint and for every Rust baseline
  alike, and were not rerun after the later changes. Those changes keep the
  draws identical at this size (21 parameters, serial path).

## Clarity

Line counts are a crude measure, so here is what the extra Rust lines are.

- **Bayesian examples.** The Rust log density is written with its gradient,
  and each derivative is a place to make a silent mistake:
  - the prior terms;
  - the stable softplus;
  - `y - sigmoid(eta)`;
  - the `X' r` accumulation;
  - the `log sigma` transform, with its Jacobian and the chain rule through
    `exp`;
  - the packing of the parameters into one vector.

  A wrong derivative does not crash; the sampler just explores the wrong
  distribution. In Mint the model block states the model, and the compiler
  derives the rest (checked against finite differences by `MINT_GRADCHECK=1`).
  The max-effort Rust adds intrinsics, transposes and tail handling on top of
  all of that.
- **Newton.** The Mint version is the textbook update. The Rust versions are
  index loops over the Hessian and a hand-written Cholesky.
- **Checked before running.** Mint rejects several things Rust accepts and
  runs, each a compile error with a hint (`examples/errors/`):
  - shape mismatches;
  - solving with `X' X`, which is only PSD;
  - a `Real` Normal scale;
  - `v * w` on two vectors;
  - an unprovable SPD annotation;
  - an ambiguous vector-matrix broadcast.

## Caveats

- Three problems here, plus the time-series model, all chosen by me. The next
  milestone exists to test models I did not choose.
- The linear-regression speedup is algorithmic and narrow. It applies to Normal
  likelihoods whose mean is affine in the parameters, with literal or
  data-vector coefficients. Computing the residual sum of squares by expansion
  loses relative precision when residuals are tiny compared with y.
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
- Compiling a Mint program takes 60 to 160 ms; the runtime is compiled once and
  cached. rustc takes about 200 to 260 ms for these single-file baselines.
- Kernels are single-threaded; chains run in parallel.
