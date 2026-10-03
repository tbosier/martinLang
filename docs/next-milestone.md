# Next milestone: do first-class mathematical types pay for themselves?

> **Status, 2026-10-02.** Written before the later compiler rounds; the gaps listed under "Known performance gaps" have since moved (see compiler-round.md). The current figures are in the
> [README](../README.md) and the [same-sampler benchmark](../bench/same_sampler/README.md).

## Where the prototype leaves the question

The prototype already has four optimisations that come from mathematical
types or declarations rather than from LLVM:

| fact the compiler knows | what it enables | measured on |
|---|---|---|
| `H` is SPD (proved from `PSD + Positive * I`) | Cholesky with no pivoting or runtime structure check; half the Gram product | Newton example |
| `sigma: Positive` | automatic log transform and Jacobian | every model with a scale |
| `X`, `y` are `data`, `beta` is a `param`, mean is affine | sufficient statistics, O(q²) instead of O(nq) per gradient | linear regression |
| observations are independent given the parameters | per-observation fused AD, loop fission, vector math | logistic regression |

It does not answer whether these are general or cherry-picked. I chose the
examples, I wrote the Rust baselines, and every optimisation was designed with
these three models in view. That is the weakness the next milestone has to
remove.

## The milestone

Take models Martin did not choose, and let every type-driven optimisation be
switched off independently.

1. **A fixed external benchmark set.** Take 10 models from
   [posteriordb](https://github.com/stan-dev/posteriordb) that fit the
   language after small, listed extensions: at least one simplex, one
   hierarchical scale, and one covariance or correlation matrix. Choose them
   before writing any optimisation, and record the choice in the repository.
2. **Three new first-class types**, each with the transform it implies and one
   optimisation it enables:
   - `Simplex[k]`: stick-breaking transform. The optimisation is to skip the
     normalisation when a Categorical or Multinomial likelihood already
     normalises.
   - `CholeskyCorr[k]` / `SPD[k]` parameters: the log-determinant and the solve
     come from one factorisation that is reused across every `~` statement
     using the matrix.
   - `Sparse[n, p]` data (CSR): the kernel choice follows from the type. The
     test is whether the source can stay identical to the dense version.
3. **Per-optimisation switches.** Most existing choices already have a flag:
   `--no-suffstats`, `--no-fission`, `--no-vecmath`, `--no-gram-blocking`,
   `--no-scan-layout`, `--no-scan-fusion`, `--no-inline-exp`,
   `--no-row-fusion` and `--strict-fp`. The one-triangle Gram computation and
   the allocator's `noalias` declaration do not. Every new pass
   gets one. The report states how often each pass fires across the 10
   models, not only how fast it is when it does.
4. **An external baseline.** Compare with Stan through
   [BridgeStan](https://github.com/roualdes/bridgestan), measuring gradient time
   at identical points, and with the posteriordb reference posteriors for
   correctness. That replaces my own Rust baselines, which are the weakest part
   of the current evidence.

## Known performance gaps

Maximum-effort Rust (nightly, AVX2 intrinsics, the same glibc vector math)
exposed places where hand-written code was faster than Martin. None needed new
language features.

1. **Scans across many series: mostly closed.** The dynamic Poisson gradient
   was 1.8x slower than the Rust; it was then 2 to 4% slower (consistently, over
   11 runs), and is now about 15% faster, through a column-major layout the compiler chooses and a
   vectorised, fused scan kernel. See
   [compiler-round.md](compiler-round.md).
2. **Several kernels streaming the same large matrix: closed.** Newton's
   method was 1.55x slower than the Rust; with row fusion and a tiled Gram
   kernel it is now 1.30x faster. Its Gram kernel still misses L1 about four
   times as often as the Rust's.
3. **The sampler at tens of thousands of dimensions.** Sharing states by
   reference, recomputing scaled momenta and splitting each chain's passes
   across threads took the large model from 1448 s to under 330 s. The sampler and
   model still run as separate passes over each state, though. The next step
   is having the compiler emit the leapfrog step fused with the model, so that
   position, momentum and gradient are updated in the same loop that computes
   the gradient. The fused scan kernel of the gradient now runs on the
   chain's threads; the rest of the gradient is one thread per chain.
4. **Adaptation that needs fewer gradients.** A gradient-informed diagonal
   metric halved trajectory lengths on the time-series model but gave about
   3.5x fewer effective draws per gradient, and made no measurable difference
   on the two small models. The remaining candidates are metrics that capture
   correlation (low-rank or structured) and reparameterisations the compiler
   can derive from the model.

## Success and stop criteria

These are fixed before the work starts:

- **Pass:** at least 3 of the 10 models get at least 1.5x faster gradients from
  a type-driven pass (the median over 7 runs, with ranges not overlapping the
  pass-off configuration). All 10 must match the posteriordb reference
  posteriors within 4 Monte Carlo standard errors. And Martin must be within 2x
  of BridgeStan on gradient time for every model it can express.
- **Stop:** if fewer than 3 models benefit, first-class types are a convenience
  for correctness (the SPD and Positive errors are still worth having) but not
  an optimisation strategy. The project should then be narrowed to a checker
  and front end for an existing backend, instead of a compiler.

## Out of scope for this milestone

Package management, a REPL, GPU backends, general control flow and
user-defined distributions stay out. None of them bears on the question.
