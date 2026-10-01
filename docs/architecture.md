# Compiler architecture

Mint's compiler (`mintc`, about 5,800 lines of Rust with no dependencies) owns
every stage from source text to LLVM IR. LLVM, invoked through `clang -O3
-march=native`, does the last step: instruction selection, register
allocation, loop vectorisation and unrolling. A small C runtime (about 1,160
lines) provides I/O, printing, a Cholesky solve and the NUTS sampler. The
runtime is compiled once and cached, so building a program takes 60 to 140 ms:
`mintc` itself takes 1 to 2 ms, clang about 30 to 120 ms, and linking about
30 ms.

```
 .mint source
     │  lexer.rs      tokens; newlines end statements except inside ( ) [ ]
     ▼
 parser.rs            untyped AST (ast.rs)
     │
     ▼
 check.rs + types.rs  names, shapes, value domains, matrix structure;
     │                `*` chains rewritten into explicit kernels
     ▼
 typed tree ──────────────┬──────────────────────────────┐
     │                    │                              │
 codegen.rs          model.rs                            │
 functions and main  model blocks: lowering, fused       │
                     reverse-mode AD, loop fission,      │
                     sufficient-statistics rewrite       │
     │                    │                              │
     └────────► ir.rs (textual LLVM IR builder) ◄────────┘
                          │
                          ▼
            clang -O3 -march=native -fveclib=libmvec
                          │   + runtime/mint_rt.c
                          ▼
                    native executable
```

## What the type checker knows

Every value has a shape with symbolic dimensions: `Vector[n]`,
`Matrix[n, p]`. Dimension names are introduced by function parameters or by
an annotated read (`let X: Matrix[n, p] = read(...)`), and every later use is
checked against them at compile time. Data coming in from files is checked
once, at the read, and nowhere after.

A vector combined elementwise with a matrix is matched to the matrix dimension
with the same name. A `Vector[G]` plus a `Matrix[G, T]` repeats across `T`,
and a `Vector[T]` repeats across `G`. If both dimensions have the same name,
the expression is a compile error, because it is ambiguous. `cumsum(M, T)` is a
running sum along the named dimension, which must be the last one.

On top of shapes it tracks two chains of facts, ordered by inclusion:

| facts | chain |
|---|---|
| value domain | `Prob (0,1)` ⊂ `Positive (0,∞)` ⊂ `NonNeg [0,∞)` ⊂ `Real` |
| matrix structure | `SPD` ⊂ `PSD` ⊂ `symmetric` ⊂ `general` |

The rules are small and local (`types.rs`):

- `exp` gives Positive, and `sigmoid` gives Prob.
- `1 - p` is Prob when `p` is Prob, and a product of Prob values is Prob.
- `A' * diag(w) * A` is PSD when `w` is NonNeg.
- PSD + SPD is SPD.
- A Positive multiple keeps SPD, a NonNeg multiple gives PSD, and a Real
  multiple gives symmetric only.
- By the Schur product theorem, the elementwise product of two PSD matrices is
  PSD.

These facts gate operations and choose algorithms:

- `solve(H, g)` only compiles when `H` is proved SPD, and then it is a Cholesky
  solve. `X' * X` is PSD, not SPD, so passing it is a compile error that
  suggests adding `lambda * I(p)`. `assume_spd(A)` is the escape hatch. It
  checks at run time that `A` is symmetric (to 1e-12 relative) and has a
  Cholesky factorisation.
- A model `param sigma: Positive` is sampled as `log(sigma)`, with the Jacobian
  added to the log density.
- A `Normal` scale or `Exponential` rate must be proved Positive. A `Real`
  parameter in that position is rejected, with a hint to declare it Positive.
- A read annotated `Positive[n]` or `Prob[n]` inserts a runtime check that
  every entry satisfies the promise, so file data cannot break the proofs.

The facts are statements about real numbers, and floating point can break
them.

- **Checked at run time:**
  - Cholesky failure in `solve`.
  - `assume_spd`, which checks symmetry and positive definiteness.
  - Domain-annotated reads.
  - `BernoulliLogit` outcomes, which must be 0 or 1.

  Each of these stops the program with a message.
- **Not checked:** plain arithmetic. `exp(-1000)` is typed Positive but
  evaluates to 0, and `sigmoid(1000)` is typed Prob but evaluates to 1. The
  program continues with those values. The compiler uses domain facts to reject
  programs and to choose algorithms, never to skip a check it could not prove
  numerically.

## From operators to kernels

`*` is resolved by looking at the whole product chain (`check.rs`,
`Checker::product` and `Checker::chain`):

| source | kernel |
|---|---|
| scalar `*` anything | elementwise scaling |
| `A * v`, `A' * v` | matrix-vector product; the transpose is never materialised |
| `A' * diag(w) * A`, `A' * A` | Gram product: one triangle, register-blocked, `diag(w)` never materialised |
| `v' * w`, `v' * A * w` | dot product (a quadratic form folds right) |
| `A * B` | general product |
| `v * w` for two vectors | compile error: ambiguous, use `dot` or `.*` |

## Function code generation (`codegen.rs`)

- Every vector and matrix value lives in a buffer whose size comes from its
  symbolic type. Buffers are allocated once, at the start of the top-level
  statement that needs them (hoisted out of `repeat` loops), and freed when the
  function returns. The runtime's Cholesky and `assume_spd` check reuse a
  per-thread buffer, so nothing inside a `repeat` allocates.
- Calls use destination passing: the caller allocates the result buffer.
- Elementwise expression trees are fused into one loop. Only non-elementwise
  subterms (products, solves, calls) are materialised. The weights of
  `X' * diag(mu .* (1 - mu)) * X` and the vector of `X' * (mu - y)` are
  computed inside the kernel's row loop and never stored as whole vectors
  (only per chunk of rows, in the fused loop).
- The Gram kernel computes the upper triangle only and mirrors it at the end.
  It works on chunks of 32 rows. The weighted rows W = diag(w) X go into L1
  scratch padded with zeros to a multiple of 4 columns, and H is updated in
  strips of four rows, each covered by 4 x 12 tiles (12 vector registers;
  per row, three vector loads from W and four broadcasts straight from X feed
  12 FMAs) and 4 x 8 tiles. Only the 4 x 4 blocks on the diagonal compute
  entries below it. Each strip prefetches its share of the next chunk of X.
  When p is not a multiple of 4 and the fused loop also computes `X' * r`, r
  goes in W's first padding column and that product comes out of the Gram
  kernel's otherwise wasted lanes, instead of a pass of its own (one store per
  row and a copy of p values at the end remain). `--no-gram-blocking`
  restores the older row-by-row kernel.
- **Row fusion** (`--no-row-fusion`; off under `--strict-fp`). Consecutive
  `let`s in a `repeat` body that stream the rows of one matrix run as one
  loop over chunks of rows: a producer (an elementwise function of `X * w`,
  one value per row) and consumers of it (`X' * f + ...` and
  `X' * diag(w) * X + ...`, which may use earlier producers elementwise). Each
  chunk of X is read from memory once. Newton's three passes over X become
  one. Within a chunk, the producers' dot products run first (four rows per
  pass over w), then the per-row values (the sigmoid with Mint's `exp`, the
  weights and coefficients) four rows at a time in vector registers.
- The runtime allocator is declared `noalias` (fresh memory, like `malloc`)
  and returns 64-byte aligned buffers, so LLVM knows a new buffer overlaps
  nothing and needs no run-time overlap checks.

## Model code generation (`model.rs`)

A `model` block becomes four LLVM functions:

- `logp(theta, grad)` returns the log density on the unconstrained space and
  writes its gradient.
- `constrain` maps unconstrained draws back, for example with `exp` for
  Positive parameters.
- `init` computes the precomputed statistics.
- `sample` calls the runtime's NUTS.

The gradient is produced by reverse-mode differentiation *at compile time*,
per observation:

1. `let` definitions are inlined, and each `x ~ D(args)` statement becomes one
   loop over its observations.
2. For observation `i`, the loop evaluates the arguments in registers. It then
   computes the log density term and its partial derivatives in closed form
   (`Mg::lpdf`), and runs the backward sweep through that observation's
   expression tree immediately (`Mg::bwd`).
3. `X * beta` inside the tree is a row dot product forward and a row axpy
   backward, so there is no tape and no n-length temporaries.

A statement's index space is its shape: scalar, vector or matrix. In a
matrix-shaped statement, each leaf knows whether it is indexed by element, by
row (a `Vector[G]` in a `[G, T]` statement) or by column (a `Vector[T]`).
Gradients of row-indexed parameters are summed in registers along each row and
written once per row, so the column loop is a reduction LLVM vectorises,
including `exp` through the vector math library. At 20 series × 150 times,
this made the dynamic Poisson gradient 2.4x faster.

Some nodes cannot be computed one element at a time, so they are materialised
into scratch buffers around the statement's loop:

- **Running sums** (`cumsum`). With the scan layout and scan fusion off, the
  forward pass computes the running sum before the loop, four series
  interleaved so the CPU has four independent chains, and after the loop the
  adjoint is a reverse running sum of the adjoint buffer, back-propagated into
  the scan's own expression. With them on (the default), see the next
  section.
- **Matrix-vector products**, when loop fission applies (see below). Before the
  loop, row dot products run four rows at a time, sharing loads of the vector.
  After the loop, gradient row updates also run four rows at a time.

Each scratch buffer is a separate per-thread allocation returned `noalias`
(`mint_ws_slot`), so LLVM knows buffers never overlap and needs no runtime
overlap checks.

Two passes then reshape the loop.

**Loop fission** (on by default; off with `--no-fission`). When an observation's
expression contains both a matrix-vector product and a transcendental function,
the single loop is split into three passes:

1. the row dot products;
2. an elementwise pass for the density, its partials and the elementwise part
   of the backward sweep;
3. the row axpys for the gradient.

With `--no-fission-kernel` these are three whole passes over the n
observations, and LLVM vectorises the middle one, calling glibc's vector
`exp` and `log` (each call spills every vector register). By default, when
the only materialised nodes are matrix-vector products and every operation
has a vector form (everything but `log1p`), the statement runs as a
**fission kernel** instead (`gen_fission_kernel` in `model.rs`): one loop
over chunks of 32 rows, each taking the three steps in Mint's own
`<4 x double>` code:

1. the dot products, four rows at a time: one accumulator per row along the
   columns (a masked tail when p is not a multiple of 4), then a 4 x 4
   transpose-and-add that leaves the four results in one vector;
2. the density and its derivatives on four observations at a time, with
   Mint's `exp` and `log` inline, so they make no calls (a power other than
   `^2` still calls glibc's vector `pow`). The density's own `exp`
   (BernoulliLogit's exp(-|eta|), PoissonLog's exp(eta)) and BernoulliLogit's
   `log1p` each run in a loop of their own over the chunk first, into L1
   scratch: an iteration is then a short dependency chain, and several
   overlap;
3. the gradient updates, four rows at a time, reading the chunk's 32 rows of
   X again from L1.

X is read from memory once per gradient instead of twice. The rows left over
take the same steps four at a time, and the last n mod 4 in scalar code. The
scratch vectors live in a per-thread workspace, which keeps parallel chains
safe. On the logistic benchmark the gradient went from 33.4 to 23.7 µs
(fastest of 28 interleaved runs each, on a loaded machine; see
`docs/compiler-round.md`). With X small enough for L2, the dot products took
about 23 cycles per four rows (640 bytes of X), close to the 20 cycles it
takes to bring those bytes from L2 into L1 at 32 bytes per cycle.

**Sufficient statistics** (on by default; off with `--no-suffstats`). This
applies to a `Normal` likelihood whose outcome is data, whose scale is one
scalar, and whose mean is affine in the parameters with data coefficients
(`alpha + X * beta`, `X * beta`, `v .* a`, or a data offset). Such a likelihood
depends on the data only through Z'Z, Z'y and y'y, where Z stacks the
coefficient columns. `init` computes them once; each gradient is then O(q²)
instead of O(nq). The compiler reports when the rewrite fires. A coefficient
must be a literal or a data vector. A scalar `data c` times a parameter is not
recognised, and that likelihood keeps the direct O(nq) loop.

The residual sum of squares is then computed as y'y − 2θ'Z'y + θ'Z'Zθ. This
loses relative precision when the residuals are tiny compared with y (the
cancellation grows with ‖y‖² / RSS). In the benchmark data the log density
agrees with the direct computation to 12 significant digits.

### Scan layout and the fused scan kernel

When a model takes `cumsum` of a `Matrix[G, T]` along T, every model
quantity of that shape is stored column-major (`--no-scan-layout` turns this
off): the parameter inside the sampler's vector, the data (copied once per
`sample`), and scratch buffers. Draws are still written in the user's order,
and the runtime converts its benchmark point and printed gradients through
two generated functions (`mint_set_layout`). Four adjacent series at one time
are then one contiguous vector load.

A matrix statement whose only materialised nodes are running sums over its
own shape then runs as one kernel (`--no-scan-fusion` turns it off) over
groups of 8 series, two vectors of four:

- A, forward in time: the running sums (vector registers) and the density's
  argument, into an L1 scratch;
- B: `exp` over the scratch in a loop of its own, with nothing else live, so
  its constants stay in registers (only when the density needs exp of its
  argument, as PoissonLog does);
- C and R, backward in time: the density and its derivatives, and the reverse
  running sums of the adjoints pushed through the scan's expression.

Mint emits this as `<4 x double>` IR itself (`Fb::lanes`); LLVM's loop
vectoriser does not produce vector lanes across rows with the loop over time.
Series left over run through the same generator with one lane, and so do
statements with BernoulliLogit, `abs` or `log1p`. (`log1p` has no vector
form yet; BernoulliLogit and `abs` have one now, used by the fission kernel,
but the scan kernel has not been switched to it or measured with it.)

Around the kernel:

- **Absorption.** An element-wise statement over the same shape (a prior on
  the scanned matrix) runs inside the kernel's reverse loop instead of making
  its own pass.
- **Gradient ownership.** When every gradient contribution to a matrix
  parameter happens in that loop, its gradient is summed in a register and
  stored once and is not zeroed first. In general only accumulated gradients
  are zeroed now.
- Gradients of column-indexed parameters go to per-lane partial sums
  (T x 4), reduced once; those of row-indexed and scalar parameters stay in
  vector registers.

`tests/run.sh` builds seven models three ways (default, layout without
fusion, and neither) and requires the same log density and gradient to 1e-12
at 7, 13 and 20 series, plus a finite-difference check at the benchmark
point. The models cover linear and nonlinear nested running sums, the
one-lane BernoulliLogit path, row, column and scalar parameters inside and
outside the sum, a column parameter also used by statements of another
shape, a running sum of data only, and two scan statements sharing a matrix
parameter. A further test checks that sampled draws come back in the user's
order. Not covered: a model with running sums over two different shapes,
and gradients away from the benchmark point (at the runtime's random point
these models' log densities are near -1e16 and finite differences fail for
every build). The measurements are in
[compiler-round.md](compiler-round.md).

### What was tried and removed

Three restructurings of scan statements were implemented and measured, and all
three were deleted because they were slower. Each was correct: gradients
matched the exact formula to about 1e-14.

- **Time-outer, series-inner loops.** The inner loop runs across independent
  series. With row-major data, every access in it is strided, and it was 1.5x
  slower.
- **Time-major data copies and shadow parameter buffers.** This removed the
  strided access, but the extra passes over memory cost more than it saved.
- **Tiling.** Eight series at a time went through the scan, the density and the
  reverse scan in a small buffer. It was 4% faster at 250 series and 35% slower
  at 20.

What worked in the end was changing the storage order itself rather than
copying (the scan layout above), which needs no transposes and no extra
passes, and emitting the vector code directly. The gradient is now within 2
to 4% of the hand-written Rust (which is still faster), which vectorises
across 8 series with 4 x 4 in-register transposes.

## Floating-point semantics

Mint treats arithmetic as arithmetic on reals, within documented limits:

- Every `fadd`/`fmul` carries `contract`, which allows fused multiply-add.
- Reductions (sums, dot products) carry `reassoc`, so LLVM may reorder them and
  use several vector accumulators. This is the main reason Mint's dot products
  are faster than a plain Rust `iter().sum()`, which must add strictly left to
  right.
- With `-fveclib=libmvec`, `exp` and `log` in loops LLVM vectorises call
  glibc's 4-lane versions, which glibc documents as accurate to within 4 ulp.
  The scalar versions are under 1 ulp.
- In vector code Mint emits itself (the scan kernel, the fission kernel) and
  in the fused row loop, `exp` is Mint's own (`ir.rs`, `mint_exp_fast`): at
  most 2 ulp over 3e7 test inputs, with NaN, infinities, overflow and
  subnormal results as in libm. `--no-inline-exp` uses `llvm.exp` everywhere.
- In the fission kernel, `log` is Mint's own too (`ir.rs`, `mint_log`):
  x = 2^k z with z in about [0.684, 1.371), z/c - 1 = r from a 128-entry
  table of 1/c and log c (two gathers; the step that holds 1 has c = 1, so
  there is no cancellation near 1), and a degree-7 polynomial for
  log1p(r). Worst error found against long double `logl`, 4.3e6 inputs over
  every binade (subnormals included), around 1 and around every table step:
  1.5 ulp. 0, -0, negatives, infinities and NaN give what libm gives.
  `tests/run.sh` checks it as emitted. `--no-inline-log` calls glibc's
  vector `log` there instead.
- The tiled Gram kernel and row fusion change the order of summation (per
  tile and per chunk); both are off under `--strict-fp`. The tiled kernel
  also groups each product as `X[i,j] * (w[i] * X[i,k])`, where the untiled
  one computes `(w[i] * X[i,j]) * X[i,k]`. The two differ by more than
  rounding only when `w[i] * X[i,k]` leaves the normal range of doubles
  (below about 2e-308 or above 1.8e308) while the full product does not.
- In `BernoulliLogit`, e = exp(-|eta|) is in (0, 1]. In scalar code and with
  `--no-fission-kernel`, `log1p(e)` is computed as `log(1 + e)`, because `log`
  has a vector version; that costs an absolute error of up to about 1e-16
  per observation. In the fission kernel's vector code (every row but the
  last n mod 4, which run in scalar code) it is Mint's own `log1p` on [0, 1]
  (`mint_log1p01`), which takes q = 1/(1 + e) (computed once, for the
  sigmoid too): m = round(256 (1 - q)), 1 - m/256 is exact, and
  r = e (1 - m/256) - m/256 is one fused multiply-add on the exact e, so
  1 + e is never rounded. The bound is about 2 ulp (at the first table
  step, where the result is half the table entry); the test over 4.5e6
  inputs found 1.93 ulp, an independent reproduction 1.97, and
  `tests/run.sh` fails above 2.1. The sigmoid there is q for eta >= 0 and
  e q otherwise, one rounding more than e/(1 + e).
- There are no `nnan` or `ninf` assumptions: NaN and infinity behave as in
  IEEE.
- `--strict-fp` turns all of this off, including the `log1p` substitution.
  `bench/results.json` includes those runs.

Results therefore differ from a strict left-to-right evaluation in the last
bits. For every benchmarked logistic and linear configuration, `tests/run.sh`
checks at a fixed point that the log density and every gradient component
agree with the Rust baselines to 1e-9 relative. The dynamic Poisson gradient
is checked against the exact formula at both sizes.

## Runtime (`runtime/mint_rt.c`)

- The NUTS sampler is a port of the structure of Stan's `base_nuts`:
  multinomial sampling, the generalised no-U-turn criterion with the extra
  checks across subtrees, a diagonal metric, and Stan's windowed warmup (dual
  averaging of the step size, variance windows 75/25…/50).
- **Immutable, reference-counted states.** A leapfrog step writes a new state
  (position, momentum, gradient) instead of updating one in place. The tree then keeps references to its end points and
  proposals rather than copying D-length vectors. States are recycled from a
  per-chain free list.
- **Fused passes.** The leapfrog's second half-step, kinetic energy and
  subtree momentum sum are one pass. Each merge's summed momentum and three
  no-U-turn checks are another. The scaled momentum (inverse metric times
  momentum) is recomputed inside those passes rather than stored, which saves
  one D-length vector of memory traffic per state.
- **Threads within a chain.** When the model has at least 8,192 parameters,
  those passes are split across OpenMP threads, by default
  (online CPUs ÷ 2) ÷ chains per chain; `MINT_THREADS_PER_CHAIN` overrides it.
  The model gradient itself still runs on one thread per chain.
- **Same draws where serial.** On the serial path the arithmetic and the order
  of random draws are unchanged, so it produces bit-identical draws to the
  original copying sampler. This was checked on eight schools and the dynamic
  Poisson model; `tests/run.sh` does not re-check it. The threaded path sums
  in a different order, so its draws differ by rounding and then diverge.
- **Speedups.** A whole 4-chain, 1000 + 1000 run went from 15.3 s to 8.3 s at
  3,171 dimensions (1.8x) and from 1448 s to 278 to 325 s at 37,901 dimensions
  (4.5 to 5.2x, depending on the run; 2.1x from the reference-counted states
  alone).
- **Metric adaptation.** The default is Stan's: the regularised variance of
  the warmup draws. `MINT_METRIC=grad` instead uses
  `sqrt(var(draws) / var(gradients))`, as nutpie does. It is not the default:
  over several seeds it made no measurable difference on eight schools or
  logistic regression, and gave about 3.5x fewer effective draws per gradient
  on the dynamic Poisson model (`bench/metric_experiment.py`).
- Chains run on separate threads. The generated `logp` functions only read the
  data, so they are safe to call concurrently.
- The summary reports the mean, sd, quantiles, split R-hat and an
  autocorrelation ESS (Geyer's initial monotone sequence).
- `MINT_GRADCHECK=1` compares the compiled gradient with central finite
  differences.
- `MINT_BENCH_GRAD=K` times K gradient evaluations at a fixed point and exits.
  The Rust baselines use the same code path.

## Deliberate limitations of the prototype

The language subset is small on purpose:

- Control flow is `repeat N { }` only: no `if`, no indexing, no user-defined
  distributions.
- Model parameters are `Real`, `Positive`, `Vector[n]`, `Positive[n]` or
  `Matrix[m, n]`.
- Distributions are `Normal`, `BernoulliLogit`, `PoissonLog` and
  `Exponential`.
- Inside a model, a matrix product must be `data_matrix * name`, in a
  vector-shaped statement.
- `cumsum` runs along the last dimension only.

Other known limits:

- Model `let`s are inlined. A `let` used by two `~` statements is computed
  twice.
- Model data is passed through globals, so only one `sample` call per model
  can run at a time. Chains within a call are fine.
- The checker stops at the first error.
- The Gram kernel's one-triangle computation has no switch. Its four-row
  blocking can be turned off with `--no-gram-blocking`.
- Errors found while lowering a model body (the constructs listed above) are
  reported without a source position.
- Model kernels are single-threaded. The parallelism is across chains, plus
  the sampler's own passes within a chain for large models.
- The Gram kernel is register-blocked but not cache-blocked. A tuned BLAS
  `dsyrk` would beat it on large p.
