# Compiler architecture

Mint's compiler (`mintc`, about 5,800 lines of Rust with no dependencies) owns
every stage from source text to LLVM IR. LLVM, invoked through `clang -O3
-march=native`, does the last step: instruction selection, register
allocation, loop vectorisation and unrolling. A small C runtime (about 1,160
lines) provides I/O, printing, a Cholesky solve and the NUTS sampler. The
runtime is compiled once and cached, so building a program takes 60 to 140 ms:
`mintc` itself takes 1 to 2 ms, clang about 30 to 120 ms, and linking about
30 ms. (Measured before narrow data, see below; a model whose kernels get
narrow-data variants now takes 0.37 to 0.56 s, the rest are unchanged.)

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
- With a fused scan kernel that owns a matrix parameter, also `leap`, the
  log density with a hook for the sampler (see the fused leapfrog in the
  Runtime section), and `leap_blocks`.

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
quantity of that shape is stored in the scan layout (`--no-scan-layout`
turns this off): the parameter inside the sampler's vector, the data (copied
once per `sample`), and scratch buffers. The series are taken in blocks of
8 (a kernel group, below), each block stored column-major: the 8 series at
time 0, then at time 1, and so on, so each block is one contiguous range of
8 T values; the G mod 8 series left over form a last block stored the same
way (`Mg::cm_row`: element (g, t) is at base(g) + t * stride(g)). Draws are
still written in the user's order, and the runtime converts its benchmark
point and printed gradients through two generated functions
(`mint_set_layout`). Four adjacent series at one time are then one
contiguous vector load, and a kernel group reads and writes one contiguous
range instead of 8 values every G. (Until this round the layout was plain
column-major, whole columns of G. With the blocks the large model's
gradient took 43 to 45 µs against 50 to 53 µs on one thread, 12 to 18%
less, and 17 to 24 against 24 to 29 µs on three threads; five interleaved
pairs at load 2.8 to 5.5, see compiler-round.md.)

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
- **Threads** (`--no-parallel-kernel` turns it off; off under `--strict-fp`).
  A group of eight series touches only its own rows of the matrix parameter
  and its gradient, so the loop over groups is emitted as a function of its
  own, `(ctx, g0, g1, tid)`. The context carries theta, grad, the buffers of
  Positive vector parameters, the scalar parameter values and two output
  arrays (and, in the fused leapfrog's `leap`, its hook). The runtime's `mint_par_groups` gives thread t of a team of T the
  groups [n t / T, n (t + 1) / T). Each thread has its own scratch (the same
  per-thread `mint_ws_slot`s), its own slice of the column partial sums, and
  writes its log density and scalar adjoints to entry t; the caller adds
  them up in thread order. Single vectors and leftover rows are run by
  thread 0 (the calling thread) after its groups, inside the parallel
  region, so that every row is done when the region ends (they used to run
  after it, which mattered once the fused leapfrog, in the Runtime section,
  needed every row's gradient final inside the region). The team is the chain's threads per chain during sampling
  (so 1 below 8,192 parameters), 1 for `MINT_BENCH_GRAD` and
  `MINT_GRADCHECK`, and `MINT_KERNEL_THREADS` overrides both. When one
  thread is requested the generated code takes the serial loop instead,
  so the result is bit-identical to a build with `--no-parallel-kernel`.
  With more threads the log density and the gradients of scalar and
  column-indexed parameters are summed in a different order. That is
  deterministic for a given team size, and usually changes only the last
  bits, but like Mint's other reassociated sums it can change a component
  by more when large terms cancel.
- On the large dynamic Poisson model (250 series, 31 groups) the gradient
  took 53 µs on one thread and 21.5 µs on three cores that share an L3
  cache, before the scan layout's blocks (above). On three cores spread over the Ryzen's two core complexes it takes
  32 µs; why the kernel itself runs slower then was not found (the threads
  write no shared cache lines except at range edges, and removing those
  stores did not close the gap). In whole 4-chain runs of that model
  (150 + 150 draws, three threads per chain) the build with the parallel
  kernel was faster in each of the last three interleaved pairs (22.4, 21.9
  and 33.0 s against 26.0, 26.8 and 35.1 s). Other jobs were running on the
  machine, and in earlier pairs either build took anywhere from 22 to 59 s
  (once 154 s),
  so the size of the gain is not established.

`tests/run.sh` builds eight models three ways (default, layout without
fusion, and neither) and requires the same log density and gradient to 1e-12
at 7, 13, 16 and 20 series (16 leaves the layout no last block), plus a
finite-difference check at the benchmark point. The models cover linear and nonlinear nested running sums, the
one-lane BernoulliLogit path, row, column and scalar parameters inside and
outside the sum, a column parameter also used by statements of another
shape, a running sum of data only, two scan statements sharing a matrix
parameter, and two scan statements each owning its own. A further test checks that sampled draws come back in the user's
order. For the parallel kernel the same models (also at 61 series: seven
groups, a single vector and a leftover row, which thread 0 runs inside the
parallel region) and the dynamic Poisson model
must be bit-identical on 1 thread to the `--no-parallel-kernel` build, and
on 3 threads agree with 1 thread to 1e-12 of the largest gradient component
and 1e-10 relative per component. The tests also check which models are
parallelised (not the one-lane BernoulliLogit kernel, nor a running sum of
data only), that 3 threads on the large model change the summation order
(so the split ran), that a team of one gives the serial result, and that
two 3-thread gradients, and the raw draws of two short
3-threads-per-chain sampling runs, are identical. They cannot show the
absence of a race; they would catch one only if it changed these results. Not covered: a model with running sums over two different shapes,
and gradients away from the benchmark point (at the runtime's random point
these models' log densities are near -1e16 and finite differences fail for
every build). The measurements are in
[compiler-round.md](compiler-round.md).

### Narrow data

Data reaches a model through `sample()`, and the generated `init` already
copies some of it (the scan layout's transposed matrices). It also looks at
the values. For each data vector or matrix that Mint's own vector kernels
load (the vectorised scan kernel and the statements it absorbs, and the
fission kernel), up to a limit described below, the runtime's `mint_narrow`
tries the types allowed for that buffer, narrowest first, and makes a copy
in the first one that holds every value exactly, bit for bit: an integer
type rejects -0.0, and float rejects NaN and keeps infinities, -0.0 and
float subnormals. The kernels then load the copy and convert in registers
(`vpmovsxbd` + `vcvtdq2pd`, or `vcvtps2pd`). `--no-narrow-data` turns this
off; it is also off under `--strict-fp` and on hosts without AVX2.

The choice depends on the data, so it is made at run time. `logp` is
generated in up to four variants (variant 0 reads only doubles), `init`
records which one this call's data allows, and `sample` hands that variant's
function pointer to the sampler, so a gradient pays nothing for the choice.
A BernoulliLogit outcome is checked to be 0 or 1, so it is allowed int8
only; anything else is allowed int8, int16 and float. More than four
variants would be too many copies for clang to compile, so the less likely
types go first (small integers in data that is not a count or 0/1 outcome,
then float and int16 for counts), and then whole buffers. The logistic
model's X and y get {double, float} and {double, int8}, the dynamic Poisson
model's y all four. Each variant is a full copy of `logp`: the dynamic
Poisson model now takes 0.56 s to build instead of 0.22 s, and the logistic
model 0.37 s instead of 0.14 s (medians of 7, this machine). For a test,
`MINTC_NARROW_VARIANTS=N` in mintc's environment raises the limit.

**Why the results do not change, and how far that is established.** The
converted values are exactly the doubles the kernel would have loaded, but
that alone was not enough. LLVM also learns facts from a conversion that it
does not have for a loaded double (an integer converted to double is never
-0.0, for example), and the code it emits can then differ: on
`y ~ Normal(a * X * beta + b * y, exp(X * beta))` with float data, the
backend fused a different multiply into an add and a gradient component
changed in the last bit (found by the independent review). So each
converted value passes through an empty inline asm, which emits no
instruction but makes the value as opaque to the optimiser as a load
(`llvm.arithmetic.fence` did not prevent the difference). And only vector
code (lanes > 1), which is Mint's own, reads the copy: scalar loops
(leftover rows, the fission kernel's last n mod 4 rows, statements that are
not vector kernels) keep reading doubles, because LLVM vectorises those and
its choice of vector width and interleaving, and so the order of a
reassociated sum, could depend on the type loaded. This makes identical
results the expected outcome; it is not a proof about LLVM. The evidence is
the tests: `tests/run.sh` builds 38 cases with and without the rewrite (the
fission and scan kernel test models with the original data, with every
value rounded to float, and at the boundary of each type; integer design
matrices; a model with 34 data buffers; the dynamic Poisson and logistic
models) and requires byte-identical exact log densities and gradients under
every `MINT_NARROW` setting, on one and three kernel threads, plus
identical raw draws of four short sampling runs, and checks that each case
picked the type it is meant to exercise. A randomised check
(`tests/narrow/fuzz.py`: 7 models, 12 data sets each, twice) compares the
gradient and the raw draws of a short run with and without the copies; the
draws amplify a difference in the last bit anywhere along the trajectory.
Without the asm it fails on 28 of 30 data sets of the model above.

The same round changed one thing for every build: register sums of
adjoints in the vector kernels (one element's running-sum or product
adjoint, one column's gradient) start at -0.0 instead of 0.0
(`--no-negzero-sums` turns it off). -0.0 + x is x for every x, so LLVM drops
the first add, which it cannot do for 0.0 + x. That is not the same
arithmetic once multiplies are fused into adds: with the add gone, the
product that was its operand can be fused into the next add without being
rounded, which can change a result in the last bits (it does on
`x ~ Normal(c * y, 1); y ~ Normal(cumsum(a * x, T), exp(b))`, found by the
second review). Mint's floating-point rules allow that (every add and
multiply carries `contract`, below). On the benchmark models the log
densities, gradients and draws are byte-identical to the previous
compiler's, and with `--no-narrow-data --no-negzero-sums` their IR is the
previous compiler's, byte for byte. The narrow and wide variants of one
build share the change, so it does not affect their identity. On its own it
is worth about 1 to 2% on the time-series gradient, but combined with the
narrow copy about 5 to 7% (see compiler-round.md).

`MINT_NARROW=0` makes the runtime pick variant 0, `MINT_NARROW=int16` or
`float` skips the narrower types, and `MINT_NARROW_REPORT=1` prints each
choice.

In the benchmark data the counts of the dynamic Poisson model (0 to 76) and
the 0/1 outcomes of the logistic model narrow to int8; the real-valued
matrices of the logistic model and Newton's method are not exact in float
(0 of 10^5 and 0 of 10^7 values). Newton's method is a function, not a
model, and has no such step. The measurements are in
[compiler-round.md](compiler-round.md#follow-up-narrow-data).

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
- Narrow data copies are exact, and the kernels' results with them are
  byte-identical to those without them in every test (see "Narrow data").
- Register sums of adjoints in the vector kernels start at -0.0, so their
  first add disappears; through contraction that can change results in the
  last bits (`--no-negzero-sums`).
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
- **One pass per leaf.** A merge of two subtrees (their summed momentum and
  the three no-U-turn checks) needs nothing that is not known once the last
  leaf of the second subtree exists. So a merge is registered before its
  second subtree is built, and that subtree's last leaf computes it in the
  same pass as its own second half-step and kinetic energy, together with
  every other merge it completes (a leaf that ends a depth-k subtree completes
  k merges, plus the trajectory's own merge when it ends the new subtree). Unless the leaf ends the trajectory's new subtree, the same pass
  also takes the first half-step of the next leaf, which is always the next
  leapfrog from this leaf in the same direction. Most leaves therefore cost
  one pass over D next to their gradient, instead of three to four. A
  subtree's summed momentum is stored only when a later merge reads it (the
  first half of a merge); the inner merges of a leaf pass their sums along in
  a block-sized buffer. The results are read in Stan's order, after the
  multinomial proposal of that level is drawn, and a half-step taken for a
  leaf that never comes (the tree stopped) is discarded. The scaled momentum
  (inverse metric times momentum) is recomputed inside the pass rather than
  stored.
- **Threads within a chain.** When the model has at least 8,192 parameters,
  those passes are split across OpenMP threads, by default
  (online CPUs ÷ 2) ÷ chains per chain; `MINT_THREADS_PER_CHAIN` overrides it.
  The same threads share the fused scan kernel of the gradient (see the scan
  kernel section); the rest of the gradient runs on the chain's own thread.
  Each threaded chain's threads are restricted to the CPUs that share one L3 cache (chains
  are dealt to the L3s in turn), because the chain's thread and its helpers
  exchange the state vectors every leapfrog and that traffic is slow between
  L3s on a chiplet CPU. This is skipped when OpenMP teams are dynamic
  (`OMP_DYNAMIC`) or a chain gets a smaller team than asked for, and each
  thread's previous mask is restored when the chain ends.
  `MINT_CHAIN_AFFINITY=0` turns it off.
- **Fused leapfrog** (off by default: a program built with `mintc
  --fused-leapfrog` uses it, unless `MINT_FUSED_LEAPFROG=0`). When a fused
  scan kernel owns a matrix parameter, the compiler then also emits
  `leap(theta, grad, hook, hctx)`, a second copy of the log density with a
  hook: each kernel thread, as soon as its groups are done (their gradient
  is then final), calls the runtime's `leaf_block` on its own rows, which in
  the scan layout are one contiguous range of the parameter; thread 0 also
  hands over the rows left over after the groups, which it runs too. The
  whole leaf work on that range (second half-step, the next leaf's first
  half-step, kinetic energy, merges) runs there, on the core that has just
  written its gradient, with the runtime's own code, so every stored value
  has the unfused arithmetic. `leap_blocks` tells the runtime which parts of
  theta are covered (at most 64 parameters); the calling thread does the
  leaf work on the rest (401 elements on the large model) after the
  gradient. A leaf then needs no parallel region of its own for its second
  half; the first half-step of a leaf that has none taken ahead (the first
  after the end of a trajectory's new subtree) is still a parallel pass
  (`leaf_start`), as before. The kinetic energy and the merges' sums are
  added in another order (in blocks of 8 from the start of each thread's
  range, the threads in order, then the rest), so the draws differ by
  rounding from the unfused path and then diverge.
  `MINT_FUSED_LEAPFROG=exact` does only the half-steps in the hooks and the
  sums in a leaf pass of the runtime's order; its draws were bit-identical
  to the unfused path's in every test. That rests on `leap` computing
  exactly the gradient of `logp`, which holds for the tested models but is
  not guaranteed by construction: the two are separately optimised copies,
  and LLVM may reassociate their `reassoc` sums differently.
  The fused leapfrog is off by default because it was not faster
  (`docs/compiler-round.md`). Our reading of why: once the scan layout
  stores each group of series contiguously, the runtime's own leaf pass,
  split by index ranges, already gives each thread nearly the same elements
  as its kernel groups (on the large model the ranges differ by 231 and 863
  elements at the two boundaries, out of 37,901, assuming each OpenMP
  thread runs on the same core in both regions), so fusing saves one
  parallel region per leaf, and the hooks (which leave the rest of theta to
  one thread) cost about as much; neither cost was measured on its own.
  `tests/run.sh` checks, in `--fused-leapfrog` builds: that `exact` ran and
  gave the unfused draws, on a scan test model (nested, 61 series) and the
  small dynamic Poisson model with 1 and 3 threads per chain and on the
  large model with 3; that the fused sums ran and are deterministic; how
  many parameters are covered (one for nested, three in two kernels for
  another model); and, with `MINT_LEAP_TEST`, one fused leaf with a merge
  against the unfused one (state bit-identical, each sum to 1e-12 of
  itself or of a thousandth of the largest) for each scan test model that
  has a covered parameter (all but two: a running sum of data only, and a
  matrix shared by two scan statements, which neither kernel owns), at 7 to
  61 series, with 1 and 3 kernel threads requested (a kernel runs at most
  one thread per group of 8 series, so only the 61-series runs split three
  ways). The fused sums themselves are checked only by that one-leaf test.
  With the low-rank metric (below) the hooks still do only the diagonal
  part of each leaf; the low-rank part runs in a pass after the kernel.
- **Fixed summation order.** Every sum over D in the leaf passes (kinetic
  energy, no-U-turn checks) is accumulated in 8 lanes
  (element i in lane i mod 8, each lane in index order, lanes combined in a
  fixed tree; per thread ranges start at multiples of 8 and thread totals are
  added in thread order). The starting energy of a transition and the
  step-size search use a plain sequential sum, as before. Draws are therefore reproducible for a given
  thread count and do not depend on blocking, fusion or the compiler's
  vectorisation, but they differ by rounding (and then diverge) from earlier
  versions of the sampler and between thread counts. The fused sampler gave
  bit-identical draws to the unfused one with the same sums (Stan's control
  flow, separate merges) on eight schools (which has divergent transitions),
  logistic and linear regression and the dynamic Poisson model, serial and
  with 3 threads per chain; `tests/run.sh` does not re-check that. (The fused leapfrog, above, sums in another order.)
- **Speedups.** A whole 4-chain, 1000 + 1000 run went from 15.3 s to 8.3 s at
  3,171 dimensions (1.8x) and from 1448 s to 278 to 325 s at 37,901 dimensions
  (4.5 to 5.2x, depending on the run; 2.1x from the reference-counted states
  alone). The one-pass leaves and the L3 affinity then took a whole
  37,901-dimension run from 200 s (one run of the previous runtime) to 113
  and 116 s (two runs, 3 and 2 threads per chain), and the 3,171-dimension
  run from 4.0 to 4.4 s to 3.7 to 3.9 s (three runs each). These were
  measured on a shared machine with other jobs running (load 7 to 13), so
  the individual times are noisy; the large run's gain is mostly the
  affinity.
- **Metric adaptation.** The default is Stan's: the regularised variance of
  the warmup draws. `MINT_METRIC=grad` instead uses
  `sqrt(var(draws) / var(gradients))`, as nutpie does. It is not the default:
  over several seeds it made no measurable difference on eight schools or
  logistic regression, and gave about 3.5x fewer effective draws per gradient
  on the dynamic Poisson model (`bench/metric_experiment.py`; those runs,
  made with an earlier sampler, are in `bench/metric_results_diagonal.json`).
- **Low-rank metric (`MINT_METRIC=lowrank`, opt-in).** The inverse metric is
  Stan's diagonal `v` plus a correction along at most k directions
  (`MINT_LOWRANK_K`; by default 16 or 24 if each thread's single-precision
  share of them fits in its L2 cache, otherwise 8: 24 at 3,171 parameters
  with one thread per chain, 8 at 37,901 with three; the threads counted are
  those OpenMP actually gives the chain):
  `diag(v) + S U diag(lam - 1) U' S`, with
  `S = diag(sqrt(v))` and orthonormal columns `U`. With the default metric
  nothing changes: the raw draws are bit-identical to those of the sampler
  before the option existed (checked on eight schools, logistic and linear
  regression, the small time-series model, a short run of the large one and
  the Gaussian test model, with 1 and 3 threads per chain). When the option
  was merged with the fused leapfrog and the fast warmup, the merged
  sampler's raw draws were checked again against those of the sampler
  before the merge, on the same models except the Gaussian, with 1 and 3
  threads per chain: identical with the default settings, with
  `MINT_WARMUP=fast`, and in `--fused-leapfrog` builds. That comparison
  needs a build of the earlier commit and is not part of `tests/run.sh`.
  - *Estimation.* At the end of each warmup window, from the window's draws
    and their gradients (the most recent ones that fit in `MINT_LOWRANK_MB`,
    default 256 MB per chain). In coordinates scaled by `S`, the gradient
    covariance of a Gaussian posterior is its precision, so the leading
    eigenvectors of the sample gradient covariance (2k candidates, from the
    n × n Gram matrix of the n draws when n < D) estimate the directions in
    which the posterior is narrower than the diagonal assumes. Within their
    span the covariance is the SPD geometric mean of the draws' covariance
    and the inverse of the gradients' covariance, as in nutpie's low-rank
    adaptation. Its eigenvalues outside [1/2, 2] (`MINT_LOWRANK_CUTOFF`) are
    kept, largest |log lam| first, each limited to [1e-4, 1e4]; a small ridge
    (`MINT_LOWRANK_GAMMA`, 1e-5) keeps the covariances definite. The leading
    eigenvectors come from a full eigendecomposition of the Gram matrix; the
    D-length products are blocked and use the chain's threads. A window whose directions are not finite or not
    independent falls back to the diagonal.
  - *Per leapfrog step: one projection and one expansion.* Every momentum,
    momentum sum and gradient carries its k projections `c = V' x`
    (`V = S U`) after its D entries. The kinetic energy and the no-U-turn
    sums then need only O(k) more work (`p_sharp . r` gains
    `sum_j d_j c_j(p) c_j(r)`, and a sum of momenta projects to the sum of
    their projections). Momenta and gradients project linearly, so a leaf
    needs the projection of its new gradient only; it is computed in the
    leaf's fused pass, block by block next to the second half-step and the
    merges. The low-rank part of the next leaf's position update,
    `eps V diag(lam - 1) c(p_half)`, needs that projection complete, so after
    a barrier the same parallel region adds it to the next leaf's position,
    which the pass had already started. (When the leaf ends the new subtree,
    `leaf_start` does it in its own pass, fused with the first half-step.)
    That is two streams over the D × k directions per leapfrog, one more
    read-modify-write of the next position, and no extra fork of the thread
    team.
  - *Precision.* The directions are stored in single precision in tiles of
    8 directions × 8 entries. Projections are accumulated in double (the
    products of the stored values with double vectors are exact before
    rounding), so the carried projections differ from `V' p` of the stored
    momentum only by rounding, as the momentum itself does along a
    trajectory. The position update forms its sum in single precision. Any
    map of the momentum that is odd keeps the leapfrog volume-preserving and
    as reversible as any floating-point leapfrog, so this affects how well
    energy is conserved, not the distribution sampled. Momenta are drawn
    through a k × k matrix that makes their covariance the inverse of the
    kinetic energy's matrix for the stored, rounded directions.
  - *Cost.* The projection is limited by the conversion of the stored
    values to double (about a quarter of a cycle per direction and entry on
    the Zen 3 test machine), the expansion by the bandwidth of the cache that
    holds the directions. On the 3,171-parameter model, with 24 directions
    and one thread per chain, two `perf` profiles put the two at 0.7 to 0.9
    and 0.4 to 0.5 times the cost of the model's gradient (on a loaded
    machine). On the 37,901-parameter model with 3 threads per chain, a
    leapfrog step cost about 1.2 times as much as with Stan's metric with 8
    directions and 1.5 times with 24 (profiles of a 300 + 100 run, relative
    to the gradient's scan kernel). The estimation at the window ends took
    about 0.3 s per chain on the small model over the whole warmup, most of
    it the eigendecomposition for the last window's 500 draws. These
    profiles and timings are not recorded in the repository.
  - `tests/run.sh` checks it on eight schools (also with every direction
    kept, 3 threads per chain) and on a 70-dimensional Gaussian whose
    posterior is 4 to 40 times narrower than the prior along 8 dense
    directions; every whitened first, second and cross moment must be within
    4 Monte Carlo standard errors of the exact value (5 for the 2,415 cross
    moments), with 1 and 3 threads, and the metric must keep 8 directions
    (the test checks their number, not their span). Pooled over 24 seeds
    (96 chains of 1000 draws each; `tests/metric/pool_gauss.py` on the draws
    of `tests/metric/gauss.mint` with seeds 1 to 4 and 6 to 25, run outside
    the test suite), the largest whitened first and second moment errors
    were 2.2 and 2.6 standard errors with one thread and 2.7 and 2.4 with
    three, against 2.2 and 2.8 for Stan's metric.
  - *With the fused leapfrog* (`mintc --fused-leapfrog`). The kernel's
    hooks (`leaf_block`) do the diagonal part of each leaf as before: the
    second half-step, the next leaf's first and the D-length merge sums.
    They do not project the gradient or add the low-rank part of the next
    position. The position update needs every thread's projections, and a
    hook runs inside the kernel's parallel region with no way to wait for
    the other threads; the projection itself works on whole tiles of 8
    entries, which a hook's range need not start on, and doing it after the
    kernel keeps the runtime's blocking and order. So after the kernel, a pass across the chain's threads
    (`leaf_work` with `half = 0`) projects the gradient and, after a
    barrier, expands the next position, split, blocked and summed as in the
    runtime's own leaf pass. The projections, the merges' low-rank parts
    and the low-rank part of the position are therefore exactly those of
    the runtime's path, and only the D-length sums are in the hooks' order.
    With more than one thread per chain this costs one more parallel region
    per leaf; with any number it costs another read of the
    gradient and the next position, on top of the two streams over the
    directions that the runtime's path has as well. With
    `MINT_FUSED_LEAPFROG=exact` the leaf pass after the kernel does the
    projections next to the sums, so the draws are the runtime's low-rank
    draws. `MINT_LEAP_TEST` checks one leaf with 12 fixed directions as
    well as with the diagonal metric, and `tests/run.sh` checks that
    `exact` gives the runtime's low-rank draws on two models with 1 and 3
    threads per chain. Both paths of the one-leaf test share the projection
    code, so what it checks is that the hooks and the pass after the kernel
    together produce the runtime's leaf (it fails when that pass, or its
    position update, is left out); it does not cover a leaf without a next
    leaf or a merge of two single leaves.
  - *With the fast warmup* (`MINT_WARMUP=fast`). The chains pool the
    directions as they pool the diagonal: each chain keeps its window's
    draws and gradients in its own rows of one shared array, and at the end
    of a window, once the pooled diagonal is in, chain 0 moves all chains'
    rows together (in chain order) and estimates the directions from all
    of them with its own threads, while the other chains wait; every chain
    then installs the same directions. The estimate is centred on the
    pooled mean, as the pooled variance is, so differences between the
    chains' positions count as spread, as they do for the diagonal. The
    number of directions is the smallest any chain would keep. With one
    chain, or `MINT_WARMUP_POOL=0`, each chain estimates its own.
    `tests/run.sh` checks the pooled combination on eight schools and the
    Gaussian test model (1 and 3 threads per chain), the per-chain one on
    the Gaussian, and that two pooled runs give identical draws. On the
    3,171-parameter time series (seed 1, one run of each, one thread per
    chain) the lowest ESS per 1000 gradients was 0.88 with Stan's metric
    and warmup, 1.50 with the fast warmup, 5.85 with the low-rank metric and
    9.78 with both. That is one run per variant: it shows the combination
    works, not by how much it beats either alone. These runs used the
    merged sampler and compiler, so they differ from the seed-1 runs
    recorded earlier in `bench/metric_results.json` and
    `bench/warmup_results.json` (1.00 and 5.04 for Stan's and the low-rank
    metric, 1.57 for the fast warmup): the compiler changes merged since
    those were made change the generated model code, and with the same
    seed the draws differ from the first one on, with Stan's metric too. On the same generated code, the merged runtime's
    low-rank draws are bit-identical to the low-rank branch's (checked on
    this model with 1 and 3 threads per chain). These four runs are not
    recorded in the repository.
  - *Posterior means.* On the 3,171-parameter time series, 4 runs (seeds 21
    to 24) of the low-rank metric through the fused leapfrog, and 4 of the
    low-rank metric with the fast warmup, each agree with 4 runs of Stan's
    metric and warmup on the means of pop, every beta and every terminal
    state to at most 1.94 and 1.52 MCSE (0.022 and 0.017 posterior sd) over
    41 quantities (`bench/compare_means.py`; the draws are not in the
    repository).
- **Warmup.** The default is Stan's: a uniform(-2, 2) start and the
  program's warmup iterations in windows 75 / 25, 50, 100, ... / 50.
  `MINT_WARMUP=fast` changes three things and nothing after warmup:
  - *Start.* Each chain climbs the log density with L-BFGS (6 pairs,
    backtracking Armijo search) from its uniform start. After every step it
    builds a diagonal Gaussian from the curvature pairs (a diagonal BFGS-type
    update that keeps only the diagonal after each pair, so not the diagonal
    of the full L-BFGS matrix; Pathfinder uses diagonal plus low rank) and
    estimates its ELBO from 4 draws. The chain starts at the point with the
    best ELBO. The climb stops 20 steps after the last improvement of the
    ELBO (or earlier, when no step is found or the log density stops
    rising). On the time-series model this costs 830 to 1,040 gradients per
    chain, and the chains start at log density 26,900 to 28,200 where the
    climbs stopped at 27,700 to 28,300 (one run, seed 1,
    `MINT_WARMUP_TRACE=1`). On a centred hierarchical model, whose density
    is unbounded as the scale goes to 0, the climb heads for the
    singularity but the ELBO falls, so the chain starts before it.
    `tests/run.sh` checks that on centred eight schools every chain's start
    was chosen by the ELBO and that the climb went on to a log density at
    least 10 higher; it does not check the posterior, which is poor under
    either warmup on that model.
  - *Schedule.* max(200, warmup / 5) iterations, never more than the
    program's warmup, in windows 10 / 10, 20, 40, ... / 50, with Stan's
    stretching of the last window. With warmup = 1000 that is windows of 10,
    20 and 110 draws.
  - *Pooling.* At the end of each window the chains wait for each other and
    estimate the variance from all chains' window draws (combined in chain
    order, so draws are reproducible), so every chain gets the same metric.
    With one chain there is nothing to pool, and with `MINT_METRIC=grad`
    each chain keeps its own estimate.

  The default warmup's gradients are spread evenly: on the time-series model
  an early iteration costs about as much as a kept draw (255 leapfrog steps),
  so warmup is half the gradients. Most of the saving is the shorter
  schedule; the start and the pooling are what let it be this short (Stan's
  own schedule cut to 300 iterations gave 1.38 to 1.56x over 6 seeds, in
  the results file as `stan_schedule_300`; a 10 / 10 schedule from the
  uniform start spent more on warmup than 75 / 25 did in 3 seeds of the
  time series, because the first windows estimated the metric from draws
  that had not converged; that run is not in the results file).
  Measured with `bench/warmup_experiment.py`, 1000 kept draws, 4 chains,
  ESS the runtime's lowest over all parameters, results in
  `bench/warmup_results.json`:

  | model | seeds | gradients, Stan → fast (median) | lowest ESS per 1000 gradients, Stan → fast (median) | fast / Stan, paired, geometric mean (range) |
  |---|---|---|---|---|
  | time series, 3,171 parameters | 8 | 2.02 M → 1.28 M | 1.02 → 1.56 | 1.56 (1.35 to 1.90) |
  | time series, 37,901 parameters | 4 | 3.66 M → 2.51 M | 0.109 → 0.164 | 1.60 (1.46 to 1.98) |
  | logistic regression | 8 | 57.6 k → 35.0 k | 96 → 159 | 1.62 (1.37 to 1.81) |
  | logistic regression, 1 chain | 8 | 14.3 k → 8.9 k | 95 → 131 | 1.39 (1.06 to 2.35) |
  | eight schools | 8 | 78.7 k → 45.7 k | 38 → 65 | 1.71 (1.39 to 2.25) |

  The fast warmup was ahead on every one of these 36 paired runs. The lowest
  ESS itself moves a lot between seeds (805 to 1,546 for one-chain logistic
  regression with the fast warmup), so single ratios are noisy.
  Wall time falls with the gradients: the 3,171-parameter runs took 2.4 to
  2.5 s against 3.7 to 4.0 s, and the 37,901-parameter runs 49 and 53 s
  against 70 and 76 s (seeds 1 and 2; seeds 3 and 4 ran at a load average
  of 25 to 34 from other jobs, and their times say nothing). Stan's warmup
  always ran first in each pair, so a drift in machine load could favour
  either side. On the three small models the
  highest R-hat was at most 1.006 with Stan's warmup and 1.002 with fast.
  On the 37,901-parameter model both are at the mixing bar, as before:
  highest R-hat 1.005 to 1.015 with Stan's warmup and 1.001 to 1.012 with
  fast, lowest ESS 309 to 409 and 323 to 543. Divergences in eight
  schools: 5 in 8 runs with Stan's warmup, 0 with fast.

  Correctness: the eight schools means are within 4 MCSE of the exact values (`tests/run.sh`,
  4000 draws, serial and 3 threads per chain); on the 3,171-parameter time
  series, 9 runs of each (seeds 21 to 29) agree on the means of pop, every
  beta and every terminal state to at most 2.10 MCSE (0.018 posterior sd)
  over 41 quantities (`bench/compare_means.py`; two groups of Stan runs, 4
  and 5 of the same seeds, differ by at most 1.48 MCSE). These draws are
  not in the repository. On the 37,901-parameter model, two runs of each (seeds 3 and 4) differ
  by at most 3.14 MCSE (0.064 sd) over 501 quantities, while the two Stan
  runs differ from each other by up to 3.69 MCSE (0.137 sd) and the two
  fast runs by up to 2.99 MCSE (0.063 sd). Means only, not
  variances or tails. Stan's warmup stays the default: the fast one was
  better on every model and seed measured here, but that is four models,
  and a warmup a fifth as long is the riskier choice for a model whose
  chains need longer to find the typical set.
- **Target acceptance.** `MINT_TARGET_ACCEPT` sets the dual averaging target
  (Stan's 0.8). At 0.7 with the fast warmup the effective draws per
  gradient rose further (2.14x Stan's warmup at 0.8 on the 3,171-parameter
  model, 1.87x on logistic regression, same seeds): on the time series the
  trajectories stay at 255 leapfrog steps, but each step is longer, so a
  trajectory travels further. But eight schools had 30
  divergent transitions in 8 runs instead of 0, so 0.8 stays the default.
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
- Only the fused scan kernel runs on several threads within a chain; every
  other model statement is single-threaded. The parallelism is across
  chains, plus the sampler's own passes and the scan kernel within a chain
  for large models.
- The Gram kernel is register-blocked but not cache-blocked. A tuned BLAS
  `dsyrk` would beat it on large p.
