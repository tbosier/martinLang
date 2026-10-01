# Compiler round: closing the gaps with max-effort Rust

The previous round left two problems where hand-written Rust (AVX2
intrinsics, glibc vector math) beat Mint: the dynamic Poisson gradient (Rust
1.8x faster) and Newton's method (Rust 1.55x faster). This round changed only
the compiler and runtime, not the language: the Mint programs are the same
text as before.

## Results

`bench/compiler_bench.py`: 11 repetitions per program, interleaved in an
order shuffled per repetition, pinned to one core of a Ryzen 9 5900X; load
average 2.0 at the start and 1.6 at the end. Gradient timings evaluate one
fixed point thousands of times, so no seed is involved. "Before" is the
previous commit's compiler and runtime. The binaries' checksums, the commands
and every measurement are in `bench/compiler_bench.json`.

| problem | Mint before | Mint after | max-effort Rust |
|---|---|---|---|
| dynamic Poisson gradient, 3,171 parameters | 7.17 µs | 4.22 µs | **4.07 µs** |
| dynamic Poisson gradient, 37,901 parameters | 95.6 µs | 53.4 µs | **52.2 µs** |
| logistic gradient (n=5000, p=20) | 34.8 µs | **32.8 µs** | 37.9 µs |
| linear regression gradient | 0.10 µs | **0.10 µs** | 0.26 µs (tuned Rust, same rewrite) |
| Newton's method, 10 iterations (n=200000, p=50) | 0.313 s | **0.161 s** | 0.209 s |

- **Dynamic Poisson: 1.7 to 1.8x faster than before; the Rust is still
  slightly faster.** By medians the Rust is 3.7% faster on the small model and
  2.4% on the large one, and it is consistently so: all 11 of its runs on the
  small model, and 10 of 11 on the large one, were faster than every Mint run.
  Before this round the Rust was 1.8x faster.
- **Newton: 1.9x faster than before, and 1.30x faster than the Rust.** Every
  Mint run was faster than every Rust run. The coefficients agree with the
  Rust's to 4e-10.
- **Logistic gradient: 1.15x faster than the Rust.** 10 of the 11 Mint runs
  were faster than every Rust run. In this same measurement the previous
  compiler was already 1.09x faster than the Rust (the previous report called
  it a tie, from an earlier measurement); the further gain comes from the
  allocator change below, not from a new pass.

Whole sampling runs of the small dynamic Poisson model, 4 chains,
1000 + 1000, three seeds each (`bench/dynpois/seed_runs.py small 1 2 3`):

| program | sampling time, seeds 1 / 2 / 3 | µs per gradient per chain |
|---|---|---|
| Mint before | 8.42 / 8.51 / 8.50 s | 16.5 to 16.9 |
| Mint after | 6.73 / 6.74 / 6.69 s | 13.3 |
| max-effort Rust | 6.63 / 6.59 / 6.65 s | 13.1 to 13.2 |

All nine runs mixed (highest R-hat 1.004). The Rust run had the higher lowest
ESS for all three seeds (Mint after 2070, 1933, 1598; Rust 2294, 2663, 2455),
and so did the Rust against the previous Mint, whose parameter layout is the
Rust's; with three seeds that could be chance, and it was not investigated.

## What changed

Each item has a switch, so its effect can be measured by turning it off.

1. **Scan layout** (`--no-scan-layout`). When a model takes `cumsum` of a
   `Matrix[G, T]` along T, every model quantity of that shape is stored
   column-major: the parameter inside the sampler's vector, the data (copied
   once when `sample` starts) and the scratch buffers. The user does not see
   this: draws are written in the original order, and the runtime converts the
   benchmark point and printed gradients through two generated functions
   (`mint_set_layout`). Four adjacent series at one time are then one
   contiguous vector load, so the scan vectorises across series without the
   4 x 4 transposes the Rust needs.
2. **Fused, vectorised scan kernel** (`--no-scan-fusion`). A matrix statement
   whose only materialised nodes are running sums over its own shape becomes
   one kernel over groups of 8 series (two vectors of four):
   - A: the running sums, as vector registers, and the density's argument,
     into an L1 scratch;
   - B: `exp` over the scratch, in a loop of its own so its constants stay in
     registers;
   - C and R together, from the last time backwards: the density, its
     derivatives, and the reverse running sums of the adjoints.

   The compiler emits the `<4 x double>` IR itself; LLVM's loop vectoriser
   cannot find this form (vector lanes across rows, the loop over time).
   Leftover series run through the same generator with one lane.
3. **Statement absorption and gradient ownership** (part of scan fusion). An
   element-wise statement over the same shape (`innov ~ Normal(0, 0.08)`) runs
   inside the kernel's reverse loop instead of making a pass of its own. When
   every gradient contribution to a matrix parameter happens there, its
   gradient is summed in a register and stored once, and is not zeroed first.
   Only the gradients that are accumulated are zeroed now.
4. **Mint's own `exp`** (`--no-inline-exp`). In the vector code Mint emits,
   `exp` is a table-driven function in the IR: 2^(j/256) from a 256-entry table
   (one AVX2 gather), a 3-FMA polynomial, and a branch to a full-range version
   for |x| > 708 or NaN. Worst error found against a long double reference,
   3e7 inputs: 2 ulp; special values match libm. `tests/run.sh` repeats the
   check on the emitted IR (3e6 inputs and the special values). In isolation it takes
   0.57 ns per value against glibc's vector `exp` at 0.86 ns. It is used only
   where Mint emits the vector code: in loops LLVM vectorises, its branch
   would stop the vectoriser (that regression was caught and fixed during the
   round; see below).
5. **Row fusion** (`--no-row-fusion`, and off under `--strict-fp`).
   Consecutive statements that stream the rows of one matrix, one producing a
   value per row from `X * w` and others consuming it through `X' * (...)` and
   `X' * diag(...) * X`, run as one loop over chunks of 32 rows. Each chunk of
   X is read from memory once. Newton's three passes over 80 MB become one.
6. **Tiled Gram kernel** (`--no-gram-blocking` restores the old kernel). Each
   chunk of rows is copied, with its weighted copy, into L1 scratch padded to
   a multiple of 8 columns, and the upper triangle of H is updated one
   4 x 8 tile at a time, held in eight vector registers. That is six loads
   per eight vector FMAs; the row-by-row kernel needed a load and a store of
   H for every four.
7. **`mint_alloc` is declared `noalias` and returns 64-byte aligned memory.**
   Without the declaration LLVM could not tell that a fresh buffer does not
   overlap the input, and guarded short inner loops with run-time overlap
   checks. This alone took Newton from 0.325 to 0.265 s.

## Tried and not kept

- Blocking the scan's rows by 16 in scalar code, with LLVM vectorising the
  inner loop: slower (inner loops too short).
- Mint's `exp` as a Taylor polynomial (degree 13, Estrin): 0.96 ns per value,
  slower than glibc's.
- Letting LLVM vectorise loops that call Mint's scalar `exp`: its
  out-of-range branch blocks the vectoriser, and the logistic gradient went
  from 33 to 62 µs. Scalar code now keeps `llvm.exp` (glibc's vector version).
- In row fusion, computing the chunk's dot products in a separate blocked
  pass and vectorising the per-row values: 0.172 s against 0.163 s for
  per-row dot products.
- Software prefetch of the next chunk of X: slower (0.18 s).
- Scan groups of 4 or 12 series instead of 8: slower on at least one size.

## What the independent review found

A second model reviewed the round adversarially (read-only). Its four compiler
defects were all real and are fixed, each with a regression test that fails
on the unfixed code:

- nested running sums with a nonlinear outer sum read an unfilled buffer in
  the reverse pass (wrong gradient);
- a column-indexed parameter also used by a statement over another shape lost
  that statement's gradient (its partial-sum buffer was visible to the whole
  function, not only to the kernel);
- row fusion accepted `X' * v + M` where M broadcasts (wrong result) or is a
  scalar (compiler crash);
- a running sum of data only crashed the compiler.

It also found that "level with the Rust" overstated the dynamic Poisson
result, that the logistic "it was a tie" did not match this measurement, that
the benchmark ran programs in a fixed order (now shuffled), and that the test
suite did not exercise several of these paths or Mint's `exp` as emitted.
Those are corrected above and in `tests/run.sh`.

## Caveats

- **The Rust baselines were not changed.** Everything above could be written
  by hand in Rust: the column-major layout (which the Rust cannot choose,
  because it shares Mint's parameter vector), a table-driven `exp`, a tiled
  Gram kernel, or a call to a BLAS `dsyrk`. The comparison is with the Rust
  as written in the previous round, which already used intrinsics and glibc's
  vector math. The claim is that Mint's compiler produces this from the
  18-line and 15-line programs; not that Rust cannot.
- One machine (Ryzen 9 5900X, AVX2, no AVX-512), one data set per problem.
- The dynamic Poisson large model's whole runs are dominated by the sampler,
  not the gradient: see `hierarchical.md`.
- During the round, rustc crashed three times while building `mintc` (two
  segmentation faults in LLVM, one "broken MIR" internal error), each on code
  that built cleanly before and after. That pattern can indicate unstable
  hardware; no machine-check errors were visible to this user. Every number
  above comes from binaries that were rebuilt and passed the test suite.

## What is left

- The dynamic Poisson gradient is 2 to 4% slower than the Rust's. The kernel reads `innov` twice (in passes A and R) and each
  group's 8 series at one time straddle cache lines, because `innov` sits at
  an arbitrary offset in the sampler's vector with a stride of G. Padding
  would change the sampler's dimension. The Rust has the same second read.
- The Gram tile kernel is about two thirds of Newton's time and still misses
  L1 about four times as often as the Rust (the padded H and the two chunk
  copies do not all fit in 32 KB).
- At 37,901 parameters, a whole run spends most of its time in the sampler,
  which this round did not touch.

## Follow-up: the fission kernel (logistic gradient)

The logistic gradient's three fission passes (dot products, elementwise
density, gradient updates) became one loop over chunks of 32 rows in Mint's
own vector code, with Mint's `exp` and a new Mint `log` inline (see
"Loop fission" in [architecture.md](architecture.md)). Switches:
`--no-fission-kernel` (the three passes as before) and `--no-inline-log`.

Measurement: `MINT_BENCH_GRAD=20000`, pinned to core 8, base build (commit
41f69c5), this build and the max-effort Rust interleaved, 7 repetitions per
set, four sets. Other jobs were running throughout (load average 7 to 22),
and the sets mixed quiet runs with runs about 1.6x slower in every binary,
so the table gives the fastest run and the range of the quiet runs; the
medians are mostly contention.

| | base | this build | max-effort Rust |
|---|---|---|---|
| gradient, fastest of 28 | 33.4 µs | **23.7 µs** | 38.5 µs |
| gradient, quiet runs | 33.4 to 39.0 µs | 23.7 to 27.6 µs | 38.5 to 40.3 µs |
| cycles per gradient, fastest | 153,000 | **109,000** | 176,000 |
| instructions per gradient | 424,000 | **277,000** | 496,000 |
| whole run (1 chain), fastest of 14 | 0.526 s | **0.379 s** | 0.622 s |

In every one of the 28 gradient repetitions this build was faster than the
base build and the Rust run next to it; in whole runs it was faster than the
base in 12 of 14 repetitions and than the Rust in 13 of 14 (the exceptions
coincided with contention). Whole runs are not like for like: the gradients differ in the
last bits, so the chains diverge and take different numbers of gradients
(14,565 and 14,136 at seed 7). The log density and gradient agree with the
max-effort Rust to 3.7e-15 of the largest component, as before.

What did not help (each measured, then removed):

- software prefetch of the next chunk of X, in one burst or spread over the
  elementwise loop, into L1 or L2: up to 5% slower;
- computing the next chunk's dot products inside this chunk's exp loop
  (software pipelining): no faster, 8.5% more instructions;
- fewer elementwise loops: exp and log1p in one loop was 16% slower, and
  everything in one loop 12% slower (each iteration becomes one long
  dependency chain);
- chunks of 16, 64 or 128 rows; eight rows per group in the dot products
  and updates: no faster;
- a general `log` that takes 1/c from the CPU's reciprocal estimate (one
  gather instead of two, as glibc does): slower in isolation (1.56 against
  1.36 ns per value) and less accurate (2.5 ulp).

Mint's general `log` is still slower than glibc's vector `log` in isolation
(about 1.25 to 1.35 against 1.1 ns per value). Inside the kernel, where
glibc's calls spill every vector register, the two measured the same on a
Normal model with an indexed scale and on a model with `log(u)`. The
logistic model does not use it: BernoulliLogit takes the specialised log1p.

Unchanged: linear regression, eight schools, the dynamic Poisson model and
Newton compile to the same machine code as before (their model and main
functions disassemble identically), so they cannot have regressed.
