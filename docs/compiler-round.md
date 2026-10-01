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

## Follow-up: fusing the leapfrog into the gradient, and the blocked scan layout

The goal was faster whole runs of the large dynamic Poisson model (37,901
parameters, 3 threads per chain), where the sampler's passes over the
D-length state vectors cost more than the parallel gradient kernel, by
letting the generated gradient do the leapfrog's elementwise updates. The
fused leapfrog was built five ways (listed under "What did not help") and
none was faster than the runtime's own pass. What made the runs faster is a
change to the scan layout that the later attempts needed: the scanned
matrix is now stored in blocks of eight series (a kernel group), each block
column by column, instead of in whole columns of G (see "Scan layout" in
[architecture.md](architecture.md)). The same change moved the parallel
kernel's single vectors and leftover rows into its thread 0, inside the
parallel region; only the one-thread gradient timings below separate the
layout from that. The fused leapfrog stays in the compiler and runtime,
tested, but is not built by default (`mintc --fused-leapfrog`).

### Whole runs

The final build against the base build (commit d51f859), interleaved, three
rounds, `MINT_THREADS_PER_CHAIN=3` for the large model (its default) and 1
for the small one (its default). Each run waited for the one-minute load
average to drop below 5 (it was 1.9 to 5.0 at the start of every run;
other jobs were running throughout). The binaries were built at commit
3fccc8f, where every build carried the fused leapfrog and the runtime used
it only with `MINT_FUSED_LEAPFROG=1` (the fused column); the current
default build no longer emits it, and its log density, kernels and sampler
code are the same as in those binaries.

| run | base | final (default) | final, fused leapfrog |
|---|---|---|---|
| large, 150 + 150 draws | 12.29 / 13.36 / 13.34 s | 11.47 / 10.83 / 11.81 s | 11.82 / 11.38 / 11.99 s |
| large, 1000 + 1000 draws | 70.03 / 68.85 / 71.61 s | 64.10 / 61.47 / 64.06 s | 63.90 / 61.75 / 64.43 s |
| small, 1000 + 1000 draws | 3.62 / 3.65 / 3.58 s | 3.48 / 3.43 / 3.48 s | 3.59 / 3.52 / 3.56 s |

Per gradient including the sampler (the chains differ in their numbers of
gradients, so this is the fairer column): large short run 18.9 to 20.6 µs
(base) against 16.8 to 18.3 µs; full run 18.6 to 19.4 against 16.5 to
17.2 µs; small 1.79 to 1.83 against 1.70 to 1.73 µs. Pair by pair, the
large model's runs took 7 to 19% less time (short) and 8 to 11% less
(full), and the small model's 3 to 6% less. Every final run was faster than every base run
of the same kind.

Two earlier sets with the layout but before the fused leapfrog was turned
off (load 3.7 to 6.0 at each start) gave the same picture: short runs
13.36 to 14.28 s (base, six runs) against 11.71 to 12.07 s (new layout,
unfused, six runs) and 11.77 to 12.31 s (fused, six runs); full runs 71.00
to 72.55 s (base, four runs) against 65.21 and 66.08 s (unfused, two runs)
and 66.51 to 67.64 s (fused, four runs). Over these sets and the table's,
the fused leapfrog was slower than the unfused path with the same layout in
12 of the 14 interleaved pairs on the large model, and faster in 2 (by 1.6%
and 0.3%). On the small model, where it runs serially, it was slower in all
8 pairs (by 1 to 7%).

The gradient alone (`MINT_BENCH_GRAD`, load 2.8 to 5.5): large model 50 to
53 µs (base) against 43 to 45 µs on one thread, 24 to 29 against 17 to
24 µs on three; small model 3.8 to 4.6 against 3.6 to 3.7 µs. The blocks
make each group's reads and writes one contiguous stream instead of 8
values every G (2,000 bytes apart on the large model).

Earlier measurements, taken while the machine was overloaded (load up to
47, every core busy; the same short run took 14 to 202 s), were
discarded.

### What did not help

Each of these was correct (the state after a leapfrog bit-identical to the
runtime's, checked as `MINT_LEAP_TEST` does, though with a looser check on
the sums than now: relative to the largest sum, which on the large model's
test state was dominated by a kinetic energy near 6e20) and was measured
with the same steady-state leaf benchmark (`MINT_LEAP_TEST=K`: blocks of 20
leaves of each kind alternating, fastest block reported, large model). The
items were measured hours apart at different loads, so compare the fused
and unfused figures within an item, not across items (the unfused leaf
took 42 µs in the first and 30 µs in the second):

- **Element-wise fusion in the column-major layout.** The kernel's forward
  pass took the first half-step where it first reads an element (reading
  the old position, momentum, gradient and inverse metric, writing the new
  position and half-step momentum), and its reverse pass the second
  half-step and kinetic energy where it stores the gradient. 101 against
  77 µs per leaf on one thread, 57 against 42 µs on three, 7.5 against
  5.5 µs on the small model. The forward pass took five times as many
  cycles as before: four more input streams and two output streams, each
  touched 8 values at a time with a stride of 2,000 bytes, which the
  hardware prefetchers do not follow. Software prefetch 2 to 16 columns
  ahead and staggering the arrays' page offsets did not close the gap.
- **The runtime's leaf work in each kernel thread on its own rows, in the
  column-major layout.** A thread's rows are then 150 runs of about 83
  values, one per column: 45 against 30 µs per leaf. Without the leaf work
  the kernel took 21 µs; with the runs trimmed to whole cache lines (wrong
  results, timing only) 35 µs, which points to false sharing of the lines
  at the threads' boundaries for about 10 µs, and to the short runs for
  the rest.
- **The leaf work on contiguous shares, after a barrier in the kernel's
  parallel region** (column-major layout): 30.5 against 30.2 µs. Without
  the barrier (racy, timing only) it was 27.3 against 29.5 µs; the barrier
  cost the gain.
- **With the blocked layout, the leaf work in each kernel thread on its
  own contiguous rows, without a barrier** (what a `--fused-leapfrog`
  build runs now): 23.3 against 22.8 µs, and the whole runs above. With the
  blocks, the runtime's own pass, split by index ranges, already gives each
  thread nearly the same elements as its kernel groups (the ranges differ
  by 231 and 863 elements at the two boundaries, out of 37,901), which is
  our reading of why the fusion gains nothing: it saves a parallel region,
  and its hooks leave the rest of theta to one thread.
- **With the blocked layout, element-wise fusion in the reverse pass**
  (second half-step, kinetic energy and the next leaf's first half-step
  where the kernel stores the gradient, now in contiguous streams; the
  merges still in the hooks): 24.1 to 25.0 against 22.4 to 22.7 µs on three
  threads, 63 against 58 µs on one, 4.8 against 4.3 µs on the small model.
  Removed.

The task's premise, that the sampler's passes are most of the time per
gradient and memory-bound, held only partly in these measurements: with
one chain the leaf pass took about 5 to 8 µs of 23 to 30 µs per leaf (the
leaf time less the kernel's time alone, in the same conditions). Fusing it
cannot remove the state's own reads and writes; at best it saves rereading
the gradient and position from beyond L2 and a parallel region, and every
way tried of moving the pass into the kernel cost about as much as the
pass.

### Not done

- The README's gradient table (4.22 / 53.4 µs against the max-effort
  Rust's 4.07 / 52.2 µs) predates the blocked layout and was not re-run
  with `bench/compiler_bench.py`; the measurements above suggest Mint's
  large-model gradient is now faster than that Rust figure, but they were
  not taken under the benchmark's conditions.
- The leaf benchmark covers steady-state leaves only (no merges, eps = 0);
  the whole runs are the measurement that counts.

## Follow-up: narrow data

When `sample()` starts, the generated code now checks the data that the
model's vector kernels read and, where every value is exactly an int8, int16
or float, has the kernels read a copy in that type (see "Narrow data" in
[architecture.md](architecture.md#narrow-data)). Switch: `--no-narrow-data`;
at run time `MINT_NARROW=0` picks the double variant of the same binary.
In the benchmark data only the outcomes narrow: the time-series counts
(0 to 76) and the logistic 0/1 outcomes to int8. The logistic X and Newton's
X are not exact in float (0 of 10^5 and 0 of 10^7 values), and Newton's
method is a function, which has no such step.

The same change starts the vector kernels' register sums of adjoints at -0.0
instead of 0.0, so LLVM drops their first add (`--no-negzero-sums` turns it
off). That applies to every build ("no narrow" below includes it, "base"
does not), and through FMA contraction it can change results in the last
bits; on the benchmark models it does not.

Measurement: `MINT_BENCH_GRAD`, one process pinned to one core (core 10,
chosen when it and its SMT sibling were idle; core 12's sibling was busy
with other jobs), all variants interleaved in an order shuffled per round,
11 rounds; load average 3 to 9 from other jobs. Base is commit d51f859.
Medians, with the range:

| gradient, µs | base | no narrow | narrow (int8) | int16 copy | float copy |
|---|---|---|---|---|---|
| time series, 3,171 parameters | 4.19 (4.18 to 4.22) | 4.14 (4.13 to 4.16) | **3.96** (3.94 to 4.14) | 3.99 | 4.04 |
| time series, 37,901 parameters | 53.95 (53.7 to 57.4) | 52.90 (52.3 to 53.3) | **50.79** (50.4 to 55.0) | 51.79 | 51.63 |
| logistic (n=5000, p=20) | 24.38 (23.8 to 25.1) | 23.96 (23.4 to 25.6) | 24.05 (24.0 to 27.3) | | |
| logistic, X rounded to float (not the benchmark data) | 24.69 (24.2 to 27.9) | 23.89 (23.6 to 27.3) | **22.67** (22.0 to 24.8) | | |

The int16 and float columns are the same binary with `MINT_NARROW=int16`
or `float`; with `MINT_NARROW=0` it took 4.17 and 52.79 µs, the same as the
no-narrow build.

The two parts separately, after the second review (core 6, 11 rounds, load
average 6 to 12; "neither" is `--no-narrow-data --no-negzero-sums`, whose
IR equals the base's):

| gradient, µs | base | neither | narrow only | -0.0 sums only | both (default) |
|---|---|---|---|---|---|
| time series, 3,171 parameters | 4.07 | 4.07 | 4.03 | 4.02 | **3.86** |
| time series, 37,901 parameters | 52.49 | 52.28 | 51.54 | 51.34 | **49.01** |

Each alone is worth 1 to 2%; together 5 to 7%. Part of what the narrow
copy gains needs the -0.0 start (see the first item under "What did not
help" below). On the logistic model y
is 40 KB of the 840 KB read per gradient, and narrowing it changed nothing
measurable; with X exact in float (an artificial case, made by rounding the
benchmark X) the float copy took 5% off.

Four copies at once, one per core on cores 6 to 9 (one L3), 7 rounds, mean
over the four: the large time-series gradient took 53.9 (base), 52.7 (no
narrow) and 50.0 µs (narrow), and the float-X logistic gradient 24.4 (base)
and 22.4 µs (narrow). The relative gain is the same as with one copy, so
at four processes these kernels were not limited by shared memory
bandwidth; the copies help through the per-core caches.

Whole sampling runs, 4 chains, interleaved, sampling time, medians of 8:

| | base | narrow | no narrow (`--no-narrow-data`) |
|---|---|---|---|
| small time series, 1000 + 1000, set 1 | 3.62 s (3.59 to 3.69) | **3.50 s** (3.44 to 3.55) | 3.62 s (3.59 to 3.68) |
| small time series, 1000 + 1000, set 2 | 3.65 s (3.61 to 3.70) | **3.49 s** (3.46 to 3.52) | 3.64 s (3.61 to 3.66) |
| large time series, 150 + 150 (6 rounds each) | 12.7 s / 14.1 s | 13.0 s / 13.9 s | 12.8 s / 14.7 s (`MINT_NARROW=0`) |

In both sets of small runs every narrow run was faster than every base run.
The large runs show no difference above the noise (two sets, other jobs
running, ranges of 1 to 4 s); they spend most of their time in the sampler
(about 78 µs per gradient per chain, of which the kernel is about 21 on
three threads), so a gain of 1 to 3 µs would be 1 to 4%. All runs of a model
produced the same raw draws (MINT_DRAWS) in all three builds.

Build time, medians of 7: the time-series model 0.22 s before, 0.56 s now;
logistic 0.14 s before, 0.37 s now (four variants of the model code each).
Models without vector kernels build as before.

What did not help or was not kept:

- The first version let LLVM see the conversion. It was not exact: the
  independent review found a model and float data where a gradient
  component differed in the last bit, because the backend fused a
  different multiply into an add. The converted value now passes through
  an empty inline asm (`llvm.arithmetic.fence` instead did not prevent the
  difference). Hiding the conversion cost speed until the -0.0 sums came
  in: on the small model (core 12, 9 rounds) base took 4.15 µs, the
  narrow build with the asm 4.10, and with the asm and the -0.0 sums 3.90,
  against 3.88 for the unprotected version (with 0.0 sums). Part of what
  the unprotected version had gained was LLVM dropping a 0.0 + x it could prove exact
  (x = count - exp(eta) is never -0.0 when the count is an integer); the
  -0.0 start removes that add in every build.
- int16 and float copies of the counts: both slower than int8 (table).
- Up to 8 or 16 variants (an int8, int16 and float choice for every
  buffer): the logistic model took 0.73 s to build with 8, for no gain on
  this data. The limit is 4.
- An earlier whole-run measurement of the large model under heavy load
  (load average up to 27) ranged from 12 to 196 s per run and is not
  reported.
