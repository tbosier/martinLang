# Runtime and compiler options

## Environment variables for compiled programs

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
- `MINT_FUSED_LEAPFROG=0` turns off the fused leapfrog in a program built
  with `mintc --fused-leapfrog` (the sampler's leaf work done by the fused
  scan kernel's threads; not built by default, because it measured no
  faster than the runtime's own pass, see
  [compiler-round.md](compiler-round.md)). `=exact` keeps the
  runtime's summation order, which gave exactly the unfused draws in the
  tests. `MINT_LEAP_TEST=1` checks one fused leaf against the runtime's own
  and exits; `=K` also times K leaves of each.
- `MINT_DRAWS=FILE` writes every draw of every parameter, in the
  parameters' own (constrained) scale: three little-endian u64 values
  (chains, draws per chain, parameters), then chains x draws x parameters
  f64. To a regular file the chains write their draws as they produce
  them, into `FILE.partial`, which becomes FILE when sampling has finished
  (a run that dies leaves FILE as it was). A pipe or terminal cannot be
  written out of order, so then every draw is kept in memory and written
  at the end.
- `MINT_KEEP_DRAWS` chooses which draws stay in memory. By default only
  those of the rows `print` shows (every entry of a parameter with at most
  12 entries, the first 3 of a larger one), which it needs for the
  quantiles; every other parameter is summarised as it is drawn, so memory
  no longer grows with draws times parameters (see the Runtime section of
  [architecture.md](architecture.md)). `=beta,sigma` also keeps every
  draw of the named parameters; `=all` keeps everything and computes the
  summary from the draws alone, as before.
- `MINT_STATS_DUMP=FILE` writes, when the posterior is printed, every
  parameter's summary statistics and their streaming versions side by side
  (with `MINT_KEEP_DRAWS=all`, to compare the two on the same draws).
  `MINT_ESS_LAGS` (default 32) and `MINT_ESS_BATCHES` (default 128) size
  the streaming ESS; per parameter and chain they cost 1.25 doubles
  per lag and half a double per batch.
- `MINT_METRIC=grad` switches to the experimental gradient-based metric
  adaptation.
- `MINT_METRIC=lowrank` adds to Stan's diagonal metric up to 8, 16 or 24
  directions, depending on the model's size, the threads per chain and the
  L2 cache size (`MINT_LOWRANK_K` sets another
  number), estimated from the gradients of the warmup draws; see
  [hierarchical.md](hierarchical.md) for what it gains and costs. It
  works with the fused leapfrog (the low-rank part of each leaf runs in a
  pass after the kernel) and with `MINT_WARMUP=fast`, under which the
  chains pool their window draws for the directions as they do for the
  diagonal.
- `MINT_NARROW=0` keeps a model's kernels on the double data (see below);
  `MINT_NARROW_REPORT=1` prints which narrow copy each data buffer got.
- `MINT_WARMUP=fast` replaces Stan's warmup with a shorter one: each chain
  starts from a Pathfinder-style L-BFGS point, warmup runs max(200, warmup / 5)
  iterations (never more than the program's warmup), and the chains pool
  their draws for each metric window. On the
  examples it took 1.4 to 1.7x fewer gradients per effective draw (see the
  Runtime section of [architecture.md](architecture.md)). Stan's warmup
  stays the default.

## Compiler switches

Each turns one optimisation off, for measuring it:
`--no-suffstats`, `--no-fission`, `--no-vecmath`, `--no-gram-blocking`,
`--no-scan-layout`, `--no-scan-fusion`, `--no-inline-exp`, `--no-row-fusion`,
`--no-fission-kernel`, `--no-inline-log`, `--no-parallel-kernel`,
`--no-narrow-data`, `--no-negzero-sums`, `--no-collapse` (sample a latent
random walk with NUTS instead of integrating it out), and `--strict-fp` (strict IEEE
evaluation order, no vector math). `--fused-leapfrog` turns one on (see
`MINT_FUSED_LEAPFROG` above).

## Narrow data

 When `sample()` starts, the generated code checks the data
vectors and matrices that the model's vector kernels read (up to a limit of
four variants of the model code), and where every value is exactly an int8,
int16 or float, the kernels read a copy in that type and convert in
registers. In every test the log density, gradient and draws are
byte-identical with and without the copies. In the benchmark data the
time-series counts and the logistic 0/1 outcomes narrow to int8: the
time-series gradient became 5 to 7% faster and a whole run of the small
model 3 to 4% faster, together with a change made at the same time (adjoint
sums that start at -0.0, `--no-negzero-sums`; each alone gives 1 to 2%).
The logistic gradient did not change measurably, because its real-valued X
is not exact in float and stays double. It costs build time:
those two models now take 0.56 and 0.37 s to build instead of 0.22 and
0.14 s. See [architecture.md](architecture.md#narrow-data).
