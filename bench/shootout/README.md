# Shootout: the large hierarchical dynamic Poisson panel in eight frameworks

One end-to-end comparison on the hardest model in this repository: the
hierarchical dynamic Poisson panel of [bench/dynpois/SPEC.md](../dynpois/SPEC.md)
on `bench/dynpois/data_large` (G = 250 series, T = 150 times, 37,901 latent
parameters, scales fixed as in the spec). Every run's numbers are in
[results.md](results.md) and [results/results.json](results/results.json).

## What came out

A run counts only if it finished within 20 minutes of total wall time and
mixed (max R-hat ≤ 1.01, lowest bulk and tail ESS ≥ 400 over pop, every beta
and every terminal state).

- **Ranked:** Martin with default settings (3 of 3 seeds), Martin with the
  opt-in sampler options (3 of 3) and the hand-tuned Rust gradient under
  Martin's sampler (2 of 3; seed 1 missed with R-hat 1.017). Nothing else
  met both conditions at any seed.
- On the quiet seed-1 pass, Martin took 82 s in total (5.3 effective draws
  per second of sampling) and the opt-in options 42 s (10.6). The opt-in
  fast warmup runs only 200 of the 1000 warmup iterations the program asks
  for, which is part of why it is faster. The seed-2 and seed-3 runs of
  these three configurations ran while other work kept 2.6 to 4.7 logical
  CPUs busy, and their times are about 25 to 65% longer; see the caveats.
- **Did not mix** (with the fixed 1000 + 1000 iterations; the runs ended
  well inside 20 minutes, and longer runs were not tried): every completed
  run of a nuts-rs-based configuration (the Rust program with nuts-rs,
  PyMC + nutpie with numba and with JAX, nutpie on the Stan model), with
  lowest bulk ESS of 4 to 159. With the diagonal adaptation they take about
  130 to 150 leapfrog steps per draw where the Stan-style warmup settles on
  511, and some chains adapt badly (nutpie on the Stan model, seed 1: one
  chain at step size 0.011 and 693 steps per draw; with the low-rank options
  some chains run 255 to 1023 steps, and one of nutpie's stalled at step
  size 8e-11). rustmc finishes in 111 s but does not mix (R-hat 1.9 to 2.2),
  as before.
- **Did not finish within 20 minutes:** Stan without threads (68 min;
  lowest bulk ESS 363, so it would not have mixed either), Stan with
  reduce_sum on 3 threads per chain (40 min; it mixed and serves as the
  reference posterior), NumPyro with parallel chains (25 min; it mixed),
  NumPyro vectorized (stopped at 2 h), nutpie's low-rank mass matrix on the
  Stan model (31 min, did not mix), and nutpie's diagonal mass matrix on
  the Stan model at seeds 2 and 3 (stopped at 20 min with no draws, so
  whether they would have mixed is unknown; 19 min at seed 1, where it did
  not mix).
- **The language against the sampler.** The same hand-tuned Rust gradient
  mixed in 2 of 3 seeds under Martin's sampler and in 0 of 3 under nuts-rs
  with its default diagonal adaptation. In those diagonal nuts-rs runs, 75
  to 85% of each chain's time was spent outside the log-density call (seed
  1: 13 to 25 s of 89 to 109 s per chain; with the low-rank adaptation,
  97 to 98%). That outside time is mostly nuts-rs, but it also includes the
  driver's per-call finiteness check of the gradient, the initial point and
  the extraction of terminal states, which were not timed separately.
- **Agreement.** Every run that mixed agrees with the threaded Stan run to
  within 0.10 posterior sd on all 501 posterior means (at most 3.3 Monte
  Carlo standard errors). Runs that did not mix range from 0.05 sd to 2.6
  sd (nutpie low-rank, with a stuck chain).
- **Lines of code:** Martin 18, rustmc 12 (a library call), PyMC 24,
  NumPyro 25, nutpie + Stan 44, CmdStan + Stan 48 (57 with reduce_sum),
  Rust 982 to 1,166 (including a self-test).

## What was compared

| configuration | what runs | settings |
|---|---|---|
| Martin | `examples/dynamic_poisson.mint`, compiled by `mintc`, NUTS in Martin's runtime | runtime defaults: Stan-style windowed warmup, diagonal metric; at this size each chain splits its sampler passes and its gradient across 3 threads |
| Martin, opt-in sampler options | the same program | `MINT_METRIC=lowrank MINT_WARMUP=fast` (a low-rank metric; an L-BFGS start, chains pooling their adaptation, and a shorter warmup: 200 of the 1000 warmup iterations the program asks for, then 1000 draws) |
| Rust end to end, nuts-rs | `rust_nuts/`: the hand-tuned gradient of `baselines/dynpois_par.rs` (AVX2/FMA, Martin's table `exp`, blocks of 8 series) in a Rust program that samples with the nuts-rs crate, version 0.19.0 (pinned as `=0.19.0`; the library nutpie is built on) | `DiagNutsSettings::default()` ("diag", nuts-rs's default) and `LowRankNutsSettings::default()` ("lowrank"), each with `num_tune = 1000`, `num_draws = 1000`, max depth 10; 4 chains on 4 threads, each splitting its gradient across 3 threads with a persistent team of spin-waiting helper threads (`rust_nuts/src/team.rs`; spawning threads per gradient with `std::thread::scope` would cost more than the 23 µs gradient) |
| Rust gradient under Martin's sampler (isolates the language) | `baselines/dynpois_par.rs` linked to Martin's runtime, as in `bench/same_sampler` | Martin's sampler and its defaults: the same NUTS, warmup and threading; only the gradient code differs |
| Stan | CmdStan 2.40 through cmdstanpy 1.3.0, `bench/dynpois/dynpois.stan` | `stanc --O1`, `CXXFLAGS=-march=native` on CmdStan's `-O3`; default NUTS (diag_e, adapt_delta 0.8, max_treedepth 10); 4 chain processes, one thread each |
| Stan, reduce_sum | `models/dynpois_reduce_sum.stan`: the same model with each series' likelihood and innovation prior summed by `reduce_sum` (sliced over `innov`, grainsize 1) | as above, plus `STAN_THREADS` and `threads_per_chain=3`; cmdstanpy then runs the 4 chains in one CmdStan process with a shared pool of 12 threads (it does not pin 3 threads to each chain) |
| nutpie, Stan model | nutpie 0.16.11 `compile_stan_model` on `dynpois.stan` (BridgeStan 2.9.0 built against CmdStan 2.40's Stan) | `--O1`, `CXXFLAGS=-march=native`, `STAN_THREADS` (nutpie requires it); `adaptation="diag"` (the default) and `adaptation="low_rank"` (the low-rank modified mass matrix; `low_rank_modified_mass_matrix=True` is its older name); tune 1000, draws 1000, chains 4, cores 4, everything else default |
| PyMC + nutpie | `models/dynpois_pymc.py` (PyMC 5.28.5, PyTensor 2.38.3), sampled by nutpie 0.16.11 | `compile_pymc_model(backend="numba")` (nutpie's default backend) and `backend="jax", gradient_backend="pytensor"` in float64 (chosen from `verify.py`'s timing, 315 µs against 428 µs for `jax.grad`; that timing did not wait for JAX's asynchronous result on the PyTensor side, and a synchronised re-timing afterwards gave 475 against 445 µs, so the two are about equal and the choice is not shown to be the faster one); `sample(tune=1000, draws=1000, chains=4, cores=4)`, everything else default |
| NumPyro | `models/dynpois_numpyro.py` (NumPyro 0.22.0, JAX 0.11.2 on CPU), float64 | NUTS defaults (max_tree_depth 10, target_accept_prob 0.8, diagonal mass matrix); 4 chains with `chain_method="parallel"` on 4 XLA host devices, and `chain_method="vectorized"` |
| rustmc | rustmc 0.13.0 `BayesianDynamicPoisson` with the spec's fixed scales: block elliptical slice sampling, a different algorithm | its documented example settings (4 chains, 1000 warmup, 1000 draws), thinned by 8 (`draws=125, thin=8`) because its 25-million stored-value cap allows at most 164 kept draws per chain at this size; the work is the same 1000 post-warmup sweeps per chain |

Seeds are each framework's own (1, 2 and 3); random streams are not
comparable across frameworks.

## Rules

1. **Same data, same model.** `verify.py` evaluates every implementation's
   log density and gradient at two points (the runtime's benchmark point and
   a point near the posterior) and compares them with the exact formula of
   the spec (`bench/dynpois/spec_logdensity.py`). Frameworks drop different
   constants, so log densities are compared through the difference between
   the two points. Every implementation agreed to 2e-13 relative or better
   ([results/verify.json](results/verify.json)); the offsets are the
   constants each drops (Stan 95,393.3; PyMC and NumPyro, which keep log(y!)
   and log(2π), 219,732). nutpie's Stan library was checked as a BridgeStan
   library built with nutpie's flags, and PyMC's graph through PyMC's own
   numba- and JAX-compiled log density. rustmc exposes no log density; it is
   checked by its posterior means only, and since it never mixed, its
  disagreement with the reference cannot separate poor mixing from a
  different target. The PyMC and NumPyro models and the
   reduce_sum Stan program were written for this comparison; the Rust
   program's kernel passes the original baseline's self-test (453 cases,
   errors below 4e-13) at 1, 2, 3, 4 and 7 threads.
2. **One run at a time.** `run_all.sh` runs the configurations strictly one
   after another. Before each run `measure.py` waits until the 1-minute load
   average is below 3 (and, from the seed-2 PyMC runs on, until fewer than
   2 logical CPUs are busy in a 5-second sample), and records the load
   average and how busy the CPUs outside the pinned set were during the run.
3. **Equal hardware budget: 12 cores.** Every run is pinned with
   `taskset -c 0-11`, one hardware thread on each of the Ryzen 9 5900X's
   12 physical cores (no SMT siblings): 4 chains x 3 threads, what Martin
   uses at this size. Each framework uses as much of that as its default or
   best setting does; the threads and CPU time it actually used are in
   results.md. A probe of Martin unpinned took 82.7 s of sampling against
   85.9 s pinned (`results/probes/`), so the pinning costs Martin little.
4. **4 chains, 1000 warmup + 1000 draws** for everything except rustmc
   (above) and Martin's opt-in fast warmup, which runs 200 of the 1000
   warmup iterations before its 1000 draws (the runtime's
   `MINT_WARMUP=fast` default; the run logs record it).
5. **A 20-minute limit per run**, total wall time including compilation. A
   run over it is "did not finish within 20 minutes" and is not ranked,
   whatever its diagnostics; its real time and whether it mixed are still
   reported. The limit was chosen after the first results had been seen:
   the seed-1 pass ran under a 2-hour limit, and seeds 2 and 3 ran with the
   20-minute limit enforced, only for the configurations that finished
   within 20 minutes with seed 1 (PyMC + nutpie with JAX was then to be
   skipped at seed 3, but its seed-3 run had already finished and is
   included). results.md also marks which runs finished within 3 minutes,
   for information only.
6. **One analysis for all.** `analyze.py` applies the same ArviZ code as
   `bench/dynpois/analyze.py` to every run's draws of pop, every beta and
   every terminal state (501 quantities): rank-normalised split R-hat, bulk
   and tail ESS. A run that misses R-hat ≤ 1.01 or bulk or tail ESS ≥ 400
   "did not mix" and is not ranked.

## What is measured

| column | definition |
|---|---|
| lines of code | non-blank, non-comment lines of the model and the code needed to load the data and run it (files listed in results.md). Instrumentation (timing, saving draws, computing terminal states for the analysis) is not counted |
| compile s | Martin: `mintc build` (to a native executable). Rust: an incremental release build of the crate (or `rustc` of the single file) with its dependencies already built; a clean build of the nuts-rs program with all dependencies took 65 s. Stan: cmdstanpy's forced compile (stanc and g++, CmdStan's precompiled header and libraries already built). nutpie on Stan: `compile_stan_model`. PyMC: `compile_pymc_model` (graph rewrites and numba or JAX compilation; building the PyMC model is shown separately). NumPyro: the tracing, lowering and XLA compile durations JAX itself reports during `mcmc.run` |
| sampling s | Martin and the Rust under its runtime: the runtime's own timer (warmup + draws). nuts-rs: the program's timer around the 4 chain threads. Stan: the `sample()` call, including CmdStan writing its CSV. nutpie: the `nutpie.sample()` call (including keeping the trace in memory). NumPyro: `mcmc.run` until the draws are ready, minus the compile durations. rustmc: the `fit()` call |
| total s | wall time of the whole process tree from launch to exit: interpreter start-up, imports, data loading, compile, sampling and post-processing (extracting pop, beta and terminal states to a file). A run stopped by the time limit is recorded with the limit plus the 10 s grace period `measure.py` gives it to exit (1210 s, 7210 s) |
| peak memory | the larger of (a) the summed resident set size of every process in the tree, polled every 0.2 s, and (b) the largest single process's exact peak RSS (`ru_maxrss` from `os.wait4`). The summed proportional set size is in results.json |
| CPU s | user + system time of the process tree (`os.wait4` rusage, which includes reaped children such as CmdStan's) |
| ESS/s | lowest bulk ESS over the 501 quantities / sampling seconds, and / total seconds, for ranked runs |
| agreement | largest difference of posterior means from the reference over the 501 quantities, in posterior standard deviations (and in Monte Carlo standard errors when both runs mixed). The reference posterior is the threaded Stan run (`cmdstan_reduce_sum_s1`), the only Stan run that mixed; it took 40 minutes and is used as a reference only, not ranked |
| gradients | Martin: the runtime's count of every gradient, warmup included. Rust, nutpie, PyMC: leapfrog steps over warmup and draws plus one per initial point (the Rust program also counts its log-density calls directly; they are 0.005% more). CmdStan and NumPyro record steps only for kept draws, so their counts cover the sampling phase only (marked) |

## Caveats

- **Configurations not tried.** nutpie's `adaptation="draw_diag"`, more
  tuning for the nuts-rs-based samplers, and longer runs for the
  configurations that ended inside 20 minutes without mixing. The seed-2
  and seed-3 list was fixed by the coordinator of this work after seed 1:
  the configurations that finished within 20 minutes, except the Rust
  program with nuts-rs's low-rank adaptation (14 min, did not mix), which
  was not repeated.
- **Different samplers and adaptation.** Martin and the Rust gradient under
  its runtime share one NUTS implementation (a port of Stan's), its warmup
  and its threading, so that row isolates the gradient code. nuts-rs,
  nutpie (which is nuts-rs), Stan and NumPyro each have their own NUTS and
  warmup: nuts-rs and nutpie adapt the mass matrix from draws and gradients
  with a different step-size schedule; Stan's and NumPyro's warmup is
  windowed. The ranking therefore mixes the cost of a gradient, the
  sampler's overhead and how well each adaptation suits this posterior. The
  nuts-rs-based runs all used 1000 tuning draws as the rules require; with
  more tuning they might mix (not tried). rustmc is a different algorithm.
- **The opt-in options weaken the convergence check.** `MINT_WARMUP=fast`
  starts every chain near the same L-BFGS point, which makes split R-hat a
  weaker test; the options are not Martin's defaults for that reason.
- **Loaded seeds.** Most seed-2 and seed-3 runs ran while other work on the
  machine kept 1.5 to 4.7 logical CPUs busy outside the pinned set (rustmc's
  had 0.3 to 0.5, and its times did not change) (the SMT siblings of
  the pinned cores, so it competes for the same cores); seed 1 had below
  0.9. Their times are inflated (Martin: 82 s at seed 1, 103 and 117 s at
  seeds 2 and 3). Since each seed ran under a different load, the
  differences between seeds mix load with the seed's own variation, and
  the per-configuration medians of time and ESS per second are not clean
  estimates. The
  diagnostics do not depend on load, except that nutpie on the Stan model
  (19 min at seed 1, with 0.5 busy CPUs) ran into the 20-minute limit at
  seeds 2 and 3 with 1.5 and 1.9 busy. The "other work" column in
  results.md shows each run's figure.
- **The Rust program's helper threads spin** while their chain is in the
  sampler, so its CPU time counts them as busy. A probe with one gradient
  thread per chain and no helpers was slower (166 s of sampling against
  118 s; `results/probes/`), so they do not slow the run down overall, but
  their cache and scheduling effects were not isolated.
- **Run-to-run variation.** The same nuts-rs configuration and seed took
  214 s in a first pass and 118 s in the recorded one, both on a quiet
  machine (`results/probes/first_pass/`). Rankings between runs whose ESS
  per second differ by less than about 2x are not findings.
- **The mixing bar is borderline at this size.** Martin's lowest ESS was
  426 to 520 and the Rust under its runtime missed once at R-hat 1.017;
  one more seed could change a verdict.
- **Stan's two runs were sequential.** Their result files were rewritten
  afterwards by `recover_cmdstan.py`, so the files' modification times
  (both 16:52) do not show when they ran. cmdstanpy's own log lines do: the
  plain program compiled at 15:04:01 and its chains finished at 16:11:40;
  the reduce_sum program started compiling at 16:12:03 and finished at
  16:51:44.
- **Two runs were completed after a script bug.** Both Stan runs crashed in
  post-processing after sampling had finished (the script looked for CSV
  columns `beta[1]`; CmdStan writes `beta.1`). `recover_cmdstan.py` read
  their intact CSV files with the fixed code; their compile and sampling
  seconds come from cmdstanpy's log (1 s resolution), and the 9 s of
  post-processing was added to the measured total. NumPyro's parallel run
  was timed by a script that stopped its sampling clock before JAX's
  asynchronous computation finished; its sampling time is reconstructed as
  an upper bound, and its total, CPU and memory are as measured. Neither
  affects a verdict: all three runs are over 20 minutes.
- **Partial gradient counts** for CmdStan and NumPyro (sampling phase only),
  and NumPyro's vectorized run gave no progress report, so how far it got
  in 2 hours is unknown.
- **Compile times are not like for like.** JAX compiles lazily; for PyMC's
  JAX backend part of the compilation falls in sampling time. CmdStan's and
  BridgeStan's Stan libraries and the Rust crate's dependencies were built
  beforehand.
- **Memory includes each framework's default trace handling.** The Rust
  nuts-rs program keeps only what the analysis needs (pop, beta and terminal
  states: 501 values per kept draw, no warmup draws); Martin streams every
  draw to a file; nutpie keeps the whole trace in memory, warmup
  included by default (about 5 GiB here); nuts-rs's low-rank adaptation
  holds draws and gradients for its eigendecomposition (6.3 GiB); CmdStan
  writes 1.75 GB of CSV per run, of which the script reads back only the
  needed columns.
- **Lines of code.** The Stan program computes the terminal states in its
  generated quantities (7 lines); for the other frameworks they are
  computed in the (uncounted) run script. The Rust counts include a scalar
  reference implementation and a self-test (about 270 lines), and the Rust
  nuts-rs program's `main.rs` is counted whole, including its timing,
  draw export and terminal-state code (instrumentation that is not counted
  for the Python frameworks). Martin's count includes its data loading and
  sampling call (`main`).
- **Environment.** Ryzen 9 5900X (12 cores, 2 x 32 MB L3), Linux 6.18,
  rustc 1.89 (nightly for the single-file Rust), clang 22, g++ 16, Python
  3.12. numpy was held at 2.3.5 and numba at 0.62.1: the newest numpy (2.5)
  breaks numba, which PyMC's numba backend and ArviZ import. The machine was
  shared with other work throughout.

## Files

- `build.sh`: builds mintc, the runtime object and the Rust baseline under
  Martin's runtime. `rust_nuts/build.sh` builds the nuts-rs program
  (`rust_nuts/make_kernel.py` regenerates `src/dynpois.rs` from
  `baselines/dynpois_par.rs`).
- `verify.py`: the log density and gradient checks.
- `run_martin.py` (Martin, opt-in, and `--rust`), `run_rust_nuts.py`,
  `run_cmdstan.py`, `run_nutpie_stan.py`, `run_pymc.py`, `run_numpyro.py`,
  `run_rustmc.py`: one per framework; `models/` holds the counted model
  code. `recover_cmdstan.py`: see the caveats.
- `measure.py`: launches one run and measures it from outside.
- `run_all.sh`: every configuration, one at a time.
- `analyze.py`: diagnostics, agreement and the tables in results.md.
- `results/runs/*.json`: every run's measurements; `results/diagnostics.json`:
  every run's per-quantity diagnostics; `results/verify.json`;
  `results/results.json`; `results/probes/`: probes and superseded first-pass
  runs, kept for the record and not analysed. Draws are not committed
  (`build/shootout/draws/`, 16 MB per run).

Python environment: `uv venv .venv -p 3.12` and
`uv pip install numpy==2.3.5 numba==0.62.1 scipy pymc==5.28.5 nutpie==0.16.11 bridgestan==2.9.0 numpyro==0.22.0 jax==0.11.2 cmdstanpy==1.3.0 psutil "arviz<1" pandas`
(with `UV_CACHE_DIR` and `TMPDIR` inside the worktree); rustmc from the
`rustmc_demo` environment (Python 3.14). CmdStan 2.40 was copied into
`build/cmdstan/` from the main checkout's build.
