#!/usr/bin/env bash
# Runs every configuration, strictly one at a time, through measure.py
# (which waits for a 1-minute load average below 3 before each run and pins
# every run to the same 12 physical cores, and stops a run after 20 minutes).
# A configuration whose result file already exists with exit status 0, or
# that hit the time limit, is skipped, so the script can be restarted.
# Seeds are given as arguments (default: 1). The seed-1 pass was made with a
# 2-hour limit (the 20-minute rule was set afterwards; see README.md).
#
# usage: bench/shootout/run_all.sh [SEED...]      (after build.sh and verify.py)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python                                              # numpy, nutpie, PyMC, NumPyro, cmdstanpy
PY_RUSTMC=/home/taylo/projects/testRandomShiz/rustmc_demo/.venv/bin/python   # rustmc 0.13.0
M="$PY bench/shootout/measure.py --timeout 1200"   # the 20-minute limit per run (total wall)
S=bench/shootout
seeds=("$@")
[ ${#seeds[@]} -eq 0 ] && seeds=(1)

run() {  # NAME COMMAND...
  local name=$1; shift
  if [ -f "$S/results/runs/$name.json" ] && grep -qE '"exit_status": 0|"timed_out": true' "$S/results/runs/$name.json"; then
    echo "skip $name (done or timed out)"; return
  fi
  echo "=== $(date +%T) $name"
  $M "$name" -- "$@" || echo "!!! $name failed (see build/shootout/logs/$name.log)"
}

for s in "${seeds[@]}"; do
  # every configuration with seed 1; with further seeds only those that finished
  # within the 20-minute limit with seed 1
  run martin_default_s$s        $PY $S/run_martin.py --seed $s
  run martin_optin_s$s          $PY $S/run_martin.py --seed $s --optin
  run rust_under_martin_s$s     $PY $S/run_martin.py --seed $s --rust
  run rust_nuts_diag_s$s        $PY $S/run_rust_nuts.py --seed $s --adapt diag
  [ "$s" = 1 ] && run rust_nuts_lowrank_s$s     $PY $S/run_rust_nuts.py --seed $s --adapt lowrank
  [ "$s" = 1 ] && run cmdstan_plain_s$s         $PY $S/run_cmdstan.py --seed $s
  [ "$s" = 1 ] && run cmdstan_reduce_sum_s$s    $PY $S/run_cmdstan.py --seed $s --reduce-sum
  run nutpie_stan_diag_s$s      $PY $S/run_nutpie_stan.py --seed $s --adapt diag
  [ "$s" = 1 ] && run nutpie_stan_low_rank_s$s  $PY $S/run_nutpie_stan.py --seed $s --adapt low_rank
  run pymc_numba_s$s            $PY $S/run_pymc.py --seed $s --backend numba
  run pymc_jax_s$s              $PY $S/run_pymc.py --seed $s --backend jax
  [ "$s" = 1 ] && run numpyro_parallel_s$s      $PY $S/run_numpyro.py --seed $s --chain-method parallel
  [ "$s" = 1 ] && run numpyro_vectorized_s$s    $PY $S/run_numpyro.py --seed $s --chain-method vectorized
  run rustmc_s$s                $PY_RUSTMC $S/run_rustmc.py --seed $s
done
echo "=== $(date +%T) all done"
