#!/usr/bin/env bash
# Final comparison, run strictly one job at a time on an otherwise idle machine.
# Every implementation: 4 chains in parallel, 1000 warmup + 1000 draws
# (rustmc: 1000 warmup + 1000 sweeps thinned by 8 on the large size, because of
# its 25M stored-value cap). Stan on the large size takes hours and is run
# separately afterwards by run_stan_large.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
P1=/home/taylo/projects/testRandomShiz/rustmc_demo/.venv/bin/python   # numpy + rustmc
P2=/home/taylo/projects/rustmc/.venv/bin/python                       # cmdstanpy + arviz
export XDG_CACHE_HOME="$HERE/../../build/cache"
step() { echo "=== $(date +%T) $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
cd "$HERE"
step mint small;      $P1 run_mint.py small 1000 1000 | tail -3
step mint large;      $P1 run_mint.py large 1000 1000 | tail -3
step rust_max small;  $P1 run_mint.py small --rust --variant rust_max | tail -3
step rust_max large;  $P1 run_mint.py large --rust --variant rust_max | tail -3
step rustmc small;    $P1 run_rustmc.py small 1000 1000 1 | tail -3
step rustmc large;    $P1 run_rustmc.py large 1000 125 8 | tail -3
step stan small;      $P2 run_stan.py small 1000 1000 | tail -3
step analyze;         $P2 analyze.py > results/analysis.md
step done
