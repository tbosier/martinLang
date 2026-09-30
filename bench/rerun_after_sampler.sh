#!/usr/bin/env bash
# Reruns everything that uses the Mint runtime after the sampler rewrite
# and records the sampler settings in each result. SKIP_BENCH=1 skips bench.py.
set -euo pipefail
cd "$(dirname "$0")/.."
P1=/home/taylo/projects/testRandomShiz/rustmc_demo/.venv/bin/python
P2=/home/taylo/projects/rustmc/.venv/bin/python
export XDG_CACHE_HOME="$PWD/build/cache"
step() { echo "=== $(date +%T) $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
clang -O3 -march=native -fopenmp -c runtime/mint_rt.c -o build/mint_rt.o
rustc +nightly --edition 2021 -C opt-level=3 -C target-cpu=native baselines/dynpois_max.rs -o build/rs_dynpois_max -C link-arg=$PWD/build/mint_rt.o -C link-arg=-lomp -l m -l mvec
cd bench/dynpois
step mint small;     $P1 run_mint.py small 1000 1000 | tail -1
step mint large;     $P1 run_mint.py large 1000 1000 | tail -1
step rust_max small; $P1 run_mint.py small --rust --variant rust_max | tail -1
step rust_max large; $P1 run_mint.py large --rust --variant rust_max | tail -1
step analyze;        $P2 analyze.py > results/analysis.md
cd ../..
if [ "${SKIP_BENCH:-0}" != 1 ]; then
  step bench.py; python3 bench/bench.py 7 > build/bench_out.md 2> build/bench_err.log
fi
step done
