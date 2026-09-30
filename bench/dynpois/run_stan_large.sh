#!/usr/bin/env bash
# Stan on the large size. 300 warmup + 300 draws (1000+1000 would take 3-4 hours),
# pinned to one CCD (CPUs 0-5,12-17: its own 32 MB L3) so other work can use the other CCD.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
P2=/home/taylo/projects/rustmc/.venv/bin/python
export XDG_CACHE_HOME="$HERE/../../build/cache"
cd "$HERE"
echo "=== $(date +%T) stan large (load $(cut -d' ' -f1 /proc/loadavg))"
taskset -c 0-5,12-17 $P2 run_stan.py large 300 300 | tail -3
$P2 analyze.py > results/analysis.md
echo "=== $(date +%T) done"
