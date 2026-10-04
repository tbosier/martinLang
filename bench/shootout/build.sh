#!/usr/bin/env bash
# Builds what the shootout runs that is not built by its run scripts:
#   - mintc and the Martin runtime object (build/mint_rt.o);
#   - the threaded Rust baseline under Martin's runtime (build/shootout/rs_dynpois_par),
#     with the same flags as bench/same_sampler/build.sh.
# The Martin program, the CmdStan executable, the nutpie/BridgeStan library,
# the PyMC/numba functions and the JAX programs are compiled inside their run
# scripts, so that compile time is measured there.
# (rustc and the C++ compilers have crashed spuriously on this machine; retry.)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
mkdir -p build/shootout
retry() { "$@" || "$@" || "$@"; }
(cd compiler && retry cargo build --release -q)
retry clang -O3 -march=native -fopenmp -c runtime/mint_rt.c -o build/mint_rt.o
retry rustc +nightly --edition 2021 -C opt-level=3 -C target-cpu=native baselines/dynpois_par.rs \
  -o build/shootout/rs_dynpois_par -C "link-arg=$ROOT/build/mint_rt.o" -C link-arg=-lomp -l m -l mvec
echo "built mintc, build/mint_rt.o, build/shootout/rs_dynpois_par"
