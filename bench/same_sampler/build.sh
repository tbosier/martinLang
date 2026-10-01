#!/usr/bin/env bash
# Builds everything the same-sampler harness runs, into build/same_sampler/:
#   - the runtime (build/mint_rt.o) and mintc;
#   - the Rust baselines that share the runtime's sampler;
#   - BridgeStan 2.9.0's C library and the Stan models, built against the
#     installed CmdStan 2.40 (its Stan, Stan Math and stanc);
#   - bs_driver, which runs a BridgeStan model under mint_sample;
#   - the Stan (JSON) copies of the data.
# Mint programs are built by the run scripts, one per seed (the seed is part
# of a Mint program's source).
#
# usage: bench/same_sampler/build.sh   (Python with numpy in $PY, default .venv/bin/python)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
OUT=$ROOT/build/same_sampler
PY=${PY:-$ROOT/.venv/bin/python}
CMDSTAN=${CMDSTAN:-/home/taylo/projects/testRandomShiz/mint/build/cmdstan/cmdstan-2.40.0}
BS_VERSION=2.9.0
BS=$ROOT/build/bs/bridgestan-$BS_VERSION
mkdir -p "$OUT/stan"

(cd compiler && cargo build --release -q)
clang -O3 -march=native -fopenmp -c runtime/mint_rt.c -o build/mint_rt.o

# Rust baselines (the max-effort ones need nightly Rust, as in bench/bench.py)
RT=(-C "link-arg=$ROOT/build/mint_rt.o" -C link-arg=-lomp -l m)
for b in dynpois_max dynpois_par logistic_bayes_max; do
  rustc +nightly --edition 2021 -C opt-level=3 -C target-cpu=native "baselines/$b.rs" -o "$OUT/rs_$b" "${RT[@]}" -l mvec
done
rustc --edition 2021 -C opt-level=3 -C target-cpu=native baselines/eight_schools.rs -o "$OUT/rs_eight_schools" "${RT[@]}"

# BridgeStan, built against CmdStan's Stan
if [ ! -d "$BS" ]; then
  mkdir -p "$ROOT/build/bs"
  curl -sSL -o "$ROOT/build/bs/bridgestan-$BS_VERSION.tar.gz" \
    "https://github.com/roualdes/bridgestan/releases/download/v$BS_VERSION/bridgestan-$BS_VERSION.tar.gz"
  tar xzf "$ROOT/build/bs/bridgestan-$BS_VERSION.tar.gz" -C "$ROOT/build/bs"
fi
mkdir -p "$BS/make"
# STAN_THREADS: the chains call one model concurrently (see bs_driver.c).
# --O1 is the stanc optimisation level Stan recommends; -march=native as for
# Mint's runtime and the Rust baselines.
cat > "$BS/make/local" <<EOF
CMDSTAN = $CMDSTAN/
STAN = \$(CMDSTAN)stan/
MATH = \$(STAN)lib/stan_math/
STANC = \$(CMDSTAN)bin/stanc
STAN_THREADS = true
CXXFLAGS += -march=native
STANCFLAGS += --O1
EOF
for m in dynpois logistic logistic_glm eight_schools; do
  cp "bench/same_sampler/stan/$m.stan" "$OUT/stan/$m.stan"
  # (compilers have crashed spuriously on this machine; retry twice)
  make -s -C "$BS" "$OUT/stan/${m}_model.so" || make -s -C "$BS" "$OUT/stan/${m}_model.so" \
    || make -s -C "$BS" "$OUT/stan/${m}_model.so"
done

clang -O3 -march=native -fopenmp bench/same_sampler/bs_driver.c build/mint_rt.o -o "$OUT/bs_driver" -ldl -lm -lpthread

"$PY" bench/same_sampler/prepare_data.py
echo "built into $OUT"
