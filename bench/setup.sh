#!/usr/bin/env bash
# Builds the compiler and the data generator, and writes the datasets the
# examples, tests and benchmark read. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build data
(cd compiler && cargo build --release -q)
rustc -O bench/gen_data.rs -o build/gen_data
./build/gen_data logistic 200000 50 11 data/newton 0.0   # Newton example: n=200000, p=50
./build/gen_data logistic 5000 20 12 data/logit 0.3      # Bayesian logistic: n=5000, p=20
./build/gen_data linear 50000 20 13 data/linear          # Bayesian linear: n=50000, p=20
echo "data written to data/"
