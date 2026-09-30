#!/usr/bin/env bash
# End-to-end tests: compile errors, runtime shape/domain/SPD checks, gradient
# checks, agreement with the Rust baselines, and a known posterior.
# Run from anywhere after bench/setup.sh has written data/. Builds everything
# it runs; any build failure is a test failure.
set -u
cd "$(dirname "$0")/.."
M=./compiler/target/release/mintc
fail=0
pass() { echo "PASS  $1"; }
bad()  { echo "FAIL  $1"; fail=1; }

(cd compiler && cargo build --release -q) || { echo "FAIL  compiler build"; exit 1; }
mkdir -p build
clang -O3 -march=native -fopenmp -c runtime/mint_rt.c -o build/mint_rt.o || { echo "FAIL  runtime build"; exit 1; }

# build SRC OUT [flags...]: removes OUT first so a failed build cannot leave a stale binary
build() {
  local src=$1 out=$2; shift 2
  rm -f "build/$out"
  $M build "$src" -o "build/$out" "$@" 2>build/$out.log || { bad "build $src $*"; cat build/$out.log; return 1; }
}
build_rust() {
  rm -f "build/rs_$1"
  rustc --edition 2021 -C opt-level=3 -C target-cpu=native "baselines/$1.rs" -o "build/rs_$1" \
    -C link-arg="$PWD/build/mint_rt.o" -C link-arg=-lomp -l m || bad "build baselines/$1.rs"
}
for b in logistic_newton logistic_bayes logistic_bayes_tuned linear_bayes linear_bayes_suffstats; do build_rust $b; done
# max-effort baselines need nightly Rust (AVX2 intrinsics, glibc vector math)
for b in logistic_bayes_max logistic_newton_max; do
  rm -f "build/rs_$b"
  rustc +nightly --edition 2021 -C opt-level=3 -C target-cpu=native "baselines/$b.rs" -o "build/rs_$b" \
    -C link-arg="$PWD/build/mint_rt.o" -C link-arg=-lomp -l m -l mvec || bad "build baselines/$b.rs (nightly)"
done

# ---- compile-time errors

expect_compile_error() { # file substring
  out=$($M check "$1" 2>&1) && { bad "$1 compiled but should not"; return; }
  grep -qF -- "$2" <<<"$out" && pass "$1: $2" || { bad "$1: expected '$2' in:"; echo "$out"; }
}
expect_compile_error examples/errors/shape_mismatch.mint "inner dimensions p and n must be equal"
expect_compile_error examples/errors/not_spd.mint "only known to be PSD"
expect_compile_error examples/errors/real_scale.mint "scale of Normal must be Positive"
expect_compile_error examples/errors/vector_times_vector.mint "is ambiguous"
expect_compile_error examples/errors/wrong_annotation.mint "does not fit the annotation SPD[p]"

# ---- runtime checks

expect_runtime_error() { # name source-text substring
  printf '%s\n' "$2" > "build/$1.mint"
  build "build/$1.mint" "$1" || return
  out=$(./build/$1 2>&1) && { bad "$1 ran but should stop"; echo "$out"; return; }
  grep -qF -- "$3" <<<"$out" && pass "$1: $3" || { bad "$1: expected '$3' in:"; echo "$out"; }
}
expect_runtime_error shape_at_read "$(cat examples/errors/dim_mismatch_runtime.mint)" "expected size 5000, got 50000"
expect_runtime_error domain_at_read 'fn main() {
    let y: Positive[n] = read("data/logit_y.f64")
    print(sum(y))
}' "but the type says every entry is positive"
python3 -c "
import struct
open('build/asym.f64','wb').write(struct.pack('<QQ', 2, 2) + struct.pack('<4d', 1, 100, 0, 1))"
expect_runtime_error assume_spd_not_symmetric 'fn main() {
    let A: Matrix[k, k] = read("build/asym.f64")
    print(solve(assume_spd(A), ones(k)))
}' "matrix is not symmetric"
python3 -c "
import struct
open('build/nan.f64','wb').write(struct.pack('<QQ', 2, 2) + struct.pack('<4d', 1, float('nan'), 0, 1))"
expect_runtime_error assume_spd_nan 'fn main() {
    let A: Matrix[k, k] = read("build/nan.f64")
    print(solve(assume_spd(A), ones(k)))
}' "matrix entry 2 is nan"
expect_runtime_error assume_spd_negative 'fn main() {
    print(solve(assume_spd(-1 * I(2)), [1, 1]))
}' "not numerically positive definite"
expect_runtime_error bernoulli_bad_data 'model B {
    data y: Vector[n]
    param a: Real
    a ~ Normal(0, 1)
    y ~ BernoulliLogit(a)
}
fn main() {
    let y: Vector[m] = read("data/linear_y.f64")
    print(sample(B(y), chains = 1))
}' "BernoulliLogit needs 0 or 1"

# ---- programs that must now work

build_and_expect() { # name source expected-output-substring
  printf '%s\n' "$2" > "build/$1.mint"
  build "build/$1.mint" "$1" || return
  out=$(./build/$1 2>&1) || { bad "$1 failed to run"; echo "$out"; return; }
  grep -qF -- "$3" <<<"$out" && pass "$1" || { bad "$1: expected '$3' in:"; echo "$out"; }
}
build_and_expect identity_product 'fn main() { print(I(2) * I(2)) }' "[[1, 0],"
build_and_expect assume_spd_valid 'fn main() {
    print(solve(assume_spd(2 * I(2)), [1, 1]))
}' "[0.5, 0.5]"

printf '%s\n' 'fn main() {
    let M = ones(2, 3) + [10, 20]
    print(cumsum(M, 3))
}' > build/broadcast_and_cumsum.mint
if build build/broadcast_and_cumsum.mint broadcast_and_cumsum; then
  out=$(./build/broadcast_and_cumsum)
  [ "$out" = "$(printf '[[11, 22, 33],\n [21, 42, 63]]')" ] && pass "broadcast_and_cumsum (exact output)" || { bad "broadcast_and_cumsum"; echo "$out"; }
fi
printf '%s\n' 'model C {
    data y: Vector[n]
    param a: Real
    a ~ Normal(0, 1)
    cumsum(y) ~ BernoulliLogit(a)
}
fn main() {
    let y = [1, 1]
    print(sample(C(y), chains = 1))
}' > build/cumsum_outcome.mint
out=$($M build build/cumsum_outcome.mint -o build/cumsum_outcome 2>&1) && bad "cumsum outcome accepted" || {
  grep -qF "cannot contain cumsum" <<<"$out" && pass "cumsum in a BernoulliLogit outcome is rejected" || { bad "cumsum outcome message"; echo "$out"; }
}
printf '%s\n' 'fn main() { print(ones(2, 2) + [1, 2]) }' > build/ambiguous.mint
expect_compile_error build/ambiguous.mint "is ambiguous: both dimensions are 2"

# ---- dynamic Poisson panel: exact gradient (numpy reference from SPEC.md)
if [ -f bench/dynpois/data_small/y.npy ]; then
  build examples/dynamic_poisson.mint dynpois && \
  MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 ./build/dynpois | python3 bench/dynpois/check_grad.py bench/dynpois/data_small \
    && pass "dynamic Poisson log density and gradient match the exact formula (small)" || bad "dynamic Poisson gradient (small)"
  sed 's#bench/dynpois/data_small/y.f64#bench/dynpois/data_large/y.f64#' examples/dynamic_poisson.mint > build/dynpois_large_check.mint
  build build/dynpois_large_check.mint dynpois_large_check && \
  MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 ./build/dynpois_large_check | python3 bench/dynpois/check_grad.py bench/dynpois/data_large \
    && pass "dynamic Poisson log density and gradient match the exact formula (large)" || bad "dynamic Poisson gradient (large)"
else
  bad "bench/dynpois/data_small missing (run bench/dynpois/make_data.py)"
fi

# ---- gradients against finite differences (a NaN counts as infinite error)

gradcheck() { # binary label
  worst=$(MINT_GRADCHECK=1 MINT_BENCH_GRAD=1 ./build/$1 2>&1 | grep gradcheck | sed 's/.*error=//' | sort -g | tail -1)
  python3 -c "import sys; w=float('${worst:-inf}'); sys.exit(0 if w < 1e-5 else 1)" && pass "gradcheck $2 ($worst)" || bad "gradcheck $2 ($worst)"
}
for e in eight_schools logistic_bayes linear_bayes; do
  build examples/$e.mint $e && gradcheck $e $e
done
printf '%s\n' 'model P {
    data y: Vector[n]
    param a: Real
    a ~ Normal(0, 1)
    y ~ Normal((exp(-740) * a)^0 + a^1 + a^2, 1)
}
fn main() {
    let y = [0.5, 1.5]
    print(sample(P(y), chains = 1))
}' > build/pow_model.mint
build build/pow_model.mint pow_model && gradcheck pow_model "powers ^0 ^1 ^2"
build examples/linear_bayes.mint linear_bayes_nss --no-suffstats
build examples/logistic_bayes.mint lb_nofis --no-fission
build examples/logistic_bayes.mint lb_strict --strict-fp
build examples/logistic_bayes.mint lb_novm --no-vecmath
build examples/logistic_bayes.mint lb_nofis_strict --no-fission --strict-fp
build examples/linear_bayes.mint linear_bayes_nss_strict --no-suffstats --strict-fp

# ---- same log density and every gradient component as the hand-written Rust

grads() { MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 ./build/$1 | sed -n 's/.*logp=\([^ ]*\) .*/\1/p; s/^grad://p' | tr '\n' ' '; }
for pair in "logistic_bayes rs_logistic_bayes" "lb_nofis rs_logistic_bayes" "lb_strict rs_logistic_bayes" \
            "lb_novm rs_logistic_bayes" "lb_nofis_strict rs_logistic_bayes" \
            "logistic_bayes rs_logistic_bayes_tuned" "logistic_bayes rs_logistic_bayes_max" \
            "linear_bayes rs_linear_bayes" "linear_bayes_nss rs_linear_bayes" \
            "linear_bayes_nss_strict rs_linear_bayes" "linear_bayes rs_linear_bayes_suffstats"; do
  set -- $pair
  a=$(grads $1); b=$(grads $2)
  python3 - "$a" "$b" <<'PY' && pass "logp and all gradient components: $1 == $2" || bad "logp/gradient $1 vs $2"
import sys
a = [float(x) for x in sys.argv[1].split()]
b = [float(x) for x in sys.argv[2].split()]
ok = len(a) == len(b) and len(a) > 2 and all(abs(x - y) <= 1e-9 * max(1.0, abs(y)) for x, y in zip(a, b))
if not ok:
    print(f"      lengths {len(a)} {len(b)}; worst diff {max((abs(x-y) for x,y in zip(a,b)), default=0)}")
sys.exit(0 if ok else 1)
PY
done

# ---- Newton: same coefficients as the Rust baseline

build examples/logistic_newton.mint logistic_newton
build examples/logistic_newton.mint logistic_newton_strict --strict-fp
build examples/logistic_newton.mint logistic_newton_noblock --no-gram-blocking
b=$(./build/rs_logistic_newton | grep '^w' | tr -d 'w[],')
for v in logistic_newton logistic_newton_strict logistic_newton_noblock rs_logistic_newton_max; do
  a=$(./build/$v | grep '^w' | tr -d 'w[],')
  python3 - "$a" "$b" <<'PY' && pass "newton coefficients: $v == rust" || bad "newton mismatch: $v"
import sys
a = [float(x) for x in sys.argv[1].split()]; b = [float(x) for x in sys.argv[2].split()]
sys.exit(0 if len(a) == len(b) == 50 and max(abs(x - y) for x, y in zip(a, b)) < 1e-8 else 1)
PY
done

# ---- eight schools: posterior means of mu and tau against exact grid
# integration (mu 4.4414, tau 3.2904); allow 4 Monte Carlo standard errors.

# The same check on the threaded sampler path and the gradient-based metric.
check_eight() {
  local label=$1; shift
  local out
  out=$(env "$@" ./build/eight_schools 2>/dev/null)
  python3 - "$out" <<'PY' && pass "eight schools posterior ($label)" || bad "eight schools posterior ($label)"
import sys
rows = {l.split()[0]: l.split() for l in sys.argv[1].splitlines() if l and l.split()[0] in ("mu", "tau")}
ok = len(rows) == 2
for name, exact in (("mu", 4.4414), ("tau", 3.2904)):
    if name not in rows:
        continue
    mean, sd, ess = float(rows[name][1]), float(rows[name][2]), float(rows[name][6])
    mcse = sd / ess ** 0.5
    print(f"      {name}: {mean:.3f} vs exact {exact:.3f} (mcse {mcse:.3f})")
    ok &= abs(mean - exact) < 4 * mcse
sys.exit(0 if ok else 1)
PY
}
check_eight serial MINT_THREADS_PER_CHAIN=1
check_eight "3 threads per chain" MINT_THREADS_PER_CHAIN=3
check_eight "gradient metric" MINT_METRIC=grad

# When OpenMP runs a smaller team than requested, the sampler must follow the
# team it got: with a limit of one thread the draws equal the serial ones.
serial=$(MINT_THREADS_PER_CHAIN=1 ./build/eight_schools 2>/dev/null)
limited=$(MINT_THREADS_PER_CHAIN=10 OMP_THREAD_LIMIT=1 ./build/eight_schools 2>/dev/null)
[ -n "$serial" ] && [ "$serial" = "$limited" ] && pass "sampler follows a reduced OpenMP team" \
  || bad "reduced OpenMP team changed the draws"

exit $fail
