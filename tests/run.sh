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
  [ -x build/dynpois_large_check ] && MINT_KERNEL_THREADS=3 MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 ./build/dynpois_large_check \
    | python3 bench/dynpois/check_grad.py bench/dynpois/data_large \
    && pass "dynamic Poisson gradient matches the exact formula (large, kernel on 3 threads)" || bad "dynamic Poisson gradient (large, kernel on 3 threads)"
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

# ---- fused scan kernels: every path (nested running sums, the one-lane
# BernoulliLogit path, row/column/scalar parameters, row counts that are not
# multiples of the vector width) must give the same log density and gradient
# as the same model built without the scan layout, scan fusion and inline exp,
# and must pass the finite-difference check.

python3 tests/scan/make_data.py build
for m in nested bernoulli mixed nested_sq colreuse datascan twohosts; do
  for G in 7 13 20; do
    sed "s/NG/$G/" tests/scan/$m.mint > build/scan_$m.$G.mint
    build build/scan_$m.$G.mint scan_${m}_${G}_opt || continue
    build build/scan_$m.$G.mint scan_${m}_${G}_ref --no-scan-layout --no-scan-fusion --no-inline-exp || continue
    # the column-major layout without the fused kernel (the path a scan
    # statement takes when it cannot be fused)
    build build/scan_$m.$G.mint scan_${m}_${G}_lay --no-scan-fusion || continue
    for v in opt ref lay; do MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 ./build/scan_${m}_${G}_$v > build/scan_${m}_${G}_$v.out; done
    python3 - build/scan_${m}_${G}_lay.out build/scan_${m}_${G}_ref.out <<'PY' && pass "scan layout $m G=$G matches the row-major build" || bad "scan layout $m G=$G differs from the row-major build"
import sys
def rd(p):
    o = open(p).read()
    return float(o.split("logp=")[1].split()[0]), [float(x) for x in o.split("grad:")[1].split()]
(la, ga), (lb, gb) = rd(sys.argv[1]), rd(sys.argv[2])
import math
if not all(map(math.isfinite, ga + gb + [la, lb])):
    sys.exit(1)
err = max(abs(a - b) for a, b in zip(ga, gb)) / max(abs(x) for x in gb)
sys.exit(0 if len(ga) == len(gb) and abs(la - lb) <= 1e-12 * abs(lb) and err < 1e-12 else 1)
PY
    python3 - build/scan_${m}_${G}_opt.out build/scan_${m}_${G}_ref.out <<'PY' && pass "scan kernel $m G=$G matches the unfused build" || bad "scan kernel $m G=$G differs from the unfused build"
import sys
def rd(p):
    o = open(p).read()
    return float(o.split("logp=")[1].split()[0]), [float(x) for x in o.split("grad:")[1].split()]
(la, ga), (lb, gb) = rd(sys.argv[1]), rd(sys.argv[2])
import math
if not all(map(math.isfinite, ga + gb + [la, lb])):
    print("      non-finite log density or gradient")
    sys.exit(1)
err = max(abs(a - b) for a, b in zip(ga, gb)) / max(abs(x) for x in gb)
print(f"      logp {la:.15g} vs {lb:.15g}; max grad diff / max |grad| {err:.1e}")
sys.exit(0 if len(ga) == len(gb) and abs(la - lb) <= 1e-12 * abs(lb) and err < 1e-12 else 1)
PY
    # the benchmark point only: at the runtime's random point (uniform on
    # [-2, 2]) these running sums give log densities near -1e16 and finite
    # differences break down for every build
    worst=$(MINT_GRADCHECK=1 MINT_BENCH_GRAD=1 ./build/scan_${m}_${G}_opt 2>&1 | grep gradcheck | head -1 | sed 's/.*error=//')
    python3 -c "import sys; sys.exit(0 if float('${worst:-inf}') < 1e-5 else 1)" && pass "gradcheck scan kernel $m G=$G ($worst)" || bad "gradcheck scan kernel $m G=$G ($worst)"
  done
done

# ---- parallel scan kernel. With one thread the kernel is the serial code, so
# the gradient must be bit-identical to the build without the parallel
# kernel. With the groups of rows split across 3 threads the summation order
# changes, so the log density and every gradient component must agree with
# the one-thread result to a tolerance (1e-12 of the largest component, and
# 1e-10 relative per component). G=61 is 7 groups of 8 rows (split 2/2/3), a
# single vector and a leftover row; G=7, 13 and 20 have fewer groups than
# threads. BernoulliLogit (one lane) and a running sum of data only are not
# parallelised, and the generated code must say so.

# close_grad A B LABEL
close_grad() {
  python3 - "$1" "$2" <<'PY' && pass "$3" || bad "$3"
import sys, math
def rd(p):
    o = open(p).read()
    return float(o.split("logp=")[1].split()[0]), [float(x) for x in o.split("grad:")[1].split()]
(la, ga), (lb, gb) = rd(sys.argv[1]), rd(sys.argv[2])
if not all(map(math.isfinite, ga + gb + [la, lb])):
    print("      non-finite log density or gradient")
    sys.exit(1)
top = max(abs(x) for x in gb)
err = max(abs(a - b) for a, b in zip(ga, gb)) / top
rel = max(abs(a - b) / max(abs(b), 1e-6 * top) for a, b in zip(ga, gb))
print(f"      logp {la:.15g} vs {lb:.15g}; max grad diff / max |grad| {err:.1e}; worst per-component {rel:.1e}")
sys.exit(0 if len(ga) == len(gb) and abs(la - lb) <= 1e-12 * abs(lb) and err < 1e-12 and rel < 1e-10 else 1)
PY
}
# grad_out BIN THREADS OUT: log density and gradient at the benchmark point, without the timing
grad_out() {
  MINT_KERNEL_THREADS=$2 MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 ./build/$1 | sed 's/ns_per_eval=[0-9.]*//' > "$3"
}
par_check() { # NAME SOURCE par|serial
  local n=$1 src=$2 want=$3
  build "$src" par_$n || return
  build "$src" par_${n}_seq --no-parallel-kernel || return
  local has=serial
  grep -q "call i64 @mint_par_groups" build/par_$n.ll && has=par
  [ "$has" = "$want" ] && pass "parallel scan kernel $n: generated code is $want" || { bad "parallel scan kernel $n: expected $want code, got $has"; return; }
  grad_out par_$n 1 build/par_${n}_t1.out
  grad_out par_$n 3 build/par_${n}_t3.out
  grad_out par_${n}_seq 3 build/par_${n}_seq.out
  [ -s build/par_${n}_t1.out ] && cmp -s build/par_${n}_t1.out build/par_${n}_seq.out \
    && pass "parallel scan kernel $n: 1 thread is bit-identical to --no-parallel-kernel" \
    || bad "parallel scan kernel $n: 1 thread differs from --no-parallel-kernel"
  [ "$want" = par ] && close_grad build/par_${n}_t3.out build/par_${n}_t1.out "parallel scan kernel $n: 3 threads match 1 thread"
}
for m in nested mixed nested_sq colreuse twohosts twoowned layout_draws bernoulli datascan; do
  want=par
  case $m in bernoulli|datascan) want=serial ;; esac
  for G in 7 13 20 61; do
    sed "s/NG/$G/" tests/scan/$m.mint > build/par_$m.$G.mint
    par_check ${m}_$G build/par_$m.$G.mint $want
  done
done
if [ -f bench/dynpois/data_large/y.f64 ]; then
  sed 's#bench/dynpois/data_small/y.f64#bench/dynpois/data_large/y.f64#' examples/dynamic_poisson.mint > build/par_dynpois_large.mint
  par_check dynpois_large build/par_dynpois_large.mint par
  par_check dynpois_small examples/dynamic_poisson.mint par
  # The split really happened (the summation order shows in the last bits),
  # it follows the team OpenMP runs (a team of one gives the serial result),
  # and it is deterministic for a given team size.
  if [ -x build/par_dynpois_large ]; then
    grad_out par_dynpois_large 3 build/par_det_a.out
    grad_out par_dynpois_large 3 build/par_det_b.out
    OMP_THREAD_LIMIT=1 grad_out par_dynpois_large 3 build/par_det_lim.out
    ! cmp -s build/par_det_a.out build/par_dynpois_large_t1.out \
      && pass "parallel scan kernel: 3 threads change the summation order (the split ran)" || bad "parallel scan kernel: 3 threads gave the one-thread result"
    [ -s build/par_det_a.out ] && cmp -s build/par_det_a.out build/par_det_b.out \
      && pass "parallel scan kernel: identical gradients in two runs on 3 threads" || bad "parallel scan kernel is not deterministic"
    cmp -s build/par_det_lim.out build/par_dynpois_large_t1.out \
      && pass "parallel scan kernel follows a reduced OpenMP team" || bad "parallel scan kernel with a team of one differs from one thread"
  fi
  # inside the sampler (threads per chain set by run_chain): two short runs
  # with 3 threads per chain give the same raw draws
  sed 's/draws = 1000, warmup = 1000/draws = 20, warmup = 20/' build/par_dynpois_large.mint > build/par_dynpois_run.mint
  if build build/par_dynpois_run.mint par_dynpois_run; then
    MINT_THREADS_PER_CHAIN=3 MINT_DRAWS=build/par_run_a.draws ./build/par_dynpois_run > /dev/null 2>&1
    MINT_THREADS_PER_CHAIN=3 MINT_DRAWS=build/par_run_b.draws ./build/par_dynpois_run > /dev/null 2>&1
    [ -s build/par_run_a.draws ] && cmp -s build/par_run_a.draws build/par_run_b.draws \
      && pass "parallel scan kernel in the sampler: identical raw draws in two runs" || bad "parallel scan kernel in the sampler: raw draws differ between runs"
  fi
fi

# ---- fused leapfrog. The leap entry point hands each kernel thread's rows of
# a covered matrix parameter to the sampler's leaf work (MINT_LEAP_TEST: one
# leaf with a merge, against the runtime's own path). The gradient, log
# density, momentum, next half-step and merged momentum must be
# bit-identical and the kinetic energy and merge sums equal to 1e-12, on 1
# and 3 kernel threads. Models with nothing to cover (a running sum of data
# only; a matrix parameter shared by two scan statements, which neither
# kernel owns) and --no-fused-leapfrog builds must not have the entry point;
# twoowned has three covered parameters in two kernels. In whole runs
# (nested at 61 series, the small dynamic Poisson model), with 1 and 3
# threads per chain, MINT_FUSED_LEAPFROG=exact (the fused leaf work with the
# sums in the runtime's order) must give exactly the draws of
# MINT_FUSED_LEAPFROG=0, and the default fused sums must be deterministic,
# must have run (the sampler reports its count of fused leaves) and must
# change the draws; on the large model, exact against 0 with 3 threads.
leap_check() { # NAME leap|none (uses build/par_NAME from the parallel kernel tests)
  local n=$1 want=$2 has=none
  [ -x build/par_$n ] || { bad "fused leapfrog $n: no binary"; return; }
  grep -q "define double @mint_model_.*_leap(" build/par_$n.ll && has=leap
  [ "$has" = "$want" ] && pass "fused leapfrog $n: generated code has $want" || { bad "fused leapfrog $n: expected $want, got $has"; return; }
  [ "$want" = leap ] || return
  for t in 1 3; do
    out=$(MINT_KERNEL_THREADS=$t MINT_LEAP_TEST=1 ./build/par_$n 2>&1)
    grep -q "^leap-test: ok" <<<"$out" && pass "fused leapfrog $n, $t kernel threads: same leaf as the runtime's" \
      || { bad "fused leapfrog $n, $t kernel threads: leaf differs"; echo "$out"; }
  done
}
for m in nested mixed nested_sq colreuse twohosts twoowned layout_draws bernoulli datascan; do
  want=leap
  case $m in datascan|twohosts) want=none ;; esac
  for G in 7 13 20 61; do leap_check ${m}_$G $want; done
done
build build/par_nested.20.mint leap_off --no-fused-leapfrog \
  && { grep -q "define double @mint_model_.*_leap(" build/leap_off.ll && bad "--no-fused-leapfrog still emits the leap entry point" \
       || pass "--no-fused-leapfrog: no leap entry point"; }
# draws_same LABEL BIN ENV_A ENV_B: the raw draws of two runs are identical
draws_same() {
  env $3 MINT_DRAWS=build/leap_a.draws ./build/$2 > /dev/null 2>&1
  env $4 MINT_DRAWS=build/leap_b.draws ./build/$2 > /dev/null 2>&1
  [ -s build/leap_a.draws ] && cmp -s build/leap_a.draws build/leap_b.draws && pass "$1" || bad "$1"
  rm -f build/leap_a.draws build/leap_b.draws
}
sed 's/draws = 4, warmup = 0/draws = 30, warmup = 30/' build/par_nested.61.mint > build/leap_nested.mint
sed 's/draws = 1000, warmup = 1000/draws = 40, warmup = 40/' examples/dynamic_poisson.mint > build/leap_dps.mint
for m in nested dps; do
  build build/leap_$m.mint leap_$m || continue
  for t in 1 3; do
    draws_same "fused leapfrog $m, $t threads per chain: exact sums give the runtime's draws" leap_$m \
      "MINT_THREADS_PER_CHAIN=$t MINT_FUSED_LEAPFROG=0" "MINT_THREADS_PER_CHAIN=$t MINT_FUSED_LEAPFROG=exact"
    draws_same "fused leapfrog $m, $t threads per chain: deterministic" leap_$m \
      "MINT_THREADS_PER_CHAIN=$t MINT_FUSED_LEAPFROG=1" "MINT_THREADS_PER_CHAIN=$t MINT_FUSED_LEAPFROG=1"
    # the fused path really ran: the sampler counts its leaves, and its
    # summation order changes the draws
    out=$(MINT_THREADS_PER_CHAIN=$t MINT_FUSED_LEAPFROG=1 MINT_DRAWS=build/leap_f.draws ./build/leap_$m 2>&1 >/dev/null)
    MINT_THREADS_PER_CHAIN=$t MINT_FUSED_LEAPFROG=0 MINT_DRAWS=build/leap_u.draws ./build/leap_$m > /dev/null 2>&1
    grep -q "leapfrog=fused (" <<<"$out" && [ -s build/leap_f.draws ] && ! cmp -s build/leap_f.draws build/leap_u.draws \
      && pass "fused leapfrog $m, $t threads per chain: ran, and its sums change the draws" \
      || { bad "fused leapfrog $m, $t threads per chain: did not run"; echo "$out"; }
    rm -f build/leap_f.draws build/leap_u.draws
  done
done
if [ -f bench/dynpois/data_large/y.f64 ] && [ -x build/par_dynpois_run ]; then
  draws_same "fused leapfrog, large model, 3 threads per chain: exact sums give the runtime's draws" par_dynpois_run \
    "MINT_THREADS_PER_CHAIN=3 MINT_FUSED_LEAPFROG=0" "MINT_THREADS_PER_CHAIN=3 MINT_FUSED_LEAPFROG=exact"
  out=$(MINT_THREADS_PER_CHAIN=3 ./build/par_dynpois_run 2>&1 >/dev/null)
  grep -q "leapfrog=fused (" <<<"$out" && pass "fused leapfrog is the default with 3 threads per chain" \
    || { bad "fused leapfrog is not the default with 3 threads per chain"; echo "$out"; }
  out=$(MINT_THREADS_PER_CHAIN=1 ./build/par_dynpois_run 2>&1 >/dev/null)
  grep -q "leapfrog=runtime$" <<<"$out" && pass "runtime leapfrog is the default with 1 thread per chain" \
    || { bad "runtime leapfrog is not the default with 1 thread per chain"; echo "$out"; }
fi

# ---- the scan layout reaches the draws: a matrix parameter pinned to the
# data by a tight prior must come back with each posterior mean on its own
# (row-major) entry.

sed "s/NG/13/" tests/scan/layout_draws.mint > build/scan_layout.mint
if build build/scan_layout.mint scan_layout; then
  MINT_DRAWS=build/scan_layout.draws ./build/scan_layout > /dev/null 2>&1
  python3 - <<'PY' && pass "scan layout: draws come back in the user's order" || bad "scan layout: draws are out of order"
import struct
h = open("build/scan_layout.draws", "rb").read()
C, N, D = struct.unpack("<QQQ", h[:24])
d = struct.unpack(f"<{C * N * D}d", h[24:24 + 8 * C * N * D])
y = open("build/scan_normal_13.f64", "rb").read()
G, T = struct.unpack("<QQ", y[:16])
yv = struct.unpack(f"<{G * T}d", y[16:])
means = [sum(d[(c * N + i) * D + j] for c in range(C) for i in range(N)) / (C * N) for j in range(D)]
err = max(abs(a - b) for a, b in zip(means, yv))
print(f"      max |posterior mean - y| = {err:.4f}")
raise SystemExit(0 if D == G * T and err < 0.02 else 1)
PY
fi

# ---- row fusion: a product plus a term of another shape (broadcast, or a
# scalar) must not be fused; Newton on a row count that is not a multiple of
# the chunk must match the unfused, untiled build.

python3 tests/rowfuse/make_data.py
if build tests/rowfuse/broadcast.mint rf_b_opt && build tests/rowfuse/broadcast.mint rf_b_ref --no-row-fusion; then
  [ -n "$(./build/rf_b_opt)" ] && [ "$(./build/rf_b_opt)" = "$(./build/rf_b_ref)" ] && pass "row fusion leaves broadcasting sums alone" || bad "row fusion changed a broadcasting sum"
fi
if build tests/rowfuse/newton_odd.mint rf_n_opt && build tests/rowfuse/newton_odd.mint rf_n_ref --no-row-fusion --no-gram-blocking; then
  a=$(./build/rf_n_opt | sed -n 's/^w \[\(.*\)\]/\1/p'); b=$(./build/rf_n_ref | sed -n 's/^w \[\(.*\)\]/\1/p')
  python3 -c "
import sys
a = [float(x) for x in '$a'.split(',')]; b = [float(x) for x in '$b'.split(',')]
sys.exit(0 if len(a) == len(b) == 13 and max(abs(x - y) for x, y in zip(a, b)) < 1e-8 else 1)" \
    && pass "row fusion + tiled Gram match the unfused build (1003 rows)" || bad "row fusion + tiled Gram differ from the unfused build"
fi
# The gradient and Hessian themselves, for column counts that exercise every
# tile shape, the clamped last strip and the gradient carried in the Gram
# kernel's padding (p not a multiple of 4), fused and unfused.
if build tests/rowfuse/hessian.mint rf_h_opt && build tests/rowfuse/hessian.mint rf_h_nrf --no-row-fusion \
   && build tests/rowfuse/hessian.mint rf_h_ref --no-row-fusion --no-gram-blocking; then
  for np in "0 5" "20 2" "1003 1" "1003 3" "1003 4" "1003 5" "1003 13" "1003 24" "1003 50" "37 9" "3000 71"; do
    python3 tests/rowfuse/make_data.py $np
    python3 - $np <<'PY' && pass "fused, tiled gradient and Hessian match the untiled ones (n p = $np)" || bad "fused, tiled gradient or Hessian differ (n p = $np)"
import re, subprocess, sys
p = int(sys.argv[2])
def values(b, label):
    out = subprocess.run(["./build/" + b], capture_output=True, text=True, check=True).stdout
    line = out[out.index(label + " "):].split("\n" + ("H " if label == "g" else "\0"))[0]
    return [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", line[2:])]
# normwise: each difference relative to the largest entry, as summation
# reordering allows (entries much smaller than the largest are not checked
# to their own precision)
ok = True
for label, size in (("g", p), ("H", p * p)):
    ref = values("rf_h_ref", label)
    s = max(abs(x) for x in ref)
    ok = ok and len(ref) == size
    for b in ("rf_h_opt", "rf_h_nrf"):
        a = values(b, label)
        d = max(abs(x - y) for x, y in zip(a, ref)) / s if len(a) == len(ref) else float("inf")
        ok = ok and d < 1e-9
        if d >= 1e-9:
            print(f"      {b} {label}: relative difference {d:.3g}")
sys.exit(0 if ok else 1)
PY
  done
  python3 tests/rowfuse/make_data.py
fi
# A fused loop whose per-row values cannot be vectorised (log1p), with an
# a Gram weight other than Newton's and both consumers in the result.
if build tests/rowfuse/fallback.mint rf_f_opt && build tests/rowfuse/fallback.mint rf_f_ref --no-row-fusion --no-gram-blocking; then
  for np in "20 2" "1003 13" "1003 50"; do
    python3 tests/rowfuse/make_data.py $np
    a=$(./build/rf_f_opt | sed -n 's/^g \[\(.*\)\]/\1/p'); b=$(./build/rf_f_ref | sed -n 's/^g \[\(.*\)\]/\1/p')
    python3 -c "
import sys
a = [float(x) for x in '$a'.split(',')]; b = [float(x) for x in '$b'.split(',')]
s = max(abs(x) for x in b)
sys.exit(0 if len(a) == len(b) == ${np#* } and max(abs(x - y) for x, y in zip(a, b)) / s < 1e-9 else 1)" \
      && pass "fused loop with scalar per-row values matches the unfused build (n p = $np)" || bad "fused loop with scalar per-row values differs (n p = $np)"
  done
  python3 tests/rowfuse/make_data.py
fi

# ---- Mint's own vector exp, as emitted: within 2 ulp of long double expl
# over 3e6 inputs across the range, and NaN, infinities, -0, the overflow and
# underflow thresholds and subnormal results.

$M emit examples/dynamic_poisson.mint -o build/exp_check.ll 2>/dev/null \
  && python3 tests/exp/check_exp.py build/exp_check.ll && pass "vector exp accuracy and special values" || bad "vector exp accuracy"

# ---- Mint's own vector log, as emitted in the fission kernel (here for a
# Normal's indexed scale): within 2 ulp of long double logl over 4e6 inputs
# (every binade including subnormals, around 1, around every table
# boundary), and 0, -0, negatives, subnormals, the extreme doubles,
# infinities and NaN as in libm. And its log1p on [0, 1] (BernoulliLogit's
# softplus): within 2.1 ulp of log1pl over 4.5e6 inputs (the bound is about
# 2 ulp), exact at 0, 1 and the smallest subnormal, NaN for NaN.

$M emit tests/fission/normal.mint -o build/log_check.ll 2>/dev/null \
  && python3 tests/log/check_log.py build/log_check.ll && pass "vector log accuracy and special values" || bad "vector log accuracy"
$M emit examples/logistic_bayes.mint -o build/log1p_check.ll 2>/dev/null \
  && python3 tests/log/check_log1p.py build/log1p_check.ll 2.1 && pass "vector log1p accuracy and special values" || bad "vector log1p accuracy"

# ---- fission kernel paths against the three-pass fission and the unsplit loop

. tests/fission/run.sh

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
