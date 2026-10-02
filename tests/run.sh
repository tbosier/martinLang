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

# build SRC OUT [flags...]: removes OUT first so a failed build cannot leave a stale binary.
# Every build gets $DEFAULT_FLAGS first. Until the Kalman section at the end
# that is --no-collapse: several scan-kernel, parallel-kernel, fused-leapfrog
# and narrow-data test models are Gaussian random walks, which the Kalman
# collapse would otherwise integrate out, and they are there to test the
# code that samples them as written.
DEFAULT_FLAGS=--no-collapse
build() {
  local src=$1 out=$2; shift 2
  rm -f "build/$out"
  $M build "$src" -o "build/$out" $DEFAULT_FLAGS "$@" 2>build/$out.log || { bad "build $src $*"; cat build/$out.log; return 1; }
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

# ---- mintc explain: compiles what emit compiles, reports every statement
# and the key decisions (compiler/tests/explain.rs)

(cd compiler && cargo test --release -q --test explain >/dev/null 2>&1) && pass "mintc explain (cargo test --test explain)" \
  || bad "mintc explain: run (cd compiler && cargo test --release --test explain)"

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
# multiples of the vector width, and 16, which leaves the scan layout no
# last block) must give the same log density and gradient as the same model
# built without the scan layout, scan fusion and inline exp, and must pass
# the finite-difference check.

python3 tests/scan/make_data.py build
for m in nested bernoulli mixed nested_sq colreuse datascan twohosts twoowned; do
  for G in 7 13 16 20; do
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
# single vector and a leftover row, which thread 0 runs inside the parallel
# region; G=7, 13, 16 and 20 have fewer groups than threads (16 leaves the
# scan layout no last block). BernoulliLogit (one lane) and a running sum of data only are not
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
  for G in 7 13 16 20 61; do
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
    # A team of one runs the parallel path, which adds the log density's
    # partial sums in a different association from the serial loop (single
    # vectors and leftover rows are in thread 0's share), so the log density
    # may differ in the last bits; the gradient must be identical.
    python3 - build/par_det_lim.out build/par_dynpois_large_t1.out <<'PY' \
      && pass "parallel scan kernel follows a reduced OpenMP team" || bad "parallel scan kernel with a team of one differs from one thread"
import sys
def rd(p):
    o = open(p).read()
    g = o.split("grad:")[1].split()
    lp = [l for l in o.splitlines() if l.startswith("exact log density:")]
    return g, float(lp[0].split(":")[1]) if lp else None
(ga, la), (gb, lb) = rd(sys.argv[1]), rd(sys.argv[2])
ok = ga == gb and (la == lb or (la is not None and lb is not None and abs(la - lb) <= 4e-16 * abs(lb)))
sys.exit(0 if ok else 1)
PY
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

# ---- fused leapfrog (mintc --fused-leapfrog). The leap entry point hands
# each kernel thread's rows of a covered matrix parameter to the sampler's
# leaf work. MINT_LEAP_TEST runs one leaf with a merge through it and
# through the runtime's own path, with the diagonal metric and with a
# low-rank one: the gradient, log density, momentum, next half-step and
# merged momentum (and their low-rank projections) must be bit-identical,
# and the kinetic energy and merge sums equal to 1e-12, with 1 and 3 kernel threads
# requested (a kernel never runs more threads than it has groups of 8
# series: at 7, 13 and 16 series one, at 20 two, at 61 three). Models with
# nothing to cover (a running sum of data only; a matrix parameter shared
# by two scan statements, which neither kernel owns) and default builds must
# not have the entry point; nested covers one parameter and twoowned three,
# in two kernels. In whole runs (nested at 61 series, the small dynamic
# Poisson model), with 1 and 3 threads per chain, MINT_FUSED_LEAPFROG=exact
# (the fused leaf work with the sums in the runtime's order) must run and
# give exactly the draws of MINT_FUSED_LEAPFROG=0, and the fused sums (the
# default in such a build) must run and be deterministic; on the large
# model, exact against 0 with 3 threads. The fused sums' correctness rests
# on MINT_LEAP_TEST: their draws differ from the unfused ones by rounding,
# and then diverge.
leap_check() { # MODEL G leap|none
  local n=${1}_$2 want=$3 has=none
  build build/par_$1.$2.mint leap_$n --fused-leapfrog || return
  grep -q "define double @mint_model_.*_leap(" build/leap_$n.ll && has=leap
  [ "$has" = "$want" ] && pass "fused leapfrog $n: generated code has $want" || { bad "fused leapfrog $n: expected $want, got $has"; return; }
  [ "$want" = leap ] || return
  for t in 1 3; do
    out=$(MINT_KERNEL_THREADS=$t MINT_LEAP_TEST=1 ./build/leap_$n 2>&1)
    grep -q "^leap-test: ok" <<<"$out" && [ "$(grep -c "metric=lowrank" <<<"$out")" = 2 ] \
      && pass "fused leapfrog $n, $t kernel threads requested: same leaf as the runtime's (diagonal and low-rank metric)" \
      || { bad "fused leapfrog $n, $t kernel threads requested: leaf differs"; echo "$out"; }
  done
}
# covered BIN N: the leap entry point covers N parameters
covered() {
  local got
  got=$(awk '/^define i64 @mint_model_.*_leap_blocks/,/^}/' build/$1.ll | sed -n 's/.*ret i64 \([0-9]*\).*/\1/p')
  [ "$got" = "$2" ] && pass "fused leapfrog $1: covers $2 parameter(s)" || bad "fused leapfrog $1: covers '$got' parameters, expected $2"
}
for m in nested mixed nested_sq colreuse twohosts twoowned layout_draws bernoulli datascan; do
  want=leap
  case $m in datascan|twohosts) want=none ;; esac
  for G in 7 13 16 20 61; do leap_check $m $G $want; done
done
[ -x build/leap_nested_61 ] && covered leap_nested_61 1
[ -x build/leap_twoowned_61 ] && covered leap_twoowned_61 3
[ -f build/par_nested_20.ll ] && { grep -q "define double @mint_model_.*_leap(" build/par_nested_20.ll \
  && bad "a default build emits the leap entry point" || pass "a default build has no leap entry point"; }
# draws_same LABEL BIN ENV_A ENV_B: the raw draws of two runs are identical
draws_same() {
  env $3 MINT_DRAWS=build/leap_a.draws ./build/$2 > /dev/null 2>&1
  env $4 MINT_DRAWS=build/leap_b.draws ./build/$2 > /dev/null 2>&1
  [ -s build/leap_a.draws ] && cmp -s build/leap_a.draws build/leap_b.draws && pass "$1" || bad "$1"
  rm -f build/leap_a.draws build/leap_b.draws
}
# ran LABEL BIN ENV MODE: the sampler reports leaves through the fused leapfrog in MODE
ran() {
  local out
  out=$(env $3 ./build/$2 2>&1 >/dev/null)
  grep -q "leapfrog=$4 (" <<<"$out" && pass "$1" || { bad "$1"; echo "$out" | grep sampler; }
}
sed 's/draws = 4, warmup = 0/draws = 30, warmup = 30/' build/par_nested.61.mint > build/leap_nested.mint
sed 's/draws = 1000, warmup = 1000/draws = 40, warmup = 40/' examples/dynamic_poisson.mint > build/leap_dps.mint
for m in nested dps; do
  build build/leap_$m.mint leap_$m --fused-leapfrog || continue
  for t in 1 3; do
    draws_same "fused leapfrog $m, $t threads per chain: exact sums give the runtime's draws" leap_$m \
      "MINT_THREADS_PER_CHAIN=$t MINT_FUSED_LEAPFROG=0" "MINT_THREADS_PER_CHAIN=$t MINT_FUSED_LEAPFROG=exact"
    ran "fused leapfrog $m, $t threads per chain: exact mode ran" leap_$m "MINT_THREADS_PER_CHAIN=$t MINT_FUSED_LEAPFROG=exact" fused-exact
    draws_same "fused leapfrog $m, $t threads per chain: deterministic" leap_$m \
      "MINT_THREADS_PER_CHAIN=$t" "MINT_THREADS_PER_CHAIN=$t"
    ran "fused leapfrog $m, $t threads per chain: on by default in a --fused-leapfrog build" leap_$m "MINT_THREADS_PER_CHAIN=$t" fused
  done
done
# The low-rank metric through the fused leapfrog: the hooks do the diagonal
# part of each leaf, a pass after the kernel the projections and the
# low-rank part of the next position (leaf_fused_run). With exact sums the
# draws must be the runtime's low-rank draws; the fused sums must run with
# directions kept (cutoff 1 keeps every direction it finds, so the rank is
# not 0 even after these short warmups).
for m in nested dps; do
  [ -x build/leap_$m ] || continue
  for t in 1 3; do
    lr="MINT_METRIC=lowrank MINT_LOWRANK_CUTOFF=1 MINT_THREADS_PER_CHAIN=$t"
    draws_same "fused leapfrog $m, low-rank metric, $t threads per chain: exact sums give the runtime's draws" leap_$m \
      "$lr MINT_FUSED_LEAPFROG=0" "$lr MINT_FUSED_LEAPFROG=exact"
    out=$(env $lr MINT_FUSED_LEAPFROG=exact ./build/leap_$m 2>&1 >/dev/null)
    grep -q "metric=lowrank (rank [1-9][0-9]* to [0-9]*) leapfrog=fused-exact (" <<<"$out" \
      && pass "fused leapfrog $m, low-rank metric, $t threads per chain: the exact run kept directions" \
      || { bad "fused leapfrog $m, low-rank metric, $t threads per chain: exact run"; echo "$out" | grep sampler; }
    out=$(env $lr ./build/leap_$m 2>&1 >/dev/null)
    grep -q "metric=lowrank (rank [1-9][0-9]* to [0-9]*) leapfrog=fused (" <<<"$out" \
      && pass "fused leapfrog $m, low-rank metric, $t threads per chain: fused sums ran with directions kept" \
      || { bad "fused leapfrog $m, low-rank metric, $t threads per chain: fused run"; echo "$out" | grep sampler; }
  done
done
if [ -f bench/dynpois/data_large/y.f64 ] && [ -f build/par_dynpois_run.mint ]; then
  if build build/par_dynpois_run.mint leap_dpl --fused-leapfrog; then
    draws_same "fused leapfrog, large model, 3 threads per chain: exact sums give the runtime's draws" leap_dpl \
      "MINT_THREADS_PER_CHAIN=3 MINT_FUSED_LEAPFROG=0" "MINT_THREADS_PER_CHAIN=3 MINT_FUSED_LEAPFROG=exact"
    ran "fused leapfrog, large model: exact mode ran" leap_dpl "MINT_THREADS_PER_CHAIN=3 MINT_FUSED_LEAPFROG=exact" fused-exact
    ran "fused leapfrog, large model: an unknown MINT_FUSED_LEAPFROG value keeps the default (on)" leap_dpl \
      "MINT_THREADS_PER_CHAIN=3 MINT_FUSED_LEAPFROG=yes" fused
    out=$(MINT_THREADS_PER_CHAIN=3 MINT_FUSED_LEAPFROG=0 ./build/leap_dpl 2>&1 >/dev/null)
    grep -q "leapfrog=runtime$" <<<"$out" && pass "MINT_FUSED_LEAPFROG=0 turns the fused leapfrog off" \
      || { bad "MINT_FUSED_LEAPFROG=0 did not turn the fused leapfrog off"; echo "$out"; }
  fi
  if [ -x build/par_dynpois_run ]; then
    out=$(MINT_THREADS_PER_CHAIN=3 MINT_FUSED_LEAPFROG=1 ./build/par_dynpois_run 2>&1 >/dev/null)
    grep -q "leapfrog=runtime$" <<<"$out" && pass "a default build has no fused leapfrog to turn on" \
      || { bad "a default build ran a fused leapfrog"; echo "$out"; }
  fi
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

# ---- narrow data copies: results byte-identical to the build without them

. tests/narrow/run.sh

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
check_eight "fast warmup" MINT_WARMUP=fast
check_eight "fast warmup, 3 threads per chain" MINT_WARMUP=fast MINT_THREADS_PER_CHAIN=3

# The fast warmup pools window estimates across chain threads; the draws must
# not depend on which chain reaches the barrier first.
rm -f build/fast_a.draws build/fast_b.draws
MINT_WARMUP=fast MINT_DRAWS=build/fast_a.draws ./build/eight_schools > /dev/null 2>&1 \
  && MINT_WARMUP=fast MINT_DRAWS=build/fast_b.draws ./build/eight_schools > /dev/null 2>&1 \
  && [ -s build/fast_a.draws ] && cmp -s build/fast_a.draws build/fast_b.draws \
  && pass "fast warmup: identical raw draws in two runs" || bad "fast warmup: raw draws differ between runs"
# The pathfinder start must not run into the singularity of a centred
# hierarchical model (its density is unbounded as tau goes to 0).
printf '%s\n' 'model C {
    data y: Vector[J]
    data s: Positive[J]
    param mu: Real
    param tau: Positive
    param theta: Vector[J]
    mu ~ Normal(0, 5)
    tau ~ Normal(0, 5)
    theta ~ Normal(mu, tau)
    y ~ Normal(theta, s)
}
fn main() {
    let y = [28, 8, -3, 7, -1, 1, 18, 12]
    let s = [15, 10, 16, 11, 9, 11, 10, 18]
    print(sample(C(y, s), draws = 200, warmup = 1000, chains = 4, seed = 3))
}' > build/centered8.mint
if build build/centered8.mint centered8; then
  out=$(MINT_WARMUP=fast MINT_WARMUP_TRACE=1 ./build/centered8 2>&1)
  python3 - "$out" <<'PY' && pass "fast warmup: centred eight schools starts away from the singularity" || bad "fast warmup: centred eight schools start"
import re, sys
out = sys.argv[1]
starts = dict(re.findall(r"chain=(\d+) start lp=(\S+)", out))
climb = {c: (int(n), int(k), float(e)) for c, n, k, e in
         re.findall(r"chain=(\d+) lbfgs steps=(\d+) chosen=(-?\d+) .* end lp=(\S+)", out)}
print(f"      chain: (climb steps, chosen step, end lp), start lp: {climb}, {starts}")
# Every chain chose a point by ELBO (not the uniform fallback, chosen = -1),
# and the climb went on well past it towards tau = 0: the log density at its
# end is far above the chosen point's. The run completes with a finite tau.
ok = len(starts) == len(climb) == 4
for c, (n, k, e) in climb.items():
    ok &= 0 <= k < n - 1 and e > float(starts[c]) + 10
tau = [l.split() for l in out.splitlines() if l.startswith("tau ")]
ok &= len(tau) == 1 and 0 < float(tau[0][1]) < 20
sys.exit(0 if ok else 1)
PY
fi
check_eight "low-rank metric" MINT_METRIC=lowrank
# cutoff 1 keeps every direction: a dense metric (rank 10 here), threaded path
check_eight "low-rank metric, full rank, 3 threads" MINT_METRIC=lowrank MINT_LOWRANK_CUTOFF=1 MINT_THREADS_PER_CHAIN=3
MINT_METRIC=lowrank MINT_LOWRANK_CUTOFF=1 MINT_THREADS_PER_CHAIN=3 ./build/eight_schools 2>&1 >/dev/null \
  | grep -qF "metric=lowrank (rank 10 to 10)" && pass "low-rank metric with cutoff 1 keeps all 10 directions" \
  || bad "low-rank metric with cutoff 1 did not keep all 10 directions"
check_eight "low-rank metric, fast warmup" MINT_METRIC=lowrank MINT_WARMUP=fast
check_eight "low-rank metric, fast warmup, 3 threads" MINT_METRIC=lowrank MINT_WARMUP=fast MINT_THREADS_PER_CHAIN=3

# ---- a Gaussian posterior with known mean and covariance, much narrower
# than the prior along 8 dense directions: every whitened first, second and
# cross moment within its Monte Carlo error (tests/metric/check_gauss.py),
# for the default metric and the low-rank one (which must keep 8
# directions).

python3 tests/metric/make_gauss.py
if build tests/metric/gauss.mint gauss; then
  check_gauss() {
    local label=$1; shift
    local out
    rm -f build/gauss.draws
    out=$(env "$@" MINT_DRAWS=build/gauss.draws ./build/gauss 2>&1) \
      || { bad "Gaussian sampler run ($label)"; echo "$out"; return; }
    python3 tests/metric/check_gauss.py build/gauss.draws && pass "Gaussian posterior moments ($label)" \
      || bad "Gaussian posterior moments ($label)"
    if [ "$label" != "default metric" ]; then
      grep -qF "metric=lowrank (rank 8 to 8)" <<<"$out" && pass "low-rank metric keeps 8 directions, as many as there are narrow ones ($label)" \
        || { bad "low-rank metric rank ($label)"; grep -F "metric=" <<<"$out"; }
    fi
  }
  check_gauss "default metric" MINT_THREADS_PER_CHAIN=1
  check_gauss "low-rank metric" MINT_METRIC=lowrank MINT_THREADS_PER_CHAIN=1
  check_gauss "low-rank metric, 3 threads" MINT_METRIC=lowrank MINT_THREADS_PER_CHAIN=3
  # with the fast warmup the chains pool their windows for the low-rank
  # estimate too (chain 0 estimates from every chain's draws), or, with
  # MINT_WARMUP_POOL=0, estimate it each on their own
  check_gauss "low-rank metric, fast warmup (pooled)" MINT_METRIC=lowrank MINT_WARMUP=fast MINT_THREADS_PER_CHAIN=1
  check_gauss "low-rank metric, fast warmup (pooled), 3 threads" MINT_METRIC=lowrank MINT_WARMUP=fast MINT_THREADS_PER_CHAIN=3
  check_gauss "low-rank metric, fast warmup, not pooled" MINT_METRIC=lowrank MINT_WARMUP=fast MINT_WARMUP_POOL=0
  out=$(MINT_METRIC=lowrank MINT_WARMUP=fast MINT_LOWRANK_VERBOSE=1 ./build/gauss 2>&1 >/dev/null)
  grep -q "^chain 3 window end .*draws (pooled), rank 8" <<<"$out" \
    && pass "low-rank metric, fast warmup: the estimate is pooled across the chains" \
    || { bad "low-rank metric, fast warmup: no pooled estimate"; grep "window end" <<<"$out"; }
  # two pooled runs give the same draws (a smoke test of the barriers: the
  # estimate is meant not to depend on which chain reaches them first)
  rm -f build/gauss_fa.draws build/gauss_fb.draws
  MINT_METRIC=lowrank MINT_WARMUP=fast MINT_THREADS_PER_CHAIN=2 MINT_DRAWS=build/gauss_fa.draws ./build/gauss > /dev/null 2>&1
  MINT_METRIC=lowrank MINT_WARMUP=fast MINT_THREADS_PER_CHAIN=2 MINT_DRAWS=build/gauss_fb.draws ./build/gauss > /dev/null 2>&1
  [ -s build/gauss_fa.draws ] && cmp -s build/gauss_fa.draws build/gauss_fb.draws \
    && pass "low-rank metric, fast warmup: identical raw draws in two runs" \
    || bad "low-rank metric, fast warmup: raw draws differ between runs"
fi

# When OpenMP runs a smaller team than requested, the sampler must follow the
# team it got: with a limit of one thread the draws equal the serial ones.
serial=$(MINT_THREADS_PER_CHAIN=1 ./build/eight_schools 2>/dev/null)
limited=$(MINT_THREADS_PER_CHAIN=10 OMP_THREAD_LIMIT=1 ./build/eight_schools 2>/dev/null)
[ -n "$serial" ] && [ "$serial" = "$limited" ] && pass "sampler follows a reduced OpenMP team" \
  || bad "reduced OpenMP team changed the draws"
rm -f build/gauss_serial.draws build/gauss_limited.draws
MINT_METRIC=lowrank MINT_THREADS_PER_CHAIN=1 MINT_DRAWS=build/gauss_serial.draws ./build/gauss > /dev/null 2>&1
MINT_METRIC=lowrank MINT_THREADS_PER_CHAIN=10 OMP_THREAD_LIMIT=1 MINT_DRAWS=build/gauss_limited.draws ./build/gauss > /dev/null 2>&1
[ -s build/gauss_serial.draws ] && cmp -s build/gauss_serial.draws build/gauss_limited.draws \
  && pass "low-rank sampler follows a reduced OpenMP team (identical raw draws)" \
  || bad "reduced OpenMP team changed the low-rank draws"

# ---- streaming summaries. By default the runtime keeps every draw only of
# the rows the summary prints (with their quantiles) and summarises the
# other parameters as they are drawn; MINT_KEEP_DRAWS=all keeps every draw
# and computes the summary from the draws alone, as before. On the small
# dynamic Poisson model (3,171 parameters), with 1 and 3 threads per chain:
# the two summaries agree (rows and R-hat exactly, ESS to rounding),
# MINT_DRAWS writes the same file both ways and its draws are those the
# summary was computed from, and, per parameter on the same draws
# (MINT_STATS_DUMP), the streaming mean, sd and split R-hat equal the
# draw-level ones to rounding and so does the ESS (Geyer's sequence stops
# within the 32 lags kept for all of them here). The batch-means fallback
# is checked twice: with batches of one draw it must reproduce the
# draw-level ESS, and with 8 lags kept (some parameters then take it) its
# ESS must stay near the draw-level one. These parameters mix fast; the
# fallback's accuracy on slowly mixing ones is not tested here.
# stream_dump_check DUMP MAXFALLBACK [exact] [N]: compares the two sets of
# statistics of N parameters (3171 by default)
stream_dump_check() {
  python3 - "$1" "$2" "${3:-}" "${4:-3171}" <<'PY'
import math, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
maxfb, exact = int(sys.argv[2]), sys.argv[3] == "exact"
ok = len(rows) == int(sys.argv[4]) and all(r[1] == "1" for r in rows)
dm = dr = de = 0.0
fb = []
for r in rows:
    m, sd, rh, e, sm, ssd, srh, se = map(float, r[2:10])
    dm = max(dm, abs(sm - m) / sd, abs(ssd / sd - 1))
    dr = max(dr, abs(srh - rh))
    if r[10] == "1":
        fb.append(se / e)
    else:
        de = max(de, abs(se / e - 1))
print(f"      max mean/sd difference {dm:.1e}, R-hat {dr:.1e}, ESS (exact lags) {de:.1e}; {len(fb)} from batch means", end="")
ok &= dm < 1e-9 and dr < 1e-9 and de < 1e-6 and len(fb) <= maxfb
if fb:
    g = math.exp(sum(map(math.log, fb)) / len(fb))
    print(f", their ESS / draw-level ESS: geometric mean {g:.3f}, range {min(fb):.3f} to {max(fb):.3f}", end="")
    if exact:
        ok &= max(abs(x - 1) for x in fb) < 1e-6
    else:
        ok &= 0.8 < g < 1.25 and min(fb) > 0.33 and max(fb) < 3
print()
sys.exit(0 if ok else 1)
PY
}
# same_summary A B: identical except the ESS values, which may differ by rounding
same_summary() {
  python3 - "$1" "$2" <<'PY'
import re, sys
a, b = (open(p).read().splitlines() for p in sys.argv[1:3])
ok = len(a) == len(b) and len(a) > 0
ess = re.compile(r"(lowest ess )(\d+)")
for x, y in zip(a, b):
    ex, ey = [int(m.group(2)) for m in ess.finditer(x)], [int(m.group(2)) for m in ess.finditer(y)]
    ok &= ess.sub(r"\1#", x) == ess.sub(r"\1#", y) and len(ex) == len(ey)
    ok &= all(abs(u - v) <= 1 for u, v in zip(ex, ey))
sys.exit(0 if ok else 1)
PY
}
# draws_match FILE DUMP: every parameter's mean over the draws file equals the
# mean the summary computed from the draws in memory (layout and offsets)
draws_match() {
  python3 - "$1" "$2" <<'PY'
import sys
import numpy as np
C, N, D = map(int, np.fromfile(sys.argv[1], dtype="<u8", count=3))
x = np.fromfile(sys.argv[1], dtype="<f8", offset=24)
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[2])][1:]
ok = x.size == C * N * D == C * N * len(rows)
mean = np.array([float(r[2]) for r in rows]); sd = np.array([float(r[3]) for r in rows])
err = float(np.max(np.abs(x.reshape(C * N, D).mean(axis=0) - mean) / sd)) if ok else float("inf")
print(f"      max |mean over the file - summary mean| / sd = {err:.1e}")
sys.exit(0 if ok and err < 1e-9 else 1)
PY
}
if [ -x build/dynpois ]; then
  for t in 1 3; do
    rm -f build/stream_a.draws build/stream_b.draws
    MINT_THREADS_PER_CHAIN=$t MINT_DRAWS=build/stream_a.draws ./build/dynpois > build/stream_a.out 2> build/stream_a.err
    MINT_THREADS_PER_CHAIN=$t MINT_KEEP_DRAWS=all MINT_DRAWS=build/stream_b.draws MINT_STATS_DUMP=build/stream_b.tsv \
      ./build/dynpois > build/stream_b.out 2> /dev/null
    grep -q "summary: draws kept for 10 of 3171 parameters" build/stream_a.err \
      && pass "streaming summary ($t threads per chain): draws kept only for the 10 printed rows" \
      || { bad "streaming summary ($t threads per chain): kept draws"; grep summary: build/stream_a.err; }
    same_summary build/stream_a.out build/stream_b.out \
      && pass "streaming summary ($t threads per chain): same summary as with every draw kept" \
      || { bad "streaming summary ($t threads per chain) differs from MINT_KEEP_DRAWS=all"; diff build/stream_a.out build/stream_b.out; }
    [ "$(stat -c %s build/stream_a.draws 2>/dev/null)" = $((24 + 4 * 1000 * 3171 * 8)) ] \
      && cmp -s build/stream_a.draws build/stream_b.draws && [ ! -e build/stream_a.draws.partial ] \
      && pass "streaming summary ($t threads per chain): MINT_DRAWS writes the same file as with every draw kept" \
      || bad "streaming summary ($t threads per chain): MINT_DRAWS files differ"
    draws_match build/stream_a.draws build/stream_b.tsv \
      && pass "streaming summary ($t threads per chain): the MINT_DRAWS file holds the summarised draws" \
      || bad "streaming summary ($t threads per chain): the MINT_DRAWS file does not match the summary"
    stream_dump_check build/stream_b.tsv 0 \
      && pass "streaming summary ($t threads per chain): streaming statistics equal the draw-level ones" \
      || bad "streaming summary ($t threads per chain): streaming statistics differ from the draw-level ones"
  done
  # MINT_DRAWS into a pipe (not seekable): written in order after sampling
  rm -f build/stream_fifo build/stream_p.draws
  mkfifo build/stream_fifo && { cat build/stream_fifo > build/stream_p.draws & }
  MINT_THREADS_PER_CHAIN=3 MINT_DRAWS=build/stream_fifo ./build/dynpois > /dev/null 2>&1
  wait
  cmp -s build/stream_p.draws build/stream_a.draws \
    && pass "streaming summary: MINT_DRAWS into a pipe writes the same draws" || bad "streaming summary: MINT_DRAWS into a pipe"
  rm -f build/stream_fifo build/stream_p.draws build/stream_a.draws build/stream_b.draws
  MINT_KEEP_DRAWS=all MINT_ESS_LAGS=2 MINT_ESS_BATCHES=1000 MINT_STATS_DUMP=build/stream_d.tsv ./build/dynpois > /dev/null 2>&1
  stream_dump_check build/stream_d.tsv 3171 exact && grep -qP "\\t1$" build/stream_d.tsv \
    && pass "streaming summary: the batch-means ESS with batches of one draw is the draw-level ESS" \
    || bad "streaming summary: batch-means ESS with batches of one draw"
  MINT_KEEP_DRAWS=all MINT_ESS_LAGS=8 MINT_STATS_DUMP=build/stream_c.tsv ./build/dynpois > /dev/null 2>&1
  stream_dump_check build/stream_c.tsv 3171 && grep -qP "\\t1$" build/stream_c.tsv \
    && pass "streaming summary: the batch-means ESS fallback is close to the draw-level ESS" \
    || bad "streaming summary: batch-means ESS fallback"
  # pop, all 20 entries of beta, the first 3 of shared and of innov
  out=$(MINT_KEEP_DRAWS=beta,nosuch ./build/dynpois 2>&1 >/dev/null)
  grep -q "MINT_KEEP_DRAWS names nosuch" <<<"$out" && grep -q "draws kept for 27 of 3171" <<<"$out" \
    && pass "MINT_KEEP_DRAWS keeps every draw of a named parameter and warns about an unknown name" \
    || { bad "MINT_KEEP_DRAWS=beta,nosuch"; echo "$out"; }
  # Peak memory must not grow with the number of draws when they are not
  # kept (200 against 1,600 draws per chain); with every draw kept it grows
  # by about the draws' size (1,400 x 4 x 3,171 doubles, 142 MB).
  for n in 200 1600; do
    sed "s/draws = 1000, warmup = 1000/draws = $n, warmup = 100/" examples/dynamic_poisson.mint > build/stream_mem$n.mint
    build build/stream_mem$n.mint stream_mem$n
  done
  python3 - <<'PY' && pass "streaming summary: peak memory does not grow with the number of draws" || bad "streaming summary: peak memory"
import os, subprocess, sys
def rss(b, env):
    p = subprocess.Popen(["./build/" + b], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, **env))
    _, st, ru = os.wait4(p.pid, 0)
    return ru.ru_maxrss / 1024 if st == 0 else float("nan")
s = [rss(f"stream_mem{n}", {}) for n in (200, 1600)]
f = [rss(f"stream_mem{n}", {"MINT_KEEP_DRAWS": "all"}) for n in (200, 1600)]
print(f"      peak MiB, 200 and 1,600 draws: streaming {s[0]:.0f}, {s[1]:.0f}; every draw kept {f[0]:.0f}, {f[1]:.0f}")
sys.exit(0 if s[1] - s[0] < 4 and f[1] - f[0] > 0.9 * 1400 * 4 * 3171 * 8 / 2**20 else 1)
PY
else
  bad "streaming summary: build/dynpois missing"
fi


# ---- Kalman collapse. A latent Gaussian random walk observed with Gaussian
# noise is integrated out; NUTS samples the rest, and the walk is drawn back
# by forward filtering, backward sampling (FFBS). Exact: on small panels
# (2 x 5, 3 x 7, 9 x 4) the compiled log density and gradient equal a dense
# Gaussian computation (the walk integrated out analytically in numpy) and
# pass finite differences, for eight models (check_marginal.py). The FFBS
# draws match the walk's exact Gaussian posterior, means and every
# covariance entry, for three models with fixed scales (check_ffbs.py), and
# the normal generator they use passes moment and binned-probability checks
# on 2e7 variates (zig_test.c). Statistical: the draws of every quantity
# (remaining parameters and every innovation) agree with full NUTS
# (--no-collapse) within Monte Carlo error, three seeds each, on four models
# (compare_posterior.py).
DEFAULT_FLAGS=
kal_py() { python3 "$@" $M build | sed 's/^/  /;s/^  PASS/PASS/;s/^  FAIL/FAIL/'; [ ${PIPESTATUS[0]} -eq 0 ] || fail=1; }
kal_py tests/kalman/check_marginal.py
kal_py tests/kalman/check_ffbs.py
kal_py tests/kalman/compare_posterior.py
if clang -O3 -march=native -fopenmp tests/kalman/zig_test.c -o build/zig_test -lm 2>/dev/null; then
  out=$(./build/zig_test) && pass "$out" || { bad "ziggurat normal generator"; echo "$out"; }
else
  bad "build tests/kalman/zig_test.c"
fi

# What is not collapsed, and why: the model must build, say why, and still
# sample as written (it runs to the end).
not_collapsed_src() { # label why source
  printf '%s\n' "$3" > build/kal_not_$1.mint
  out=$($M build build/kal_not_$1.mint -o build/kal_not_$1 2>&1) || { bad "kalman eligibility $1: build failed"; echo "$out"; return; }
  if grep -qF "collapsed innov" <<<"$out" || ! grep -qF -- "innov was not integrated out: $2" <<<"$out"; then
    bad "kalman eligibility $1"; echo "$out"; return
  fi
  run=$(./build/kal_not_$1 2>&1) && grep -q "^all .* parameters" <<<"$run" \
    && pass "kalman eligibility: $1 is sampled as written ($2)" || { bad "kalman eligibility $1: run failed"; echo "$run" | tail -3; }
}
not_collapsed() { # label why model-body
  not_collapsed_src "$1" "$2" "$(printf 'model W {\n    data y: Matrix[G, T]\n    param s: Positive\n    param innov: Matrix[G, T]\n    s ~ Normal(0, 1)\n%s\n}\nfn main() {\n    let y: Matrix[G, T] = read("bench/kalman/data_small/y.f64")\n    print(sample(W(y), draws = 10, warmup = 10, chains = 1))\n}' "$3")"
}
not_collapsed nonlinear "the observation's mean must be" '    innov ~ Normal(0, 0.1)
    y ~ Normal(exp(cumsum(innov, T)), s)'
not_collapsed squared "inside the running sum it must enter linearly" '    innov ~ Normal(0, 0.1)
    y ~ Normal(cumsum(innov .* innov, T), s)'
not_collapsed reused "it appears in 3 statements" '    innov ~ Normal(0, 0.1)
    y ~ Normal(cumsum(innov, T), s)
    y ~ Normal(innov, 1)'
not_collapsed_src poisson "its observation is PoissonLog, not Normal" 'model P {
    data y: Matrix[G, T]
    param b: Real
    param innov: Matrix[G, T]
    b ~ Normal(0, 1)
    innov ~ Normal(0, 0.1)
    y ~ PoissonLog(b + cumsum(innov, T))
}
fn main() {
    let y: Matrix[G, T] = read("build/scan_count_13.f64")
    print(sample(P(y), draws = 10, warmup = 10, chains = 1))
}'
not_collapsed prior_depends "its prior must be" '    innov ~ Normal(0.1 * cumsum(innov, T), 0.1)
    y ~ Normal(cumsum(innov, T), s)'
not_collapsed other_scan "the other terms contain a running sum" '    innov ~ Normal(0, 0.1)
    y ~ Normal(cumsum(innov, T) + cumsum(y, T), s)'
not_collapsed param_coef "the observation's mean must be (terms without it) + c * cumsum(...) with c a literal number" '    innov ~ Normal(0, 0.1)
    y ~ Normal(s * cumsum(innov, T), 1)'
# nothing would be left for NUTS (fixed scales): sampled as written
not_collapsed_src nothing_left "no other parameter would be left for NUTS to sample" 'model Z {
    data y: Matrix[G, T]
    param innov: Matrix[G, T]
    innov ~ Normal(0, 0.3)
    y ~ Normal(cumsum(innov, T), 0.5)
}
fn main() {
    let y: Matrix[G, T] = read("bench/kalman/data_small/y.f64")
    print(sample(Z(y), draws = 10, warmup = 10, chains = 1))
}'
# a scale that depends on the walk: only a vector can say so, since the
# checker proves exp(...) Positive for vectors, not matrices
not_collapsed_src scale "the observation's scale depends on it" 'model V {
    data y: Vector[T]
    param innov: Vector[T]
    innov ~ Normal(0, 0.1)
    y ~ Normal(cumsum(innov), exp(innov))
}
fn main() {
    let y: Vector[T] = read("data/linear_y.f64")
    print(sample(V(y), draws = 10, warmup = 10, chains = 1))
}'

# A collapse next to a fused scan kernel (tests/scan/twoowned.mint: w is
# integrated out; u, which shares its running sum, stays a NUTS parameter and
# now also gets gradient from the filter, after every kernel, so no kernel
# may own it): with the fused leapfrog, one fused leaf must match the
# runtime's, and the gradient must pass finite differences.
sed "s/NG/13/" tests/scan/twoowned.mint > build/kal_twoowned.mint
if build build/kal_twoowned.mint kal_twoowned --fused-leapfrog; then
  grep -qF "collapsed w (G x T latent scalars)" build/kal_twoowned.log && grep -qF "u was not integrated out" build/kal_twoowned.log \
    && pass "kalman next to a fused scan kernel: w collapsed, u kept" || { bad "kalman twoowned: collapse report"; cat build/kal_twoowned.log; }
  for t in 1 3; do
    out=$(MINT_KERNEL_THREADS=$t MINT_LEAP_TEST=1 ./build/kal_twoowned 2>&1)
    grep -q "leap-test: ok" <<<"$out" && pass "kalman next to a fused scan kernel: fused leaf matches ($t kernel threads)" \
      || { bad "kalman twoowned leap test ($t kernel threads)"; echo "$out" | tail -3; }
  done
  gradcheck kal_twoowned "kalman next to a fused scan kernel"
fi
build examples/random_walk_panel.mint rwp_full --no-collapse && ! grep -q "collapsed" build/rwp_full.log \
  && MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 ./build/rwp_full | python3 -c "
import sys; g = sys.stdin.read().split('grad:')[1].split(); sys.exit(0 if len(g) == 20 + 3 + 20 * 150 else 1)" \
  && pass "--no-collapse: no collapse, NUTS sees all 3,023 parameters" || bad "--no-collapse"
build examples/random_walk_panel.mint rwp && grep -qF "collapsed innov (G x T latent scalars)" build/rwp.log \
  && out=$(./build/rwp 2>&1) && grep -qF "NUTS sampled 23 of the 3023 parameters; 3000 latent scalars" <<<"$out" \
  && grep -qE "^innov +2997 more entries" <<<"$out" \
  && pass "random_walk_panel: NUTS samples 23 parameters, the 3,000 innovations come back as draws" \
  || { bad "random_walk_panel collapse"; cat build/rwp.log; echo "$out" | tail -5; }
# --strict-fp: the same log density and gradient to rounding; and the
# collapsed model sampled with 3 threads per chain gives finite draws of
# every quantity and the same posterior summary of sigma_w to 2 digits
build examples/random_walk_panel.mint rwp_strict --strict-fp \
  && a=$(MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 ./build/rwp | sed -n 's/^exact log density: //p; s/^grad://p' | tr '\n' ' ') \
  && b=$(MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 ./build/rwp_strict | sed -n 's/^exact log density: //p; s/^grad://p' | tr '\n' ' ') \
  && python3 -c "
import sys
a = [float(x) for x in sys.argv[1].split()]; b = [float(x) for x in sys.argv[2].split()]
m = max(abs(x) for x in b)
sys.exit(0 if len(a) == len(b) == 24 and all(abs(x - y) <= 1e-12 * max(m, 1) for x, y in zip(a, b)) else 1)" "$a" "$b" \
  && pass "random_walk_panel: --strict-fp gives the same log density and gradient to 1e-12" || bad "random_walk_panel --strict-fp"
out=$(MINT_THREADS_PER_CHAIN=3 MINT_DRAWS=build/rwp_t3.draws ./build/rwp 2>&1) && python3 -c "
import struct, sys
import numpy as np
f = open('build/rwp_t3.draws', 'rb'); c, n, d = struct.unpack('<QQQ', f.read(24))
x = np.frombuffer(f.read(), dtype='<f8').reshape(c * n, d)
sw = [l.split() for l in sys.argv[1].splitlines() if l.startswith('sigma_w ')][0]
sys.exit(0 if d == 3023 and np.all(np.isfinite(x)) and abs(float(sw[1]) - 0.0972) < 0.0015 else 1)" "$out" \
  && pass "random_walk_panel: 3 threads per chain, finite draws, same sigma_w" || { bad "random_walk_panel, 3 threads per chain"; echo "$out" | grep sigma_w; }

# Streaming summaries with the collapse: a draw holds the 23 values NUTS
# samples and the 3,000 innovations FFBS draws for it, and all of them are
# kept, summarised and written like any other parameter's. With 1 and 3
# threads per chain: draws are kept only for the printed rows (pop, the first
# 3 of beta, sigma_w, sigma_y, the first 3 of innov); the summary is the one
# computed with every draw kept; MINT_DRAWS writes the same 3,023 values per
# draw both ways (and into a pipe), the summarised ones; per value, the
# streaming statistics equal the draw-level ones; MINT_KEEP_DRAWS=innov keeps
# every innovation.
for t in 1 3; do
  rm -f build/kal_stream_a.draws build/kal_stream_b.draws
  MINT_THREADS_PER_CHAIN=$t MINT_DRAWS=build/kal_stream_a.draws ./build/rwp > build/kal_stream_a.out 2> build/kal_stream_a.err
  MINT_THREADS_PER_CHAIN=$t MINT_KEEP_DRAWS=all MINT_DRAWS=build/kal_stream_b.draws MINT_STATS_DUMP=build/kal_stream_b.tsv \
    ./build/rwp > build/kal_stream_b.out 2> /dev/null
  grep -q "summary: draws kept for 9 of 3023 parameters" build/kal_stream_a.err \
    && pass "kalman + streaming ($t threads per chain): draws kept only for the 9 printed rows" \
    || { bad "kalman + streaming ($t threads per chain): kept draws"; grep summary: build/kal_stream_a.err; }
  same_summary build/kal_stream_a.out build/kal_stream_b.out \
    && pass "kalman + streaming ($t threads per chain): same summary as with every draw kept" \
    || { bad "kalman + streaming ($t threads per chain): summary differs from MINT_KEEP_DRAWS=all"; diff build/kal_stream_a.out build/kal_stream_b.out; }
  [ "$(stat -c %s build/kal_stream_a.draws 2>/dev/null)" = $((24 + 4 * 1000 * 3023 * 8)) ] \
    && cmp -s build/kal_stream_a.draws build/kal_stream_b.draws \
    && pass "kalman + streaming ($t threads per chain): MINT_DRAWS writes all 3,023 values per draw, as with every draw kept" \
    || bad "kalman + streaming ($t threads per chain): MINT_DRAWS files"
  draws_match build/kal_stream_a.draws build/kal_stream_b.tsv \
    && pass "kalman + streaming ($t threads per chain): the MINT_DRAWS file holds the summarised draws" \
    || bad "kalman + streaming ($t threads per chain): the MINT_DRAWS file does not match the summary"
  stream_dump_check build/kal_stream_b.tsv 3023 "" 3023 \
    && pass "kalman + streaming ($t threads per chain): streaming statistics of every value, innovations included, equal the draw-level ones" \
    || bad "kalman + streaming ($t threads per chain): streaming statistics"
done
rm -f build/kal_stream_fifo build/kal_stream_p.draws
mkfifo build/kal_stream_fifo && { cat build/kal_stream_fifo > build/kal_stream_p.draws & }
MINT_THREADS_PER_CHAIN=3 MINT_DRAWS=build/kal_stream_fifo ./build/rwp > /dev/null 2>&1
wait
cmp -s build/kal_stream_p.draws build/kal_stream_a.draws \
  && pass "kalman + streaming: MINT_DRAWS into a pipe writes the same draws" || bad "kalman + streaming: MINT_DRAWS into a pipe"
out=$(MINT_KEEP_DRAWS=innov ./build/rwp 2>&1 >/dev/null)
grep -q "draws kept for 3006 of 3023" <<<"$out" \
  && pass "kalman + streaming: MINT_KEEP_DRAWS=innov keeps every draw of the collapsed parameter" \
  || { bad "kalman + streaming: MINT_KEEP_DRAWS=innov"; grep summary: <<<"$out"; }
rm -f build/kal_stream_fifo build/kal_stream_p.draws build/kal_stream_a.draws build/kal_stream_b.draws

exit $fail
