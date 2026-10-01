# Sourced by tests/run.sh (uses its build, pass and bad functions).
# Narrow data: a model's vector kernels may read a narrow copy (int8, int16
# or float) of a data buffer whose values that type holds exactly, chosen
# when sample() starts. The results must not change at all: the log density
# and every gradient component at the benchmark point must be byte-identical
# to the build without the rewrite (--no-narrow-data), whichever copy the
# runtime picks (MINT_NARROW unset, int16, float, or 0 for none), on one and
# on three kernel threads, and so must the raw draws of short sampling runs.
# The runtime's report (MINT_NARROW_REPORT) must show the expected choice,
# so each case is known to have run the narrow code it is meant to test.

python3 tests/fission/make_data.py build
python3 tests/scan/make_data.py build
python3 tests/narrow/make_data.py build

nw_out() { # BIN OUT [VAR=value...]: log density and gradient without the timing
  local bin=$1 out=$2; shift 2
  env "$@" MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 ./build/$bin 2>/dev/null | sed 's/ns_per_eval=[0-9.]*//' > "$out"
}

# narrow_case NAME SOURCE THREADS EXPECTED...: EXPECTED entries are report
# lines without the model, e.g. "y: int8", or - for no narrow copy at all
narrow_case() {
  local n=$1 src=$2 nt=$3; shift 3
  build "$src" nw_$n || return
  build "$src" nw_${n}_ref --no-narrow-data || return
  nw_out nw_${n}_ref build/nw_${n}_ref.out MINT_KERNEL_THREADS=1
  [ -s build/nw_${n}_ref.out ] || { bad "narrow $n: no output"; return; }
  local ok=1 mode t
  for t in 1 $nt; do
    [ $t = 1 ] || nw_out nw_${n}_ref build/nw_${n}_ref.out MINT_KERNEL_THREADS=$t
    for mode in "" int16 float 0; do
      nw_out nw_$n build/nw_${n}_$t$mode.out MINT_KERNEL_THREADS=$t MINT_NARROW=$mode
      cmp -s build/nw_${n}_$t$mode.out build/nw_${n}_ref.out || { ok=0; echo "      differs: threads=$t MINT_NARROW=$mode"; }
    done
  done
  [ $ok = 1 ] && pass "narrow data $n: log density and gradient identical to --no-narrow-data" \
    || bad "narrow data $n: log density or gradient differs from --no-narrow-data"
  local rep e
  rep=$(MINT_NARROW_REPORT=1 MINT_BENCH_GRAD=1 ./build/nw_$n 2>&1 >/dev/null | grep '^narrow:')
  for e in "$@"; do
    if [ "$e" = - ]; then # no copy at all
      [ -z "$rep" ] && pass "narrow data $n: no narrow copy" || bad "narrow data $n: unexpected copy: $(tr '\n' ';' <<<"$rep")"
      continue
    fi
    grep -qF "data $e" <<<"$rep" && pass "narrow data $n: chooses $e" || { bad "narrow data $n: expected $e, got: $(tr '\n' ';' <<<"$rep")"; }
  done
}

# fission kernel models (n = 1003, p = 13: masked column tails, leftover
# groups of four and scalar rows), with the original data (only the 0/1 and
# count vectors are exact in a narrow type) and with every value rounded to
# float; a BernoulliLogit outcome takes int8 whatever else happens
for m in logit poisson normal expo funcs datamv; do
  cp tests/fission/$m.mint build/nw_fis_$m.mint
  sed 's#\(build/fis_[A-Za-z0-9]*\)\.f64#\1_f32.f64#g' tests/fission/$m.mint > build/nw_fis_${m}32.mint
done
narrow_case fis_logit build/nw_fis_logit.mint 1 "X: double" "y: int8"
narrow_case fis_logit32 build/nw_fis_logit32.mint 1 "X: float" "y: int8"
narrow_case fis_poisson build/nw_fis_poisson.mint 1 "X: double" "c: int8"
narrow_case fis_poisson32 build/nw_fis_poisson32.mint 1 "X: float" "c: int8"
narrow_case fis_normal build/nw_fis_normal.mint 1 "X: double" "z: double"
narrow_case fis_normal32 build/nw_fis_normal32.mint 1 "X: float" "z: float"
narrow_case fis_expo32 build/nw_fis_expo32.mint 1 "X: float" "w: float"
narrow_case fis_funcs32 build/nw_fis_funcs32.mint 1 "X: float" "y: int8"
narrow_case fis_datamv32 build/nw_fis_datamv32.mint 1 "X: float" "y: int8"
# the boundaries of each type. With X a candidate too, the limit on variants
# leaves the counts only int8 (see narrow_candidates), so anything int8 cannot
# hold keeps them double.
sed "s#build/fis_c_f32.f64#build/fis_c_i16.f64#" build/nw_fis_poisson32.mint > build/nw_fis_c_i16.mint
sed "s#build/fis_c_f32.f64#build/fis_c_big.f64#" build/nw_fis_poisson32.mint > build/nw_fis_c_big.mint
sed "s#build/fis_c_f32.f64#build/fis_c_negz.f64#" build/nw_fis_poisson32.mint > build/nw_fis_c_negz.mint
sed 's#build/fis_z_f32.f64#build/fis_z_edge.f64#' build/nw_fis_normal32.mint > build/nw_fis_z_edge.mint
sed 's#build/fis_z_f32.f64#build/fis_z_wide.f64#' build/nw_fis_normal32.mint > build/nw_fis_z_wide.mint
narrow_case fis_c_i16 build/nw_fis_c_i16.mint 1 "c: double"
narrow_case fis_c_big build/nw_fis_c_big.mint 1 "c: double"
narrow_case fis_c_negz build/nw_fis_c_negz.mint 1 "c: double"
narrow_case fis_z_edge build/nw_fis_z_edge.mint 1 "z: float"
narrow_case fis_z_wide build/nw_fis_z_wide.mint 1 "z: double"

# fused scan kernels, G = 13 (one group of 8, one vector, one leftover row)
# and 61 (seven groups split across 3 threads by the parallel kernel)
for m in nested mixed nested_sq colreuse datascan twohosts bernoulli; do
  for G in 13 61; do
    sed "s/NG/$G/" tests/scan/$m.mint > build/nw_scan_${m}_$G.mint
    sed "s/NG/$G/; s#\(build/scan_[a-z]*_[0-9]*\)\.f64#\1_f32.f64#g" tests/scan/$m.mint > build/nw_scan_${m}32_$G.mint
  done
done
for G in 13 61; do
  narrow_case scan_nested_$G build/nw_scan_nested_$G.mint 3 "y: double"
  narrow_case scan_nested32_$G build/nw_scan_nested32_$G.mint 3 "y: float"
  narrow_case scan_mixed_$G build/nw_scan_mixed_$G.mint 3 "y: int8"
  narrow_case scan_nested_sq32_$G build/nw_scan_nested_sq32_$G.mint 3 "y: float"
  narrow_case scan_colreuse32_$G build/nw_scan_colreuse32_$G.mint 3 "y: float"
  # a running sum of data only is not a fused kernel: nothing to narrow
  narrow_case scan_datascan32_$G build/nw_scan_datascan32_$G.mint 3 -
  narrow_case scan_twohosts32_$G build/nw_scan_twohosts32_$G.mint 3 "y: float"
done
# a count panel that is the model's only candidate: every type is available
for v in i16 big negz huge; do
  sed "s#build/scan_count_61.f64#build/scan_count_61_$v.f64#" build/nw_scan_mixed_61.mint > build/nw_scan_mixed_$v.mint
done
narrow_case scan_mixed_i16 build/nw_scan_mixed_i16.mint 3 "y: int16"
narrow_case scan_mixed_big build/nw_scan_mixed_big.mint 3 "y: float"
narrow_case scan_mixed_negz build/nw_scan_mixed_negz.mint 3 "y: float"
narrow_case scan_mixed_huge build/nw_scan_mixed_huge.mint 3 "y: double"
# the one-lane BernoulliLogit scan kernel loads no data in vector code:
# no narrow copy, no variants
build build/nw_scan_bernoulli_13.mint nw_scan_bern && ! grep -q mint_narrow build/nw_scan_bern.ll \
  && pass "narrow data: no copy for a kernel without vector code" || bad "narrow data: copy made for a one-lane kernel"

if [ -f bench/dynpois/data_small/y.f64 ]; then
  narrow_case dynpois examples/dynamic_poisson.mint 3 "y: int8"
  sed 's#data_small#data_large#' examples/dynamic_poisson.mint > build/nw_dynpois_large.mint
  narrow_case dynpois_large build/nw_dynpois_large.mint 3 "y: int8"
fi
narrow_case logistic examples/logistic_bayes.mint 1 "X: double" "y: int8"

# integer copies of a design matrix (allowed by raising the variant limit):
# the dot products' and gradient updates' masked column tails (p = 13) load
# int8 and int16
sed 's#build/fis_Xbig.f64#build/fis_X_i8.f64#' tests/fission/logit.mint > build/nw_fis_xi8.mint
sed 's#build/fis_Xbig.f64#build/fis_X_i16.f64#' tests/fission/logit.mint > build/nw_fis_xi16.mint
MINTC_NARROW_VARIANTS=16 narrow_case fis_xi8 build/nw_fis_xi8.mint 1 "X: int8" "y: int8"
MINTC_NARROW_VARIANTS=16 narrow_case fis_xi16 build/nw_fis_xi16.mint 1 "X: int16" "y: int8"
grep -q "masked.load.v4i8" build/nw_fis_xi8.ll && grep -q "masked.load.v4i16" build/nw_fis_xi16.ll \
  && pass "narrow data: integer masked loads emitted" || bad "narrow data: integer masked loads missing"

# many candidates: the limit on variants must still hold (a product of
# 4^32 once wrapped to 0 and left a variable undefined)
{
  echo "model Many {"
  echo "    data X: Matrix[n, p]"
  echo "    data z: Vector[n]"
  for k in $(seq 0 31); do echo "    data x$k: Vector[n]"; done
  echo "    param b: Vector[p]"
  echo "    param s: Vector[p]"
  echo "    b ~ Normal(0, 1)"
  echo "    s ~ Normal(0, 0.3)"
  echo "    z ~ Normal(X * b $(for k in $(seq 0 31); do printf '+ 0.01 * x%d ' $k; done), exp(X * s))"
  echo "}"
  echo "fn main() {"
  echo '    let X: Matrix[n, p] = read("build/fis_X_f32.f64")'
  echo '    let z: Vector[n] = read("build/fis_z_f32.f64")'
  for k in $(seq 0 31); do echo "    let x$k: Vector[n] = read(\"build/fis_z_f32.f64\")"; done
  echo "    let post = sample(Many(X, z $(for k in $(seq 0 31); do printf ', x%d' $k; done)), draws = 4, warmup = 0, chains = 1, seed = 1)"
  echo "    print(post)"
  echo "}"
} > build/nw_many.mint
narrow_case many build/nw_many.mint 1 "X: float" "z: float"
[ "$(grep -c '^define double @mint_model_Many_logp' build/nw_many.ll)" = 4 ] \
  && pass "narrow data: at most 4 variants of logp" || bad "narrow data: variant limit exceeded"

# random models and data: narrow against wide, including raw draws
python3 tests/narrow/fuzz.py $M 12 7 && pass "narrow data: randomised models and data" || bad "narrow data: randomised check"
# and with every type available to every buffer (integer design matrices)
MINTC_NARROW_VARIANTS=64 python3 tests/narrow/fuzz.py $M 12 8 && pass "narrow data: randomised models and data, all types" || bad "narrow data: randomised check, all types"
# and with the adjoint sums starting at 0.0
python3 tests/narrow/fuzz.py $M 8 9 --no-negzero-sums && pass "narrow data: randomised models and data, --no-negzero-sums" || bad "narrow data: randomised check, --no-negzero-sums"

# the switch: no copies and no variants in the IR (with --no-negzero-sums
# too, the IR of the benchmark models was checked by hand to equal that of
# the compiler before this change); and none under --strict-fp
grep -q "call ptr @mint_narrow" build/nw_logistic.ll && ! grep -q "mint_narrow\|logp_n1" build/nw_logistic_ref.ll \
  && pass "narrow data: --no-narrow-data removes the copies and variants" || bad "narrow data: --no-narrow-data"
build examples/dynamic_poisson.mint nw_strict --strict-fp && ! grep -q "mint_narrow" build/nw_strict.ll \
  && pass "narrow data: off under --strict-fp" || bad "narrow data: copies made under --strict-fp"

# raw draws of short sampling runs, narrow against wide
nw_draws() { # NAME SOURCE [VAR=value...]
  local n=$1 src=$2; shift 2
  build "$src" nwd_$n && build "$src" nwd_${n}_ref --no-narrow-data || return
  rm -f build/nwd_$n.draws build/nwd_${n}_ref.draws
  env "$@" MINT_DRAWS=build/nwd_$n.draws ./build/nwd_$n > /dev/null 2>&1 || { bad "narrow data $n: sampling run failed"; return; }
  env "$@" MINT_DRAWS=build/nwd_${n}_ref.draws ./build/nwd_${n}_ref > /dev/null 2>&1 || { bad "narrow data $n: reference run failed"; return; }
  [ -s build/nwd_$n.draws ] && cmp -s build/nwd_$n.draws build/nwd_${n}_ref.draws \
    && pass "narrow data $n: identical raw draws" || bad "narrow data $n: raw draws differ"
}
sed 's/draws = 1000, warmup = 1000/draws = 40, warmup = 40/' examples/dynamic_poisson.mint > build/nwd_dynpois.mint
[ -f bench/dynpois/data_small/y.f64 ] && nw_draws dynpois build/nwd_dynpois.mint
sed 's#data_small#data_large#; s/draws = 1000, warmup = 1000/draws = 10, warmup = 10/' examples/dynamic_poisson.mint > build/nwd_dynpois_large.mint
[ -f bench/dynpois/data_large/y.f64 ] && nw_draws dynpois_large build/nwd_dynpois_large.mint MINT_THREADS_PER_CHAIN=3
sed 's/draws = 1000, warmup = 1000/draws = 100, warmup = 100/' examples/logistic_bayes.mint > build/nwd_logistic.mint
nw_draws logistic build/nwd_logistic.mint
sed 's/draws = 4, warmup = 0/draws = 50, warmup = 50/' build/nw_fis_poisson32.mint > build/nwd_poisson32.mint
nw_draws poisson32 build/nwd_poisson32.mint
