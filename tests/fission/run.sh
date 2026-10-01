# Sourced by tests/run.sh (uses its build, pass and bad functions).
# Fission kernel: every path (row counts and column counts that are not
# multiples of the chunk or of 4, |eta| beyond exp's fast range, other
# densities, functions inside the linear predictor, two products, a product
# of data only, a vector parameter indexed by observation, a Positive vector
# parameter, and the log1p fallback) must give the same log density and
# gradient as the three-pass fission (--no-fission-kernel) and as the
# unsplit loop (--no-fission) at the benchmark point, and pass the
# finite-difference check (see below for the points used).

python3 tests/fission/make_data.py build
for m in logit poisson normal expo funcs datamv log1p; do
  build tests/fission/$m.mint fis_${m}_opt || continue
  build tests/fission/$m.mint fis_${m}_ref --no-fission-kernel || continue
  build tests/fission/$m.mint fis_${m}_nof --no-fission || continue
  for v in opt ref nof; do MINT_BENCH_GRAD=1 MINT_PRINT_GRAD=1 ./build/fis_${m}_$v > build/fis_${m}_$v.out; done
  python3 tests/fission/compare.py build/fis_${m}_opt.out build/fis_${m}_ref.out \
    && pass "fission kernel $m matches the three-pass fission" || bad "fission kernel $m differs from the three-pass fission"
  python3 tests/fission/compare.py build/fis_${m}_opt.out build/fis_${m}_nof.out \
    && pass "fission kernel $m matches the unsplit loop" || bad "fission kernel $m differs from the unsplit loop"
  # finite differences at the benchmark point and at the runtime's random
  # point, except Poisson at the random point: its data has rows of 1e5, so
  # eta reaches about 2e5 there and exp(eta) overflows in every build
  npts=2
  [ $m = poisson ] && npts=1
  worst=$(MINT_GRADCHECK=1 MINT_BENCH_GRAD=1 ./build/fis_${m}_opt 2>&1 | grep gradcheck | head -$npts | sed 's/.*error=//' | tr '\n' ' ')
  python3 -c "
import sys
w = [float(x) for x in '$worst'.split()]
sys.exit(0 if len(w) == $npts and all(x < 1e-5 for x in w) else 1)" && pass "gradcheck fission kernel $m ($worst)" || bad "gradcheck fission kernel $m ($worst)"
done
