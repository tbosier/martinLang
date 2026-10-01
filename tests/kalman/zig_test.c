// Checks the ziggurat normal generator used for the collapsed draws
// (runtime/mint_rt.c, rng_normals) on 2e7 variates: mean, variance, the
// third and fourth moments, and the probability of each of 40 bins of
// width 0.25 on (-5, 5) and of the two tails beyond, against the exact
// normal probabilities (|z| of each against its binomial standard error).
// Build: clang -O3 -march=native -fopenmp tests/kalman/zig_test.c -lm
#include "../../runtime/mint_rt.c"

int main(void) {
  const int64_t n = 20000000, chunk = 100000;
  double *buf = malloc(chunk * sizeof(double));
  Rng r;
  rng_seed(&r, 12345);
  double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  int64_t bins[42] = {0};
  for (int64_t k = 0; k < n; k += chunk) {
    rng_normals(&r, buf, chunk);
    for (int64_t i = 0; i < chunk; i++) {
      double z = buf[i];
      s1 += z, s2 += z * z, s3 += z * z * z, s4 += z * z * z * z;
      int b = z < -5 ? 0 : z >= 5 ? 41 : 1 + (int)floor((z + 5) / 0.25);
      bins[b]++;
    }
  }
  double worst = 0;
  for (int b = 0; b < 42; b++) {
    double lo = b == 0 ? -INFINITY : -5 + 0.25 * (b - 1), hi = b == 41 ? INFINITY : -5 + 0.25 * b;
    double p = 0.5 * (erfc(-hi / sqrt(2)) - erfc(-lo / sqrt(2)));
    double z = (bins[b] - n * p) / sqrt(n * p * (1 - p));
    if (fabs(z) > worst) worst = fabs(z);
  }
  double m = s1 / n, v = s2 / n, m3 = s3 / n, m4 = s4 / n;
  // standard errors of the sample moments of a standard normal
  double zm = m / sqrt(1.0 / n), zv = (v - 1) / sqrt(2.0 / n), z3 = m3 / sqrt(15.0 / n), z4 = (m4 - 3) / sqrt(96.0 / n);
  printf("ziggurat: mean %.2e (z %.2f) var %.6f (z %.2f) third %.2e (z %.2f) fourth %.4f (z %.2f); worst bin |z| %.2f of 42\n",
         m, zm, v, zv, m3, z3, m4, z4, worst);
  int ok = fabs(zm) < 4.5 && fabs(zv) < 4.5 && fabs(z3) < 4.5 && fabs(z4) < 4.5 && worst < 4.5;
  return ok ? 0 : 1;
}
