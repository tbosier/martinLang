// Mint runtime: memory, binary I/O, printing, Cholesky solve, RNG and a NUTS sampler.
//
// Generated programs call into this file for everything that is not model or
// numeric-kernel code. The Rust baselines link the same object so that the
// sampler is identical on both sides of the benchmark.

#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// ---------------------------------------------------------------- basics

void mint_panic(const char *msg) {
  fprintf(stderr, "mint runtime error: %s\n", msg);
  exit(1);
}

// Fresh, 64-byte aligned (one cache line) memory; released with free().
void *mint_alloc(int64_t n_doubles) {
  if (n_doubles < 0) mint_panic("negative allocation size");
  size_t bytes = (size_t)(n_doubles > 0 ? n_doubles : 1) * sizeof(double);
  bytes = (bytes + 63) & ~(size_t)63;
  void *p = aligned_alloc(64, bytes);
  if (!p) mint_panic("out of memory");
  return p;
}

void mint_free(void *p) { free(p); }

double mint_clock(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void mint_check_dim(int64_t got, int64_t want, const char *what) {
  if (got != want) {
    fprintf(stderr, "mint runtime error: %s: expected size %lld, got %lld\n", what,
            (long long)want, (long long)got);
    exit(1);
  }
}

// ---------------------------------------------------------------- binary I/O
// File format (".f64"): two little-endian u64 (rows, cols) then rows*cols
// little-endian f64 in row-major order. A vector is stored with cols == 1.

static double *read_f64(const char *path, int64_t *rows, int64_t *cols) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "mint runtime error: cannot open '%s'\n", path);
    exit(1);
  }
  uint64_t hdr[2];
  if (fread(hdr, sizeof(uint64_t), 2, f) != 2) mint_panic("truncated header");
  int64_t n = (int64_t)(hdr[0] * hdr[1]);
  double *data = mint_alloc(n);
  if ((int64_t)fread(data, sizeof(double), (size_t)n, f) != n) {
    fprintf(stderr, "mint runtime error: '%s' is shorter than its header says\n", path);
    exit(1);
  }
  fclose(f);
  *rows = (int64_t)hdr[0];
  *cols = (int64_t)hdr[1];
  return data;
}

double *mint_read_matrix(const char *path, int64_t *rows, int64_t *cols) {
  return read_f64(path, rows, cols);
}

double *mint_read_vector(const char *path, int64_t *len) {
  int64_t r, c;
  double *d = read_f64(path, &r, &c);
  if (c != 1) {
    fprintf(stderr, "mint runtime error: '%s' holds a %lldx%lld matrix, not a vector\n", path,
            (long long)r, (long long)c);
    exit(1);
  }
  *len = r;
  return d;
}

// ---------------------------------------------------------------- printing

void mint_print_str(const char *s) { fputs(s, stdout); }
void mint_print_sep(void) { fputc(' ', stdout); }
void mint_print_newline(void) { fputc('\n', stdout); }
void mint_print_f64(double x) { printf("%.10g", x); }

void mint_print_vec(const double *v, int64_t n) {
  fputc('[', stdout);
  for (int64_t i = 0; i < n; i++) printf(i ? ", %.10g" : "%.10g", v[i]);
  fputc(']', stdout);
}

void mint_print_mat(const double *m, int64_t r, int64_t c) {
  fputc('[', stdout);
  for (int64_t i = 0; i < r; i++) {
    if (i) fputs(",\n ", stdout);
    mint_print_vec(m + i * c, c);
  }
  fputc(']', stdout);
}

// ---------------------------------------------------------------- linear algebra

// Solves H x = g for symmetric positive definite H (row-major, p x p) by
// Cholesky factorisation. The type checker only lets this be called on a
// matrix it has proved SPD (or one the program asserted with assume_spd), so
// failure here means floating-point breakdown or a false assumption.
static __thread double *chol_buf;
static __thread int64_t chol_cap;

void mint_chol_solve(const double *H, int64_t p, const double *g, double *out, const char *what) {
  if (p * p > chol_cap) {  // grown once, reused, so solves inside loops do not allocate
    free(chol_buf);
    chol_buf = mint_alloc(p * p);
    chol_cap = p * p;
  }
  double *L = chol_buf;
  for (int64_t j = 0; j < p; j++) {
    for (int64_t i = j; i < p; i++) {
      double s = H[i * p + j];
      for (int64_t k = 0; k < j; k++) s -= L[i * p + k] * L[j * p + k];
      if (i == j) {
        if (!(s > 0.0)) {
          fprintf(stderr,
                  "mint runtime error: Cholesky failed at pivot %lld (value %g) in %s; the "
                  "matrix is not numerically positive definite\n",
                  (long long)j, s, what);
          exit(1);
        }
        L[j * p + j] = sqrt(s);
      } else {
        L[i * p + j] = s / L[j * p + j];
      }
    }
  }
  for (int64_t i = 0; i < p; i++) {  // L y = g
    double s = g[i];
    for (int64_t k = 0; k < i; k++) s -= L[i * p + k] * out[k];
    out[i] = s / L[i * p + i];
  }
  for (int64_t i = p - 1; i >= 0; i--) {  // L' x = y
    double s = out[i];
    for (int64_t k = i + 1; k < p; k++) s -= L[k * p + i] * out[k];
    out[i] = s / L[i * p + i];
  }
}

// assume_spd(A): checks symmetry (to a relative tolerance) and positive
// definiteness (a Cholesky factorisation) before the value is used.
void mint_check_spd(const double *A, int64_t p, const char *what) {
  double scale = 0;
  for (int64_t i = 0; i < p * p; i++) {
    if (!isfinite(A[i])) {
      fprintf(stderr, "mint runtime error: %s: matrix entry %lld is %g\n", what, (long long)(i + 1), A[i]);
      exit(1);
    }
    scale = fmax(scale, fabs(A[i]));
  }
  for (int64_t i = 0; i < p; i++)
    for (int64_t j = 0; j < i; j++)
      if (fabs(A[i * p + j] - A[j * p + i]) > 1e-12 * scale) {
        fprintf(stderr, "mint runtime error: %s: matrix is not symmetric (entries [%lld,%lld] and [%lld,%lld] differ)\n",
                what, (long long)(i + 1), (long long)(j + 1), (long long)(j + 1), (long long)(i + 1));
        exit(1);
      }
  static __thread double *buf;  // reused, so checks inside loops do not allocate
  static __thread int64_t cap;
  if (2 * p > cap) {
    free(buf);
    buf = mint_alloc(2 * p);
    cap = 2 * p;
  }
  for (int64_t i = 0; i < p; i++) buf[i] = 1.0;
  mint_chol_solve(A, p, buf, buf + p, what);  // exits with a message if not positive definite
}

// ---------------------------------------------------------------- RNG
// xoshiro256++ seeded through splitmix64.

typedef struct {
  uint64_t s[4];
} Rng;

static uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

static void rng_seed(Rng *r, uint64_t seed) {
  for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&seed);
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t rng_next(Rng *r) {
  uint64_t *s = r->s;
  uint64_t result = rotl(s[0] + s[3], 23) + s[0];
  uint64_t t = s[1] << 17;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl(s[3], 45);
  return result;
}

static double rng_uniform(Rng *r) { return (double)(rng_next(r) >> 11) * 0x1.0p-53; }

static double rng_normal(Rng *r) {
  // Marsaglia polar method; the second variate is discarded to keep the
  // stream easy to reproduce.
  for (;;) {
    double u = 2.0 * rng_uniform(r) - 1.0, v = 2.0 * rng_uniform(r) - 1.0;
    double s = u * u + v * v;
    if (s > 0.0 && s < 1.0) return u * sqrt(-2.0 * log(s) / s);
  }
}

// ---------------------------------------------------------------- dense helpers
// Used only by the low-rank metric adaptation (MINT_METRIC=lowrank), once per
// warmup window, on matrices of at most a few hundred rows.

// Symmetric eigendecomposition: Householder tridiagonalisation and the
// implicit QL algorithm (tred2 and tql2, as in EISPACK and JAMA). A is n x n,
// row-major; on return its columns are the eigenvectors and w holds the
// eigenvalues in ascending order.
static void sym_eig(double *A, int n, double *w) {
#define V_(i, j) A[(int64_t)(i) * n + (j)]
  double *d = w, *e = calloc((size_t)n, sizeof(double));
  for (int j = 0; j < n; j++) d[j] = V_(n - 1, j);
  for (int i = n - 1; i > 0; i--) {
    double scale = 0.0, h = 0.0;
    for (int k = 0; k < i; k++) scale += fabs(d[k]);
    if (scale == 0.0) {
      e[i] = d[i - 1];
      for (int j = 0; j < i; j++) {
        d[j] = V_(i - 1, j);
        V_(i, j) = 0.0;
        V_(j, i) = 0.0;
      }
    } else {
      for (int k = 0; k < i; k++) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      double f = d[i - 1], g = sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g;
      h -= f * g;
      d[i - 1] = f - g;
      for (int j = 0; j < i; j++) e[j] = 0.0;
      for (int j = 0; j < i; j++) {
        f = d[j];
        V_(j, i) = f;
        g = e[j] + V_(j, j) * f;
        for (int k = j + 1; k <= i - 1; k++) {
          g += V_(k, j) * d[k];
          e[k] += V_(k, j) * f;
        }
        e[j] = g;
      }
      f = 0.0;
      for (int j = 0; j < i; j++) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      double hh = f / (h + h);
      for (int j = 0; j < i; j++) e[j] -= hh * d[j];
      for (int j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (int k = j; k <= i - 1; k++) V_(k, j) -= (f * e[k] + g * d[k]);
        d[j] = V_(i - 1, j);
        V_(i, j) = 0.0;
      }
    }
    d[i] = h;
  }
  for (int i = 0; i < n - 1; i++) {
    V_(n - 1, i) = V_(i, i);
    V_(i, i) = 1.0;
    double h = d[i + 1];
    if (h != 0.0) {
      for (int k = 0; k <= i; k++) d[k] = V_(k, i + 1) / h;
      for (int j = 0; j <= i; j++) {
        double g = 0.0;
        for (int k = 0; k <= i; k++) g += V_(k, i + 1) * V_(k, j);
        for (int k = 0; k <= i; k++) V_(k, j) -= g * d[k];
      }
    }
    for (int k = 0; k <= i; k++) V_(k, i + 1) = 0.0;
  }
  for (int j = 0; j < n; j++) {
    d[j] = V_(n - 1, j);
    V_(n - 1, j) = 0.0;
  }
  V_(n - 1, n - 1) = 1.0;
  e[0] = 0.0;

  // The QL iterations rotate pairs of eigenvector columns; work on the
  // transpose so that those rotations run along contiguous rows.
  for (int i = 0; i < n; i++)
    for (int j = 0; j < i; j++) {
      double t = V_(i, j);
      V_(i, j) = V_(j, i);
      V_(j, i) = t;
    }

  for (int i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0.0;
  double f = 0.0, tst1 = 0.0;
  const double eps = 0x1.0p-52;
  for (int l = 0; l < n; l++) {
    tst1 = fmax(tst1, fabs(d[l]) + fabs(e[l]));
    int m = l;
    while (m < n) {
      if (fabs(e[m]) <= eps * tst1) break;
      m++;
    }
    if (m == n) m = n - 1;
    if (m > l) {
      int iter = 0;
      do {
        if (++iter > 100) mint_panic("eigendecomposition did not converge");
        double g = d[l];
        double p = (d[l + 1] - g) / (2.0 * e[l]);
        double r = hypot(p, 1.0);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        double dl1 = d[l + 1];
        double h = g - d[l];
        for (int i = l + 2; i < n; i++) d[i] -= h;
        f += h;
        p = d[m];
        double c = 1.0, c2 = c, c3 = c, el1 = e[l + 1], s = 0.0, s2 = 0.0;
        for (int i = m - 1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = hypot(p, e[i]);
          e[i + 1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          double *restrict r0 = A + (int64_t)i * n, *restrict r1 = A + (int64_t)(i + 1) * n;
          for (int k = 0; k < n; k++) {
            double a0 = r0[k], a1 = r1[k];
            r1[k] = s * a0 + c * a1;
            r0[k] = c * a0 - s * a1;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (fabs(e[l]) > eps * tst1);
    }
    d[l] += f;
    e[l] = 0.0;
  }
  for (int i = 0; i < n - 1; i++) {
    int k = i;
    double p = d[i];
    for (int j = i + 1; j < n; j++)
      if (d[j] < p) k = j, p = d[j];
    if (k != i) {
      d[k] = d[i];
      d[i] = p;
      for (int j = 0; j < n; j++) {
        p = V_(i, j);
        V_(i, j) = V_(k, j);
        V_(k, j) = p;
      }
    }
  }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < i; j++) {
      double t = V_(i, j);
      V_(i, j) = V_(j, i);
      V_(j, i) = t;
    }
  free(e);
#undef V_
}

// Orthonormalises the k rows of U (each of length D) in place, modified
// Gram-Schmidt applied twice. Returns the number of rows kept (rows that
// become numerically zero are dropped and the rest moved up).
static int orthonormalise(double *U, int k, int64_t D) {
  int out = 0;
  for (int j = 0; j < k; j++) {
    double *u = U + (int64_t)j * D;
    double n0 = 0;
    for (int64_t i = 0; i < D; i++) n0 += u[i] * u[i];
    for (int pass = 0; pass < 2; pass++)
      for (int m = 0; m < out; m++) {
        const double *v = U + (int64_t)m * D;
        double dot = 0;
        for (int64_t i = 0; i < D; i++) dot += u[i] * v[i];
        for (int64_t i = 0; i < D; i++) u[i] -= dot * v[i];
      }
    double nrm = 0;
    for (int64_t i = 0; i < D; i++) nrm += u[i] * u[i];
    if (!(nrm > 1e-20 * n0) || !(nrm > 0)) continue;
    nrm = 1.0 / sqrt(nrm);
    double *dst = U + (int64_t)out * D;
    for (int64_t i = 0; i < D; i++) dst[i] = u[i] * nrm;
    out++;
  }
  return out;
}

// C = A' B / n for A (n x a) and B (n x b), row-major; C is a x b.
static void gram_at_b(const double *A, int a, const double *B, int b, int n, double *C) {
  for (int i = 0; i < a; i++)
    for (int j = 0; j < b; j++) {
      double s = 0;
      for (int r = 0; r < n; r++) s += A[(int64_t)r * a + i] * B[(int64_t)r * b + j];
      C[i * b + j] = s / n;
    }
}

// R = V diag(f(w)) V' for a symmetric eigendecomposition (V columns), r x r.
static void eig_apply(const double *V, const double *fw, int r, double *R) {
  for (int i = 0; i < r; i++)
    for (int j = 0; j < r; j++) {
      double s = 0;
      for (int m = 0; m < r; m++) s += V[i * r + m] * fw[m] * V[j * r + m];
      R[i * r + j] = s;
    }
}

static void matmul_sq(const double *A, const double *B, int r, double *C) {
  for (int i = 0; i < r; i++)
    for (int j = 0; j < r; j++) {
      double s = 0;
      for (int m = 0; m < r; m++) s += A[i * r + m] * B[m * r + j];
      C[i * r + j] = s;
    }
}

// Low-rank correction to a diagonal metric, estimated from n warmup draws Q
// and their log-density gradients G (each n x D, row-major; both are
// overwritten). im is the diagonal inverse metric already chosen for the
// window, S = sqrt(im) its scale.
//
// In the scaled coordinates x = q / S the gradients are y = S g. For a
// Gaussian posterior with covariance C (scaled), cov(y) = C^-1, so the
// leading eigenvectors of cov(y) are the directions in which the posterior
// is much narrower than the diagonal metric assumes; those set the step size.
// Within the span of the r = 2 kmax leading ones, the covariance is
// estimated by the SPD geometric mean Sigma solving
// Sigma cov(y) Sigma = cov(x) (as nutpie's low-rank adaptation does), which
// is exact for a Gaussian's true covariances in a span the precision maps to
// itself, and an estimate with sample covariances. Its
// eigenvalues lam that lie outside [1/cutoff, cutoff] are kept, at most kmax
// of them, ordered by |log lam|.
//
// Writes the kept directions as orthonormal rows of U (k x D, in scaled
// coordinates) and their variances lam; returns k.
static int lowrank_estimate(int64_t D, int n, double *Q, double *G, const double *im, int kmax, double cutoff,
                            double gamma, double *U, double *lam) {
  if (n < 3 || kmax < 1) return 0;
  double *qm = calloc((size_t)D, sizeof(double)), *gm = calloc((size_t)D, sizeof(double));
  for (int r = 0; r < n; r++)
    for (int64_t i = 0; i < D; i++) qm[i] += Q[(int64_t)r * D + i], gm[i] += G[(int64_t)r * D + i];
  for (int64_t i = 0; i < D; i++) qm[i] /= n, gm[i] /= n;
  for (int r = 0; r < n; r++)
    for (int64_t i = 0; i < D; i++) {
      double sc = sqrt(im[i]);
      Q[(int64_t)r * D + i] = (Q[(int64_t)r * D + i] - qm[i]) / sc;
      G[(int64_t)r * D + i] = (G[(int64_t)r * D + i] - gm[i]) * sc;
    }
  free(qm), free(gm);
  // Leading eigenvectors of cov(y) = Y' Y / n: directly when D <= n,
  // otherwise from the Gram matrix Y Y' / n (n x n), whose eigenvectors give
  // those of Y' Y / n without forming a D x D matrix.
  int r = 0;
  int rmax = 2 * kmax < n - 1 ? 2 * kmax : n - 1;  // candidates; at most kmax are kept
  if (rmax > D) rmax = (int)D;
  double *U0 = calloc((size_t)rmax * (size_t)D, sizeof(double));
  if (D <= n) {
    int d = (int)D;
    double *K = malloc(sizeof(double) * d * d), *w = malloc(sizeof(double) * d);
    for (int i = 0; i < d; i++)
      for (int j = 0; j <= i; j++) {
        double sum = 0;
        for (int a = 0; a < n; a++) sum += G[(int64_t)a * D + i] * G[(int64_t)a * D + j];
        K[i * d + j] = K[j * d + i] = sum / n;
      }
    sym_eig(K, d, w);
    for (int j = d - 1; j >= 0 && r < rmax; j--) {
      if (!(w[j] > 1e-12 * w[d - 1])) break;
      for (int i = 0; i < d; i++) U0[(int64_t)r * D + i] = K[i * d + j];
      r++;
    }
    free(K), free(w);
  } else {
    double *K = malloc(sizeof(double) * n * n), *w = malloc(sizeof(double) * n);
    for (int a = 0; a < n; a++)
      for (int b = 0; b <= a; b++) {
        const double *ya = G + (int64_t)a * D, *yb = G + (int64_t)b * D;
        double sum = 0;
#pragma omp simd reduction(+ : sum)
        for (int64_t i = 0; i < D; i++) sum += ya[i] * yb[i];
        K[a * n + b] = K[b * n + a] = sum / n;
      }
    sym_eig(K, n, w);
    for (int j = n - 1; j >= 0 && r < rmax; j--) {
      if (!(w[j] > 1e-12 * w[n - 1])) break;
      double *u = U0 + (int64_t)r * D;
      for (int a = 0; a < n; a++) {
        double c = K[a * n + j];
        const double *ya = G + (int64_t)a * D;
        for (int64_t i = 0; i < D; i++) u[i] += c * ya[i];
      }
      r++;
    }
    free(K), free(w);
  }
  r = orthonormalise(U0, r, D);
  int k = 0;
  if (r > 0) {
    // projections of the scaled draws and gradients on the subspace (n x r)
    double *PX = malloc(sizeof(double) * n * r), *PY = malloc(sizeof(double) * n * r);
    for (int a = 0; a < n; a++)
      for (int j = 0; j < r; j++) {
        const double *u = U0 + (int64_t)j * D, *xa = Q + (int64_t)a * D, *ya = G + (int64_t)a * D;
        double sx = 0, sy = 0;
#pragma omp simd reduction(+ : sx, sy)
        for (int64_t i = 0; i < D; i++) sx += u[i] * xa[i], sy += u[i] * ya[i];
        PX[a * r + j] = sx, PY[a * r + j] = sy;
      }
    size_t rr = (size_t)r * r;
    double *Cd = malloc(sizeof(double) * rr), *Cg = malloc(sizeof(double) * rr);
    double *T1 = malloc(sizeof(double) * rr), *T2 = malloc(sizeof(double) * rr), *T3 = malloc(sizeof(double) * rr);
    double *ev = malloc(sizeof(double) * r), *fw = malloc(sizeof(double) * r);
    gram_at_b(PX, r, PX, r, n, Cd);
    gram_at_b(PY, r, PY, r, n, Cg);
    for (int j = 0; j < r; j++) Cd[j * r + j] += gamma, Cg[j * r + j] += gamma;
    // Sigma = Cg^-1/2 (Cg^1/2 Cd Cg^1/2)^1/2 Cg^-1/2
    memcpy(T1, Cg, sizeof(double) * rr);
    sym_eig(T1, r, ev);  // T1 = eigenvectors of Cg
    for (int j = 0; j < r; j++) fw[j] = sqrt(fmax(ev[j], 1e-300));
    eig_apply(T1, fw, r, T2);  // T2 = Cg^1/2
    matmul_sq(T2, Cd, r, T3);
    matmul_sq(T3, T2, r, Cd);  // Cd <- Cg^1/2 Cd Cg^1/2
    for (int j = 0; j < r; j++) fw[j] = 1.0 / fw[j];
    eig_apply(T1, fw, r, T2);  // T2 = Cg^-1/2
    for (size_t m = 0; m < rr; m++) T1[m] = Cd[m];
    for (int i = 0; i < r; i++)
      for (int j = 0; j < i; j++) T1[i * r + j] = T1[j * r + i] = 0.5 * (Cd[i * r + j] + Cd[j * r + i]);
    sym_eig(T1, r, ev);
    for (int j = 0; j < r; j++) fw[j] = sqrt(fmax(ev[j], 0.0));
    eig_apply(T1, fw, r, T3);  // T3 = (Cg^1/2 Cd Cg^1/2)^1/2
    matmul_sq(T2, T3, r, T1);
    matmul_sq(T1, T2, r, T3);  // T3 = Sigma
    for (int i = 0; i < r; i++)
      for (int j = 0; j < i; j++) T3[i * r + j] = T3[j * r + i] = 0.5 * (T3[i * r + j] + T3[j * r + i]);
    sym_eig(T3, r, ev);
    // keep eigenvalues outside [1/cutoff, cutoff], largest |log| first
    int *ord = malloc(sizeof(int) * r);
    int nk = 0;
    for (int j = 0; j < r; j++)
      if (isfinite(ev[j]) && ev[j] > 0 && (ev[j] > cutoff || ev[j] < 1.0 / cutoff)) ord[nk++] = j;
    for (int a = 0; a < nk; a++)
      for (int b = a + 1; b < nk; b++)
        if (fabs(log(ev[ord[b]])) > fabs(log(ev[ord[a]]))) {
          int t = ord[a];
          ord[a] = ord[b];
          ord[b] = t;
        }
    if (nk > kmax) nk = kmax;
    memset(U, 0, sizeof(double) * (size_t)nk * (size_t)D);
    for (int m = 0; m < nk; m++) {
      double *u = U + (int64_t)m * D;
      for (int j = 0; j < r; j++) {
        double c = T3[j * r + ord[m]];
        const double *u0 = U0 + (int64_t)j * D;
        for (int64_t i = 0; i < D; i++) u[i] += c * u0[i];
      }
      // Early windows have few draws, sometimes from a chain still moving
      // to the typical set; there the estimate can come out absurdly small
      // (1e-7 on the 37,901-parameter model after 25 draws), which would
      // freeze that direction for the next window. Limit it to 100x
      // narrower or wider than the diagonal's scale.
      lam[m] = fmin(fmax(ev[ord[m]], 1e-4), 1e4);
    }
    // orthonormal up to rounding already; to working precision, so that lam
    // are the variances along exactly these directions
    k = orthonormalise(U, nk, D);
    if (k != nk) k = 0;  // a dependent direction: skip the correction for this window
    free(ord), free(PX), free(PY), free(Cd), free(Cg), free(T1), free(T2), free(T3), free(ev), free(fw);
  }
  free(U0);
  return k;
}

// ---------------------------------------------------------------- NUTS
// Multinomial NUTS with the generalised no-U-turn criterion, diagonal metric
// and Stan's windowed warmup (step-size dual averaging plus metric windows).
// This is a port of the structure of Stan's base_nuts.hpp.

typedef double (*mint_logp_fn)(const double *theta, double *grad);
typedef void (*mint_constrain_fn)(const double *unc, double *out);

#define MAX_DEPTH 10
#define LR_KMAX 128

// Phase-space states are immutable once built and shared by reference: a
// leapfrog step writes a new state instead of updating one in place, and the
// tree keeps references to its end points and proposals instead of copying
// D-length vectors. States are reference counted and recycled from a
// per-chain free list. The arithmetic and the order of random draws are
// exactly those of Stan's base_nuts structure.
typedef struct St {
  double *q, *p, *g;  // position, momentum, gradient
  double lp, h;            // log density, Hamiltonian
  int rc;
  struct St *next_free;
} St;

typedef struct {
  St *init_end, *final_beg, *prop_final;
  double *rho_init, *rho_final;
} Level;

typedef struct {
  int64_t D;
  mint_logp_fn f;
  double *inv_m;
  double eps;
  Rng rng;
  St *free_list;
  St *all[4096];
  int n_all;
  St *cur;   // current sample
  St *edge;  // end of the trajectory being extended
  Level lv[MAX_DEPTH + 1];
  int64_t n_leapfrog;
  int64_t n_grad;
  double sum_metro;
  int divergent;
  int depth;
  int nt;        // threads splitting this chain's D-length passes (1 = serial)
  int team_min;  // smallest OpenMP team that actually ran one of those passes
  // Low-rank metric (MINT_METRIC=lowrank; k = 0 otherwise):
  //   inverse metric = diag(inv_m) + sum_j lr_d[j] v_j v_j',  v_j = sqrt(inv_m) u_j,
  // with orthonormal u_j (before rounding). Every momentum-like vector
  // (momenta and their sums) and every gradient carries its k projections
  // c_j = v_j . x in the LR_KMAX slots after its D entries, so the no-U-turn
  // checks need no extra D-length work.
  int k;
  // The v_j are stored in single precision to halve the memory traffic of
  // the two O(D k) passes per leapfrog step; all arithmetic is in double, and
  // the metric is exactly the one these stored values define (lr_a makes the
  // momentum draws exact for it, whether or not the rounded u_j are
  // orthonormal).
  float *lr_v;                      // tiled, see LR_AT
  int64_t lr_nb;                    // tiles per direction group: ceil(D / 8)
  double lr_d[LR_KMAX];             // lam_j - 1
  double lr_a[LR_KMAX * LR_KMAX];   // k x k, for drawing momenta (lr_set)
} Nuts;

#define MAX_NT 64

static St *st_acquire(Nuts *s) {
  St *x = s->free_list;
  if (x) {
    s->free_list = x->next_free;
  } else {
    if (s->n_all == 4096) mint_panic("sampler state pool exhausted");
    x = calloc(1, sizeof *x);
    x->q = mint_alloc(s->D);
    x->p = mint_alloc(s->D + LR_KMAX);
    x->g = mint_alloc(s->D + LR_KMAX);
    s->all[s->n_all++] = x;
  }
  x->rc = 1;
  return x;
}

static void st_release(Nuts *s, St *x) {
  if (x && --x->rc == 0) {
    x->next_free = s->free_list;
    s->free_list = x;
  }
}

// *slot = x, adjusting reference counts
static void st_set(Nuts *s, St **slot, St *x) {
  if (x) x->rc++;
  st_release(s, *slot);
  *slot = x;
}

static double log_sum_exp(double a, double b) {
  if (a == -INFINITY) return b;
  if (b == -INFINITY) return a;
  double m = a > b ? a : b;
  return m + log(exp(a - m) + exp(b - m));
}

static void eval(Nuts *s, St *z) {
  z->lp = s->f(z->q, z->g);
  s->n_grad++;
}

static double hamiltonian(Nuts *s, const St *z) {
  double k = 0;
  for (int64_t i = 0; i < s->D; i++) k += z->p[i] * z->p[i] * s->inv_m[i];
  if (s->k) {  // low-rank part, from the stored projections
    const double *c = z->p + s->D;
    for (int j = 0; j < s->k; j++) k += s->lr_d[j] * c[j] * c[j];
  }
  return -z->lp + 0.5 * k;
}

// The directions are stored in tiles of 8 directions x 8 consecutive entries
// (256 bytes): direction j, entry i lives in tile (j / 8, i / 8). Both
// passes then stream one array with full-width vector operations. Padding
// (directions k .. 8 ng - 1 and entries D .. 8 nb - 1) is zero.
#define LR_AT(s, j, i) \
  ((s)->lr_v[(((int64_t)((j) >> 3) * (s)->lr_nb + ((i) >> 3)) * 8 + ((j) & 7)) * 8 + ((i) & 7)])
#define LR_CHUNK 64  // tiles per chunk of a pass

typedef float v8f __attribute__((vector_size(32), aligned(4)));
typedef double v4d __attribute__((vector_size(32), aligned(8)));

static inline v4d v8f_lo(v8f x) { return __builtin_convertvector(__builtin_shufflevector(x, x, 0, 1, 2, 3), v4d); }
static inline v4d v8f_hi(v8f x) { return __builtin_convertvector(__builtin_shufflevector(x, x, 4, 5, 6, 7), v4d); }

// q[i] += sum_j w[j] v_j[i] for the entries of tiles [tb, te). The sum is
// formed in single precision (eight lanes per instruction) and added to q in
// double. The position update q += eps f(p) is volume-preserving, and
// reversible in the same sense as any floating-point leapfrog, for any f that
// is odd in p, which a rounded linear map is; so this changes only how well
// energy is conserved, not what the sampler targets. The kinetic energy and
// the no-U-turn sums use double-precision projections of the momenta.
// wf has 8 ng entries, zero beyond k.
static void lr_expand(const Nuts *s, double *restrict q, const float *wf, int64_t tb, int64_t te) {
  int64_t D = s->D, nb = s->lr_nb;
  int ng = (s->k + 7) >> 3;
  const float *restrict V = s->lr_v;
  int64_t full = D >> 3;  // tiles with 8 real entries
  int64_t ib = tb;
  for (; ib + 8 <= te && ib + 8 <= full; ib += 8) {
    v8f a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0}, a4 = {0}, a5 = {0}, a6 = {0}, a7 = {0};
    for (int g = 0; g < ng; g++)
      for (int j = 0; j < 8; j++) {
        float ws = wf[8 * g + j];
        v8f w = {ws, ws, ws, ws, ws, ws, ws, ws};
        const float *t = V + (((int64_t)g * nb + ib) * 8 + j) * 8;
        a0 += w * *(const v8f *)(t);
        a1 += w * *(const v8f *)(t + 64);
        a2 += w * *(const v8f *)(t + 128);
        a3 += w * *(const v8f *)(t + 192);
        a4 += w * *(const v8f *)(t + 256);
        a5 += w * *(const v8f *)(t + 320);
        a6 += w * *(const v8f *)(t + 384);
        a7 += w * *(const v8f *)(t + 448);
      }
    v8f acc[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    for (int m = 0; m < 8; m++) {
      double *x = q + (ib + m) * 8;
      *(v4d *)x += v8f_lo(acc[m]);
      *(v4d *)(x + 4) += v8f_hi(acc[m]);
    }
  }
  for (; ib < te; ib++) {
    float acc[8] = {0};
    for (int g = 0; g < ng; g++)
      for (int j = 0; j < 8; j++) {
        const float *t = V + (((int64_t)g * nb + ib) * 8 + j) * 8;
        for (int ii = 0; ii < 8; ii++) acc[ii] += wf[8 * g + j] * t[ii];
      }
    for (int ii = 0; ii < 8 && ib * 8 + ii < D; ii++) q[ib * 8 + ii] += (double)acc[ii];
  }
}

// c[j] += v_j . x (and, when two is set, cy[j] += v_j . y) over the entries
// of tiles [tb, te), accumulated in double. Four directions per pass share
// each conversion of the stored values. The arithmetic for one vector does
// not depend on whether a second is projected alongside it.
static inline __attribute__((always_inline)) void lr_dots_impl(const Nuts *s, double *c, const double *restrict x,
                                                               double *cy, const double *restrict y, const int two,
                                                               int64_t tb, int64_t te) {
  int64_t D = s->D, nb = s->lr_nb;
  int ng = (s->k + 7) >> 3;
  const float *restrict V = s->lr_v;
  int64_t full = D >> 3;
  int64_t tf = te < full ? te : full;
  for (int g = 0; g < ng; g++)
    for (int h = 0; h < 8; h += 4) {
      if (8 * g + h >= s->k) break;
      v4d a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0}, b0 = {0}, b1 = {0}, b2 = {0}, b3 = {0};
      for (int64_t ib = tb; ib < tf; ib++) {
        v4d xl = *(const v4d *)(x + ib * 8), xh = *(const v4d *)(x + ib * 8 + 4);
        const float *t = V + ((int64_t)g * nb + ib) * 64 + h * 8;
        v4d l0 = v8f_lo(*(const v8f *)(t)), h0 = v8f_hi(*(const v8f *)(t));
        v4d l1 = v8f_lo(*(const v8f *)(t + 8)), h1 = v8f_hi(*(const v8f *)(t + 8));
        v4d l2 = v8f_lo(*(const v8f *)(t + 16)), h2 = v8f_hi(*(const v8f *)(t + 16));
        v4d l3 = v8f_lo(*(const v8f *)(t + 24)), h3 = v8f_hi(*(const v8f *)(t + 24));
        a0 += l0 * xl + h0 * xh;
        a1 += l1 * xl + h1 * xh;
        a2 += l2 * xl + h2 * xh;
        a3 += l3 * xl + h3 * xh;
        if (two) {
          v4d yl = *(const v4d *)(y + ib * 8), yh = *(const v4d *)(y + ib * 8 + 4);
          b0 += l0 * yl + h0 * yh;
          b1 += l1 * yl + h1 * yh;
          b2 += l2 * yl + h2 * yh;
          b3 += l3 * yl + h3 * yh;
        }
      }
      v4d acc[4] = {a0, a1, a2, a3}, bcc[4] = {b0, b1, b2, b3};
      for (int j = 0; j < 4 && 8 * g + h + j < s->k; j++) {
        c[8 * g + h + j] += (acc[j][0] + acc[j][1]) + (acc[j][2] + acc[j][3]);
        if (two) cy[8 * g + h + j] += (bcc[j][0] + bcc[j][1]) + (bcc[j][2] + bcc[j][3]);
      }
      if (tb <= full && full < te) {  // the partial last tile (D not a multiple of 8)
        for (int j = 0; j < 4 && 8 * g + h + j < s->k; j++) {
          const float *t = V + (((int64_t)g * nb + full) * 8 + h + j) * 8;
          double sx = 0, sy = 0;
          for (int64_t ii = 0; full * 8 + ii < D; ii++) {
            sx += (double)t[ii] * x[full * 8 + ii];
            if (two) sy += (double)t[ii] * y[full * 8 + ii];
          }
          c[8 * g + h + j] += sx;
          if (two) cy[8 * g + h + j] += sy;
        }
      }
    }
}

static void lr_dots(const Nuts *s, double *c, const double *restrict x, int64_t tb, int64_t te) {
  lr_dots_impl(s, c, x, NULL, NULL, 0, tb, te);
}

static void lr_dots2(const Nuts *s, double *c, const double *restrict x, double *cy, const double *restrict y,
                     int64_t tb, int64_t te) {
  lr_dots_impl(s, c, x, cy, y, 1, tb, te);
}

// c_j = v_j . x for the low-rank directions (x has D entries), split and
// chunked exactly as in the leaf step's final pass
static void lr_project(Nuts *s, const double *x, double *c) {
  int64_t nb = s->lr_nb;
  int nt = s->nt, k = s->k;
  double part[MAX_NT][LR_KMAX];
  int used = 1;
#pragma omp parallel num_threads(nt) if (nt > 1)
  {
    int t = omp_get_thread_num(), T = omp_get_num_threads();
    if (t == 0) used = T;
    for (int j = 0; j < k; j++) part[t][j] = 0;
    int64_t tlo = nb * t / T, thi = nb * (t + 1) / T;
    for (int64_t tb = tlo; tb < thi; tb += LR_CHUNK)
      lr_dots(s, part[t], x, tb, tb + LR_CHUNK < thi ? tb + LR_CHUNK : thi);
  }
  if (used < s->team_min) s->team_min = used;
  for (int j = 0; j < k; j++) {
    double acc = 0;
    for (int t = 0; t < used; t++) acc += part[t][j];
    c[j] = acc;
  }
}

// Leaf step with the low-rank metric. The position update needs
// M^-1 p_half = inv_m * p_half + sum_j d_j (v_j . p_half) v_j, where the
// projections of p_half come from those stored with z (p and g are linear).
// The final pass projects the new gradient (for the next step's position
// update) and the new momentum itself, so that the kinetic energy is a
// function of the momentum alone rather than of the path that led to it.
// Two passes over the k directions per step, O(D k).
static double leaf_into_lr(Nuts *s, const St *z, St *n, double eps) {
  int64_t D = s->D, nb = s->lr_nb;
  int nt = s->nt, k = s->k;
  const double *restrict im = s->inv_m;
  const double *cpz = z->p + D, *cgz = z->g + D;
  double ch[LR_KMAX];
  float wf[LR_KMAX + 8];
  for (int j = 0; j < k; j++) {
    ch[j] = cpz[j] + 0.5 * eps * cgz[j];
    wf[j] = (float)(eps * s->lr_d[j] * ch[j]);
  }
  for (int j = k; j < ((k + 7) & ~7); j++) wf[j] = 0;
#pragma omp parallel num_threads(nt) if (nt > 1)
  {
    int t = omp_get_thread_num(), T = omp_get_num_threads();
    int64_t tlo = nb * t / T, thi = nb * (t + 1) / T;
    for (int64_t tb = tlo; tb < thi; tb += LR_CHUNK) {
      int64_t te = tb + LR_CHUNK < thi ? tb + LR_CHUNK : thi;
      int64_t b = tb * 8, e = te * 8 < D ? te * 8 : D;
      const double *restrict zp = z->p, *restrict zg = z->g, *restrict zq = z->q;
      double *restrict np = n->p, *restrict nq = n->q;
      for (int64_t i = b; i < e; i++) {
        double ph = zp[i] + 0.5 * eps * zg[i];
        np[i] = ph;
        nq[i] = zq[i] + eps * im[i] * ph;
      }
      lr_expand(s, nq, wf, tb, te);
    }
  }
  eval(s, n);
  double part[MAX_NT][2 * LR_KMAX + 1];
  int used = 1;
#pragma omp parallel num_threads(nt) if (nt > 1)
  {
    int t = omp_get_thread_num(), T = omp_get_num_threads();
    if (t == 0) used = T;
    int64_t tlo = nb * t / T, thi = nb * (t + 1) / T;
    double kd = 0, c[LR_KMAX], cp[LR_KMAX];
    for (int j = 0; j < k; j++) c[j] = 0, cp[j] = 0;
    for (int64_t tb = tlo; tb < thi; tb += LR_CHUNK) {
      int64_t te = tb + LR_CHUNK < thi ? tb + LR_CHUNK : thi;
      int64_t b = tb * 8, e = te * 8 < D ? te * 8 : D;
      double *restrict np = n->p;
      const double *restrict ng = n->g;
      #pragma omp simd reduction(+ : kd)
      for (int64_t i = b; i < e; i++) {
        double p = np[i] + 0.5 * eps * ng[i];
        np[i] = p;
        kd += p * p * im[i];
      }
      lr_dots2(s, c, ng, cp, np, tb, te);
    }
    part[t][0] = kd;
    for (int j = 0; j < k; j++) part[t][1 + j] = c[j], part[t][1 + LR_KMAX + j] = cp[j];
  }
  if (used < s->team_min) s->team_min = used;
  double kin = 0;
  for (int t = 0; t < used; t++) kin += part[t][0];
  double *cgn = n->g + D, *cpn = n->p + D;
  for (int j = 0; j < k; j++) {
    double ag = 0, ap = 0;
    for (int t = 0; t < used; t++) ag += part[t][1 + j], ap += part[t][1 + LR_KMAX + j];
    cgn[j] = ag;
    cpn[j] = ap;
    kin += s->lr_d[j] * cpn[j] * cpn[j];
  }
  return -n->lp + 0.5 * kin;
}

// n = one leapfrog step from z (z is not modified)
static void leapfrog_into(Nuts *s, const St *z, St *n, double eps) {
  int64_t D = s->D;
  const double *restrict im = s->inv_m;
  for (int64_t i = 0; i < D; i++) {
    double ph = z->p[i] + 0.5 * eps * z->g[i];
    n->p[i] = ph;
    n->q[i] = z->q[i] + eps * im[i] * ph;
  }
  eval(s, n);
  for (int64_t i = 0; i < D; i++) n->p[i] += 0.5 * eps * n->g[i];
}

// Leaf step of the tree: leapfrog, then in one pass the final momentum and
// the kinetic energy. The sampler's sums are vectorised (`omp simd
// reduction`): a fixed order, so draws are reproducible for a given build,
// but not the strict left-to-right order of scalar code. Returns the Hamiltonian. (A leaf's momentum sum is its
// own momentum, so nothing is copied for it; see build_tree.)
static double leaf_into(Nuts *s, const St *z, St *n, double eps) {
  if (s->k) return leaf_into_lr(s, z, n, eps);
  int64_t D = s->D;
  const double *restrict im = s->inv_m;
  if (s->nt > 1) {
    // Split across s->nt threads. Partial sums are combined in thread order,
    // so the result is deterministic for a given thread count.
    int nt = s->nt;
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int64_t i = 0; i < D; i++) {
      double ph = z->p[i] + 0.5 * eps * z->g[i];
      n->p[i] = ph;
      n->q[i] = z->q[i] + eps * im[i] * ph;
    }
    eval(s, n);
    // OpenMP may give a smaller team than asked for (thread limits, nesting),
    // so the split follows the team actually running, not nt.
    double part[MAX_NT];
    int used = 1;
#pragma omp parallel num_threads(nt)
    {
      int t = omp_get_thread_num(), T = omp_get_num_threads();
      if (t == 0) used = T;
      int64_t lo = D * t / T, hi = D * (t + 1) / T;
      double k = 0;
      #pragma omp simd reduction(+ : k)
      for (int64_t i = lo; i < hi; i++) {
        double p = n->p[i] + 0.5 * eps * n->g[i];
        n->p[i] = p;
        k += p * p * im[i];
      }
      part[t] = k;
    }
    if (used < s->team_min) s->team_min = used;
    double k = 0;
    for (int t = 0; t < used; t++) k += part[t];
    return -n->lp + 0.5 * k;
  }
  for (int64_t i = 0; i < D; i++) {
    double ph = z->p[i] + 0.5 * eps * z->g[i];
    n->p[i] = ph;
    n->q[i] = z->q[i] + eps * im[i] * ph;
  }
  eval(s, n);
  double k = 0;
  #pragma omp simd reduction(+ : k)
  for (int64_t i = 0; i < D; i++) {
    double p = n->p[i] + 0.5 * eps * n->g[i];
    n->p[i] = p;
    k += p * p * im[i];
  }
  return -n->lp + 0.5 * k;
}

// rho = ra + rb, and the three no-U-turn checks:
//   (a1, b1) = (end_ps . rho, beg_ps . rho)
//   (a2, b2) = (mid2_ps . (ra + mid2_p), beg_ps . (ra + mid2_p))
//   (a3, b3) = (end_ps . (rb + mid1_p), mid1_ps . (rb + mid1_p))
// Returns 1 when all three pass.
// Scaled momenta (inverse metric * p) are recomputed here instead of being
// stored with every state: (im * p) rounds exactly as a stored value would,
// and the sampler moves less memory.
static int merge_checks_diag(Nuts *s, double *rho, const double *ra, const double *rb, const double *beg_p,
                             const double *end_p, const double *mid2_p, const double *mid1_p, const double *corr);

// With the low-rank metric each dot product p_sharp . r gains
// sum_j d_j c_j(p) c_j(r), computed from the stored projections (a sum of
// momenta projects to the sum of their projections). These are computed
// before rho's projections are written, since rho may alias ra or rb.
static int merge_checks(Nuts *s, double *rho, const double *ra, const double *rb, const double *beg_p,
                        const double *end_p, const double *mid2_p, const double *mid1_p) {
  if (!s->k) return merge_checks_diag(s, rho, ra, rb, beg_p, end_p, mid2_p, mid1_p, NULL);
  int64_t D = s->D;
  const double *a = ra + D, *b = rb + D, *be = beg_p + D, *en = end_p + D, *m2 = mid2_p + D, *m1 = mid1_p + D;
  double corr[6] = {0, 0, 0, 0, 0, 0}, rc[LR_KMAX];
  for (int j = 0; j < s->k; j++) {
    double d = s->lr_d[j];
    double r = a[j] + b[j], r2 = a[j] + m2[j], r3 = b[j] + m1[j];
    corr[0] += d * en[j] * r;
    corr[1] += d * be[j] * r;
    corr[2] += d * m2[j] * r2;
    corr[3] += d * be[j] * r2;
    corr[4] += d * en[j] * r3;
    corr[5] += d * m1[j] * r3;
    rc[j] = r;
  }
  int ok = merge_checks_diag(s, rho, ra, rb, beg_p, end_p, mid2_p, mid1_p, corr);
  memcpy(rho + D, rc, sizeof(double) * (size_t)s->k);
  return ok;
}

// The diagonal part of merge_checks; corr (or NULL) is added to the six sums.
static int merge_checks_diag(Nuts *s, double *rho, const double *ra, const double *rb, const double *beg_p,
                             const double *end_p, const double *mid2_p, const double *mid1_p, const double *corr) {
  int64_t D = s->D;
  int nt = s->nt;
  const double *restrict im = s->inv_m;
  if (ra == beg_p && ra == mid1_p && rb == end_p && rb == mid2_p) {
    // Merging two single leaves (half of all merges): ra and rb are the
    // leaves' momenta, and the three checks reduce to the same two sums
    // (with exactly the same arithmetic as the general loop below).
    double part2[MAX_NT][2];
    int used2 = 1;
#pragma omp parallel num_threads(nt) if (nt > 1)
    {
      int t = omp_get_thread_num(), T = omp_get_num_threads();
      if (t == 0) used2 = T;
      int64_t lo = D * t / T, hi = D * (t + 1) / T;
      double a1 = 0, b1 = 0;
      #pragma omp simd reduction(+ : a1, b1)
      for (int64_t i = lo; i < hi; i++) {
        double xa = ra[i], xb = rb[i];
        double r = xa + xb;
        rho[i] = r;
        a1 += (im[i] * xb) * r;
        b1 += (im[i] * xa) * r;
      }
      part2[t][0] = a1, part2[t][1] = b1;
    }
    if (used2 < s->team_min) s->team_min = used2;
    double a = 0, b = 0;
    for (int t = 0; t < used2; t++) a += part2[t][0], b += part2[t][1];
    if (corr) a += corr[0], b += corr[1];
    return a > 0 && b > 0;
  }
  double part[MAX_NT][6];
  int used = 1;
#pragma omp parallel num_threads(nt) if (nt > 1)
  {
    int t = omp_get_thread_num(), T = omp_get_num_threads();
    if (t == 0) used = T;
    int64_t lo = D * t / T, hi = D * (t + 1) / T;
    double a1 = 0, b1 = 0, a2 = 0, b2 = 0, a3 = 0, b3 = 0;
    #pragma omp simd reduction(+ : a1, b1, a2, b2, a3, b3)
    for (int64_t i = lo; i < hi; i++) {
      double beg_ps = im[i] * beg_p[i], end_ps = im[i] * end_p[i];
      // read both sums before writing rho, which may be one of them (the
      // trajectory's sum is updated in place)
      double xa = ra[i], xb = rb[i];
      double r = xa + xb;
      rho[i] = r;
      a1 += end_ps * r;
      b1 += beg_ps * r;
      double r2 = xa + mid2_p[i];
      a2 += (im[i] * mid2_p[i]) * r2;
      b2 += beg_ps * r2;
      double r3 = xb + mid1_p[i];
      a3 += end_ps * r3;
      b3 += (im[i] * mid1_p[i]) * r3;
    }
    part[t][0] = a1, part[t][1] = b1, part[t][2] = a2, part[t][3] = b2, part[t][4] = a3, part[t][5] = b3;
  }
  if (used < s->team_min) s->team_min = used;
  double v[6] = {0, 0, 0, 0, 0, 0};
  for (int t = 0; t < used; t++)
    for (int k = 0; k < 6; k++) v[k] += part[t][k];
  if (corr)
    for (int k = 0; k < 6; k++) v[k] += corr[k];
  return (v[0] > 0 && v[1] > 0) & (v[2] > 0 && v[3] > 0) & (v[4] > 0 && v[5] > 0);
}

static int cholesky_upper(const double *G, int k, double *R) {  // G = R' R, R upper triangular
  memset(R, 0, sizeof(double) * k * k);
  for (int j = 0; j < k; j++) {
    for (int i = 0; i <= j; i++) {
      double sum = G[i * k + j];
      for (int m = 0; m < i; m++) sum -= R[m * k + i] * R[m * k + j];
      if (i == j) {
        if (!(sum > 0)) return 0;
        R[j * k + j] = sqrt(sum);
      } else {
        R[i * k + j] = sum / R[i * k + i];
      }
    }
  }
  return 1;
}

// Installs a low-rank metric: k orthonormal directions U (rows, scaled
// coordinates) with variances lam. Stores v_j = S u_j in single precision and
// derives, for the stored values, the k x k matrix A with
//   (I + W A W')^2 = (I + W diag(lam - 1) W')^-1,  W = V / S (D x k),
// so that p = S^-1 (x + W A W' x), x ~ N(0, I), has exactly the covariance
// whose inverse the kinetic energy uses. With G = W'W = R'R (Cholesky) and
// R diag(lam - 1) R' = E diag(beta) E', A = R^-1 E diag((1 + beta)^-1/2 - 1) E' R^-T.
static void lr_set(Nuts *s, int k, const double *U, const double *lam) {
  int64_t D = s->D;
  memset(s->lr_v, 0, sizeof(float) * (size_t)((k + 7) >> 3) * (size_t)s->lr_nb * 64);
  for (int j = 0; j < k; j++) {
    const double *u = U + (int64_t)j * D;
    for (int64_t i = 0; i < D; i++) LR_AT(s, j, i) = (float)(sqrt(s->inv_m[i]) * u[i]);
    s->lr_d[j] = lam[j] - 1.0;
  }
  if (k == 0) {
    s->k = 0;
    return;
  }
  size_t kk = (size_t)k * k;
  double *Gm = calloc(kk, sizeof(double)), *R = malloc(sizeof(double) * kk), *B = malloc(sizeof(double) * kk);
  double *Ri = calloc(kk, sizeof(double)), *T = malloc(sizeof(double) * kk), *beta = malloc(sizeof(double) * k);
  for (int a = 0; a < k; a++)
    for (int b = 0; b <= a; b++) {
      double sum = 0;
      for (int64_t i = 0; i < D; i++) sum += (double)LR_AT(s, a, i) * (double)LR_AT(s, b, i) / s->inv_m[i];
      Gm[a * k + b] = Gm[b * k + a] = sum;
    }
  int finite = 1;
  for (int j = 0; j < k && finite; j++)
    for (int64_t i = 0; i < D && finite; i++) finite = isfinite(LR_AT(s, j, i));
  for (int j = 0; j < k && finite; j++) finite = isfinite(s->lr_d[j]);
  if (!finite || !cholesky_upper(Gm, k, R)) {
    s->k = 0;  // a direction overflowed single precision (an extreme scale) or G is not
               // positive definite: fall back to the diagonal metric for this window
    goto done;
  }
  // B = R diag(d) R'
  for (int a = 0; a < k; a++)
    for (int b = 0; b < k; b++) {
      double sum = 0;
      for (int m = 0; m < k; m++) sum += R[a * k + m] * s->lr_d[m] * R[b * k + m];
      B[a * k + b] = sum;
    }
  sym_eig(B, k, beta);  // B = E diag(beta) E', E in B's columns
  if (!(1.0 + beta[0] > 0)) {
    s->k = 0;  // not positive definite: fall back to the diagonal
    goto done;
  }
  // Ri = R^-1 (upper triangular)
  for (int j = 0; j < k; j++) {
    Ri[j * k + j] = 1.0 / R[j * k + j];
    for (int i = j - 1; i >= 0; i--) {
      double sum = 0;
      for (int m = i + 1; m <= j; m++) sum += R[i * k + m] * Ri[m * k + j];
      Ri[i * k + j] = -sum / R[i * k + i];
    }
  }
  // T = Ri E
  for (int a = 0; a < k; a++)
    for (int b = 0; b < k; b++) {
      double sum = 0;
      for (int m = 0; m < k; m++) sum += Ri[a * k + m] * B[m * k + b];
      T[a * k + b] = sum;
    }
  // A = T diag(alpha) T'
  for (int a = 0; a < k; a++)
    for (int b = 0; b < k; b++) {
      double sum = 0;
      for (int m = 0; m < k; m++) sum += T[a * k + m] * (1.0 / sqrt(1.0 + beta[m]) - 1.0) * T[b * k + m];
      s->lr_a[a * k + b] = sum;
    }
  s->k = k;
done:
  free(Gm), free(R), free(B), free(Ri), free(T), free(beta);
}

static void sample_momentum(Nuts *s, St *z) {
  for (int64_t i = 0; i < s->D; i++) z->p[i] = rng_normal(&s->rng) / sqrt(s->inv_m[i]);
  if (s->k) {
    // p = S^-1 (x + U A U' x) with x ~ N(0, I), S = sqrt(inv_m), U = V / S
    // and A from lr_set, so that cov(p) is the inverse of the inverse metric.
    // Above p holds S^-1 x, so U' x = V' p; and S^-1 U = V / inv_m.
    int64_t D = s->D;
    int k = s->k;
    double c[LR_KMAX], ac[LR_KMAX + 8];
    lr_project(s, z->p, c);
    for (int a = 0; a < k; a++) {
      double sum = 0;
      for (int b = 0; b < k; b++) sum += s->lr_a[a * k + b] * c[b];
      ac[a] = sum;
    }
    int ng = (k + 7) >> 3;
    for (int j = k; j < 8 * ng; j++) ac[j] = 0;
    for (int64_t ib = 0; ib < s->lr_nb; ib++) {
      double sum[8] = {0};
      for (int g = 0; g < ng; g++)
        for (int j = 0; j < 8; j++) {
          const float *t = s->lr_v + (((int64_t)g * s->lr_nb + ib) * 8 + j) * 8;
          for (int ii = 0; ii < 8; ii++) sum[ii] += ac[8 * g + j] * (double)t[ii];
        }
      for (int ii = 0; ii < 8 && ib * 8 + ii < D; ii++) z->p[ib * 8 + ii] += sum[ii] / s->inv_m[ib * 8 + ii];
    }
    lr_project(s, z->p, z->p + D);
    lr_project(s, z->g, z->g + D);  // the metric may have changed since z's gradient was projected
  }
}


// a fresh state with z's position, gradient and log density
static St *st_clone_position(Nuts *s, const St *z) {
  St *x = st_acquire(s);
  memcpy(x->q, z->q, s->D * sizeof(double));
  memcpy(x->g, z->g, s->D * sizeof(double));
  x->lp = z->lp;
  return x;
}

static void vzero(double *v, int64_t D) { memset(v, 0, D * sizeof(double)); }
static void vcopy(double *d, const double *s, int64_t D) { memcpy(d, s, D * sizeof(double)); }

// Extends the trajectory from s->edge by 2^depth leapfrog steps. On return
// *beg and *end reference the subtree's first and last states, *prop its
// multinomial proposal, and rho has the subtree's summed momenta added.
// The subtree's momentum sum is returned in *rho_out: the leaf's own momentum
// at depth 0 (the leaf stays referenced as *beg and *end until the parent's
// merge has used it), and otherwise rho, which the merge fills.
static int build_tree(Nuts *s, int depth, St **prop, St **beg, St **end, double *rho, const double **rho_out,
                      double H0, double sign, double *log_sum_weight) {
  if (depth == 0) {
    St *n = st_acquire(s);
    double h = leaf_into(s, s->edge, n, sign * s->eps);
    *rho_out = n->p;
    st_release(s, s->edge);
    s->edge = n;
    s->n_leapfrog++;
    if (isnan(h)) h = INFINITY;
    n->h = h;
    if (h - H0 > 1000.0) s->divergent = 1;
    *log_sum_weight = log_sum_exp(*log_sum_weight, H0 - h);
    s->sum_metro += (H0 - h > 0) ? 1.0 : exp(H0 - h);
    st_set(s, prop, n);
    st_set(s, beg, n);
    st_set(s, end, n);
    return !s->divergent;
  }
  Level *L = &s->lv[depth];
  int persist = 0;
  *rho_out = rho;

  double lsw_init = -INFINITY;
  const double *r_init, *r_final;
  if (!build_tree(s, depth - 1, prop, beg, &L->init_end, L->rho_init, &r_init, H0, sign, &lsw_init)) goto out;

  double lsw_final = -INFINITY;
  if (!build_tree(s, depth - 1, &L->prop_final, &L->final_beg, end, L->rho_final, &r_final, H0, sign,
                  &lsw_final))
    goto out;

  double lsw_subtree = log_sum_exp(lsw_init, lsw_final);
  *log_sum_weight = log_sum_exp(*log_sum_weight, lsw_subtree);
  if (lsw_final > lsw_subtree) {
    st_set(s, prop, L->prop_final);
  } else if (rng_uniform(&s->rng) < exp(lsw_final - lsw_subtree)) {
    st_set(s, prop, L->prop_final);
  }

  // rho = rho_init + rho_final and the three no-U-turn checks, in one pass
  persist = merge_checks(s, rho, r_init, r_final, (*beg)->p, (*end)->p, L->final_beg->p, L->init_end->p);
out:
  st_set(s, &L->init_end, NULL);
  st_set(s, &L->final_beg, NULL);
  st_set(s, &L->prop_final, NULL);
  return persist;
}

typedef struct {
  double *rho, *rho_fwd, *rho_bck;
} Traj;

// One NUTS transition from s->cur. On return s->cur is the selected sample.
// Returns the acceptance statistic.
static double transition(Nuts *s, Traj *t) {
  int64_t D = s->D;
  (void)D;
  St *z0 = st_clone_position(s, s->cur);
  sample_momentum(s, z0);
  St *fwd_fwd = NULL, *fwd_bck = NULL, *bck_fwd = NULL, *bck_bck = NULL;
  St *edge_fwd = NULL, *edge_bck = NULL, *sample = NULL, *propose = NULL;
  St **refs[] = {&fwd_fwd, &fwd_bck, &bck_fwd, &bck_bck, &edge_fwd, &edge_bck, &sample, &propose};
  for (size_t k = 0; k < sizeof refs / sizeof refs[0]; k++) st_set(s, refs[k], z0);
  vcopy(t->rho, z0->p, D + s->k);

  double log_sum_weight = 0;
  double H0 = hamiltonian(s, z0);
  s->n_leapfrog = 0;
  s->sum_metro = 0;
  s->depth = 0;
  s->divergent = 0;

  while (s->depth < MAX_DEPTH) {
    int valid;
    double lsw_sub = -INFINITY;
    // The trajectory so far is one side of the merge and the new subtree the
    // other; the merge writes their sum back into t->rho in place.
    const double *r_new, *r_bck, *r_fwd;
    if (rng_uniform(&s->rng) > 0.5) {
      st_set(s, &s->edge, edge_fwd);
      st_set(s, &bck_fwd, fwd_fwd);
      valid = build_tree(s, s->depth, &propose, &fwd_bck, &fwd_fwd, t->rho_fwd, &r_new, H0, 1.0, &lsw_sub);
      st_set(s, &edge_fwd, s->edge);
      r_bck = t->rho, r_fwd = r_new;
    } else {
      st_set(s, &s->edge, edge_bck);
      st_set(s, &fwd_bck, bck_bck);
      valid = build_tree(s, s->depth, &propose, &bck_fwd, &bck_bck, t->rho_bck, &r_new, H0, -1.0, &lsw_sub);
      st_set(s, &edge_bck, s->edge);
      r_bck = r_new, r_fwd = t->rho;
    }
    if (!valid) break;
    s->depth++;
    if (lsw_sub > log_sum_weight) {
      st_set(s, &sample, propose);
    } else if (rng_uniform(&s->rng) < exp(lsw_sub - log_sum_weight)) {
      st_set(s, &sample, propose);
    }
    log_sum_weight = log_sum_exp(log_sum_weight, lsw_sub);

    int persist = merge_checks(s, t->rho, r_bck, r_fwd, bck_bck->p, fwd_fwd->p, fwd_bck->p, bck_fwd->p);
    if (!persist) break;
  }
  st_set(s, &s->cur, sample);
  st_set(s, &s->edge, NULL);
  for (size_t k = 0; k < sizeof refs / sizeof refs[0]; k++) st_set(s, refs[k], NULL);
  st_release(s, z0);
  return s->n_leapfrog > 0 ? s->sum_metro / (double)s->n_leapfrog : 0.0;
}

// Hamiltonian after one leapfrog step from z into n
static double step_energy(Nuts *s, const St *z, St *n, double eps) {
  if (s->k) return leaf_into_lr(s, z, n, eps);
  leapfrog_into(s, z, n, eps);
  return hamiltonian(s, n);
}

// Stan's heuristic: double or halve the step size until the acceptance
// probability of a single leapfrog step crosses 0.8.
static void init_stepsize(Nuts *s) {
  St *a = st_clone_position(s, s->cur), *b = st_acquire(s);
  sample_momentum(s, a);
  double H0 = hamiltonian(s, a);
  double h = step_energy(s, a, b, s->eps);
  if (isnan(h)) h = INFINITY;
  int direction = (H0 - h) > log(0.8) ? 1 : -1;
  for (;;) {
    sample_momentum(s, a);
    H0 = hamiltonian(s, a);
    h = step_energy(s, a, b, s->eps);
    if (isnan(h)) h = INFINITY;
    double dH = H0 - h;
    if (direction == 1 && !(dH > log(0.8))) break;
    if (direction == -1 && !(dH < log(0.8))) break;
    s->eps = direction == 1 ? 2 * s->eps : 0.5 * s->eps;
    if (s->eps > 1e7) mint_panic("step size search diverged upward: posterior may be improper");
    if (s->eps == 0) mint_panic("step size search collapsed to zero: check the model's gradient");
  }
  st_release(s, a);
  st_release(s, b);
}

typedef struct {
  double mu, s_bar, x_bar;
  double counter;
} DualAvg;

static void da_restart(DualAvg *d) { d->counter = 0, d->s_bar = 0, d->x_bar = 0; }

static void da_learn(DualAvg *d, double *eps, double accept) {
  const double gamma = 0.05, t0 = 10, kappa = 0.75, delta = 0.8;
  d->counter += 1;
  if (accept > 1) accept = 1;
  double eta = 1.0 / (d->counter + t0);
  d->s_bar = (1.0 - eta) * d->s_bar + eta * (delta - accept);
  double x = d->mu - d->s_bar * sqrt(d->counter) / gamma;
  double x_eta = pow(d->counter, -kappa);
  d->x_bar = (1.0 - x_eta) * d->x_bar + x_eta * x;
  *eps = exp(x);
}

typedef struct {
  int64_t init_buffer, term_buffer, base_window, warmup;
  int64_t window_size, next_window, counter;
  int enabled;
} Windows;

static void windows_init(Windows *w, int64_t warmup) {
  w->warmup = warmup;
  w->init_buffer = 75, w->term_buffer = 50, w->base_window = 25;
  w->enabled = warmup >= 20;
  if (w->enabled && w->init_buffer + w->term_buffer + w->base_window > warmup) {
    w->init_buffer = (int64_t)(0.15 * warmup);
    w->term_buffer = (int64_t)(0.1 * warmup);
    w->base_window = warmup - (w->init_buffer + w->term_buffer);
  }
  w->counter = 0;
  w->window_size = w->base_window;
  w->next_window = w->init_buffer + w->window_size - 1;
}

static int in_window(Windows *w) {
  return w->counter >= w->init_buffer && w->counter < w->warmup - w->term_buffer &&
         w->counter != w->warmup;
}

static int end_window(Windows *w) {
  return w->counter == w->next_window && w->counter != w->warmup;
}

static void next_window(Windows *w) {
  if (w->next_window == w->warmup - w->term_buffer - 1) return;
  w->window_size *= 2;
  w->next_window = w->counter + w->window_size;
  if (w->next_window != w->warmup - w->term_buffer - 1) {
    int64_t boundary = w->next_window + 2 * w->window_size;
    if (boundary >= w->warmup - w->term_buffer) w->next_window = w->warmup - w->term_buffer - 1;
  }
}

typedef struct {
  // inputs
  mint_logp_fn f;
  mint_constrain_fn constrain;
  int64_t D, draws, warmup;
  uint64_t seed;
  int chain;
  int threads_per_chain;
  // outputs
  int team_min;
  double *out;  // draws x D, constrained
  double step_size;
  int64_t n_grad, divergent;
  double mean_leapfrog;
  int rank;  // low-rank metric: number of directions at the end of warmup
} ChainJob;

static void *run_chain(void *arg) {
  ChainJob *job = arg;
  int64_t D = job->D;
  Nuts *sp = calloc(1, sizeof(Nuts));
  Nuts *s = sp;
  s->D = D;
  s->nt = job->threads_per_chain;
  s->team_min = s->nt;
  s->f = job->f;
  s->inv_m = mint_alloc(D);
  for (int64_t i = 0; i < D; i++) s->inv_m[i] = 1.0;
  rng_seed(&s->rng, job->seed * 0x9E3779B97F4A7C15ull + (uint64_t)job->chain + 1);
  for (int d = 0; d <= MAX_DEPTH; d++) {
    Level *L = &s->lv[d];
    L->rho_init = mint_alloc(D + LR_KMAX);
    L->rho_final = mint_alloc(D + LR_KMAX);
  }
  Traj t;
  double **tv[] = {&t.rho, &t.rho_fwd, &t.rho_bck};
  for (size_t k = 0; k < sizeof tv / sizeof tv[0]; k++) *tv[k] = mint_alloc(D + LR_KMAX);

  // Initialise uniformly on (-2, 2) in the unconstrained space, as Stan does.
  s->cur = st_acquire(s);
  int tries = 0;
  for (;;) {
    for (int64_t i = 0; i < D; i++) s->cur->q[i] = 4.0 * rng_uniform(&s->rng) - 2.0;
    eval(s, s->cur);
    int ok = isfinite(s->cur->lp);
    for (int64_t i = 0; i < D && ok; i++) ok = isfinite(s->cur->g[i]);
    if (ok) break;
    if (++tries == 100) mint_panic("could not find a finite initial point in 100 tries");
  }

  // Metric adaptation: "stan" uses the variance of the draws in each window;
  // "grad" (nutpie's idea) uses sqrt(var(draws) / var(gradients)), which is
  // exact for a Gaussian with diagonal covariance, and starts from the
  // gradient at the initial point instead of the identity.
  const char *metric_env = getenv("MINT_METRIC");
  int grad_metric = metric_env && strcmp(metric_env, "grad") == 0;
  // "lowrank": Stan's diagonal plus a low-rank correction estimated from the
  // gradients of the draws in each window (lowrank_estimate).
  int lowrank = metric_env && strcmp(metric_env, "lowrank") == 0;
  int lr_kmax = 0, lr_nmax = 0;
  double lr_cutoff = 2.0, lr_gamma = 1e-5;
  double *lr_q = NULL, *lr_g = NULL, *lr_u = NULL, lr_lam[LR_KMAX];
  if (lowrank) {
    const char *e;
    lr_kmax = (e = getenv("MINT_LOWRANK_K")) ? atoi(e) : 24;
    if (lr_kmax > LR_KMAX) lr_kmax = LR_KMAX;
    if (lr_kmax > D) lr_kmax = (int)D;
    if (lr_kmax < 0) lr_kmax = 0;
    if ((e = getenv("MINT_LOWRANK_CUTOFF"))) lr_cutoff = atof(e);
    if (!(lr_cutoff >= 1.0)) lr_cutoff = 2.0;
    if ((e = getenv("MINT_LOWRANK_GAMMA"))) lr_gamma = atof(e);
    if (!(lr_gamma >= 0)) lr_gamma = 1e-5;
    // Draws and gradients of the current window are kept, at most lr_nmax of
    // each (the most recent), within a memory budget of MINT_LOWRANK_MB
    // (default 256) per chain.
    double mb = (e = getenv("MINT_LOWRANK_MB")) ? atof(e) : 256;
    if (!(mb > 0)) mb = 256;
    double cap = mb * 1048576.0 / (16.0 * (double)D);
    lr_nmax = (int)(cap < (double)job->warmup ? cap : (double)job->warmup);
    if (lr_nmax < 3) lr_nmax = 0;
    if (lr_nmax && lr_kmax) {
      lr_q = mint_alloc((int64_t)lr_nmax * D);
      lr_g = mint_alloc((int64_t)lr_nmax * D);
      lr_u = mint_alloc((int64_t)lr_kmax * D);
      s->lr_nb = (D + 7) / 8;
      size_t bytes = sizeof(float) * (size_t)((lr_kmax + 7) >> 3) * (size_t)s->lr_nb * 64;
      if (posix_memalign((void **)&s->lr_v, 64, bytes)) mint_panic("out of memory");
    } else {
      lowrank = 0;
    }
  }
  const char *init_env = getenv("MINT_METRIC_INIT");
  if (grad_metric && !(init_env && strcmp(init_env, "0") == 0)) {
    for (int64_t i = 0; i < D; i++) {
      double a = fabs(s->cur->g[i]);
      double v = a > 0 ? 1.0 / a : 1.0;
      s->inv_m[i] = v < 1e-6 ? 1e-6 : v > 1e6 ? 1e6 : v;
    }
  }
  s->eps = 1.0;
  init_stepsize(s);
  DualAvg da;
  da_restart(&da);
  da.mu = log(10 * s->eps);
  Windows w;
  windows_init(&w, job->warmup);
  double *wmean = mint_alloc(D), *wm2 = mint_alloc(D);
  double *gmean = mint_alloc(D), *gm2 = mint_alloc(D);
  vzero(wmean, D);
  vzero(wm2, D);
  vzero(gmean, D);
  vzero(gm2, D);
  int64_t wn = 0;

  job->divergent = 0;
  int64_t total_leapfrog = 0;
  for (int64_t it = 0; it < job->warmup + job->draws; it++) {
    double accept = transition(s, &t);
    const double *q = s->cur->q, *gq = s->cur->g;
    if (it < job->warmup) {
      da_learn(&da, &s->eps, accept);
      if (w.enabled) {
        if (in_window(&w)) {
          wn++;
          for (int64_t i = 0; i < D; i++) {
            double d = q[i] - wmean[i];
            wmean[i] += d / (double)wn;
            wm2[i] += d * (q[i] - wmean[i]);
          }
          if (grad_metric) {
            for (int64_t i = 0; i < D; i++) {
              double d = gq[i] - gmean[i];
              gmean[i] += d / (double)wn;
              gm2[i] += d * (gq[i] - gmean[i]);
            }
          }
          if (lowrank) {
            int64_t slot = (wn - 1) % lr_nmax;
            vcopy(lr_q + slot * D, q, D);
            vcopy(lr_g + slot * D, gq, D);
          }
        }
        if (end_window(&w)) {
          next_window(&w);
          double n = (double)wn;
          for (int64_t i = 0; i < D; i++) {
            double var = wn > 1 ? wm2[i] / (n - 1.0) : 1.0;
            if (grad_metric && wn > 1 && gm2[i] > 0) var = sqrt(var / (gm2[i] / (n - 1.0)));
            s->inv_m[i] = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0));
          }
          if (lowrank) {
            int nst = wn < lr_nmax ? (int)wn : lr_nmax;
            int k = lowrank_estimate(D, nst, lr_q, lr_g, s->inv_m, lr_kmax, lr_cutoff, lr_gamma, lr_u, lr_lam);
            lr_set(s, k, lr_u, lr_lam);
            if (getenv("MINT_LOWRANK_VERBOSE")) {
              fprintf(stderr, "chain %d window end at %lld: %d draws, rank %d, lam", job->chain, (long long)it,
                      nst, k);
              for (int j = 0; j < k && j < 8; j++) fprintf(stderr, " %.3g", lr_lam[j]);
              fprintf(stderr, "%s\n", k > 8 ? " ..." : "");
            }
          }
          vzero(wmean, D);
          vzero(wm2, D);
          vzero(gmean, D);
          vzero(gm2, D);
          wn = 0;
          w.counter++;
          init_stepsize(s);
          da.mu = log(10 * s->eps);
          da_restart(&da);
        } else {
          w.counter++;
        }
      }
      if (it == job->warmup - 1) s->eps = exp(da.x_bar);
    } else {
      job->constrain(q, job->out + (it - job->warmup) * D);
      total_leapfrog += s->n_leapfrog;
      job->divergent += s->divergent;
    }
  }
  job->step_size = s->eps;
  job->n_grad = s->n_grad;
  job->team_min = s->team_min;
  job->mean_leapfrog = job->draws ? (double)total_leapfrog / (double)job->draws : 0;

  job->rank = s->k;
  free(lr_q), free(lr_g), free(lr_u), free(s->lr_v);
  free(wmean);
  free(wm2);
  free(gmean);
  free(gm2);
  for (size_t k = 0; k < sizeof tv / sizeof tv[0]; k++) free(*tv[k]);
  for (int d = 0; d <= MAX_DEPTH; d++) {
    free(s->lv[d].rho_init);
    free(s->lv[d].rho_final);
  }
  for (int k = 0; k < s->n_all; k++) {
    St *x = s->all[k];
    free(x->q), free(x->p), free(x->g), free(x);
  }
  free(s->inv_m);
  free(sp);
  return NULL;
}

// Time spent preparing a model before sampling (for example computing
// sufficient statistics), recorded by generated code or a baseline and
// reported with the sampling time.
static double prep_seconds;
void mint_set_prep_seconds(double s) { prep_seconds = s; }

// Aborts if a BernoulliLogit outcome is not 0 or 1.
void mint_check_binary(double v, int64_t i, const char *what) {
  if (v != 0.0 && v != 1.0) {
    fprintf(stderr, "mint runtime error: %s: observation %lld is %g, but BernoulliLogit needs 0 or 1\n",
            what, (long long)(i + 1), v);
    exit(1);
  }
}

// ---------------------------------------------------------------- diagnostics

// Checks the model gradient against central finite differences at theta.
static void gradcheck(mint_logp_fn f, int64_t D, const double *theta) {
  double *q = mint_alloc(D), *g = mint_alloc(D), *g2 = mint_alloc(D);
  memcpy(q, theta, D * sizeof(double));
  double lp = f(q, g);
  double worst = 0;
  for (int64_t i = 0; i < D; i++) {
    double h = 1e-5 * (fabs(q[i]) > 1 ? fabs(q[i]) : 1);
    double orig = q[i];
    q[i] = orig + h;
    double up = f(q, g2);
    q[i] = orig - h;
    double dn = f(q, g2);
    q[i] = orig;
    double fd = (up - dn) / (2 * h);
    double err = fabs(fd - g[i]) / (fabs(fd) > 1 ? fabs(fd) : 1);
    if (!isfinite(g[i]) || !isfinite(fd) || isnan(err)) err = INFINITY;
    if (err > worst) worst = err;
  }
  if (!isfinite(lp)) worst = INFINITY;
  fprintf(stderr, "gradcheck: D=%lld logp=%.17g worst relative error=%.3e\n", (long long)D, lp,
          worst);
  free(q), free(g), free(g2);
}

// Deterministic test point shared by every implementation.
// A compiled model may store some parameters in a different order from the
// user's (see scan layouts in compiler/src/model.rs). These convert an
// unconstrained vector between the two; null means the orders agree.
typedef void (*mint_permute_fn)(const double *src, double *dst);
static mint_permute_fn layout_to_internal, layout_to_user;
void mint_set_layout(mint_permute_fn to_internal, mint_permute_fn to_user) {
  layout_to_internal = to_internal;
  layout_to_user = to_user;
}

// The benchmark point is defined in the user's order, so every implementation
// evaluates the same point whatever its internal layout.
static void bench_point(double *theta, int64_t D) {
  for (int64_t i = 0; i < D; i++) theta[i] = 0.05 * (double)((i * 37) % 11 - 5) / 5.0;
  if (layout_to_internal) {
    double *t = mint_alloc(D);
    layout_to_internal(theta, t);
    memcpy(theta, t, (size_t)D * sizeof(double));
    free(t);
  }
}

// Times repeated gradient evaluations at a fixed point and exits. Used to
// compare the model code of different implementations without sampler noise.
static void bench_grad(mint_logp_fn f, int64_t D, int64_t reps) {
  double *q = mint_alloc(D), *g = mint_alloc(D);
  bench_point(q, D);
  double lp = f(q, g);
  for (int w = 0; w < 3; w++) lp = f(q, g);
  double t0 = mint_clock();
  double sink = 0;
  for (int64_t r = 0; r < reps; r++) {
    lp = f(q, g);
    sink += g[r % D];
  }
  double t1 = mint_clock();
  if (layout_to_user) {
    double *t = mint_alloc(D);
    layout_to_user(g, t);
    memcpy(g, t, (size_t)D * sizeof(double));
    free(t);
  }
  double gn = 0;
  for (int64_t i = 0; i < D; i++) gn += g[i] * g[i];
  printf("grad-bench: reps=%lld ns_per_eval=%.1f logp=%.12e grad_norm=%.12e sink=%g\n",
         (long long)reps, 1e9 * (t1 - t0) / (double)reps, lp, sqrt(gn), sink * 0);
  if (getenv("MINT_PRINT_GRAD")) {
    printf("grad:");
    for (int64_t i = 0; i < D; i++) printf(" %.17g", g[i]);
    printf("\n");
  }
  free(q), free(g);
  exit(0);
}

typedef struct {
  int64_t D, draws, chains, nparams;
  char **labels;  // D labels
  char **block_name;
  int64_t *block_start, *block_len;  // per parameter: first flat index, element count
  double *draw;   // chains x draws x D
  double seconds;
  int64_t n_grad, divergent;
  double *step_size, *mean_leapfrog;
  int threads_per_chain, team_min, grad_metric;
  int lowrank, rank_min, rank_max;
} MintPosterior;

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static double quantile_sorted(const double *v, int64_t n, double q) {
  double h = (double)(n - 1) * q;
  int64_t lo = (int64_t)floor(h);
  int64_t hi = lo + 1 < n ? lo + 1 : lo;
  return v[lo] + (h - (double)lo) * (v[hi] - v[lo]);
}

// Split-R-hat and an autocorrelation-based ESS (Geyer's initial positive
// sequence on chain-averaged autocovariances), computed on the raw draws.
static void rhat_ess(const double *x, int64_t C, int64_t N, int64_t stride, double *rhat,
                     double *ess) {
  // split R-hat: 2C half chains of length N/2
  int64_t h = N / 2;
  int64_t M = 2 * C;
  double *means = mint_alloc(M), *vars = mint_alloc(M);
  for (int64_t m = 0; m < M; m++) {
    int64_t c = m / 2, start = (m % 2) * h;
    double s = 0;
    for (int64_t i = 0; i < h; i++) s += x[(c * N + start + i) * stride];
    double mu = s / (double)h, v = 0;
    for (int64_t i = 0; i < h; i++) {
      double d = x[(c * N + start + i) * stride] - mu;
      v += d * d;
    }
    means[m] = mu;
    vars[m] = v / (double)(h - 1);
  }
  double grand = 0, W = 0, B = 0;
  for (int64_t m = 0; m < M; m++) grand += means[m], W += vars[m];
  grand /= (double)M;
  W /= (double)M;
  for (int64_t m = 0; m < M; m++) B += (means[m] - grand) * (means[m] - grand);
  B = B * (double)h / (double)(M - 1);
  double var_plus = ((double)(h - 1) / (double)h) * W + B / (double)h;
  *rhat = W > 0 ? sqrt(var_plus / W) : NAN;

  // ESS over the full chains
  double *cm = mint_alloc(C), *cv = mint_alloc(C);
  for (int64_t c = 0; c < C; c++) {
    double s = 0;
    for (int64_t i = 0; i < N; i++) s += x[(c * N + i) * stride];
    cm[c] = s / (double)N;
    double v = 0;
    for (int64_t i = 0; i < N; i++) {
      double d = x[(c * N + i) * stride] - cm[c];
      v += d * d;
    }
    cv[c] = v / (double)N;
  }
  double Wf = 0, gm = 0, Bf = 0;
  for (int64_t c = 0; c < C; c++) Wf += cv[c] * (double)N / (double)(N - 1), gm += cm[c];
  Wf /= (double)C;
  gm /= (double)C;
  if (C > 1) {
    for (int64_t c = 0; c < C; c++) Bf += (cm[c] - gm) * (cm[c] - gm);
    Bf = Bf * (double)N / (double)(C - 1);
  }
  double vp = ((double)(N - 1) / (double)N) * Wf + Bf / (double)N;
  if (!(vp > 0)) {
    *ess = NAN;
    goto done;
  }
  double tau = 0;
  double prev_pair = INFINITY;
  for (int64_t t = 0; t + 1 < N; t += 2) {
    double rho[2];
    for (int k = 0; k < 2; k++) {
      int64_t lag = t + k;
      double acov = 0;
      for (int64_t c = 0; c < C; c++) {
        double s = 0;
        for (int64_t i = 0; i + lag < N; i++)
          s += (x[(c * N + i) * stride] - cm[c]) * (x[(c * N + i + lag) * stride] - cm[c]);
        acov += s / (double)N;
      }
      acov /= (double)C;
      rho[k] = lag == 0 ? 1.0 : 1.0 - (Wf - acov) / vp;
    }
    double pair = rho[0] + rho[1];
    if (pair < 0) break;
    if (pair > prev_pair) pair = prev_pair;  // monotone sequence
    prev_pair = pair;
    tau += 2 * pair;
  }
  tau -= 1;
  if (tau < 1.0 / log10((double)(C * N))) tau = 1.0 / log10((double)(C * N));
  *ess = (double)(C * N) / tau;
done:
  free(means), free(vars), free(cm), free(cv);
}

MintPosterior *mint_sample(mint_logp_fn f, mint_constrain_fn constrain, int64_t D, int64_t draws,
                           int64_t warmup, int64_t chains, int64_t seed, int64_t nparams,
                           const char **names, const int64_t *sizes) {
  if (draws < 4) mint_panic("sample: draws must be at least 4");
  if (chains < 1) mint_panic("sample: chains must be at least 1");
  if (warmup < 0) mint_panic("sample: warmup must be non-negative");

  const char *bench = getenv("MINT_BENCH_GRAD");
  if (getenv("MINT_GRADCHECK")) {
    double *q = mint_alloc(D);
    bench_point(q, D);
    gradcheck(f, D, q);
    Rng r;
    rng_seed(&r, (uint64_t)seed);
    for (int64_t i = 0; i < D; i++) q[i] = 4.0 * rng_uniform(&r) - 2.0;
    gradcheck(f, D, q);
    free(q);
  }
  if (bench) bench_grad(f, D, atoll(bench));

  MintPosterior *post = calloc(1, sizeof *post);
  post->D = D, post->draws = draws, post->chains = chains, post->nparams = nparams;
  post->draw = mint_alloc(chains * draws * D);
  post->labels = calloc((size_t)D, sizeof(char *));
  int64_t k = 0;
  post->block_name = calloc((size_t)nparams, sizeof(char *));
  post->block_start = calloc((size_t)nparams, sizeof(int64_t));
  post->block_len = calloc((size_t)nparams, sizeof(int64_t));
  // sizes[j] < 0 marks a scalar parameter; otherwise it is a vector length.
  for (int64_t j = 0; j < nparams; j++) {
    post->block_name[j] = strdup(names[j]);
    post->block_start[j] = k;
    post->block_len[j] = sizes[j] < 0 ? 1 : sizes[j];
    if (sizes[j] < 0) {
      if (k < D) post->labels[k] = strdup(names[j]);
      k++;
      continue;
    }
    for (int64_t i = 0; i < sizes[j]; i++) {
      char buf[128];
      snprintf(buf, sizeof buf, "%s[%lld]", names[j], (long long)(i + 1));
      if (k < D) post->labels[k] = strdup(buf);
      k++;
    }
  }
  if (k != D) mint_panic("sample: parameter sizes do not add up to the model dimension");

  // Threads per chain for the sampler's D-length passes. Splitting only pays
  // off when D is large; the default gives each chain about (physical cores /
  // chains) threads, assuming two hardware threads per core.
  int tpc = 1;
  const char *tenv = getenv("MINT_THREADS_PER_CHAIN");
  if (tenv) {
    tpc = atoi(tenv);
  } else if (D >= 8192) {
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    tpc = (int)((online / 2) / chains);
  }
  if (tpc < 1) tpc = 1;
  if (tpc > MAX_NT) tpc = MAX_NT;
  ChainJob *jobs = calloc((size_t)chains, sizeof *jobs);
  pthread_t *th = calloc((size_t)chains, sizeof *th);
  double t0 = mint_clock();
  for (int64_t c = 0; c < chains; c++) {
    jobs[c] = (ChainJob){.f = f,
                         .constrain = constrain,
                         .D = D,
                         .draws = draws,
                         .warmup = warmup,
                         .seed = (uint64_t)seed,
                         .chain = (int)c,
                         .threads_per_chain = tpc,
                         .out = post->draw + c * draws * D};
    if (chains == 1) {
      run_chain(&jobs[c]);
    } else if (pthread_create(&th[c], NULL, run_chain, &jobs[c]) != 0) {
      mint_panic("could not start a sampler thread");
    }
  }
  if (chains > 1)
    for (int64_t c = 0; c < chains; c++) pthread_join(th[c], NULL);
  post->seconds = mint_clock() - t0;
  post->step_size = mint_alloc(chains);
  post->mean_leapfrog = mint_alloc(chains);
  post->threads_per_chain = tpc;
  post->team_min = tpc;
  const char *metric_env = getenv("MINT_METRIC");
  post->grad_metric = metric_env && strcmp(metric_env, "grad") == 0;
  post->lowrank = metric_env && strcmp(metric_env, "lowrank") == 0;
  post->rank_min = jobs[0].rank, post->rank_max = jobs[0].rank;
  for (int64_t c = 0; c < chains; c++) {
    if (jobs[c].rank < post->rank_min) post->rank_min = jobs[c].rank;
    if (jobs[c].rank > post->rank_max) post->rank_max = jobs[c].rank;
    if (jobs[c].team_min < post->team_min) post->team_min = jobs[c].team_min;
    post->n_grad += jobs[c].n_grad;
    post->divergent += jobs[c].divergent;
    post->step_size[c] = jobs[c].step_size;
    post->mean_leapfrog[c] = jobs[c].mean_leapfrog;
  }
  free(jobs);
  free(th);
  const char *dump = getenv("MINT_DRAWS");
  if (dump) {  // raw constrained draws: u64 chains, draws, D, then chains x draws x D f64
    FILE *f = fopen(dump, "wb");
    if (!f) mint_panic("cannot open MINT_DRAWS file");
    uint64_t hdr[3] = {(uint64_t)chains, (uint64_t)draws, (uint64_t)D};
    fwrite(hdr, sizeof hdr[0], 3, f);
    fwrite(post->draw, sizeof(double), (size_t)(chains * draws * D), f);
    fclose(f);
  }
  return post;
}

typedef struct {
  double mean, sd, q5, q50, q95, ess, rhat;
} ColStat;

typedef struct {
  MintPosterior *p;
  ColStat *out;
  int64_t lo, hi;
} StatJob;

static void *stat_worker(void *arg) {
  StatJob *j = arg;
  MintPosterior *p = j->p;
  int64_t N = p->draws, C = p->chains, D = p->D, M = N * C;
  double *col = mint_alloc(M);
  for (int64_t c = j->lo; c < j->hi; c++) {
    double s = 0;
    for (int64_t i = 0; i < M; i++) col[i] = p->draw[i * D + c], s += col[i];
    ColStat *o = &j->out[c];
    o->mean = s / (double)M;
    double v = 0;
    for (int64_t i = 0; i < M; i++) v += (col[i] - o->mean) * (col[i] - o->mean);
    o->sd = sqrt(v / (double)(M - 1));
    rhat_ess(p->draw + c, C, N, D, &o->rhat, &o->ess);
    qsort(col, (size_t)M, sizeof(double), cmp_double);
    o->q5 = quantile_sorted(col, M, 0.05);
    o->q50 = quantile_sorted(col, M, 0.5);
    o->q95 = quantile_sorted(col, M, 0.95);
  }
  free(col);
  return NULL;
}

void mint_print_posterior(MintPosterior *p) {
  int64_t N = p->draws, C = p->chains, D = p->D;
  // per-column statistics, computed on up to 24 threads
  ColStat *st = calloc((size_t)D, sizeof *st);
  int64_t nt = D < 24 ? D : 24;
  pthread_t th[24];
  StatJob jobs[24];
  for (int64_t t = 0; t < nt; t++) {
    jobs[t] = (StatJob){p, st, D * t / nt, D * (t + 1) / nt};
    pthread_create(&th[t], NULL, stat_worker, &jobs[t]);
  }
  for (int64_t t = 0; t < nt; t++) pthread_join(th[t], NULL);

  printf("%-14s %12s %12s %12s %12s %12s %8s %6s\n", "parameter", "mean", "sd", "5%", "50%", "95%",
         "ess", "rhat");
  int64_t worst_r = 0, worst_e = 0;
  for (int64_t c = 0; c < D; c++) {
    if (st[c].rhat > st[worst_r].rhat || isnan(st[c].rhat)) worst_r = c;
    if (st[c].ess < st[worst_e].ess || isnan(st[c].ess)) worst_e = c;
  }
  for (int64_t b = 0; b < p->nparams; b++) {
    int64_t s0 = p->block_start[b], len = p->block_len[b];
    int64_t shown = len > 12 ? 3 : len;
    for (int64_t c = s0; c < s0 + shown; c++) {
      ColStat *o = &st[c];
      printf("%-14s %12.5g %12.5g %12.5g %12.5g %12.5g %8.0f %6.3f\n", p->labels[c], o->mean, o->sd, o->q5,
             o->q50, o->q95, o->ess, o->rhat);
    }
    if (len > shown) {
      double mn = INFINITY, mx = -INFINITY;
      for (int64_t c = s0 + shown; c < s0 + len; c++) {
        mn = fmin(mn, st[c].ess);
        mx = fmax(mx, st[c].rhat);
      }
      printf("%-14s %lld more entries: lowest ess %.0f, highest rhat %.3f\n", p->block_name[b],
             (long long)(len - shown), mn, mx);
    }
  }
  printf("all %lld parameters: highest rhat %.3f (%s), lowest ess %.0f (%s)\n", (long long)D, st[worst_r].rhat,
         p->labels[worst_r], st[worst_e].ess, p->labels[worst_e]);
  free(st);
  printf("chains=%lld draws/chain=%lld gradients=%lld divergences=%lld step size=", (long long)C,
         (long long)N, (long long)p->n_grad, (long long)p->divergent);
  for (int64_t c = 0; c < C; c++) printf(c ? ",%.3g" : "%.3g", p->step_size[c]);
  printf(" leapfrog/draw=");
  for (int64_t c = 0; c < C; c++) printf(c ? ",%.1f" : "%.1f", p->mean_leapfrog[c]);
  printf("\n");
  fprintf(stderr, "sampling took %.6f s (%.0f ns per gradient incl. sampler); preparation took %.6f s\n",
          p->seconds, 1e9 * p->seconds / (double)(p->n_grad ? p->n_grad : 1), prep_seconds);
  fprintf(stderr, "sampler: threads per chain=%d (smallest team that ran=%d) metric=%s", p->threads_per_chain,
          p->team_min, p->grad_metric ? "grad" : p->lowrank ? "lowrank" : "stan");
  if (p->lowrank) fprintf(stderr, " (rank %d to %d)", p->rank_min, p->rank_max);
  fprintf(stderr, "\n");
}

// Posterior mean of flat component j (0-based), used by generated code.
double mint_posterior_mean(MintPosterior *p, int64_t j) {
  double s = 0;
  int64_t n = p->draws * p->chains;
  for (int64_t i = 0; i < n; i++) s += p->draw[i * p->D + j];
  return s / (double)n;
}

// Checks that data read from a file lies in the domain its type promises
// (0: Prob, 1: Positive, 2: NonNeg), so the compiler's facts hold.
void mint_check_domain(const double *v, int64_t n, int64_t dom, const char *what) {
  static const char *names[] = {"in (0, 1)", "positive", "non-negative"};
  for (int64_t i = 0; i < n; i++) {
    double x = v[i];
    int ok = dom == 0 ? (x > 0 && x < 1) : dom == 1 ? x > 0 : x >= 0;
    if (!ok) {
      fprintf(stderr, "mint runtime error: %s: entry %lld is %g, but the type says every entry is %s\n",
              what, (long long)(i + 1), x, names[dom]);
      exit(1);
    }
  }
}

// Per-thread scratch space for generated log-density functions. It grows on
// demand and is reused across calls, so gradients do not allocate. Each
// sampler chain runs on its own thread, so chains never share it.
static __thread double *ws_ptr;
static __thread int64_t ws_cap;

double *mint_workspace(int64_t n_doubles) {
  if (n_doubles > ws_cap) {
    free(ws_ptr);
    ws_ptr = mint_alloc(n_doubles);
    ws_cap = n_doubles;
  }
  return ws_ptr;
}

// Aborts if a PoissonLog outcome is not a non-negative integer.
void mint_check_count(double v, int64_t i, const char *what) {
  if (!(v >= 0.0) || v != floor(v)) {
    fprintf(stderr, "mint runtime error: %s: observation %lld is %g, but PoissonLog needs a count (0, 1, 2, ...)\n",
            what, (long long)(i + 1), v);
    exit(1);
  }
}

// Separate per-thread scratch buffers, one per slot. Generated code declares
// this function with a `noalias` result, so LLVM knows distinct slots never
// overlap and can vectorise loops that use several of them without runtime
// overlap checks.
#define MINT_WS_SLOTS 256
static __thread double *slot_ptr[MINT_WS_SLOTS];
static __thread int64_t slot_cap[MINT_WS_SLOTS];

double *mint_ws_slot(int64_t slot, int64_t n_doubles) {
  if (slot < 0 || slot >= MINT_WS_SLOTS) mint_panic("too many scratch buffers in one model");
  if (n_doubles > slot_cap[slot]) {
    free(slot_ptr[slot]);
    slot_ptr[slot] = mint_alloc(n_doubles);
    slot_cap[slot] = n_doubles;
  }
  return slot_ptr[slot];
}
