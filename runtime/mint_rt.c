// Mint runtime: memory, binary I/O, printing, Cholesky solve, RNG and a NUTS sampler.
//
// Generated programs call into this file for everything that is not model or
// numeric-kernel code. The Rust baselines link the same object so that the
// sampler is identical on both sides of the benchmark.

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
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

typedef float v8f __attribute__((vector_size(32), aligned(4)));
typedef double v4d __attribute__((vector_size(32), aligned(8)));

// Products of matrices stored as rows of length D (D up to tens of
// thousands, a few hundred rows), blocked so that the rows are streamed
// from memory once per block of entries, and split across nt threads. Each
// result entry is summed by one thread in a fixed order, so the results do
// not depend on the number of threads.

// one 4 x 2 block of row dot products over entries [l0, l1), in four lanes
static void gram_block(const double *const a[4], const double *const b[2], int64_t l0, int64_t l1,
                       double o[4][2]) {
  v4d s[4][2] = {{{0}}};
  int64_t l = l0;
  for (; l + 4 <= l1; l += 4) {
    v4d b0 = *(const v4d *)(b[0] + l), b1 = *(const v4d *)(b[1] + l);
    for (int i = 0; i < 4; i++) {
      v4d x = *(const v4d *)(a[i] + l);
      s[i][0] = __builtin_elementwise_fma(x, b0, s[i][0]);
      s[i][1] = __builtin_elementwise_fma(x, b1, s[i][1]);
    }
  }
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++) {
      double t = (s[i][j][0] + s[i][j][1]) + (s[i][j][2] + s[i][j][3]);
      for (int64_t r = l; r < l1; r++) t += a[i][r] * b[j][r];
      o[i][j] = t;
    }
}

#define GR_DC 128  // entries per block of gram_rows

// C[i ldc + j] = scale * (row i of A) . (row j of B) for i < m, j < p. With
// sym (A == B, m == p) only j <= i is computed and C is filled symmetrically.
static void gram_rows(const double *A, int m, const double *B, int p, int64_t D, int sym, double scale, double *C,
                      int ldc, int nt) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < p; j++) C[(int64_t)i * ldc + j] = 0;
#pragma omp parallel num_threads(nt) if (nt > 1)
  for (int64_t l0 = 0; l0 < D; l0 += GR_DC) {
    int64_t l1 = l0 + GR_DC < D ? l0 + GR_DC : D;
#pragma omp for schedule(static, 1)
    for (int i0 = 0; i0 < m; i0 += 4) {
      const double *a[4];
      for (int i = 0; i < 4; i++) a[i] = A + (int64_t)(i0 + i < m ? i0 + i : i0) * D;
      int jend = sym ? (i0 + 4 < m ? i0 + 4 : m) : p;
      for (int j0 = 0; j0 < jend; j0 += 2) {
        const double *b[2];
        for (int j = 0; j < 2; j++) b[j] = B + (int64_t)(j0 + j < jend ? j0 + j : j0) * D;
        double o[4][2];
        gram_block(a, b, l0, l1, o);
        for (int i = 0; i < 4 && i0 + i < m; i++)
          for (int j = 0; j < 2 && j0 + j < jend; j++)
            if (!sym || j0 + j <= i0 + i) C[(int64_t)(i0 + i) * ldc + j0 + j] += o[i][j];
      }
    }
  }
  for (int i = 0; i < m; i++)
    for (int j = 0; j < (sym ? i + 1 : p); j++) {
      C[(int64_t)i * ldc + j] *= scale;
      if (sym) C[(int64_t)j * ldc + i] = C[(int64_t)i * ldc + j];
    }
}

#define CB_DC 64  // entries per block of combine_rows

// Row j of C (m x D) = sum_a W[a ldw + j] (row a of B), a < n, summed in
// order of a. C must not overlap B.
static void combine_rows(const double *W, int ldw, int n, int m, const double *B, int64_t D, double *C, int nt) {
#pragma omp parallel for num_threads(nt) schedule(static) if (nt > 1)
  for (int64_t l0 = 0; l0 < D; l0 += CB_DC) {
    int64_t len = l0 + CB_DC < D ? CB_DC : D - l0;
    for (int j0 = 0; j0 < m; j0 += 4) {
      double acc[4][CB_DC];
      memset(acc, 0, sizeof acc);
      for (int a = 0; a < n; a++) {
        const double *br = B + (int64_t)a * D + l0;
        double w[4];
        for (int j = 0; j < 4; j++) w[j] = j0 + j < m ? W[(int64_t)a * ldw + j0 + j] : 0.0;
        for (int j = 0; j < 4; j++)
          for (int64_t l = 0; l < len; l++) acc[j][l] = __builtin_elementwise_fma(w[j], br[l], acc[j][l]);
      }
      for (int j = 0; j < 4 && j0 + j < m; j++) memcpy(C + (int64_t)(j0 + j) * D + l0, acc[j], sizeof(double) * len);
    }
  }
}

// The r largest eigenvalues (w, descending) and their unit eigenvectors
// (columns of E, n x r with leading dimension lde) of the symmetric n x n
// matrix K, from its full eigendecomposition (n is at most the number of
// draws in a window). Returns r (clipped to n).
static int top_eigs(const double *K, int n, int r, double *E, int lde, double *w) {
  if (r > n) r = n;
  double *A = malloc(sizeof(double) * n * n), *ev = malloc(sizeof(double) * n);
  memcpy(A, K, sizeof(double) * n * n);
  sym_eig(A, n, ev);
  for (int j = 0; j < r; j++) {
    w[j] = ev[n - 1 - j];
    for (int a = 0; a < n; a++) E[(int64_t)a * lde + j] = A[a * n + n - 1 - j];
  }
  free(A), free(ev);
  return r;
}

// Symmetric (Loewdin) orthonormalisation of the m rows of U in place:
// U <- (U U')^-1/2 U, the orthonormal rows closest to the given ones, so a
// set that is already orthonormal up to rounding is changed only by
// rounding, row by row. Returns 0 (and leaves U alone) if the rows are far
// from independent (eigenvalues of U U' outside [1/4, 4]); the callers pass
// rows that are orthonormal in exact arithmetic.
static int lowdin(double *U, int m, int64_t D, int nt) {
  if (m == 0) return 1;
  double *G = malloc(sizeof(double) * m * m), *w = malloc(sizeof(double) * m), *T = malloc(sizeof(double) * m * m);
  double *tmp = mint_alloc((int64_t)m * D);
  gram_rows(U, m, U, m, D, 1, 1.0, G, m, nt);
  sym_eig(G, m, w);
  int ok = isfinite(w[0]) && isfinite(w[m - 1]) && w[0] > 0.25 && w[m - 1] < 4.0;
  if (ok) {
    for (int j = 0; j < m; j++) w[j] = 1.0 / sqrt(w[j]);
    eig_apply(G, w, m, T);
    combine_rows(T, m, m, m, U, D, tmp, nt);
    memcpy(U, tmp, sizeof(double) * (size_t)m * (size_t)D);
  }
  free(G), free(w), free(T), free(tmp);
  return ok;
}

// Low-rank correction to a diagonal metric, estimated from n warmup draws Q
// and their log-density gradients G (each n x D, row-major; both are
// overwritten). im is the diagonal inverse metric already chosen for the
// window, S = sqrt(im) its scale. nt threads share the O(n D) work.
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
                            double gamma, double *U, double *lam, int nt) {
  if (n < 3 || kmax < 1) return 0;
#pragma omp parallel for num_threads(nt) schedule(static) if (nt > 1)
  for (int64_t l0 = 0; l0 < D; l0 += CB_DC) {
    int64_t l1 = l0 + CB_DC < D ? l0 + CB_DC : D;
    for (int64_t i = l0; i < l1; i++) {
      double qm = 0, gm = 0;
      for (int r = 0; r < n; r++) qm += Q[(int64_t)r * D + i], gm += G[(int64_t)r * D + i];
      qm /= n, gm /= n;
      double sc = sqrt(im[i]);
      for (int r = 0; r < n; r++) {
        Q[(int64_t)r * D + i] = (Q[(int64_t)r * D + i] - qm) / sc;
        G[(int64_t)r * D + i] = (G[(int64_t)r * D + i] - gm) * sc;
      }
    }
  }
  // Leading eigenvectors of cov(y) = Y' Y / n: directly when D <= n,
  // otherwise from the Gram matrix Y Y' / n (n x n), whose eigenvectors e_j
  // (eigenvalues w_j) give those of Y' Y / n as Y' e_j / sqrt(n w_j),
  // without forming a D x D matrix.
  int r = 0;
  int rmax = 2 * kmax < n - 1 ? 2 * kmax : n - 1;  // candidates; at most kmax are kept
  if (rmax > D) rmax = (int)D;
  double *U0 = mint_alloc((int64_t)rmax * D);
  if (D <= n) {
    int d = (int)D;
    double *K = malloc(sizeof(double) * d * d), *w = malloc(sizeof(double) * rmax);
    double *E = malloc(sizeof(double) * d * rmax);
    for (int i = 0; i < d; i++)
      for (int j = 0; j <= i; j++) {
        double sum = 0;
        for (int a = 0; a < n; a++) sum += G[(int64_t)a * D + i] * G[(int64_t)a * D + j];
        K[i * d + j] = K[j * d + i] = sum / n;
      }
    int got = top_eigs(K, d, rmax, E, rmax, w);
    for (int j = 0; j < got; j++) {
      if (!(w[j] > 1e-12 * w[0])) break;
      for (int i = 0; i < d; i++) U0[(int64_t)r * D + i] = E[(int64_t)i * rmax + j];
      r++;
    }
    free(K), free(w), free(E);
  } else {
    double *K = malloc(sizeof(double) * n * n), *w = malloc(sizeof(double) * rmax);
    gram_rows(G, n, G, n, D, 1, 1.0 / n, K, n, nt);
    double *E = malloc(sizeof(double) * n * rmax);  // n x r: scaled leading eigenvectors
    int got = top_eigs(K, n, rmax, E, rmax, w);
    for (int j = 0; j < got; j++) {
      if (!(w[j] > 1e-12 * w[0])) break;
      double f = 1.0 / sqrt(n * w[j]);
      for (int a = 0; a < n; a++) E[(int64_t)a * rmax + j] *= f;
      r++;
    }
    combine_rows(E, rmax, n, r, G, D, U0, nt);
    free(K), free(w), free(E);
  }
  int k = 0;
  if (r > 0 && lowdin(U0, r, D, nt)) {
    // projections of the scaled draws and gradients on the subspace (n x r)
    double *PX = malloc(sizeof(double) * n * r), *PY = malloc(sizeof(double) * n * r);
    gram_rows(Q, n, U0, r, D, 0, 1.0, PX, r, nt);
    gram_rows(G, n, U0, r, D, 0, 1.0, PY, r, nt);
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
    // keep eigenvalues outside [1/cutoff, cutoff], largest |log lam| first
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
    double *Wk = malloc(sizeof(double) * r * (nk > 0 ? nk : 1));  // r x nk: the kept eigenvectors of Sigma
    for (int m = 0; m < nk; m++) {
      for (int j = 0; j < r; j++) Wk[(int64_t)j * nk + m] = T3[j * r + ord[m]];
      // Early windows have few draws, sometimes from a chain still moving
      // to the typical set; there the estimate can come out absurdly small
      // (1e-7 on the 37,901-parameter model after 25 draws), which would
      // freeze that direction for the next window. Limit it to 100x
      // narrower or wider than the diagonal's scale.
      lam[m] = fmin(fmax(ev[ord[m]], 1e-4), 1e4);
    }
    combine_rows(Wk, nk, r, nk, U0, D, U, nt);
    // orthonormal up to rounding already; to working precision, so that lam
    // are the variances along exactly these directions
    k = lowdin(U, nk, D, nt) ? nk : 0;  // otherwise skip the correction for this window
    free(Wk), free(ord), free(PX), free(PY), free(Cd), free(Cg), free(T1), free(T2), free(T3), free(ev), free(fw);
  }
  free(U0);
  return k;
}

// ---------------------------------------------------------------- Kalman
// The collapsed likelihood of a latent Gaussian random walk observed with
// Gaussian noise (see `detect_kalman` in compiler/src/model.rs). For each of
// G independent series and t = 0..T-1:
//
//   x[t] = x[t-1] + e[t],  e[t] ~ N(d[t], q[t])   (x[-1] = 0; q is a variance)
//   y[t] = a[t] + c x[t] + noise,  noise ~ N(0, r[t])   (r is a variance)
//
// mint_kalman_ll returns log p(y | a, d, q, r) with the x integrated out,
// without the -0.5 log(2 pi) per observation (Mint drops constants), and
// overwrites a, d, q and r with the gradient of that log density with
// respect to each of them. Every array is time-major, element (g, t) at
// t * G + g, so the loops over series are contiguous and vectorise. ws holds
// 3 G T + 6 G doubles.
//
// Forward (per series): mp = m + d, Pp = P + q, F = c^2 Pp + r,
// v = y - a - c mp, ll += -0.5 log F - 0.5 v^2 / F, K = c Pp / F,
// m' = mp + K v, P' = Pp r / F (the update in a form that stays positive).
// Reverse: the adjoint of each line, in the opposite order, with the
// adjoints of m' and P' carried from the step after (zero after the last).
//
// sum_t log F[t] is taken as the log of a running product per series whose
// binary exponent is moved into an integer counter at every step (exact),
// so the loop needs no log call; if any F is outside [2^-1022, 2^1022] the
// sum is recomputed with log (a zero, infinite or NaN F then gives NaN).
// The forward pass stores 1/F, so the reverse pass divides by nothing.

static inline double kal_split(double x, int64_t *e) {
  uint64_t b;
  memcpy(&b, &x, sizeof b);
  *e += (int64_t)((b >> 52) & 0x7ff) - 1023;
  b = (b & 0x800fffffffffffffull) | 0x3ff0000000000000ull;
  memcpy(&x, &b, sizeof b);
  return x;
}

double mint_kalman_ll(int64_t G, int64_t T, double c, const double *restrict y, double *restrict a,
                      double *restrict d, double *restrict q, double *restrict r, double *restrict ws) {
  // per element: Pp, 1/F and v, for the reverse pass
  double *restrict PPs = ws, *restrict IFs = ws + G * T, *restrict VVs = ws + 2 * G * T;
  double *restrict m = ws + 3 * G * T, *restrict P = m + G, *restrict prod = P + G;
  double *restrict mb = prod + G, *restrict pb = mb + G;
  int64_t *restrict ex = (int64_t *)(pb + G);
  double c2 = c * c, quad = 0.0;
  int bad = 0;
  for (int64_t g = 0; g < G; g++) m[g] = 0.0, P[g] = 0.0, prod[g] = 1.0, ex[g] = 0;
  for (int64_t t = 0; t < T; t++) {
    const int64_t o = t * G;
#pragma omp simd reduction(+ : quad) reduction(| : bad)
    for (int64_t g = 0; g < G; g++) {
      int64_t i = o + g;
      double mp = m[g] + d[i], Pp = P[g] + q[i];
      double F = c2 * Pp + r[i];
      double v = y[i] - a[i] - c * mp;
      double iF = 1.0 / F;
      quad += v * v * iF;
      bad |= !(F >= 0x1p-1022 && F <= 0x1p+1022);  // so that prod * F stays normal
      double K = c * Pp * iF;
      m[g] = mp + K * v;
      P[g] = Pp * r[i] * iF;
      prod[g] = kal_split(prod[g] * F, &ex[g]);
      PPs[i] = Pp, IFs[i] = iF, VVs[i] = v;
    }
  }
  double logdet = 0.0;
  if (!bad) {
    int64_t esum = 0;
    for (int64_t g = 0; g < G; g++) logdet += log(prod[g]), esum += ex[g];
    logdet += (double)esum * 0.69314718055994530942;
  } else {
    for (int64_t i = 0; i < G * T; i++) logdet += log(c2 * PPs[i] + r[i]);  // F (r is still the input)
  }
  for (int64_t g = 0; g < G; g++) mb[g] = 0.0, pb[g] = 0.0;
  for (int64_t t = T - 1; t >= 0; t--) {
    const int64_t o = t * G;
#pragma omp simd
    for (int64_t g = 0; g < G; g++) {
      int64_t i = o + g;
      double Pp = PPs[i], iF = IFs[i], v = VVs[i], rr = r[i];
      double K = c * Pp * iF;
      double mbar = mb[g], pbar = pb[g];
      double Kb = mbar * v;
      double vb = mbar * K - v * iF;
      double mpb = mbar;
      // P' = Pp r / F
      double Ppb = pbar * rr * iF;
      double rb = pbar * Pp * iF;
      double Fb = -pbar * Pp * rr * iF * iF;
      // K = c Pp / F
      Ppb += Kb * c * iF;
      Fb -= Kb * K * iF;
      // ll = -0.5 log F - 0.5 v^2 / F
      Fb += 0.5 * iF * (v * v * iF - 1.0);
      // v = y - a - c mp
      double ab = -vb;
      mpb -= c * vb;
      // F = c^2 Pp + r
      Ppb += c2 * Fb;
      rb += Fb;
      // Pp = P + q, mp = m + d
      mb[g] = mpb, pb[g] = Ppb;
      a[i] = ab, d[i] = mpb, q[i] = Ppb, r[i] = rb;
    }
  }
  return -0.5 * (logdet + quad);
}

// The same when q and r are the same for every series (they depend on time
// at most: the compiler proved it from the expressions), so that q and r
// hold T values: the variances (the Riccati recursion: Pp, F, K, P) are
// then computed once per time step instead of once per series, and only
// the means are filtered per series. Gradients: a and d per element as
// above, q[t] and r[t] summed over the series. ws holds G T + 2 G + 3 T
// doubles.
double mint_kalman_ll_shared(int64_t G, int64_t T, double c, const double *restrict y, double *restrict a,
                             double *restrict d, double *restrict q, double *restrict r, double *restrict ws) {
  double *restrict VVs = ws, *restrict m = ws + G * T, *restrict mb = m + G;
  double *restrict PPt = mb + G, *restrict IFt = PPt + T, *restrict Kt = IFt + T;
  double c2 = c * c, P = 0.0, logdet = 0.0, quad = 0.0;
  for (int64_t t = 0; t < T; t++) {
    double Pp = P + q[t], F = c2 * Pp + r[t], iF = 1.0 / F;
    PPt[t] = Pp, IFt[t] = iF, Kt[t] = c * Pp * iF;
    P = Pp * r[t] * iF;
    logdet += log(F);
  }
  for (int64_t g = 0; g < G; g++) m[g] = 0.0, mb[g] = 0.0;
  for (int64_t t = 0; t < T; t++) {
    const int64_t o = t * G;
    const double K = Kt[t];
    double vv = 0.0;
#pragma omp simd reduction(+ : vv)
    for (int64_t g = 0; g < G; g++) {
      int64_t i = o + g;
      double mp = m[g] + d[i];
      double v = y[i] - a[i] - c * mp;
      m[g] = mp + K * v;
      vv += v * v;
      VVs[i] = v;
    }
    quad += vv * IFt[t];
  }
  double pbar = 0.0;  // adjoint of the filtered variance P[t] (shared)
  for (int64_t t = T - 1; t >= 0; t--) {
    const int64_t o = t * G;
    const double K = Kt[t], iF = IFt[t], Pp = PPt[t], rr = r[t];
    double Kb = 0.0, vv = 0.0;
#pragma omp simd reduction(+ : Kb, vv)
    for (int64_t g = 0; g < G; g++) {
      int64_t i = o + g;
      double mbar = mb[g], v = VVs[i];
      Kb += mbar * v;
      vv += v * v;
      double vb = mbar * K - v * iF;
      mb[g] = mbar - c * vb;
      a[i] = -vb, d[i] = mbar - c * vb;
    }
    // the shared recursion's adjoints, as in mint_kalman_ll, summed over series
    double Ppb = pbar * rr * iF + Kb * c * iF;
    double rb = pbar * Pp * iF;
    double Fb = -pbar * Pp * rr * iF * iF - Kb * K * iF + 0.5 * iF * (vv * iF - (double)G);
    Ppb += c2 * Fb;
    rb += Fb;
    pbar = Ppb;
    q[t] = Ppb, r[t] = rb;
  }
  return -0.5 * ((double)G * logdet + quad);
}

// n standard normal variates by the ziggurat method (Marsaglia and Tsang
// 2000, in Doornik's 2005 form with 128 blocks): most variates cost one
// 64-bit random number, a compare and a multiply, against a log, a square
// root and a division for the polar method. Tables are built once.
#define ZIG_C 128
#define ZIG_R 3.442619855899
#define ZIG_V 9.91256303526217e-3
static double zig_x[ZIG_C + 1], zig_ratio[ZIG_C];
static pthread_once_t zig_once = PTHREAD_ONCE_INIT;
static void zig_init(void) {
  double f = exp(-0.5 * ZIG_R * ZIG_R);
  zig_x[0] = ZIG_V / f;
  zig_x[1] = ZIG_R;
  zig_x[ZIG_C] = 0.0;
  for (int i = 2; i < ZIG_C; i++) {
    zig_x[i] = sqrt(-2.0 * log(ZIG_V / zig_x[i - 1] + f));
    f = exp(-0.5 * zig_x[i] * zig_x[i]);
  }
  for (int i = 0; i < ZIG_C; i++) zig_ratio[i] = zig_x[i + 1] / zig_x[i];
}
static double zig_normal(Rng *r) {
  for (;;) {
    uint64_t b = rng_next(r);
    int i = (int)(b & (ZIG_C - 1));
    double u = 2.0 * ((double)(b >> 11) * 0x1.0p-53) - 1.0;  // bits independent of i
    if (fabs(u) < zig_ratio[i]) return u * zig_x[i];
    if (i == 0) {  // the tail beyond R
      double x, y;
      do {
        x = log(1.0 - rng_uniform(r)) / ZIG_R;
        y = log(1.0 - rng_uniform(r));
      } while (-2.0 * y < x * x);
      return u < 0 ? x - ZIG_R : ZIG_R - x;
    }
    double x = u * zig_x[i];
    double f0 = exp(-0.5 * (zig_x[i] * zig_x[i] - x * x));
    double f1 = exp(-0.5 * (zig_x[i + 1] * zig_x[i + 1] - x * x));
    if (f1 + rng_uniform(r) * (f0 - f1) < 1.0) return x;
  }
}
static void rng_normals(Rng *r, double *out, int64_t n) {
  pthread_once(&zig_once, zig_init);
  for (int64_t i = 0; i < n; i++) out[i] = zig_normal(r);
}

// Forward filtering, backward sampling: one draw of the latent paths given
// the same inputs (not modified), written as increments e[t] = x[t] - x[t-1]
// (time-major like the inputs). rng is the sampler's per-chain stream for
// these draws (an Rng); the G T normal variates are drawn first, so the
// backward pass vectorises. With `shared`, q and r hold T values as for
// mint_kalman_ll_shared. ws as for mint_kalman_ll.
void mint_kalman_ffbs(int64_t G, int64_t T, double c, const double *restrict y, const double *restrict a,
                      const double *restrict d, const double *restrict q, const double *restrict r,
                      int64_t shared, double *restrict ws, void *rng, double *restrict e) {
  double *restrict Ms = ws, *restrict Ps = ws + G * T, *restrict Z = ws + 2 * G * T;
  double *restrict m = ws + 3 * G * T, *restrict P = m + G, *restrict x = P + G;
  const int64_t qs = shared ? 0 : 1;  // stride of q and r within a time step
  double c2 = c * c;
  for (int64_t g = 0; g < G; g++) m[g] = 0.0, P[g] = 0.0;
  for (int64_t t = 0; t < T; t++) {
    const int64_t o = t * G;
    const double *restrict qt = q + (shared ? t : o), *restrict rt = r + (shared ? t : o);
#pragma omp simd
    for (int64_t g = 0; g < G; g++) {
      int64_t i = o + g;
      double mp = m[g] + d[i], Pp = P[g] + qt[g * qs], rr = rt[g * qs];
      double F = c2 * Pp + rr;
      double v = y[i] - a[i] - c * mp;
      double iF = 1.0 / F;
      m[g] = mp + c * Pp * iF * v;
      P[g] = Pp * rr * iF;
      Ms[i] = m[g], Ps[i] = P[g];
    }
  }
  rng_normals(rng, Z, G * T);
  const int64_t last = (T - 1) * G;
  for (int64_t g = 0; g < G; g++) {
    double var = Ps[last + g];
    x[g] = Ms[last + g] + sqrt(var > 0.0 ? var : 0.0) * Z[last + g];
  }
  for (int64_t t = T - 2; t >= 0; t--) {
    const int64_t o = t * G;
    const double *restrict q1t = q + (shared ? t + 1 : o + G);
#pragma omp simd
    for (int64_t g = 0; g < G; g++) {
      int64_t i = o + g;
      // x[t] | x[t+1], y[0..t]: Pp1 = P[t] + q[t+1], J = P[t] / Pp1
      double mean = Ms[i], var = Ps[i], q1 = q1t[g * qs], Pp1 = var + q1;
      double J = Pp1 > 0.0 ? var / Pp1 : 0.0;
      double cv = Pp1 > 0.0 ? var * q1 / Pp1 : 0.0;
      double xt = mean + J * (x[g] - mean - d[i + G]) + sqrt(cv > 0.0 ? cv : 0.0) * Z[i];
      e[i + G] = x[g] - xt;
      x[g] = xt;
    }
  }
  for (int64_t g = 0; g < G; g++) e[g] = x[g];
}

// One standard normal from the collapsed draws' stream (for an element whose
// coefficient on the walk is zero, so that only its prior applies).
double mint_kalman_normal(void *rng) { return rng_normal(rng); }

// A model with collapsed parameters (mintc's Kalman collapse): NUTS runs
// over the remaining D parameters, and for each kept draw `fn` writes the
// whole constrained draw (d_out values, in the user's order, the collapsed
// ones drawn by mint_kalman_ffbs from the rng it is given). Registered by
// the generated sample function before it calls mint_sample (0 / NULL: no
// collapse; the Rust baselines never set it).
typedef void (*mint_collapse_fn)(const double *unc, double *out, void *rng);
static mint_collapse_fn model_collapse;
static int64_t model_d_out, model_n_latent;
void mint_set_collapsed(int64_t d_out, mint_collapse_fn fn, int64_t n_latent) {
  model_collapse = fn;
  model_d_out = d_out;
  model_n_latent = n_latent;
}

// ---------------------------------------------------------------- NUTS
// Multinomial NUTS with the generalised no-U-turn criterion, diagonal metric
// and Stan's windowed warmup (step-size dual averaging plus metric windows).
// This is a port of the structure of Stan's base_nuts.hpp.

typedef double (*mint_logp_fn)(const double *theta, double *grad);
typedef void (*mint_constrain_fn)(const double *unc, double *out);

// The fused leapfrog (optional, generated for models with a fused scan
// kernel; see gen_logp in compiler/src/model.rs). leap(theta, grad, hook,
// hctx) is the model's logp, except that each thread of a fused scan kernel
// that owns matrix parameters, once the gradient of its rows is final,
// calls hook(hctx, slot, lo, len): in the scan layout a thread's rows are
// one contiguous range theta[lo .. lo + len) of each such parameter. The
// sampler does its leaf work on that range there (leaf_block): on the
// thread that has just written its gradient and read its position, inside
// the kernel's parallel region, instead of in a parallel region of its own
// split differently. slot is the kernel's thread index (0 for the calling
// thread, which also takes the rows left over after the groups).
// leap_blocks(out) writes (offset, length) of each parameter the hooks
// cover and returns their number; the sampler does the rest of theta
// itself.
typedef void (*mint_leap_hook)(void *hctx, int64_t slot, int64_t lo, int64_t len);
typedef double (*mint_leap_fn)(const double *theta, double *grad, mint_leap_hook hook, void *hctx);
typedef int64_t (*mint_leap_blocks_fn)(int64_t *out);
#define MAX_LEAP_BLOCKS 64

#define MAX_DEPTH 10
#define LR_KMAX 128  // most directions of the low-rank metric

// Phase-space states are immutable once built and shared by reference: a
// leapfrog step writes a new state instead of updating one in place, and the
// tree keeps references to its end points and proposals instead of copying
// D-length vectors. States are reference counted and recycled from a
// per-chain free list. The arithmetic and the order of random draws are
// exactly those of Stan's base_nuts structure.
//
// With the low-rank metric (MINT_METRIC=lowrank) p and g carry, after their D
// entries, their k projections on the metric's directions (see Nuts); so do
// the momentum sums of the tree.
typedef struct St {
  double *q, *p, *g;  // position, momentum, gradient
  double lp, h;            // log density, Hamiltonian
  int rc;
  struct St *next_free;
} St;

typedef struct {
  St *init_end, *final_beg, *prop_final;
  double *rho_init;
} Level;

// A merge that waits for the last leaf of its subtree. The merge of two
// subtrees (summed momentum and the three no-U-turn checks) needs nothing
// that is not known once that leaf exists, so the leaf's second half-step
// computes the sums of every merge it completes in the same pass over D (see
// leaf_finish). States are read through their slots when the leaf runs.
typedef struct {
  double *rho;            // this subtree's summed momentum, stored only when the
                          // merge is the last one the leaf completes
  const double *ra, *rb;  // the two halves' sums; NULL is "the subtree below"
                          // (the leaf's own momentum at the bottom)
  St **beg, **end, **mid2, **mid1;
  int persist;            // the result, filled in by the leaf
} Merge;

#define MAX_NT 64
// part[t]: the kinetic energy and six sums per merge, then (low-rank metric)
// the k projections of the leaf's gradient
#define NPART_MERGE (1 + 6 * (MAX_DEPTH + 2))
#define NPART (NPART_MERGE + LR_KMAX)
#define SLOT_OTHER MAX_NT  // partial sums of the parameters no hook covers

typedef struct {
  int64_t D;
  mint_logp_fn f;
  mint_leap_fn leap;  // the model's fused leapfrog, or NULL (see mint_leap_fn)
  int leap_exact;     // with leap: the sums in the runtime's own order (MINT_FUSED_LEAPFROG=exact)
  int nother;         // with leap: the ranges of theta no hook covers
  int64_t other[2 * (MAX_LEAP_BLOCKS + 1)];
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
  int64_t n_fused;  // leaves that ran through the fused leapfrog
  double sum_metro;
  int divergent;
  int depth;
  int nt;        // threads splitting this chain's D-length passes (1 = serial)
  int team_min;  // smallest OpenMP team that actually ran one of those passes
  // merges waiting for a leaf: pend[pend_lo .. npend-1], innermost last
  Merge pend[MAX_DEPTH + 2];
  int npend, pend_lo;
  // the next leaf's first half-step, computed ahead by the previous leaf
  St *spec;
  const St *spec_from;
  double spec_eps;
  double part[MAX_NT + 1][NPART];  // per-thread partial sums of a leaf pass (and SLOT_OTHER)
  // Low-rank metric (MINT_METRIC=lowrank; k = 0 otherwise):
  //   inverse metric = diag(inv_m) + sum_j lr_d[j] v_j v_j',  v_j = sqrt(inv_m) u_j,
  // with orthonormal u_j (before rounding). Every momentum, momentum sum and
  // gradient carries its k projections c_j = v_j . x in the LR_KMAX slots after
  // its D entries, so the kinetic energy and the no-U-turn checks need O(k)
  // extra work. The v_j are stored in single precision (half the memory
  // traffic of the O(D k) passes); all arithmetic on them is in double except
  // the position update's sum, and the metric is exactly the one the stored
  // values define (lr_a makes the momentum draws exact for it).
  int k;
  float *lr_v;                     // tiled, see LR_AT
  int64_t lr_nb;                   // tiles per direction group: ceil(D / 8)
  double lr_d[LR_KMAX];            // lam_j - 1
  double lr_a[LR_KMAX * LR_KMAX];   // k x k, for drawing momenta (lr_set)
  double lr_gm[LR_KMAX * LR_KMAX];  // k x k: v_a . (v_b / inv_m)
} Nuts;

#define MAX_NT 64

// ---------------------------------------------------------------- threads inside the gradient
//
// A fused scan kernel (compiler/src/model.rs, gen_fused_scan) works on
// independent groups of rows. The generated logp hands the groups to
// mint_par_groups, which splits them across the calling chain's threads:
// thread t of a team of T runs groups [n t / T, n (t + 1) / T) and writes its
// partial sums to slot t, which the generated code adds up in thread order.
// So the gradient is deterministic for a given team size. The thread count is
// per calling thread: run_chain sets it to the chain's threads per chain;
// elsewhere (the gradient benchmark, the gradient check) it is 1.
// MINT_KERNEL_THREADS overrides both.
typedef void (*mint_group_fn)(void *ctx, int64_t g0, int64_t g1, int64_t tid);
static __thread int64_t kernel_nt = 1;

static int64_t kernel_threads(int64_t dflt) {
  const char *e = getenv("MINT_KERNEL_THREADS");
  int64_t n = e ? atoll(e) : dflt;
  if (n < 1) n = 1;
  if (n > MAX_NT) n = MAX_NT;  // the generated code has room for MAX_NT slots
  return n;
}

int64_t mint_par_threads(void) { return kernel_nt; }

// Runs fn over groups 0..ngroups on up to nt threads; returns the number of
// threads (slots) that ran. With one thread fn runs here, without OpenMP.
int64_t mint_par_groups(mint_group_fn fn, void *ctx, int64_t ngroups, int64_t nt) {
  if (nt > ngroups) nt = ngroups;
  if (nt > MAX_NT) nt = MAX_NT;
  if (nt <= 1) {
    fn(ctx, 0, ngroups, 0);
    return 1;
  }
  // OpenMP may run a smaller team than asked for; the split follows the team
  int used = 1;
#pragma omp parallel num_threads((int)nt)
  {
    int t = omp_get_thread_num(), T = omp_get_num_threads();
    if (t == 0) used = T;
    fn(ctx, ngroups * t / T, ngroups * (t + 1) / T, t);
  }
  return used;
}

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

static double hamiltonian(Nuts *s, const St *z) {
  double k = 0;
  for (int64_t i = 0; i < s->D; i++) k += z->p[i] * z->p[i] * s->inv_m[i];
  if (s->k) {  // low-rank part, from the stored projections
    const double *c = z->p + s->D;
    for (int j = 0; j < s->k; j++) k += s->lr_d[j] * c[j] * c[j];
  }
  return -z->lp + 0.5 * k;
}

static void eval(Nuts *s, St *z) {
  z->lp = s->f(z->q, z->g);
  s->n_grad++;
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

// ---- the sampler's D-length passes
//
// Every sum over D in these passes (the leaves' kinetic energy and the
// no-U-turn checks; not hamiltonian(), a plain loop used for the starting
// energy and the step-size search) is accumulated in LN lanes: element i goes
// to lane i % LN, each lane adds its elements in index order, and the lanes
// are combined in a fixed tree (lanes_total). When a chain's passes are split across threads,
// each thread's range starts at a multiple of LN and the threads' totals are
// added in thread order. The result therefore does not depend on how a pass
// is blocked or fused, or on the compiler's vectorisation; it does depend on
// the number of threads. The fused leapfrog (leaf_fused) sums in another
// order: see there.
#define LN 8
#define CHUNK 512  // block of the fused leaf pass, a multiple of LN

static void split_range(int64_t D, int t, int T, int64_t *lo, int64_t *hi) {
  int64_t nb = (D + LN - 1) / LN;
  int64_t a = nb * t / T * LN, b = nb * (t + 1) / T * LN;
  *lo = a < D ? a : D;
  *hi = b < D ? b : D;
}

static double lanes_total(const double *a) {
  return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
}

// First half-step of a leapfrog from z into n: half-step momentum and the new
// position, over [lo, hi).
static void kern_half1(int64_t lo, int64_t hi, double eps, const double *restrict im,
                       const double *restrict zp, const double *restrict zg, const double *restrict zq,
                       double *restrict np, double *restrict nq) {
  for (int64_t i = lo; i < hi; i++) {
    double ph = zp[i] + 0.5 * eps * zg[i];
    np[i] = ph;
    nq[i] = zq[i] + eps * im[i] * ph;
  }
}

// Second half-step over [lo, hi) (lo a multiple of LN in the leaf pass; the
// fused leapfrog's hook passes any lo, which changes only the order of the
// lanes' sums): the final momentum
// p, its kinetic-energy lanes k, and, when xp is not NULL, the first
// half-step of the next leapfrog from this state (same arithmetic as
// kern_half1).
static void kern_half2(int64_t lo, int64_t hi, double eps, const double *restrict im, double *restrict p,
                       const double *restrict g, const double *restrict q, double *restrict xp,
                       double *restrict xq, double *restrict k) {
  double kk[LN];
  for (int l = 0; l < LN; l++) kk[l] = k[l];
#define HALF2(e, l)                       \
  do {                                    \
    double v = p[e] + 0.5 * eps * g[e];   \
    p[e] = v;                             \
    kk[l] += v * v * im[e];               \
  } while (0)
#define HALF2_NEXT(e, l)                  \
  do {                                    \
    double v = p[e] + 0.5 * eps * g[e];   \
    p[e] = v;                             \
    kk[l] += v * v * im[e];               \
    double ph = v + 0.5 * eps * g[e];     \
    xp[e] = ph;                           \
    xq[e] = q[e] + eps * im[e] * ph;      \
  } while (0)
  int64_t i = lo;
  if (xp) {
    for (; i + LN <= hi; i += LN)
      for (int l = 0; l < LN; l++) HALF2_NEXT(i + l, l);
    for (; i < hi; i++) HALF2_NEXT(i, (int)(i % LN));
  } else {
    for (; i + LN <= hi; i += LN)
      for (int l = 0; l < LN; l++) HALF2(i + l, l);
    for (; i < hi; i++) HALF2(i, (int)(i % LN));
  }
#undef HALF2
#undef HALF2_NEXT
  for (int l = 0; l < LN; l++) k[l] = kk[l];
}

// A merge over n elements (pointers already offset to the block; the block
// starts at a multiple of LN): out = ra + rb and the lanes of the three
// no-U-turn checks
//   (a1, b1) = (end_ps . out, beg_ps . out)
//   (a2, b2) = (mid2_ps . (ra + mid2_p), beg_ps . (ra + mid2_p))
//   (a3, b3) = (end_ps . (rb + mid1_p), mid1_ps . (rb + mid1_p))
// Scaled momenta (inverse metric * p) are recomputed instead of being stored
// with every state: (im * p) rounds exactly as a stored value would. out must
// not overlap any input.
static void kern_merge(int64_t n, const double *restrict im, const double *restrict ra,
                       const double *restrict rb, const double *restrict bp, const double *restrict ep,
                       const double *restrict m2, const double *restrict m1, double *restrict out,
                       double (*restrict acc)[LN]) {
  double a1[LN], b1[LN], a2[LN], b2[LN], a3[LN], b3[LN];
  for (int l = 0; l < LN; l++)
    a1[l] = acc[0][l], b1[l] = acc[1][l], a2[l] = acc[2][l], b2[l] = acc[3][l], a3[l] = acc[4][l],
    b3[l] = acc[5][l];
#define MERGE(e, l)                                  \
  do {                                               \
    double beg_ps = im[e] * bp[e], end_ps = im[e] * ep[e]; \
    double xa = ra[e], xb = rb[e];                   \
    double r = xa + xb;                              \
    out[e] = r;                                      \
    a1[l] += end_ps * r;                             \
    b1[l] += beg_ps * r;                             \
    double r2 = xa + m2[e];                          \
    a2[l] += (im[e] * m2[e]) * r2;                   \
    b2[l] += beg_ps * r2;                            \
    double r3 = xb + m1[e];                          \
    a3[l] += end_ps * r3;                            \
    b3[l] += (im[e] * m1[e]) * r3;                   \
  } while (0)
  int64_t i = 0;
  for (; i + LN <= n; i += LN)
    for (int l = 0; l < LN; l++) MERGE(i + l, l);
  for (; i < n; i++) MERGE(i, (int)(i % LN));
#undef MERGE
  for (int l = 0; l < LN; l++)
    acc[0][l] = a1[l], acc[1][l] = b1[l], acc[2][l] = a2[l], acc[3][l] = b2[l], acc[4][l] = a3[l],
    acc[5][l] = b3[l];
}

// The merge of two single leaves with momenta ra and rb (half of all merges):
// the three checks reduce to one pair, (rb_ps . out, ra_ps . out), with
// exactly the arithmetic of kern_merge's (a1, b1).
static void kern_merge_leaves(int64_t n, const double *restrict im, const double *restrict ra,
                              const double *restrict rb, double *restrict out, double (*restrict acc)[LN]) {
  double a1[LN], b1[LN];
  for (int l = 0; l < LN; l++) a1[l] = acc[0][l], b1[l] = acc[1][l];
#define MERGE2(e, l)                 \
  do {                               \
    double xa = ra[e], xb = rb[e];   \
    double r = xa + xb;              \
    out[e] = r;                      \
    a1[l] += (im[e] * xb) * r;       \
    b1[l] += (im[e] * xa) * r;       \
  } while (0)
  int64_t i = 0;
  for (; i + LN <= n; i += LN)
    for (int l = 0; l < LN; l++) MERGE2(i + l, l);
  for (; i < n; i++) MERGE2(i, (int)(i % LN));
#undef MERGE2
  for (int l = 0; l < LN; l++) acc[0][l] = a1[l], acc[1][l] = b1[l];
}

// ---- the low-rank metric's O(D k) kernels
//
// The directions are stored in tiles of 8 directions x 8 consecutive entries
// (256 bytes): direction j, entry i lives in tile (j / 8, i / 8). The passes
// stream one array with full-width vector operations. Padding (directions
// k .. 8 ng - 1 and entries D .. 8 nb - 1) is zero. The threads of a chain
// split D at multiples of 8 entries (split_range), which are tile boundaries.
#define LR_AT(s, j, i) \
  ((s)->lr_v[(((int64_t)((j) >> 3) * (s)->lr_nb + ((i) >> 3)) * 8 + ((j) & 7)) * 8 + ((i) & 7)])

static inline v4d v8f_lo(v8f x) { return __builtin_convertvector(__builtin_shufflevector(x, x, 0, 1, 2, 3), v4d); }
static inline v4d v8f_hi(v8f x) { return __builtin_convertvector(__builtin_shufflevector(x, x, 4, 5, 6, 7), v4d); }

// q[i] += sum_j w[j] v_j[i] for the entries of tiles [tb, te). The sum is
// formed in single precision (eight lanes per instruction, directions in
// index order, each step a fused multiply-add, the same for every entry
// whichever loop handles it) and added to q in double. The position update
// q += eps f(p) is volume-preserving, and reversible in the same sense as any
// floating-point leapfrog, for any f that is odd in p, which a rounded
// linear map is; so the single-precision sum changes only how well energy is
// conserved, not what the sampler targets. wf has 8 ng entries, zero beyond k.
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
        a0 = __builtin_elementwise_fma(w, *(const v8f *)(t), a0);
        a1 = __builtin_elementwise_fma(w, *(const v8f *)(t + 64), a1);
        a2 = __builtin_elementwise_fma(w, *(const v8f *)(t + 128), a2);
        a3 = __builtin_elementwise_fma(w, *(const v8f *)(t + 192), a3);
        a4 = __builtin_elementwise_fma(w, *(const v8f *)(t + 256), a4);
        a5 = __builtin_elementwise_fma(w, *(const v8f *)(t + 320), a5);
        a6 = __builtin_elementwise_fma(w, *(const v8f *)(t + 384), a6);
        a7 = __builtin_elementwise_fma(w, *(const v8f *)(t + 448), a7);
      }
    v8f acc[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    for (int m = 0; m < 8; m++) {
      double *x = q + (ib + m) * 8;
      *(v4d *)x += v8f_lo(acc[m]);
      *(v4d *)(x + 4) += v8f_hi(acc[m]);
    }
  }
  for (; ib < te; ib++) {
    v8f a = {0};
    for (int g = 0; g < ng; g++)
      for (int j = 0; j < 8; j++) {
        float ws = wf[8 * g + j];
        v8f w = {ws, ws, ws, ws, ws, ws, ws, ws};
        a = __builtin_elementwise_fma(w, *(const v8f *)(V + (((int64_t)g * nb + ib) * 8 + j) * 8), a);
      }
    double *x = q + ib * 8;
    if (ib < full) {
      *(v4d *)x += v8f_lo(a);
      *(v4d *)(x + 4) += v8f_hi(a);
    } else {
      for (int ii = 0; ib * 8 + ii < D; ii++) x[ii] += (double)a[ii];
    }
  }
}

// c[j] += v_j . x over the entries of tiles [tb, te), in double (products of
// the stored single-precision values with x are formed exactly as double
// arithmetic would). Four directions at a time; within each, the entries are
// summed in four lanes and the lanes combined in a fixed order, so the result
// depends only on [tb, te).
static void lr_dots(const Nuts *s, double *c, const double *restrict x, int64_t tb, int64_t te) {
  int64_t D = s->D, nb = s->lr_nb;
  int ng = (s->k + 7) >> 3;
  const float *restrict V = s->lr_v;
  int64_t full = D >> 3;
  int64_t tf = te < full ? te : full;
  for (int g = 0; g < ng; g++)
    for (int h = 0; h < 8; h += 4) {
      if (8 * g + h >= s->k) break;
      v4d a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
      for (int64_t ib = tb; ib < tf; ib++) {
        v4d xl = *(const v4d *)(x + ib * 8), xh = *(const v4d *)(x + ib * 8 + 4);
        const float *t = V + ((int64_t)g * nb + ib) * 64 + h * 8;
        a0 = __builtin_elementwise_fma(v8f_lo(*(const v8f *)(t)), xl, a0);
        a1 = __builtin_elementwise_fma(v8f_lo(*(const v8f *)(t + 8)), xl, a1);
        a2 = __builtin_elementwise_fma(v8f_lo(*(const v8f *)(t + 16)), xl, a2);
        a3 = __builtin_elementwise_fma(v8f_lo(*(const v8f *)(t + 24)), xl, a3);
        a0 = __builtin_elementwise_fma(v8f_hi(*(const v8f *)(t)), xh, a0);
        a1 = __builtin_elementwise_fma(v8f_hi(*(const v8f *)(t + 8)), xh, a1);
        a2 = __builtin_elementwise_fma(v8f_hi(*(const v8f *)(t + 16)), xh, a2);
        a3 = __builtin_elementwise_fma(v8f_hi(*(const v8f *)(t + 24)), xh, a3);
      }
      v4d acc[4] = {a0, a1, a2, a3};
      for (int j = 0; j < 4 && 8 * g + h + j < s->k; j++)
        c[8 * g + h + j] += (acc[j][0] + acc[j][1]) + (acc[j][2] + acc[j][3]);
      if (tb <= full && full < te) {  // the partial last tile (D not a multiple of 8)
        for (int j = 0; j < 4 && 8 * g + h + j < s->k; j++) {
          const float *t = V + (((int64_t)g * nb + full) * 8 + h + j) * 8;
          double sx = 0;
          for (int64_t ii = 0; full * 8 + ii < D; ii++) sx += (double)t[ii] * x[full * 8 + ii];
          c[8 * g + h + j] += sx;
        }
      }
    }
}

#define LR_CHUNK (CHUNK / 8)  // tiles per block of a projection

// c_j = v_j . x for the k directions (x has D entries), split across the
// chain's threads like the leaf passes and blocked like them
static void lr_project(Nuts *s, const double *x, double *c) {
  int nt = s->nt, k = s->k;
  int used = 1;
#pragma omp parallel num_threads(nt) if (nt > 1)
  {
    int t = omp_get_thread_num(), T = omp_get_num_threads();
    if (t == 0) used = T;
    double *part = s->part[t];
    for (int j = 0; j < k; j++) part[j] = 0;
    int64_t lo, hi;
    split_range(s->D, t, T, &lo, &hi);
    for (int64_t c0 = lo; c0 < hi; c0 += CHUNK) {
      int64_t c1 = c0 + CHUNK < hi ? c0 + CHUNK : hi;
      lr_dots(s, part, x, c0 >> 3, (c1 + 7) >> 3);
    }
  }
  if (used < s->team_min) s->team_min = used;
  for (int j = 0; j < k; j++) {
    double acc = 0;
    for (int t = 0; t < used; t++) acc += s->part[t][j];
    c[j] = acc;
  }
}

// The projections of a leaf's momenta from those of its half-step momentum
// (cph) and of its new gradient (cg): the final momentum p = ph + eps/2 g and
// the next leaf's half-step momentum p + eps/2 g (momenta and gradients
// project linearly). Shared by the leaf pass and the code after it, so that
// both get the same values.
static inline void lr_next(int k, double eps, const double *cph, const double *cg, double *cp, double *cx) {
  for (int j = 0; j < k; j++) {
    double p = cph[j] + 0.5 * eps * cg[j];
    cp[j] = p;
    cx[j] = p + 0.5 * eps * cg[j];
  }
}

// weights of the position update q += eps (inv_m * ph + sum_j d_j c_j(ph) v_j)
static inline void lr_weights(const Nuts *s, double eps, const double *cph, float *wf) {
  int k = s->k;
  for (int j = 0; j < k; j++) wf[j] = (float)(eps * s->lr_d[j] * cph[j]);
  for (int j = k; j < ((k + 7) & ~7); j++) wf[j] = 0;
}

// ---- a leaf of the tree
//
// A leaf costs two passes over D around its gradient. The first half-step
// writes the new state's position. The second (leaf_finish) writes its
// momentum, sums its kinetic energy, computes every merge this leaf completes
// (when it is the last leaf of one or more subtrees), and, unless the leaf
// ends the trajectory's new subtree, also takes the first half-step of the
// next leaf, so that most leaves need only this one pass. A merge's sum is
// stored only for the outermost merge of the leaf; the inner ones are only
// read by the merge above them and stay in a block-sized buffer. The
// arithmetic of every stored value is that of a plain leapfrog followed by
// separate merges; only the order of the sums differs (see LN above).

typedef struct {
  const double *ra, *rb;  // NULL: the merge below (the leaf's momentum at the bottom)
  const double *bp, *ep, *m2p, *m1p;
  double *out;            // NULL except for the outermost merge
  int leaves;             // a merge of two single leaves
} LevelJob;

typedef struct {
  Nuts *s;
  St *n, *nx;  // the leaf, and the next leaf (NULL: none)
  double eps;
  int nlev;
  int half;    // 1: the second half-step (and the next leaf's first); 2: only the kinetic energy of n->p
  int merges;  // whether to compute the merges
  LevelJob lev[MAX_DEPTH + 2];
} LeafJob;

// The kinetic-energy lanes of a final momentum p over [lo, hi), with the
// arithmetic and order of kern_half2's.
static void kern_kinetic(int64_t lo, int64_t hi, const double *restrict im, const double *restrict p,
                         double *restrict k) {
  double kk[LN];
  for (int l = 0; l < LN; l++) kk[l] = k[l];
  int64_t i = lo;
  for (; i + LN <= hi; i += LN)
    for (int l = 0; l < LN; l++) kk[l] += p[i + l] * p[i + l] * im[i + l];
  for (; i < hi; i++) kk[i % LN] += p[i] * p[i] * im[i];
  for (int l = 0; l < LN; l++) k[l] = kk[l];
}

// Leaf J's work on elements [lo, hi), accumulating the kinetic-energy lanes
// k and the merges' lanes acc. With cg (the low-rank metric, and only for
// ranges that start on a tile boundary, as split_range's do), also adds the
// new gradient's projections on the metric's directions, block by block
// while the gradient is in cache.
static void leaf_range(const LeafJob *J, int64_t lo, int64_t hi, double *k, double (*acc)[6][LN], double *cg) {
  Nuts *s = J->s;
  St *n = J->n, *nx = J->nx;
  const double *im = s->inv_m;
  double buf[2][CHUNK];
  for (int64_t c0 = lo; c0 < hi; c0 += CHUNK) {
    int64_t c1 = c0 + CHUNK < hi ? c0 + CHUNK : hi, m = c1 - c0;
    if (J->half == 1)
      kern_half2(c0, c1, J->eps, im, n->p, n->g, n->q, nx ? nx->p : NULL, nx ? nx->q : NULL, k);
    else
      kern_kinetic(c0, c1, im, n->p, k);
    if (J->merges) {
      const double *below = n->p + c0;
      for (int v = 0; v < J->nlev; v++) {
        const LevelJob *L = &J->lev[v];
        double *out = L->out ? L->out + c0 : buf[v & 1];
        const double *ra = L->ra ? L->ra + c0 : below, *rb = L->rb ? L->rb + c0 : below;
        if (L->leaves)
          kern_merge_leaves(m, im + c0, ra, rb, out, acc[v]);
        else
          kern_merge(m, im + c0, ra, rb, L->bp + c0, L->ep + c0, L->m2p + c0, L->m1p + c0, out, acc[v]);
        below = out;
      }
    }
    if (cg) lr_dots(s, cg, n->g, c0 >> 3, (c1 + 7) >> 3);
  }
}

// Stores (add = 0) or adds the totals of the lanes to a partial-sum slot.
static void leaf_part(const LeafJob *J, double *part, const double *k, double (*acc)[6][LN], int add) {
  if (!add) memset(part, 0, (size_t)(1 + 6 * J->nlev) * sizeof(double));
  part[0] += lanes_total(k);
  if (!J->merges) return;
  for (int v = 0; v < J->nlev; v++)
    for (int j = 0; j < 6; j++) part[1 + 6 * v + j] += lanes_total(acc[v][j]);
}

// Thread t of T's share of the leaf pass. With the low-rank metric the pass
// also projects the new gradient on the directions (its momentum's
// projections follow from it, see lr_next) into part[t][NPART_MERGE ..],
// and, when there is a next leaf, adds the low-rank part of that leaf's
// position update after all threads' projections are in: a second pass over
// the directions, in the same parallel region. J->half == 0 is that
// low-rank work alone (after the fused leapfrog, whose hooks did the rest):
// the projections are summed in the same blocks and order as with the rest,
// so they and the position update come out the same either way.
static void leaf_work(LeafJob *J, int t, int T) {
  Nuts *s = J->s;
  St *n = J->n, *nx = J->nx;
  int k = s->k;
  int64_t lo, hi;
  split_range(s->D, t, T, &lo, &hi);
  double *cg = s->part[t] + NPART_MERGE;  // this thread's projections of the gradient
  for (int j = 0; j < k; j++) cg[j] = 0;
  if (J->half) {
    double kl[LN] = {0};
    double acc[MAX_DEPTH + 2][6][LN];
    memset(acc, 0, (size_t)J->nlev * sizeof acc[0]);
    leaf_range(J, lo, hi, kl, acc, k ? cg : NULL);
    leaf_part(J, s->part[t], kl, acc, 0);
  } else {
    for (int64_t c0 = lo; c0 < hi; c0 += CHUNK) {
      int64_t c1 = c0 + CHUNK < hi ? c0 + CHUNK : hi;
      lr_dots(s, cg, n->g, c0 >> 3, (c1 + 7) >> 3);
    }
  }
  if (k && nx) {
#pragma omp barrier
    // every thread adds the threads' projections in thread order, as
    // leaf_pass does afterwards
    double cgt[LR_KMAX], cp[LR_KMAX], cx[LR_KMAX];
    float wf[LR_KMAX + 8];
    for (int j = 0; j < k; j++) {
      double a = s->part[0][NPART_MERGE + j];
      for (int u = 1; u < T; u++) a += s->part[u][NPART_MERGE + j];
      cgt[j] = a;
    }
    lr_next(k, J->eps, n->p + s->D, cgt, cp, cx);
    lr_weights(s, J->eps, cx, wf);
    lr_expand(s, nx->q, wf, lo >> 3, (hi + 7) >> 3);
  }
}

// The fused leapfrog's hook (see mint_leap_fn): leaf work on
// theta[lo .. lo + len), in blocks of LN from lo (the last few elements by
// index modulo LN, as in kern_half2); the totals are added to the slot. The
// low-rank metric's work is not done here: its position update needs every
// thread's projections, which a hook inside the kernel cannot wait for
// (leaf_fused_run does it after the kernel).
static void leaf_block(void *ctx, int64_t slot, int64_t lo, int64_t len) {
  LeafJob *J = ctx;
  if (len <= 0) return;
  double k[LN] = {0};
  double acc[MAX_DEPTH + 2][6][LN];
  memset(acc, 0, (size_t)J->nlev * sizeof acc[0]);
  leaf_range(J, lo, lo + len, k, acc, NULL);
  leaf_part(J, J->s->part[slot], k, acc, 1);
}

// Records the smallest OpenMP team that ran a pass: OpenMP may give fewer
// threads than asked for (thread limits, nesting), and the passes split by
// the team actually running.
static void note_team(Nuts *s, int used) {
  if (used < s->team_min) s->team_min = used;
}

// n = first half-step of a leapfrog from z
static void leaf_start(Nuts *s, const St *z, St *n, double eps) {
  int64_t D = s->D;
  int nt = s->nt, k = s->k;
  float wf[LR_KMAX + 8];
  if (k) {  // the half-step momentum's projections (as lr_next forms them), and the weights
    for (int j = 0; j < k; j++) n->p[D + j] = z->p[D + j] + 0.5 * eps * z->g[D + j];
    lr_weights(s, eps, n->p + D, wf);
  }
  if (nt == 1) {
    kern_half1(0, D, eps, s->inv_m, z->p, z->g, z->q, n->p, n->q);
    if (k) lr_expand(s, n->q, wf, 0, s->lr_nb);
    return;
  }
  int used = 1;
#pragma omp parallel num_threads(nt)
  {
    int t = omp_get_thread_num(), T = omp_get_num_threads();
    if (t == 0) used = T;
    int64_t lo, hi;
    split_range(D, t, T, &lo, &hi);
    kern_half1(lo, hi, eps, s->inv_m, z->p, z->g, z->q, n->p, n->q);
    if (k) lr_expand(s, n->q, wf, lo >> 3, (hi + 7) >> 3);
  }
  note_team(s, used);
}

// The work of leaf n once its gradient is known: the merges waiting for it
// (whose states are read through their slots now) and the next leaf.
static void leaf_prepare(Nuts *s, St *n, double eps, LeafJob *J) {
  int lo = s->pend_lo, hi = s->npend;
  *J = (LeafJob){.s = s, .n = n, .eps = eps, .nlev = hi - lo, .half = 1, .merges = 1};
  // When the merges reach the transition's own (pend[0]), this leaf ends the
  // new subtree and the next direction is not yet drawn: no next leaf.
  if (!(lo == 0 && hi > 0)) J->nx = st_acquire(s);
  for (int v = 0; v < J->nlev; v++) {
    const Merge *m = &s->pend[hi - 1 - v];
    LevelJob *L = &J->lev[v];
    L->ra = m->ra, L->rb = m->rb;
    L->bp = (*m->beg)->p, L->ep = (*m->end)->p, L->m2p = (*m->mid2)->p, L->m1p = (*m->mid1)->p;
    L->out = v == J->nlev - 1 ? m->rho : NULL;
    if (v == J->nlev - 1 && !L->out) mint_panic("internal error: outermost merge has no output");
    // two single leaves: the halves are the earlier leaf and this one
    L->leaves = v == 0 && m->ra && !m->rb && m->ra == L->bp && m->ra == L->m1p && L->ep == n->p &&
                L->m2p == n->p;
  }
}

// The low-rank metric's parts of leaf J's sums v, from the projections (the
// gradient's total in v[NPART_MERGE ..]): the leaf's momentum projections and
// the next leaf's half-step ones; the kinetic energy gains sum_j d_j c_j(p)^2;
// each p_sharp . r of a merge gains sum_j d_j c_j(p) c_j(r), and a merge's sum
// projects to the sum of its halves' projections.
static void lr_conclude(Nuts *s, const LeafJob *J, double *v) {
  int64_t D = s->D;
  int k = s->k;
  const double *d = s->lr_d;
  double *cg = J->n->g + D, *cp = J->n->p + D, cx[LR_KMAX];
  for (int j = 0; j < k; j++) cg[j] = v[NPART_MERGE + j];
  lr_next(k, J->eps, cp, cg, cp, cx);
  if (J->nx) memcpy(J->nx->p + D, cx, sizeof(double) * (size_t)k);
  for (int j = 0; j < k; j++) v[0] += d[j] * cp[j] * cp[j];
  double cbelow[2][LR_KMAX];
  const double *below = cp;
  for (int l = 0; l < J->nlev; l++) {
    const LevelJob *L = &J->lev[l];
    const double *a = L->ra ? L->ra + D : below, *b = L->rb ? L->rb + D : below;
    double *out = L->out ? L->out + D : cbelow[l & 1];
    double *x = v + 1 + 6 * l;
    double corr[6] = {0, 0, 0, 0, 0, 0};
    if (L->leaves) {
      for (int j = 0; j < k; j++) {
        double r = a[j] + b[j];
        corr[0] += d[j] * b[j] * r;
        corr[1] += d[j] * a[j] * r;
        out[j] = r;
      }
    } else {
      const double *be = L->bp + D, *en = L->ep + D, *m2 = L->m2p + D, *m1 = L->m1p + D;
      for (int j = 0; j < k; j++) {
        double r = a[j] + b[j], r2 = a[j] + m2[j], r3 = b[j] + m1[j];
        corr[0] += d[j] * en[j] * r;
        corr[1] += d[j] * be[j] * r;
        corr[2] += d[j] * m2[j] * r2;
        corr[3] += d[j] * be[j] * r2;
        corr[4] += d[j] * en[j] * r3;
        corr[5] += d[j] * m1[j] * r3;
        out[j] = r;
      }
    }
    for (int j = 0; j < 6; j++) x[j] += corr[j];
    below = out;
  }
}

// Reads leaf J's sums v (kinetic energy, then six per merge, then with the
// low-rank metric the gradient's projections, whose parts are added first):
// each merge's result into its Merge, and the next leaf's half-step for
// build_tree. Returns the Hamiltonian.
static double leaf_conclude(Nuts *s, const LeafJob *J, double *v) {
  int hi = s->npend;
  if (s->k) lr_conclude(s, J, v);
  for (int l = 0; l < J->nlev; l++) {
    const double *x = v + 1 + 6 * l;
    int ok = J->lev[l].leaves ? (x[0] > 0 && x[1] > 0)
                              : (x[0] > 0 && x[1] > 0) & (x[2] > 0 && x[3] > 0) & (x[4] > 0 && x[5] > 0);
    s->pend[hi - 1 - l].persist = ok;
  }
  if (J->nx) {
    s->spec = J->nx;
    s->spec_from = J->n;
    s->spec_eps = J->eps;
  }
  return -J->n->lp + 0.5 * v[0];
}

// The leaf pass across the chain's threads, split by index ranges; the
// totals end up in s->part[0] (with J->half == 0 only the projections').
static void leaf_pass(Nuts *s, LeafJob *J) {
  if (s->nt == 1) {
    leaf_work(J, 0, 1);
    note_team(s, 1);
  } else {
    int used = 1;
#pragma omp parallel num_threads(s->nt)
    {
      int t = omp_get_thread_num(), T = omp_get_num_threads();
      if (t == 0) used = T;
      leaf_work(J, t, T);
    }
    note_team(s, used);
    for (int t = 1; t < used; t++) {
      if (J->half)
        for (int j = 0; j < 1 + 6 * J->nlev; j++) s->part[0][j] += s->part[t][j];
      for (int j = 0; j < s->k; j++) s->part[0][NPART_MERGE + j] += s->part[t][NPART_MERGE + j];
    }
  }
}

// Second half of leaf n's leapfrog and the merges waiting for it (see above).
// Returns the Hamiltonian; each completed merge's result is in its Merge.
static double leaf_finish(Nuts *s, St *n, double eps) {
  LeafJob J;
  leaf_prepare(s, n, eps, &J);
  leaf_pass(s, &J);
  return leaf_conclude(s, &J, s->part[0]);
}

// Leaf n with the model's fused leapfrog (see mint_leap_fn): its gradient,
// with the leaf work on the covered parameters done by the kernel's threads
// through the hook (leaf_block), and on the rest of theta here, after it.
// The partial sums are added in slot order: the kernel's threads (each its
// range of each covered parameter in turn, in blocks of LN from the range's
// first element; thread 0's rows left over after the groups last), then
// the rest. That order differs from leaf_pass's, so the draws differ from
// the runtime's own leapfrog by rounding. With s->leap_exact the hooks and
// the rest take only the half-steps, and a leaf pass then sums in
// leaf_pass's order, which gives the runtime's draws (for testing; that
// relies on leap computing the same gradient as the model's logp, which
// MINT_LEAP_TEST and tests/run.sh check). The sums end up in s->part[0].
//
// With the low-rank metric the hooks do the diagonal part of the leaf as
// above, and a pass across the chain's threads after the kernel (leaf_work
// with half == 0; in exact mode the leaf pass that sums) projects the
// gradient and adds the low-rank part of the next leaf's position update,
// split, blocked and summed as in the runtime's own leaf pass. So the
// projections, and that part of the position, are those of the runtime's
// path; only the diagonal sums are in the hooks' order.
static void leaf_fused_run(Nuts *s, LeafJob *J) {
  St *n = J->n;
  int nt = (int)kernel_nt;
  int nsum = 1 + 6 * J->nlev;
  if (s->leap_exact) J->merges = 0;
  for (int t = 0; t < nt; t++) memset(s->part[t], 0, (size_t)nsum * sizeof(double));
  n->lp = s->leap(n->q, n->g, leaf_block, J);
  s->n_grad++;
  s->n_fused++;
  {
    double k[LN] = {0};
    double acc[MAX_DEPTH + 2][6][LN];
    memset(acc, 0, (size_t)J->nlev * sizeof acc[0]);
    for (int b = 0; b < s->nother; b++) leaf_range(J, s->other[2 * b], s->other[2 * b + 1], k, acc, NULL);
    leaf_part(J, s->part[SLOT_OTHER], k, acc, 0);
  }
  if (s->leap_exact) {
    J->half = 2;
    J->merges = 1;
    leaf_pass(s, J);
    return;
  }
  double *v = s->part[0];
  for (int t = 1; t < nt; t++)
    for (int j = 0; j < nsum; j++) v[j] += s->part[t][j];
  for (int j = 0; j < nsum; j++) v[j] += s->part[SLOT_OTHER][j];
  if (s->k) {
    J->half = 0;
    leaf_pass(s, J);
    J->half = 1;
  }
}

// Leaf n through the fused leapfrog (leaf_fused_run); returns the Hamiltonian.
static double leaf_fused(Nuts *s, St *n, double eps) {
  LeafJob J;
  leaf_prepare(s, n, eps, &J);
  leaf_fused_run(s, &J);
  return leaf_conclude(s, &J, s->part[0]);
}

// The ranges of theta that no fused-leapfrog block covers, into s->other.
static void leap_other(Nuts *s, mint_leap_blocks_fn blocks) {
  // the generated code writes at most MAX_LEAP_BLOCKS blocks (LEAP_MAX_BLOCKS
  // in compiler/src/model.rs)
  int64_t D = s->D, blk[2 * MAX_LEAP_BLOCKS];
  int64_t nb = blocks(blk);
  if (nb < 1 || nb > MAX_LEAP_BLOCKS) mint_panic("internal error: fused leapfrog blocks");
  for (int64_t a = 0; a < nb; a++)  // sort by offset
    for (int64_t b = a + 1; b < nb; b++)
      if (blk[2 * b] < blk[2 * a]) {
        int64_t t0 = blk[2 * a], t1 = blk[2 * a + 1];
        blk[2 * a] = blk[2 * b], blk[2 * a + 1] = blk[2 * b + 1];
        blk[2 * b] = t0, blk[2 * b + 1] = t1;
      }
  int64_t at = 0;
  s->nother = 0;
  for (int64_t a = 0; a <= nb; a++) {
    int64_t lo = a < nb ? blk[2 * a] : D;
    if (lo < at || lo > D) mint_panic("internal error: fused leapfrog blocks overlap");
    if (lo > at) s->other[2 * s->nother] = at, s->other[2 * s->nother + 1] = lo, s->nother++;
    if (a < nb) at = lo + blk[2 * a + 1];
  }
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
// Falls back to the diagonal metric (k = 0) when the stored directions are
// not finite or not independent.
static void lr_set(Nuts *s, int k, const double *U, const double *lam) {
  int64_t D = s->D;
  memset(s->lr_v, 0, sizeof(float) * (size_t)((k + 7) >> 3) * (size_t)s->lr_nb * 64);
  for (int j = 0; j < k; j++) {
    const double *u = U + (int64_t)j * D;
    for (int64_t i = 0; i < D; i++) LR_AT(s, j, i) = (float)(sqrt(s->inv_m[i]) * u[i]);
    s->lr_d[j] = lam[j] - 1.0;
  }
  s->k = 0;
  if (k == 0) return;
  size_t kk = (size_t)k * k;
  double *Gm = calloc(kk, sizeof(double)), *R = malloc(sizeof(double) * kk), *B = malloc(sizeof(double) * kk);
  double *Ri = calloc(kk, sizeof(double)), *T = malloc(sizeof(double) * kk), *beta = malloc(sizeof(double) * k);
  for (int a = 0; a < k; a++)
    for (int b = 0; b <= a; b++) {
      double sum = 0;
      for (int64_t i = 0; i < D; i++) sum += (double)LR_AT(s, a, i) * (double)LR_AT(s, b, i) / s->inv_m[i];
      Gm[a * k + b] = Gm[b * k + a] = sum;
    }
  memcpy(s->lr_gm, Gm, sizeof(double) * kk);
  int finite = 1;
  for (int j = 0; j < k && finite; j++)
    for (int64_t i = 0; i < D && finite; i++) finite = isfinite(LR_AT(s, j, i));
  for (int j = 0; j < k && finite; j++) finite = isfinite(s->lr_d[j]);
  if (!finite || !cholesky_upper(Gm, k, R)) goto done;
  // B = R diag(d) R'
  for (int a = 0; a < k; a++)
    for (int b = 0; b < k; b++) {
      double sum = 0;
      for (int m = 0; m < k; m++) sum += R[a * k + m] * s->lr_d[m] * R[b * k + m];
      B[a * k + b] = sum;
    }
  sym_eig(B, k, beta);  // B = E diag(beta) E', E in B's columns
  if (!(1.0 + beta[0] > 0)) goto done;
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

// Most directions of the low-rank metric: MINT_LOWRANK_K, or by default 16
// or 24 if each thread's share of that many directions (single precision,
// D / nt entries each) fits in its L2 cache (512 KiB if the size is not
// reported), and otherwise 8, whether or not those fit. Each leapfrog step
// streams the directions twice. On the 37,901-parameter time series (3
// threads per chain) 24 directions gave more effective draws per gradient
// than 8 but about as many per second, each step costing more; on the
// 3,171-parameter one 24 gave the most per gradient and per second
// (bench/metric_experiment.py).
static int lowrank_k(int64_t D, int nt) {
  const char *e = getenv("MINT_LOWRANK_K");
  int k;
  if (e) {
    k = atoi(e);
  } else {
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2 <= 0) l2 = 512 * 1024;
    double groups = (double)l2 * nt / (32.0 * (double)D);  // 8 directions x 4 bytes per entry
    k = groups >= 3 ? 24 : groups >= 2 ? 16 : 8;
  }
  if (k > LR_KMAX) k = LR_KMAX;
  if (k < 0) k = 0;
  return k;
}

// Window draws (and as many gradients) the low-rank metric keeps per chain,
// at most the warmup iterations, within a memory budget of MINT_LOWRANK_MB
// (default 256) per chain; 0 if fewer than 3 fit.
static int lowrank_nmax(int64_t D, int64_t warmup) {
  const char *e = getenv("MINT_LOWRANK_MB");
  double mb = e ? atof(e) : 256;
  if (!(mb > 0)) mb = 256;
  double cap = mb * 1048576.0 / (16.0 * (double)D);
  int n = (int)(cap < (double)warmup ? cap : (double)warmup);
  return n < 3 ? 0 : n;
}

static void sample_momentum(Nuts *s, St *z) {
  for (int64_t i = 0; i < s->D; i++) z->p[i] = rng_normal(&s->rng) / sqrt(s->inv_m[i]);
  if (s->k) {
    // p = S^-1 (x + U A U' x) with x ~ N(0, I), S = sqrt(inv_m), U = V / S
    // and A from lr_set, so that cov(p) is the inverse of the inverse metric.
    // Above p holds S^-1 x, so U' x = V' p = c0; and S^-1 U = V / inv_m. The
    // projections of the result are c0 + Gm A c0 (Gm = V' diag(1 / inv_m) V).
    int64_t D = s->D, nb = s->lr_nb, full = D >> 3;
    int k = s->k, ng = (k + 7) >> 3;
    double c0[LR_KMAX], ac[LR_KMAX + 8];
    lr_project(s, z->p, c0);
    for (int a = 0; a < k; a++) {
      double sum = 0;
      for (int b = 0; b < k; b++) sum += s->lr_a[a * k + b] * c0[b];
      ac[a] = sum;
    }
    for (int j = k; j < 8 * ng; j++) ac[j] = 0;
    double *cp = z->p + D;
    for (int a = 0; a < k; a++) {
      double sum = 0;
      for (int b = 0; b < k; b++) sum += s->lr_gm[a * k + b] * ac[b];
      cp[a] = c0[a] + sum;
    }
    const double *im = s->inv_m;
    for (int64_t ib = 0; ib < nb; ib++) {
      v4d lo = {0}, hi = {0};
      for (int g = 0; g < ng; g++)
        for (int j = 0; j < 8; j++) {
          v8f t = *(const v8f *)(s->lr_v + (((int64_t)g * nb + ib) * 8 + j) * 8);
          v4d a = {ac[8 * g + j], ac[8 * g + j], ac[8 * g + j], ac[8 * g + j]};
          lo = __builtin_elementwise_fma(a, v8f_lo(t), lo);
          hi = __builtin_elementwise_fma(a, v8f_hi(t), hi);
        }
      double *p = z->p + ib * 8;
      if (ib < full) {
        *(v4d *)p += lo / *(const v4d *)(im + ib * 8);
        *(v4d *)(p + 4) += hi / *(const v4d *)(im + ib * 8 + 4);
      } else {
        for (int ii = 0; ib * 8 + ii < D; ii++) p[ii] += (ii < 4 ? lo[ii] : hi[ii - 4]) / im[ib * 8 + ii];
      }
    }
  }
}


// a fresh state with z's position, gradient and log density
static St *st_clone_position(Nuts *s, const St *z) {
  St *x = st_acquire(s);
  memcpy(x->q, z->q, s->D * sizeof(double));
  memcpy(x->g, z->g, (s->D + s->k) * sizeof(double));  // with the gradient's projections
  x->lp = z->lp;
  return x;
}

static void vzero(double *v, int64_t D) { memset(v, 0, D * sizeof(double)); }
static void vcopy(double *d, const double *s, int64_t D) { memcpy(d, s, D * sizeof(double)); }

// Extends the trajectory from s->edge by 2^depth leapfrog steps. On return
// *beg and *end reference the subtree's first and last states and *prop its
// multinomial proposal. The subtree's momentum sum is returned in *rho_out:
// the leaf's own momentum at depth 0 (the leaf stays referenced as *beg and
// *end until the parent's merge has used it), and otherwise rho. rho is only
// written when this subtree is the first half of its parent (the parent's
// merge reads it later); the sum of a second half is only read by the merge
// that its last leaf also completes, so it is never stored, and rho may be
// NULL there.
//
// The control flow is Stan's base_nuts. The difference is when the merges'
// sums are computed: a merge is registered in s->pend before its second half
// is built, and that half's last leaf computes it (leaf_finish). Its result
// is read here in Stan's order, after the multinomial proposal is drawn.
static int build_tree(Nuts *s, int depth, St **prop, St **beg, St **end, double *rho, const double **rho_out,
                      double H0, double sign, double *log_sum_weight) {
  if (depth == 0) {
    double eps = sign * s->eps;
    St *n = s->spec;
    s->spec = NULL;
    if (n) {
      if (s->spec_from != s->edge || s->spec_eps != eps) mint_panic("internal error: stale leapfrog half-step");
    } else {
      n = st_acquire(s);
      leaf_start(s, s->edge, n, eps);
    }
    // (with the fused leapfrog the gradient comes with the leaf's work)
    if (!s->leap) eval(s, n);
    st_release(s, s->edge);
    s->edge = n;
    s->n_leapfrog++;
    st_set(s, prop, n);
    st_set(s, beg, n);
    st_set(s, end, n);
    *rho_out = n->p;
    double h = s->leap ? leaf_fused(s, n, eps) : leaf_finish(s, n, eps);
    if (isnan(h)) h = INFINITY;
    n->h = h;
    if (h - H0 > 1000.0) s->divergent = 1;
    *log_sum_weight = log_sum_exp(*log_sum_weight, H0 - h);
    s->sum_metro += (H0 - h > 0) ? 1.0 : exp(H0 - h);
    return !s->divergent;
  }
  Level *L = &s->lv[depth];
  int persist = 0;
  *rho_out = rho;

  // The first half's last leaf completes the first half's merges but none
  // waiting above this subtree.
  double lsw_init = -INFINITY;
  const double *r_init, *r_final;
  int saved_lo = s->pend_lo;
  s->pend_lo = s->npend;
  int ok = build_tree(s, depth - 1, prop, beg, &L->init_end, L->rho_init, &r_init, H0, sign, &lsw_init);
  s->pend_lo = saved_lo;
  if (!ok) goto out;

  // This merge: rho = r_init + (second half's sum), checks against *beg,
  // *end, the second half's first state and the first half's last.
  Merge *m = &s->pend[s->npend++];
  *m = (Merge){.rho = rho, .ra = r_init, .rb = NULL, .beg = beg, .end = end, .mid2 = &L->final_beg,
               .mid1 = &L->init_end};
  double lsw_final = -INFINITY;
  ok = build_tree(s, depth - 1, &L->prop_final, &L->final_beg, end, NULL, &r_final, H0, sign, &lsw_final);
  s->npend--;
  if (!ok) goto out;

  double lsw_subtree = log_sum_exp(lsw_init, lsw_final);
  *log_sum_weight = log_sum_exp(*log_sum_weight, lsw_subtree);
  if (lsw_final > lsw_subtree) {
    st_set(s, prop, L->prop_final);
  } else if (rng_uniform(&s->rng) < exp(lsw_final - lsw_subtree)) {
    st_set(s, prop, L->prop_final);
  }

  persist = m->persist;
out:
  st_set(s, &L->init_end, NULL);
  st_set(s, &L->final_beg, NULL);
  st_set(s, &L->prop_final, NULL);
  return persist;
}

typedef struct {
  double *rho, *rho_next;  // the trajectory's summed momentum; the merge writes rho_next
} Traj;

// One NUTS transition from s->cur. On return s->cur is the selected sample.
// Returns the acceptance statistic.
static double transition(Nuts *s, Traj *t) {
  int64_t D = s->D;
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
    // other (NULL: its sum comes from the subtree's last leaf). The merge is
    // registered as pend[0]; the subtree's last leaf computes it.
    const double *r_new;
    s->npend = 1;
    s->pend_lo = 0;
    Merge *m = &s->pend[0];
    *m = (Merge){.rho = t->rho_next, .beg = &bck_bck, .end = &fwd_fwd, .mid2 = &fwd_bck, .mid1 = &bck_fwd};
    if (rng_uniform(&s->rng) > 0.5) {
      st_set(s, &s->edge, edge_fwd);
      st_set(s, &bck_fwd, fwd_fwd);
      m->ra = t->rho, m->rb = NULL;
      valid = build_tree(s, s->depth, &propose, &fwd_bck, &fwd_fwd, NULL, &r_new, H0, 1.0, &lsw_sub);
      st_set(s, &edge_fwd, s->edge);
    } else {
      st_set(s, &s->edge, edge_bck);
      st_set(s, &fwd_bck, bck_bck);
      m->ra = NULL, m->rb = t->rho;
      valid = build_tree(s, s->depth, &propose, &bck_fwd, &bck_bck, NULL, &r_new, H0, -1.0, &lsw_sub);
      st_set(s, &edge_bck, s->edge);
    }
    s->npend = 0;
    if (!valid) break;
    // the subtree is complete, so its last leaf has merged it into rho_next
    double *tmp = t->rho;
    t->rho = t->rho_next, t->rho_next = tmp;
    s->depth++;
    if (lsw_sub > log_sum_weight) {
      st_set(s, &sample, propose);
    } else if (rng_uniform(&s->rng) < exp(lsw_sub - log_sum_weight)) {
      st_set(s, &sample, propose);
    }
    log_sum_weight = log_sum_exp(log_sum_weight, lsw_sub);

    if (!m->persist) break;
  }
  if (s->spec) {  // a half-step taken ahead for a leaf that never came
    st_release(s, s->spec);
    s->spec = NULL;
  }
  st_set(s, &s->cur, sample);
  st_set(s, &s->edge, NULL);
  for (size_t k = 0; k < sizeof refs / sizeof refs[0]; k++) st_set(s, refs[k], NULL);
  st_release(s, z0);
  return s->n_leapfrog > 0 ? s->sum_metro / (double)s->n_leapfrog : 0.0;
}

// Stan's heuristic: double or halve the step size until the acceptance
// probability of a single leapfrog step crosses 0.8.
// Hamiltonian after one leapfrog step from z into n
static double step_energy(Nuts *s, const St *z, St *n, double eps) {
  if (!s->k) {
    leapfrog_into(s, z, n, eps);
    return hamiltonian(s, n);
  }
  int64_t D = s->D;
  leaf_start(s, z, n, eps);
  eval(s, n);
  for (int64_t i = 0; i < D; i++) n->p[i] += 0.5 * eps * n->g[i];
  double cx[LR_KMAX];
  lr_project(s, n->g, n->g + D);
  lr_next(s->k, eps, n->p + D, n->g + D, n->p + D, cx);
  return hamiltonian(s, n);
}

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
  double delta;  // target acceptance statistic (Stan: 0.8)
} DualAvg;

static void da_restart(DualAvg *d) { d->counter = 0, d->s_bar = 0, d->x_bar = 0; }

static void da_learn(DualAvg *d, double *eps, double accept) {
  const double gamma = 0.05, t0 = 10, kappa = 0.75, delta = d->delta;
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

// Stan's schedule is windows_init(w, warmup, 75, 25, 50).
static void windows_init(Windows *w, int64_t warmup, int64_t init_buffer, int64_t base_window,
                         int64_t term_buffer) {
  w->warmup = warmup;
  w->init_buffer = init_buffer, w->term_buffer = term_buffer, w->base_window = base_window;
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

// ---- the alternative warmup (MINT_WARMUP=fast)
//
// The default is Stan's: a uniform(-2, 2) start, then the program's warmup
// iterations in windows 75 / 25, 50, 100, ... / 50. The alternative:
//   - starts each chain at a point chosen along an L-BFGS climb from its
//     uniform start (lbfgs_init below);
//   - runs max(200, warmup / 5) iterations (never more than the program's
//     warmup) in windows 10 / 10, 20, 40, ... / 50; Stan's rule that
//     stretches the last window to the terminal buffer is unchanged;
//   - estimates each window's variance from the draws of all chains together
//     (Pool), so a window of n iterations has chains x n draws.
// Only warmup changes: after it the metric and step size are fixed and every
// chain is an ordinary NUTS chain. MINT_WARMUP_ITERS, _INIT, _WINDOW, _TERM,
// _LBFGS (0 or 1) and _POOL (0 or 1) override the parts, for experiments.
// MINT_TARGET_ACCEPT sets the dual averaging target (0.8) under either warmup.
typedef struct {
  int fast;
  int64_t iters;  // warmup iterations to run (at most the program's warmup)
  int64_t init_buffer, base_window, term_buffer;
  int lbfgs;         // optimise from the uniform start before warmup
  int64_t lbfgs_iters;
  int pool;          // estimate each window's variance from every chain's draws
  double delta;      // the step size's target acceptance statistic
} WarmupCfg;

static int64_t env_int(const char *name, int64_t dflt) {
  const char *e = getenv(name);
  return e && *e ? atoll(e) : dflt;
}

static WarmupCfg warmup_cfg(int64_t warmup) {
  WarmupCfg c = {.fast = 0, .iters = warmup, .init_buffer = 75, .base_window = 25, .term_buffer = 50, .delta = 0.8};
  const char *e = getenv("MINT_WARMUP");
  if (e && strcmp(e, "fast") == 0) {
    c.fast = 1;
    c.iters = env_int("MINT_WARMUP_ITERS", warmup / 5 > 200 ? warmup / 5 : 200);
    c.init_buffer = env_int("MINT_WARMUP_INIT", 10);
    c.base_window = env_int("MINT_WARMUP_WINDOW", 10);
    c.term_buffer = env_int("MINT_WARMUP_TERM", 50);
    // a terminal buffer of 0 would end warmup on a metric update, which
    // restarts the dual averaging and leaves step size exp(0) = 1
    if (c.init_buffer < 0 || c.base_window < 1 || c.term_buffer < 1)
      mint_panic("MINT_WARMUP_INIT must be at least 0, MINT_WARMUP_WINDOW and MINT_WARMUP_TERM at least 1");
    c.lbfgs = (int)env_int("MINT_WARMUP_LBFGS", 1);
    c.lbfgs_iters = env_int("MINT_WARMUP_LBFGS_ITERS", 1000);
    c.pool = (int)env_int("MINT_WARMUP_POOL", 1);
  } else if (e && *e && strcmp(e, "stan") != 0) {
    mint_panic("MINT_WARMUP must be stan or fast");
  }
  const char *de = getenv("MINT_TARGET_ACCEPT");
  if (de && *de) {
    c.delta = atof(de);
    if (!(c.delta > 0 && c.delta < 1)) mint_panic("MINT_TARGET_ACCEPT must be strictly between 0 and 1");
  }
  if (c.iters > warmup) c.iters = warmup;
  if (c.iters < 0) c.iters = 0;
  return c;
}

// ---- the starting point of MINT_WARMUP=fast: L-BFGS with Pathfinder's choice
//
// L-BFGS climbs the log density from the uniform start. After every step it
// builds a diagonal Gaussian around the current point (variances from the
// stored curvature pairs, lb_diag; Pathfinder uses diagonal plus low rank) and
// estimates that Gaussian's ELBO from a few draws. The chain starts at the
// point whose Gaussian had the highest ELBO, not at the end of the climb. For
// a well-behaved posterior that is close to the mode; where the density is
// unbounded (a centred hierarchical model, whose density grows without limit
// as a scale goes to zero) the climb runs off towards the singularity but the
// ELBO falls, so the chain starts before that. If no ELBO is finite the chain
// keeps its uniform start.
#define LB_M 6
typedef struct {
  int64_t D;
  double *S[LB_M], *Y[LB_M], rho[LB_M];
  int k, head;  // pairs stored, next slot
} Lbfgs;

// pair i of the stored ones, 0 the newest
static int lb_slot(const Lbfgs *L, int i) { return (L->head - 1 - i + 2 * LB_M) % LB_M; }

// A diagonal BFGS-type estimate of the inverse Hessian: from the newest pair's
// scalar (s'y / y'y), each stored pair (oldest first) is applied as a BFGS
// update to the current diagonal matrix and only the diagonal of the result
// is kept. That is not the diagonal of the full L-BFGS matrix, whose
// off-diagonal terms are dropped at every step. The BFGS update of a
// positive diagonal matrix is positive definite, so every entry stays positive.
static void lb_diag(const Lbfgs *L, double *diag) {
  int64_t D = L->D;
  int i0 = lb_slot(L, 0);
  double yy = 0;
  for (int64_t q = 0; q < D; q++) yy += L->Y[i0][q] * L->Y[i0][q];
  for (int64_t q = 0; q < D; q++) diag[q] = 1.0 / (L->rho[i0] * yy);
  for (int j = L->k - 1; j >= 0; j--) {
    int i = lb_slot(L, j);
    const double *sv = L->S[i], *yv = L->Y[i];
    double yay = 0;
    for (int64_t q = 0; q < D; q++) yay += diag[q] * yv[q] * yv[q];
    double r = L->rho[i];
    for (int64_t q = 0; q < D; q++)
      diag[q] += -2.0 * r * sv[q] * yv[q] * diag[q] + r * r * sv[q] * sv[q] * yay + r * sv[q] * sv[q];
  }
}

// d = H g by the two-loop recursion (g the ascent direction)
static void lb_direction(const Lbfgs *L, const double *g, double *d) {
  int64_t D = L->D;
  double alpha[LB_M];
  vcopy(d, g, D);
  for (int j = 0; j < L->k; j++) {
    int i = lb_slot(L, j);
    double a = 0;
    for (int64_t q = 0; q < D; q++) a += L->S[i][q] * d[q];
    alpha[i] = L->rho[i] * a;
    for (int64_t q = 0; q < D; q++) d[q] -= alpha[i] * L->Y[i][q];
  }
  double gamma;
  if (L->k > 0) {
    int i = lb_slot(L, 0);
    double yy = 0;
    for (int64_t q = 0; q < D; q++) yy += L->Y[i][q] * L->Y[i][q];
    gamma = 1.0 / (L->rho[i] * yy);
  } else {
    double gg = 0;
    for (int64_t q = 0; q < D; q++) gg += g[q] * g[q];
    gamma = 1.0 / sqrt(gg > 0 ? gg : 1.0);
  }
  for (int64_t q = 0; q < D; q++) d[q] *= gamma;
  for (int j = L->k - 1; j >= 0; j--) {
    int i = lb_slot(L, j);
    double b = 0;
    for (int64_t q = 0; q < D; q++) b += L->Y[i][q] * d[q];
    b *= L->rho[i];
    for (int64_t q = 0; q < D; q++) d[q] += L->S[i][q] * (alpha[i] - b);
  }
}

#define ELBO_DRAWS 4
#define ELBO_PATIENCE 20

// The ELBO (up to a constant) of N(x, diag(var)) from ELBO_DRAWS draws, -inf if
// any draw has a non-finite log density. z is scratch.
static double elbo(Nuts *s, const St *x, const double *var, St *z) {
  int64_t D = s->D;
  double ent = 0;
  for (int64_t q = 0; q < D; q++) ent += 0.5 * log(var[q]);
  double tot = 0;
  for (int j = 0; j < ELBO_DRAWS; j++) {
    double ee = 0;
    for (int64_t q = 0; q < D; q++) {
      double e = rng_normal(&s->rng);
      ee += e * e;
      z->q[q] = x->q[q] + sqrt(var[q]) * e;
    }
    eval(s, z);
    if (!isfinite(z->lp)) return -INFINITY;
    tot += z->lp + 0.5 * ee;
  }
  return tot / ELBO_DRAWS + ent;
}

// Runs the search from s->cur and replaces s->cur with the chosen point. The
// climb stops when no step raises the log density by the Armijo condition with
// a finite log density and gradient, after maxit steps, when the log density
// rose by less than 1e-10 (1 + |lp|) three steps running, or when the ELBO
// has not improved for ELBO_PATIENCE steps. Returns 1 if a point was chosen.
static int lbfgs_init(Nuts *s, int64_t maxit, int trace, int chain) {
  int64_t D = s->D;
  Lbfgs L = {.D = D};
  for (int i = 0; i < LB_M; i++) L.S[i] = mint_alloc(D), L.Y[i] = mint_alloc(D);
  double *d = mint_alloc(D), *var = mint_alloc(D);
  St *x = s->cur, *n = st_acquire(s), *z = st_acquire(s), *best = st_clone_position(s, s->cur);
  double best_elbo = -INFINITY;  // best holds the uniform start until an ELBO is finite
  int small = 0;
  int64_t since_best = 0, it = 0, best_it = -1, rejected = 0;
  for (; it < maxit; it++) {
    lb_direction(&L, x->g, d);
    double gd = 0;
    for (int64_t q = 0; q < D; q++) gd += x->g[q] * d[q];
    if (!(gd > 0)) {  // not an ascent direction: forget the pairs and follow the gradient
      L.k = 0, L.head = 0;
      lb_direction(&L, x->g, d);
      gd = 0;
      for (int64_t q = 0; q < D; q++) gd += x->g[q] * d[q];
      if (!(gd > 0)) break;
    }
    double t = 1.0;
    int found = 0;
    for (int ls = 0; ls < 60 && !found; ls++, t *= 0.5) {
      for (int64_t q = 0; q < D; q++) n->q[q] = x->q[q] + t * d[q];
      eval(s, n);
      found = isfinite(n->lp) && n->lp >= x->lp + 1e-4 * t * gd;
      for (int64_t q = 0; q < D && found; q++) found = isfinite(n->g[q]);
    }
    if (!found) break;
    // the pair is stored only if its curvature is positive; once the history is
    // full the slot at head holds the oldest live pair, so test before writing
    double sy = 0;
    for (int64_t q = 0; q < D; q++) sy += (n->q[q] - x->q[q]) * (x->g[q] - n->g[q]);
    if (sy > 0 && isfinite(sy)) {
      double *sv = L.S[L.head], *yv = L.Y[L.head];
      for (int64_t q = 0; q < D; q++) {
        sv[q] = n->q[q] - x->q[q];
        yv[q] = x->g[q] - n->g[q];
      }
      L.rho[L.head] = 1.0 / sy;
      L.head = (L.head + 1) % LB_M;
      if (L.k < LB_M) L.k++;
    } else {
      rejected++;
    }
    double rise = n->lp - x->lp;
    St *tmp = x;
    x = n, n = tmp;
    if (L.k > 0) {
      lb_diag(&L, var);
      double e = elbo(s, x, var, z);
      if (e > best_elbo) {
        best_elbo = e, since_best = 0, best_it = it;
        vcopy(best->q, x->q, D);
        vcopy(best->g, x->g, D);
        best->lp = x->lp;
      } else if (++since_best >= ELBO_PATIENCE) {
        it++;
        break;
      }
    }
    small = rise < 1e-10 * (1.0 + fabs(x->lp)) ? small + 1 : 0;
    if (small >= 3) {
      it++;
      break;
    }
  }
  if (trace)
    fprintf(stderr, "warmup chain=%d lbfgs steps=%lld chosen=%lld elbo=%.6g pairs rejected=%lld end lp=%.6g\n", chain,
            (long long)it, (long long)best_it, best_elbo, (long long)rejected, x->lp);
  st_release(s, n);
  st_release(s, z);
  st_release(s, x);
  s->cur = best;
  for (int i = 0; i < LB_M; i++) free(L.S[i]), free(L.Y[i]);
  free(d), free(var);
  return best_it >= 0;
}

// Window statistics shared between the chains of one sample() call. At the
// end of each metric window every chain publishes its count, mean and sum of
// squared deviations, waits for the others, and combines all chains' in chain
// order, so each chain gets the same metric and the result does not depend
// on thread timing.
//
// With the low-rank metric (MINT_METRIC=lowrank) the directions are pooled
// too: chain c keeps its window's draws and gradients in rows
// [c lr_nmax, (c + 1) lr_nmax) of lr_q and lr_g, and after the variances are
// combined chain 0 estimates the directions from every chain's rows (in
// chain order) against the pooled diagonal, while the others wait; each
// chain then installs the same estimate (lowrank_pool).
typedef struct {
  pthread_barrier_t bar;
  int64_t chains;
  int64_t *n;
  const double **mean, **m2;
  int lr_nmax;                 // rows per chain; 0: no low-rank pooling
  double *lr_q, *lr_g, *lr_u;  // chains x lr_nmax x D, twice; the most directions any chain keeps x D
  int *lr_rows, *lr_kmax;      // per chain: rows kept this window, most directions
  int lr_k;                    // directions chain 0 estimated
  double lr_lam[LR_KMAX];
} Pool;

// Pooled variance of coordinate i over every chain's window; *n_out the pooled count.
static void pool_combine(Pool *p, int64_t D, double *var, double *n_out) {
  double n = 0;
  for (int64_t c = 0; c < p->chains; c++) n += (double)p->n[c];
  *n_out = n;
  for (int64_t i = 0; i < D; i++) {
    double mean = 0;
    for (int64_t c = 0; c < p->chains; c++) mean += (double)p->n[c] * p->mean[c][i];
    mean /= n;
    double m2 = 0;
    for (int64_t c = 0; c < p->chains; c++) {
      double dm = p->mean[c][i] - mean;
      m2 += p->m2[c][i] + (double)p->n[c] * dm * dm;
    }
    var[i] = n > 1 ? m2 / (n - 1.0) : 1.0;
  }
}

// The pooled low-rank estimate at the end of a window (see Pool): chain
// `chain` kept `rows` draws in its rows of P->lr_q and P->lr_g and asks for
// at most kmax directions. Chain 0 moves every chain's rows together (in
// chain order) and estimates from all of them with its own threads, using
// the smallest kmax any chain asked for (each chain's directions are
// allocated for its own); im is the pooled diagonal, the same in every
// chain. Returns the number of directions (in P->lr_u, P->lr_lam) and the
// pooled number of draws in *n_out.
static int lowrank_pool(Pool *P, int chain, int64_t D, int rows, int kmax, const double *im, double cutoff,
                        double gamma, int nt, int *n_out) {
  P->lr_rows[chain] = rows;
  P->lr_kmax[chain] = kmax;
  pthread_barrier_wait(&P->bar);
  if (chain == 0) {
    int64_t n = 0;
    int km = kmax;
    for (int64_t c = 0; c < P->chains; c++) {
      if (P->lr_kmax[c] < km) km = P->lr_kmax[c];
      int64_t from = c * (int64_t)P->lr_nmax;
      if (from != n) {  // n <= from: moving down never overwrites rows not yet moved
        memmove(P->lr_q + n * D, P->lr_q + from * D, sizeof(double) * (size_t)P->lr_rows[c] * (size_t)D);
        memmove(P->lr_g + n * D, P->lr_g + from * D, sizeof(double) * (size_t)P->lr_rows[c] * (size_t)D);
      }
      n += P->lr_rows[c];
    }
    P->lr_k = lowrank_estimate(D, (int)n, P->lr_q, P->lr_g, im, km, cutoff, gamma, P->lr_u, P->lr_lam, nt);
  }
  pthread_barrier_wait(&P->bar);
  int n = 0;
  for (int64_t c = 0; c < P->chains; c++) n += P->lr_rows[c];
  *n_out = n;
  return P->lr_k;
}

// ---- keeping a chain's threads on one L3 cache
//
// When a chain's passes are split across threads, the chain's own thread
// (which runs the gradient) and its helpers hand the state vectors to each
// other every leapfrog. On CPUs with several L3 caches (AMD's chiplets), a
// chain whose threads sit behind different L3s moves that traffic across the
// chip's interconnect: a 37,901-parameter chain with 3 threads took 17 to 18 s
// instead of 10.6 s that way. So each threaded chain's threads are restricted
// to the CPUs sharing one L3 (any of them, not one CPU each), and chains are
// dealt to the L3s in turn. MINT_CHAIN_AFFINITY=0 turns this off.

#define MAX_L3 16

// Parses a sysfs CPU list such as "0-5,12-17".
static int read_cpu_list(const char *path, cpu_set_t *set) {
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  char buf[8192];
  int ok = fgets(buf, sizeof buf, f) != NULL;
  // the whole line, or nothing: a truncated list would parse as other CPUs
  ok = ok && (strchr(buf, '\n') || feof(f));
  fclose(f);
  if (!ok) return 0;
  CPU_ZERO(set);
  char *p = buf;
  while (*p && *p != '\n') {
    char *e;
    long a = strtol(p, &e, 10), b = a;
    if (e == p) return 0;
    if (*e == '-') {
      p = e + 1;
      b = strtol(p, &e, 10);
      if (e == p) return 0;
    }
    for (long c = a; c <= b && c < CPU_SETSIZE; c++) CPU_SET((int)c, set);
    p = e;
    if (*p == ',') p++;
  }
  return CPU_COUNT(set) > 0;
}

// The CPUs this process may use, grouped by shared L3 cache. Returns the
// number of groups, or 0 when the topology cannot be read.
static int l3_groups(cpu_set_t *g, int max) {
  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return 0;
  int n = 0;
  for (int c = 0; c < CPU_SETSIZE; c++) {
    if (!CPU_ISSET(c, &allowed)) continue;
    int seen = 0;
    for (int k = 0; k < n; k++) seen |= CPU_ISSET(c, &g[k]) != 0;
    if (seen) continue;
    cpu_set_t l3;
    int found = 0;
    for (int idx = 0; idx < 8 && !found; idx++) {
      char path[160];
      snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%d/level", c, idx);
      FILE *f = fopen(path, "r");
      if (!f) break;
      int level = 0;
      if (fscanf(f, "%d", &level) != 1) level = 0;
      fclose(f);
      if (level == 3) {
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_list", c, idx);
        found = read_cpu_list(path, &l3);
      }
    }
    if (!found || n == max) return 0;
    CPU_AND(&g[n], &l3, &allowed);
    if (!CPU_ISSET(c, &g[n])) return 0;
    for (int k = 0; k < n; k++) {  // groups must not overlap
      cpu_set_t both;
      CPU_AND(&both, &g[k], &g[n]);
      if (CPU_COUNT(&both)) return 0;
    }
    n++;
  }
  return n;
}

// Each thread's own mask from before bind_team, put back by restore_team.
static __thread cpu_set_t saved_mask;
static __thread int saved_ok;

static void restore_team(int nt) {
#pragma omp parallel num_threads(nt)
  if (saved_ok) {
    pthread_setaffinity_np(pthread_self(), sizeof saved_mask, &saved_mask);
    saved_ok = 0;
  }
}

// Restricts every thread of this chain's team to *set. This relies on the
// chain getting the same OpenMP threads for every pass (LLVM's "hot team"
// of a thread that is not itself inside a parallel region), so it is not
// done with dynamic team sizes, and it is undone when the team is smaller
// than asked for or a thread could not be moved. Returns 1 when bound.
static int bind_team(int nt, const cpu_set_t *set) {
  if (omp_get_dynamic()) return 0;
  int used = 0, fails = 0;
#pragma omp parallel num_threads(nt) reduction(+ : fails)
  {
    if (omp_get_thread_num() == 0) used = omp_get_num_threads();
    saved_ok = pthread_getaffinity_np(pthread_self(), sizeof saved_mask, &saved_mask) == 0;
    if (!saved_ok || pthread_setaffinity_np(pthread_self(), sizeof *set, set) != 0) fails++;
  }
  if (used != nt || fails) {
    restore_team(nt);
    return 0;
  }
  return 1;
}

// ---------------------------------------------------------------- streaming summaries
//
// print(post) reports, for every parameter, an ESS and a split R-hat, and
// for the rows it shows also the mean, sd and quantiles. Storing every draw
// of every parameter for that costs chains x draws x D doubles. Instead each
// chain keeps, per parameter, what those statistics need, updated after
// every kept draw, so this memory grows with D times a constant that does
// not depend on the number of draws (with the defaults and 1000 draws per
// chain, 120 doubles' worth; at most 121 for any number of draws):
//  - Welford means and sums of squared deviations of the first half, the
//    second half and the whole chain (the halves are those of split R-hat:
//    draws [0, h) and [h, 2h) with h = N / 2): 6 doubles.
//  - For the ESS, the sums of lagged products the draw-level estimate needs
//    at lags 1 .. L - 1 (L = MINT_ESS_LAGS, 32 by default). That estimate
//    (Geyer's initial monotone sequence, as in rhat_ess) only ever uses the
//    autocovariances at lags 2m and 2m + 1 added together, so one sum per
//    pair is kept: v_i (v_{i-2m} + v_{i-2m-1}), and v_i v_{i-1} for m = 0,
//    where v_i is draw i minus the chain's first draw, rounded to float
//    (about 7 significant digits of its distance from that draw). The
//    last L values of v and, per pair, the sum of the first values that the
//    pair's mean correction needs (also as a float) give the
//    autocovariances of v, so when the sequence stops before lag L the ESS
//    is the draw-level one up to that rounding (1e-7 relative in the tests). The products are added every CS_BLOCK = 16 draws, so that
//    each sum is read and written once per block (see cs_products); the
//    ring buffer of v therefore holds L + 16 values. L / 2 doubles,
//    3 L / 2 + 16 floats and 2 doubles (the first draw and the sum of v).
//  - When it does not stop before lag L (parameters with longer
//    autocorrelation), the ESS comes from the means of nb consecutive
//    batches of b draws instead (b = ceil(N / k), nb = floor(N / b) with
//    k = MINT_ESS_BATCHES, 128 by default, at least 4: 125 batches of 8 for
//    1000 draws; the last N - nb b draws are left out): see stream_stats.
//    This is noisier than the draw-level estimate: on simulated AR(1)
//    chains (4 x 1000 draws, autocorrelation 0.8 to 0.95) it was within
//    0.80 to 1.05 of it for 98% of parameters. nb floats and 1 double.
// Full draws are kept only for the parameters the summary prints quantiles
// for and those named in MINT_KEEP_DRAWS (see mint_sample). Each chain
// writes only its own ChainStats (with its own threads on disjoint ranges
// of parameters), so the threaded sampler stays race-free and the
// statistics of given draws do not depend on the thread count; the chains
// are combined in chain order when the summary is computed.
#define CS_BLOCK 16
typedef struct {
  int64_t D, N, h;
  int64_t L, P;   // lags with exact sums (even, at most N) and their pairs, P = L / 2
  int64_t R;      // ring slots: L + CS_BLOCK
  int64_t b, nb;  // batch size and number of batches of the fallback
  double *m1, *s1, *m2, *s2, *mt, *st;  // Welford mean and sum of squares: halves, whole chain
  double *shift, *vsum;  // the chain's first draw; the sum of v
  double *q;             // P x D: pair sums of lagged products of v
  float *ring;           // R x D: v_i in slot i mod R
  float *head;           // P x D: for pair m >= 1, 2 (v_0 + ... + v_{2m-1}) + v_{2m}
  double *bsum;          // the running batch sum of v
  float *bm;             // nb x D batch means of v
} ChainStats;

static double *zalloc(int64_t n) {
  double *p = mint_alloc(n);
  memset(p, 0, (size_t)(n > 0 ? n : 1) * sizeof(double));
  return p;
}

// A positive integer from the environment; anything else is ignored with a warning.
static int64_t env_pos(const char *name, int64_t dflt) {
  const char *e = getenv(name);
  if (!e || !*e) return dflt;
  char *end;
  errno = 0;
  long long v = strtoll(e, &end, 10);
  if (*end || errno || v <= 0) {
    fprintf(stderr, "warning: %s=%s is not a positive integer; using %lld\n", name, e, (long long)dflt);
    return dflt;
  }
  return v;
}

// Allocated by the chain's own thread, so that its pages are local to it.
static void cs_init(ChainStats *S, int64_t D, int64_t N) {
  S->D = D, S->N = N, S->h = N / 2;
  int64_t L = env_pos("MINT_ESS_LAGS", 32);
  if (L > N) L = N;
  L -= L % 2;
  if (L < 2) L = 2;
  S->L = L, S->P = L / 2, S->R = L + CS_BLOCK;
  int64_t k = env_pos("MINT_ESS_BATCHES", 128);
  if (k < 4) k = 4;  // at least 4 batches per chain (N >= 4)
  if (k > N) k = N;
  S->b = (N + k - 1) / k;
  S->nb = N / S->b;
  double **v[] = {&S->m1, &S->s1, &S->m2, &S->s2, &S->mt, &S->st, &S->shift, &S->vsum, &S->bsum};
  for (size_t j = 0; j < sizeof v / sizeof v[0]; j++) *v[j] = zalloc(D);
  S->q = zalloc(S->P * D);
  S->ring = mint_alloc((S->R * D + 1) / 2);
  S->head = mint_alloc((S->P * D + 1) / 2);
  S->bm = mint_alloc((S->nb * D + 1) / 2);
}

static inline void welford(double *m, double *s, double x, double r) {
  double d = x - *m;
  *m += d * r;
  *s += d * (x - *m);
}

// The lagged products of draws t0 .. t1 (one block), parameters lo .. hi - 1,
// in chunks of parameters small enough for the block's ring rows and sums
// to stay in cache: each sum is then read and written once per block
// rather than once per draw.
static void cs_products(ChainStats *S, int64_t t0, int64_t t1, int64_t lo, int64_t hi) {
  int64_t D = S->D, R = S->R, P = S->P;
  for (int64_t j0 = lo; j0 < hi; j0 += 256) {
    int64_t j1 = j0 + 256 < hi ? j0 + 256 : hi;
    for (int64_t t = t0; t <= t1; t++) {
      const float *restrict v = S->ring + (t % R) * D;
      if (t >= 1) {
        const float *r1 = S->ring + ((t - 1) % R) * D;
        double *restrict q = S->q;
        for (int64_t j = j0; j < j1; j++) q[j] += (double)v[j] * (double)r1[j];
      }
      for (int64_t m = 1; m < P && 2 * m <= t; m++) {
        const float *ra = S->ring + ((t - 2 * m) % R) * D;
        double *restrict q = S->q + m * D;
        if (2 * m + 1 <= t) {
          const float *rb = S->ring + ((t - 2 * m - 1) % R) * D;
          for (int64_t j = j0; j < j1; j++) q[j] += (double)v[j] * ((double)ra[j] + (double)rb[j]);
        } else {
          for (int64_t j = j0; j < j1; j++) q[j] += (double)v[j] * (double)ra[j];
        }
      }
    }
  }
}

// Adds kept draw i (0-based) of the chain, parameters lo .. hi - 1. The
// lagged products are added for blocks of CS_BLOCK draws at a time (the
// ring keeps the block and the L values before it), and for the last
// partial block at the last draw.
static void cs_add(ChainStats *S, const double *x, int64_t i, int64_t lo, int64_t hi) {
  int64_t D = S->D, h = S->h, L = S->L, b = S->b;
  double *restrict sh = S->shift, *restrict vs = S->vsum;
  if (i == 0)
    for (int64_t j = lo; j < hi; j++) sh[j] = x[j];
  float *restrict v = S->ring + (i % S->R) * D;
  for (int64_t j = lo; j < hi; j++) v[j] = (float)(x[j] - sh[j]);
  if ((i + 1) % CS_BLOCK == 0 || i == S->N - 1) cs_products(S, i - i % CS_BLOCK, i, lo, hi);
  if (i < L && i % 2 == 0 && i >= 2) {
    float *restrict hd = S->head + (i / 2) * D;
    for (int64_t j = lo; j < hi; j++) hd[j] = (float)(2.0 * vs[j] + (double)v[j]);
  }
  for (int64_t j = lo; j < hi; j++) vs[j] += (double)v[j];

  double rt = 1.0 / (double)(i + 1);
  for (int64_t j = lo; j < hi; j++) welford(&S->mt[j], &S->st[j], x[j], rt);
  if (i < h) {
    for (int64_t j = lo; j < hi; j++) welford(&S->m1[j], &S->s1[j], x[j], rt);
  } else if (i < 2 * h) {
    double r = 1.0 / (double)(i - h + 1);
    for (int64_t j = lo; j < hi; j++) welford(&S->m2[j], &S->s2[j], x[j], r);
  }
  if (i < S->nb * b) {
    double *restrict bs = S->bsum;
    for (int64_t j = lo; j < hi; j++) bs[j] += (double)v[j];
    if ((i + 1) % b == 0) {
      float *restrict o = S->bm + (i / b) * D;
      for (int64_t j = lo; j < hi; j++) o[j] = (float)(bs[j] / (double)b), bs[j] = 0;
    }
  }
}

// Writes n bytes at offset off (the chains write disjoint parts of the
// MINT_DRAWS file, so no ordering between them is needed).
static void write_at(int fd, const void *buf, size_t n, off_t off) {
  const char *p = buf;
  while (n > 0) {
    ssize_t w = pwrite(fd, p, n, off);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) mint_panic("cannot write the MINT_DRAWS file");
    p += w, n -= (size_t)w, off += w;
  }
}

typedef struct {
  // inputs
  mint_logp_fn f;
  mint_leap_fn leap;
  mint_leap_blocks_fn leap_blocks;
  int leap_mode;  // 0: off, 2: on, 3: on with exact sums
  mint_constrain_fn constrain;
  mint_collapse_fn collapse;  // non-NULL: writes d_out-long draws instead of constrain
  int64_t D, d_out, draws, warmup;
  uint64_t seed;
  int chain;
  int threads_per_chain;
  const cpu_set_t *l3;  // CPUs for this chain's threads, or NULL
  const WarmupCfg *cfg;
  Pool *pool;  // NULL unless the chains pool their window estimates
  const int64_t *keep_idx;  // the parameters whose draws are kept (all when nkeep == D)
  int64_t nkeep;
  int draws_fd;  // MINT_DRAWS file, or -1
  // outputs
  int team_min;
  double *out;      // draws x nkeep, constrained
  ChainStats *stats;
  double step_size;
  int64_t n_grad, divergent, n_fused;
  int64_t warmup_grad;  // gradients used before the first kept draw (initialisation and warmup)
  double mean_leapfrog;
  int rank;  // low-rank metric: number of directions at the end of warmup
  double stats_seconds;  // time spent constraining, keeping and summarising draws (and writing MINT_DRAWS)
} ChainJob;

static void *run_chain(void *arg) {
  ChainJob *job = arg;
  int64_t D = job->D;
  Nuts *sp = calloc(1, sizeof(Nuts));
  Nuts *s = sp;
  s->D = D;
  s->nt = job->threads_per_chain;
  s->team_min = s->nt;
  int64_t saved_knt = kernel_nt;
  kernel_nt = kernel_threads(s->nt);
  int bound = job->l3 && s->nt > 1 && bind_team(s->nt, job->l3);
  s->f = job->f;
  if (job->leap && job->leap_mode >= 2) {
    leap_other(s, job->leap_blocks);
    s->leap = job->leap;
    s->leap_exact = job->leap_mode == 3;
  }
  s->inv_m = mint_alloc(D);
  for (int64_t i = 0; i < D; i++) s->inv_m[i] = 1.0;
  rng_seed(&s->rng, job->seed * 0x9E3779B97F4A7C15ull + (uint64_t)job->chain + 1);
  // a separate stream for the draws of collapsed parameters, so that the
  // sampler's own stream (and so its draws) do not depend on them
  Rng crng;
  rng_seed(&crng, (job->seed * 0x9E3779B97F4A7C15ull + (uint64_t)job->chain + 1) ^ 0xC2B2AE3D27D4EB4Full);
  for (int d = 0; d <= MAX_DEPTH; d++) {
    Level *L = &s->lv[d];
    L->rho_init = mint_alloc(D + LR_KMAX);
  }
  Traj t;
  double **tv[] = {&t.rho, &t.rho_next};
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
  const WarmupCfg *cfg = job->cfg;
  int64_t warmup = cfg->iters;
  int trace = getenv("MINT_WARMUP_TRACE") != NULL;
  if (cfg->lbfgs) lbfgs_init(s, cfg->lbfgs_iters, trace, job->chain);
  if (trace)
    fprintf(stderr, "warmup chain=%d start lp=%.6g gradients=%lld\n", job->chain, s->cur->lp,
            (long long)s->n_grad);

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
  Pool *lr_pool = NULL;  // the chains pool their low-rank estimates
  if (lowrank) {
    const char *e;
    int team = 1;  // the team OpenMP actually gives this chain, which may be smaller than asked for
    if (s->nt > 1) {
#pragma omp parallel num_threads(s->nt)
      {
        if (omp_get_thread_num() == 0) team = omp_get_num_threads();
      }
    }
    lr_kmax = lowrank_k(D, team);
    if (lr_kmax > D) lr_kmax = (int)D;
    if ((e = getenv("MINT_LOWRANK_CUTOFF"))) lr_cutoff = atof(e);
    if (!(lr_cutoff >= 1.0)) lr_cutoff = 2.0;
    if ((e = getenv("MINT_LOWRANK_GAMMA"))) lr_gamma = atof(e);
    if (!(lr_gamma >= 0)) lr_gamma = 1e-5;
    // Draws and gradients of the current window are kept, at most lr_nmax of
    // each (the most recent): in this chain's rows of the pool's arrays when
    // the chains pool their windows (MINT_WARMUP=fast), else its own.
    lr_nmax = lowrank_nmax(D, warmup);
    lr_pool = job->pool && job->pool->lr_nmax ? job->pool : NULL;
    if (lr_nmax && lr_kmax && lr_pool) {
      if (lr_pool->lr_nmax != lr_nmax) mint_panic("internal error: low-rank pool size");
      lr_q = lr_pool->lr_q + (int64_t)job->chain * lr_nmax * D;
      lr_g = lr_pool->lr_g + (int64_t)job->chain * lr_nmax * D;
    }
    if (lr_nmax && lr_kmax) {
      if (!lr_pool) {
        lr_q = mint_alloc((int64_t)lr_nmax * D);
        lr_g = mint_alloc((int64_t)lr_nmax * D);
        lr_u = mint_alloc((int64_t)lr_kmax * D);
      }
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
  da.delta = cfg->delta;
  da.mu = log(10 * s->eps);
  Windows w;
  windows_init(&w, warmup, cfg->init_buffer, cfg->base_window, cfg->term_buffer);
  double *wmean = mint_alloc(D), *wm2 = mint_alloc(D);
  double *gmean = mint_alloc(D), *gm2 = mint_alloc(D);
  vzero(wmean, D);
  vzero(wm2, D);
  vzero(gmean, D);
  vzero(gm2, D);
  int64_t wn = 0;
  int64_t Dout = job->d_out;  // values per draw: D, plus any collapsed parameters
  int all_kept = job->nkeep == Dout;
  double *xbuf = all_kept ? NULL : mint_alloc(Dout);
  cs_init(job->stats, Dout, job->draws);

  job->divergent = 0;
  job->warmup_grad = s->n_grad;
  int64_t total_leapfrog = 0;
  for (int64_t it = 0; it < warmup + job->draws; it++) {
    double accept = transition(s, &t);
    const double *q = s->cur->q, *gq = s->cur->g;
    if (it < warmup) {
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
          if (trace) fprintf(stderr, "warmup chain=%d it=%lld gradients=%lld eps=%.4g\n", job->chain, (long long)it,
                             (long long)s->n_grad, s->eps);
          next_window(&w);
          double n = (double)wn;
          if (job->pool) {  // the variance of every chain's window draws (not with grad_metric)
            Pool *P = job->pool;
            P->n[job->chain] = wn, P->mean[job->chain] = wmean, P->m2[job->chain] = wm2;
            pthread_barrier_wait(&P->bar);
            pool_combine(P, D, s->inv_m, &n);
            pthread_barrier_wait(&P->bar);
            for (int64_t i = 0; i < D; i++) s->inv_m[i] = (n / (n + 5.0)) * s->inv_m[i] + 1e-3 * (5.0 / (n + 5.0));
          } else {
            for (int64_t i = 0; i < D; i++) {
              double var = wn > 1 ? wm2[i] / (n - 1.0) : 1.0;
              if (grad_metric && wn > 1 && gm2[i] > 0) var = sqrt(var / (gm2[i] / (n - 1.0)));
              s->inv_m[i] = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0));
            }
          }
          if (lowrank) {
            int nst = wn < lr_nmax ? (int)wn : lr_nmax;
            double t_lr = mint_clock();
            int k;
            const double *lam = lr_lam;
            if (lr_pool) {
              k = lowrank_pool(lr_pool, job->chain, D, nst, lr_kmax, s->inv_m, lr_cutoff, lr_gamma, s->nt, &nst);
              lr_set(s, k, lr_pool->lr_u, lr_pool->lr_lam);
              lam = lr_pool->lr_lam;
            } else {
              k = lowrank_estimate(D, nst, lr_q, lr_g, s->inv_m, lr_kmax, lr_cutoff, lr_gamma, lr_u, lr_lam, s->nt);
              lr_set(s, k, lr_u, lr_lam);
            }
            if (s->k) lr_project(s, s->cur->g, s->cur->g + D);  // the current gradient, for the new directions
            if (getenv("MINT_LOWRANK_VERBOSE")) {
              // one write per line, so that the chains' lines do not interleave
              char msg[512];
              int n = snprintf(msg, sizeof msg, "chain %d window end at %lld: %d draws%s, rank %d (%.3f s), lam",
                               job->chain, (long long)it, nst, lr_pool ? " (pooled)" : "", s->k, mint_clock() - t_lr);
              for (int j = 0; j < s->k && j < 8; j++) n += snprintf(msg + n, sizeof msg - (size_t)n, " %.3g", lam[j]);
              n += snprintf(msg + n, sizeof msg - (size_t)n, "%s\n", s->k > 8 ? " ..." : "");
              fflush(stderr);
              if (write(2, msg, (size_t)n) < 0) {}
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
      if (it == warmup - 1) {
        s->eps = exp(da.x_bar);
        job->warmup_grad = s->n_grad;
        if (trace) fprintf(stderr, "warmup chain=%d it=%lld gradients=%lld eps=%.4g (end)\n", job->chain,
                           (long long)it, (long long)s->n_grad, s->eps);
      }
    } else {
      // With collapsed parameters (mint_set_collapsed) a draw has d_out
      // values, more than NUTS's D: the collapse function writes the whole
      // constrained draw, the collapsed parameters drawn by FFBS from crng.
      // The kept draws, the streaming statistics and MINT_DRAWS cover all of them.
      double ts = mint_clock();
      int64_t i = it - warmup, nk = job->nkeep;
      double *x = all_kept ? job->out + i * Dout : xbuf;
      if (job->collapse)
        job->collapse(q, x, &crng);
      else
        job->constrain(q, x);
      if (!all_kept)
        for (int64_t k = 0; k < nk; k++) job->out[i * nk + k] = x[job->keep_idx[k]];
      if (s->nt > 1) {
#pragma omp parallel num_threads(s->nt)
        {
          int64_t lo, hi;
          split_range(Dout, omp_get_thread_num(), omp_get_num_threads(), &lo, &hi);
          cs_add(job->stats, x, i, lo, hi);
        }
      } else {
        cs_add(job->stats, x, i, 0, Dout);
      }
      if (job->draws_fd >= 0)
        write_at(job->draws_fd, x, (size_t)Dout * sizeof(double),
                 (off_t)(3 * sizeof(uint64_t)) + (off_t)((job->chain * job->draws + i) * Dout) * (off_t)sizeof(double));
      job->stats_seconds += mint_clock() - ts;
      total_leapfrog += s->n_leapfrog;
      job->divergent += s->divergent;
    }
  }
  job->step_size = s->eps;
  job->n_grad = s->n_grad;
  job->n_fused = s->n_fused;
  job->team_min = s->team_min;
  if (bound) restore_team(s->nt);
  job->mean_leapfrog = job->draws ? (double)total_leapfrog / (double)job->draws : 0;
  job->rank = s->k;
  if (!lr_pool) free(lr_q), free(lr_g);
  free(lr_u), free(s->lr_v);

  free(xbuf);
  free(wmean);
  free(wm2);
  free(gmean);
  free(gm2);
  for (size_t k = 0; k < sizeof tv / sizeof tv[0]; k++) free(*tv[k]);
  for (int d = 0; d <= MAX_DEPTH; d++) {
    free(s->lv[d].rho_init);
  }
  for (int k = 0; k < s->n_all; k++) {
    St *x = s->all[k];
    free(x->q), free(x->p), free(x->g), free(x);
  }
  free(s->inv_m);
  free(sp);
  kernel_nt = saved_knt;
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
// The model's fused leapfrog, registered by the generated sample function
// before it calls mint_sample (NULL: none; the Rust baselines never set it).
static mint_leap_fn model_leap;
static mint_leap_blocks_fn model_leap_blocks;
void mint_set_leap(mint_leap_fn f, mint_leap_blocks_fn blocks) {
  model_leap = f;
  model_leap_blocks = blocks;
}

typedef void (*mint_permute_fn)(const double *src, double *dst);
static mint_permute_fn layout_to_internal, layout_to_user;
void mint_set_layout(mint_permute_fn to_internal, mint_permute_fn to_user) {
  layout_to_internal = to_internal;
  layout_to_user = to_user;
}

// The benchmark point is defined in the user's order, so every implementation
// evaluates the same point whatever its internal layout.
// MINT_THETA=FILE (a .f64 file of D values, unconstrained, in the user's
// order) replaces it, for tests that need the log density elsewhere.
static void bench_point(double *theta, int64_t D) {
  for (int64_t i = 0; i < D; i++) theta[i] = 0.05 * (double)((i * 37) % 11 - 5) / 5.0;
  const char *tf = getenv("MINT_THETA");
  if (tf) {
    int64_t n;
    double *v = mint_read_vector(tf, &n);
    if (n != D) {
      fprintf(stderr, "mint runtime error: MINT_THETA has %lld values, the model has %lld parameters\n",
              (long long)n, (long long)D);
      exit(1);
    }
    memcpy(theta, v, (size_t)D * sizeof(double));
    free(v);
  }
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
    printf("exact log density: %.17g\n", lp);
    printf("grad:");
    for (int64_t i = 0; i < D; i++) printf(" %.17g", g[i]);
    printf("\n");
  }
  free(q), free(g);
  exit(0);
}

// MINT_LEAP_TEST: one leaf from the benchmark point (momentum, inverse
// metric, step size and the states of one merge fixed below; eps of both
// signs) through the runtime's own path (the model's logp, then the leaf
// pass) and through the fused leapfrog (MINT_KERNEL_THREADS threads in the
// kernel), first with the diagonal metric and then with a low-rank one
// (LEAP_TEST_K fixed directions; its projections and its part of the next
// position are computed after the kernel, see leaf_fused_run). The
// gradient, log density, final momentum, the next leaf's half-step and the
// merge's summed momentum (with the low-rank metric also their projections,
// and the gradient's) must be bit-identical; the kinetic energy and the
// merge's six sums are summed in a different order and are compared
// relative to the largest of them. With MINT_LEAP_TEST=K > 1, also times K
// steady-state leaves of each (diagonal metric, no merge, eps = 0 so the
// values stay put, three rotating states, the runtime's pass on as many
// threads as the kernel), alternating blocks of each so that both see the
// same machine. Then exits.
#define LEAP_TEST_K 12  // a full group of 8 directions and part of a second
static St *test_state(int64_t D) {
  St *x = calloc(1, sizeof *x);
  x->q = mint_alloc(D), x->p = mint_alloc(D + LR_KMAX), x->g = mint_alloc(D + LR_KMAX);
  return x;
}

static void leap_test(mint_logp_fn f, int64_t D, int64_t reps) {
  if (!model_leap) {
    printf("leap-test: the model has no fused leapfrog\n");
    exit(2);
  }
  Nuts *s = calloc(1, sizeof *s);
  s->D = D, s->f = f, s->leap = model_leap, s->nt = (int)kernel_nt, s->team_min = s->nt;
  leap_other(s, model_leap_blocks);
  double *im = mint_alloc(D);
  St *z = test_state(D), *na = test_state(D), *nb = test_state(D), *xa = test_state(D), *xb = test_state(D);
  double *v[6];
  for (int j = 0; j < 6; j++) v[j] = mint_alloc(D + LR_KMAX);
  bench_point(z->q, D);
  f(z->q, z->g);
  for (int64_t i = 0; i < D; i++) {
    z->p[i] = 0.3 * (double)((i * 53) % 17 - 8) / 8.0;
    im[i] = 0.5 + (double)((i * 29) % 13) / 13.0;
    for (int j = 0; j < 5; j++) v[j][i] = 0.2 * (double)((i * (31 + 6 * j) + j) % 19 - 9) / 9.0;
  }
  s->inv_m = im;
  // the low-rank case's directions (not orthonormal: the leaf's arithmetic
  // does not rely on that), their d_j, and the projections that the leaf's
  // half-step momentum and the merge's states carry
  int kt = D < LEAP_TEST_K ? (int)D : LEAP_TEST_K;
  s->lr_nb = (D + 7) / 8;
  size_t lr_bytes = sizeof(float) * (size_t)((kt + 7) >> 3) * (size_t)s->lr_nb * 64;
  if (posix_memalign((void **)&s->lr_v, 64, lr_bytes)) mint_panic("out of memory");
  memset(s->lr_v, 0, lr_bytes);
  for (int j = 0; j < kt; j++) {
    for (int64_t i = 0; i < D; i++)
      LR_AT(s, j, i) = (float)(0.05 * (double)((i * (7 + 2 * j) + 3 * j) % 23 - 11) / 11.0);
    s->lr_d[j] = j % 3 == 0 ? -0.6 : 0.4 * (double)(j + 1);
    for (int m = 0; m < 5; m++) v[m][D + j] = 0.1 * (double)((j * (5 + m) + m) % 9 - 4);
  }
  // step sizes scaled to the gradient at the benchmark point, so that the
  // momenta and sums stay of moderate size (and the 1e-12 below means
  // something)
  double gmax = 0;
  for (int64_t i = 0; i < D; i++) gmax = fmax(gmax, fabs(z->g[i]));
  double escale = 1.0 / fmax(1.0, 0.01 * gmax);
  int bad = 0;
  for (int lr = 0; lr < 2; lr++)
    for (int sgn = 0; sgn < 2; sgn++) {
      s->k = lr ? kt : 0;
      int k = s->k;
      double eps = (sgn ? -0.00137 : 0.00213) * escale;
      kern_half1(0, D, eps, im, z->p, z->g, z->q, na->p, na->q);
      for (int j = 0; j < k; j++) na->p[D + j] = 0.3 * (double)(j % 5 - 2);  // the half-step's projections
      memcpy(nb->p, na->p, (size_t)(D + k) * sizeof(double));
      memcpy(nb->q, na->q, (size_t)D * sizeof(double));
      double sums[2][7];
      double *out[2] = {v[5], mint_alloc(D + LR_KMAX)};
      for (int w = 0; w < 2; w++) {
        St *n = w ? nb : na, *x = w ? xb : xa;
        LeafJob J = {.s = s, .n = n, .nx = x, .eps = eps, .nlev = 1, .half = 1, .merges = 1};
        J.lev[0] = (LevelJob){.ra = v[0], .bp = v[1], .ep = v[2], .m2p = v[3], .m1p = v[4], .out = out[w]};
        // NaN in every partial sum, so that neither path can pass on values
        // the other left there
        for (int t = 0; t <= MAX_NT; t++)
          for (int j = 0; j < NPART; j++) s->part[t][j] = NAN;
        if (w == 0) {
          n->lp = f(n->q, n->g);
          leaf_pass(s, &J);
        } else {
          leaf_fused_run(s, &J);
        }
        if (k) lr_conclude(s, &J, s->part[0]);
        memcpy(sums[w], s->part[0], sizeof sums[w]);
      }
      int64_t d[6] = {0};
      for (int64_t i = 0; i < D + k; i++) {
        d[0] += memcmp(&na->g[i], &nb->g[i], sizeof(double)) != 0;
        d[1] += memcmp(&na->p[i], &nb->p[i], sizeof(double)) != 0;
        d[2] += memcmp(&xa->p[i], &xb->p[i], sizeof(double)) != 0;
        if (i < D) d[3] += memcmp(&xa->q[i], &xb->q[i], sizeof(double)) != 0;
        d[4] += memcmp(&out[0][i], &out[1][i], sizeof(double)) != 0;
      }
      d[5] = memcmp(&na->lp, &nb->lp, sizeof(double)) != 0;
      // each sum relative to itself, or to 1e-3 of the largest when it is
      // smaller than that (a sum near zero after cancellation)
      double big = 0, worst = 0;
      for (int j = 0; j < 7; j++) big = fmax(big, fabs(sums[0][j]));
      for (int j = 0; j < 7; j++) {
        double e = fabs(sums[0][j] - sums[1][j]) / fmax(fabs(sums[0][j]), 1e-3 * big);
        if (!isfinite(sums[0][j]) || !isfinite(sums[1][j]) || isnan(e)) e = INFINITY;  // fmax drops NaN
        worst = fmax(worst, e);
      }
      printf("leap-test: threads=%d metric=%s eps=%g elements that differ: gradient %lld, momentum %lld, "
             "next momentum %lld, next position %lld, merged momentum %lld; logp differs %lld; kinetic energy "
             "%.17g fused %.17g; largest relative sum difference %.3g\n",
             s->nt, k ? "lowrank" : "diagonal", eps, (long long)d[0], (long long)d[1], (long long)d[2],
             (long long)d[3], (long long)d[4], (long long)d[5], sums[0][0], sums[1][0], worst);
      bad |= d[0] || d[1] || d[2] || d[3] || d[4] || d[5] || !(worst < 1e-12);
      free(out[1]);
    }
  s->k = 0;
  if (reps > 1) {
    St *S[3];
    for (int i = 0; i < 3; i++) {
      S[i] = test_state(D);
      memcpy(S[i]->q, z->q, (size_t)D * sizeof(double));
      memcpy(S[i]->p, z->p, (size_t)D * sizeof(double));
      memcpy(S[i]->g, z->g, (size_t)D * sizeof(double));
    }
    const int64_t B = 20;
    double tsum[2] = {0, 0}, tmin[2] = {1e30, 1e30}, sink = 0;
    int cur = 0;
    for (int64_t r0 = 0; r0 < reps; r0 += B) {
      for (int w = 0; w < 2; w++) {
        double t0 = mint_clock();
        for (int64_t r = 0; r < B; r++) {
          St *n = S[cur], *x = S[(cur + 1) % 3];
          LeafJob J = {.s = s, .n = n, .nx = x, .eps = 0.0, .nlev = 0, .half = 1, .merges = 1};
          if (w == 0) {
            n->lp = f(n->q, n->g);
            leaf_pass(s, &J);
          } else {
            leaf_fused_run(s, &J);
          }
          sink += n->lp + s->part[0][0];
          cur = (cur + 1) % 3;
        }
        double dt = (mint_clock() - t0) / (double)B;
        tsum[w] += dt;
        if (dt < tmin[w]) tmin[w] = dt;
      }
    }
    double nblk = (double)((reps + B - 1) / B);
    printf("leap-test: reps=%lld threads=%d ns per leaf: runtime mean %.1f fastest %.1f, fused mean %.1f "
           "fastest %.1f (sink %g)\n",
           (long long)reps, s->nt, 1e9 * tsum[0] / nblk, 1e9 * tmin[0], 1e9 * tsum[1] / nblk, 1e9 * tmin[1],
           sink * 0);
  }
  printf("leap-test: %s\n", bad ? "MISMATCH" : "ok");
  exit(bad);
}

typedef struct {
  int64_t D, draws, chains, nparams;  // D: values per draw (with any collapsed parameters)
  int64_t D_sampled, n_latent;        // the dimension NUTS ran in; latent scalars collapsed
  char **labels;  // D labels
  char **block_name;
  int64_t *block_start, *block_len;  // per parameter: first flat index, element count
  int64_t nkeep;     // parameters whose draws are kept (D: all of them, the full-draw path)
  int64_t *keep_idx;  // nkeep flat indices, increasing
  int64_t *keep_pos;  // D: position in keep_idx, or -1
  double *draw;       // chains x draws x nkeep
  ChainStats *stats;  // per chain, for every parameter
  double seconds;
  int64_t n_grad, divergent, warmup_grad;
  double *step_size, *mean_leapfrog;
  int threads_per_chain, team_min, grad_metric;
  int leapfrog;     // the fused leapfrog was enabled (1), with exact sums (2), or not (0)
  int64_t n_fused;  // leaves that ran through it
  int64_t warmup_iters;
  int warmup_fast;
  int lowrank, rank_min, rank_max;
  double stats_seconds;  // summed over chains
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

// Split R-hat from the means and (n - 1)-normalised variances of M half
// chains of h draws each.
static double split_rhat(const double *means, const double *vars, int64_t M, int64_t h) {
  double grand = 0, W = 0, B = 0;
  for (int64_t m = 0; m < M; m++) grand += means[m], W += vars[m];
  grand /= (double)M;
  W /= (double)M;
  for (int64_t m = 0; m < M; m++) B += (means[m] - grand) * (means[m] - grand);
  B = B * (double)h / (double)(M - 1);
  double var_plus = ((double)(h - 1) / (double)h) * W + B / (double)h;
  return W > 0 ? sqrt(var_plus / W) : NAN;
}

// var_plus of C chains of N draws from their means cm and their
// N-normalised variances cv: ((N - 1) / N) W + B / N, with W the mean
// within-chain variance and B / N the variance of the chain means. *W_out
// gets W.
static double var_plus(const double *cm, const double *cv, int64_t C, int64_t N, double *W_out) {
  double Wf = 0, gm = 0, Bf = 0;
  for (int64_t c = 0; c < C; c++) Wf += cv[c] * (double)N / (double)(N - 1), gm += cm[c];
  Wf /= (double)C;
  gm /= (double)C;
  if (C > 1) {
    for (int64_t c = 0; c < C; c++) Bf += (cm[c] - gm) * (cm[c] - gm);
    Bf = Bf * (double)N / (double)(C - 1);
  }
  *W_out = Wf;
  return ((double)(N - 1) / (double)N) * Wf + Bf / (double)N;
}

// Geyer's initial monotone sequence on the chain-averaged autocovariances of
// C series of N values (value i of chain c at x[(c N + i) stride]), as
// Stan's ESS does: the integrated autocorrelation time relative to var_plus,
// which goes to *vp (NAN, and *vp not positive, when the series are
// constant). cm and cv are C doubles of scratch.
static double geyer_tau(const double *x, int64_t C, int64_t N, int64_t stride, double *cm, double *cv, double *vp) {
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
  double Wf;
  *vp = var_plus(cm, cv, C, N, &Wf);
  if (!(*vp > 0)) return NAN;
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
      rho[k] = lag == 0 ? 1.0 : 1.0 - (Wf - acov) / *vp;
    }
    double pair = rho[0] + rho[1];
    if (pair < 0) break;
    if (pair > prev_pair) pair = prev_pair;  // monotone sequence
    prev_pair = pair;
    tau += 2 * pair;
  }
  return tau - 1;
}

// Split-R-hat and an autocorrelation-based ESS (Geyer's initial monotone
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
  *rhat = split_rhat(means, vars, M, h);

  // ESS over the full chains
  double vp;
  double tau = geyer_tau(x, C, N, stride, means, vars, &vp);
  if (!(vp > 0)) {
    *ess = NAN;
  } else {
    if (tau < 1.0 / log10((double)(C * N))) tau = 1.0 / log10((double)(C * N));
    *ess = (double)(C * N) / tau;
  }
  free(means), free(vars);
}

// The same statistics from the chains' streaming summaries (ChainStats) of
// parameter j. Mean, sd and split R-hat are those of the draws, up to
// rounding. The ESS is the draw-level estimate, from the autocovariances
// of v, when Geyer's sequence stops within the lags kept (or they are all
// of them); *fallback is then 0. Otherwise (*fallback = 1) it applies the
// same estimator to the series of batch means: with batches of b draws, the
// variance of the overall mean is estimated as var_plus of the batch means
// times their integrated autocorrelation time, divided by the C nb batch
// means, and the ESS is the draws' var_plus over that (with b = 1 it is the
// draw-level estimate). As in rhat_ess, a sequence whose first pair is
// negative gives the largest ESS, C N log10(C N). scratch: 6 C + C nb + P
// doubles.
static void stream_stats(const ChainStats *S, int64_t C, int64_t j, double *scratch, double *mean, double *sd,
                         double *rhat, double *ess, int *fallback) {
  int64_t N = S[0].N, h = S[0].h, nb = S[0].nb, P = S[0].P, D = S[0].D;
  double *means = scratch, *vars = scratch + 2 * C, *cm = scratch + 4 * C, *cv = scratch + 5 * C;
  double *y = scratch + 6 * C, *acs = y + C * nb;
  for (int64_t c = 0; c < C; c++) {
    means[2 * c] = S[c].m1[j], vars[2 * c] = S[c].s1[j] / (double)(h - 1);
    means[2 * c + 1] = S[c].m2[j], vars[2 * c + 1] = S[c].s2[j] / (double)(h - 1);
  }
  *rhat = split_rhat(means, vars, 2 * C, h);

  double g = 0;
  for (int64_t c = 0; c < C; c++) g += S[c].mt[j];
  g /= (double)C;
  double ss = 0;
  for (int64_t c = 0; c < C; c++) ss += S[c].st[j] + (double)N * (S[c].mt[j] - g) * (S[c].mt[j] - g);
  *mean = g;
  *sd = sqrt(ss / (double)(C * N - 1));
  *fallback = 0;

  for (int64_t c = 0; c < C; c++) cm[c] = S[c].mt[j], cv[c] = S[c].st[j] / (double)N;
  double W;
  double vp = var_plus(cm, cv, C, N, &W);
  if (!(vp > 0)) {
    *ess = NAN;
    return;
  }
  // chain-averaged autocovariance sums of each pair of lags: lag 1 for
  // m = 0, lags 2m and 2m + 1 otherwise. For lag k,
  //   N acov_k = sum_{i < N - k} v_i v_{i+k} - mu (A_k + B_k) + (N - k) mu^2
  // with A_k the sum of all but the last k values and B_k of all but the
  // first k (v_0 = 0: the first draw is the shift).
  for (int64_t m = 0; m < P; m++) acs[m] = 0;
  for (int64_t c = 0; c < C; c++) {
    const ChainStats *Z = &S[c];
    double sv = Z->vsum[j], mu = sv / (double)N;
#define RING(i) ((double)Z->ring[((i) % Z->R) * D + j])
    double tail = 0;  // the sum of the last 2m values
    double A = sv - RING(N - 1), B = sv;
    acs[0] += (Z->q[j] - mu * (A + B) + (double)(N - 1) * mu * mu) / (double)N;
    tail = RING(N - 1) + RING(N - 2);
    for (int64_t m = 1; m < P; m++) {
      double A2 = 2 * sv - (2 * tail + RING(N - 1 - 2 * m));
      double B2 = 2 * sv - (double)Z->head[m * D + j];
      acs[m] += (Z->q[m * D + j] - mu * (A2 + B2) + (double)(2 * N - 4 * m - 1) * mu * mu) / (double)N;
      tail += RING(N - 1 - 2 * m) + RING(N - 2 - 2 * m);
    }
#undef RING
  }
  double tau = 0, prev_pair = INFINITY;
  int stopped = 0;
  for (int64_t m = 0; m < P; m++) {
    double a = acs[m] / (double)C;
    double pair = m == 0 ? 1.0 + (1.0 - (W - a) / vp) : 2.0 - (2.0 * W - a) / vp;
    if (pair < 0) {
      stopped = 1;
      break;
    }
    if (pair > prev_pair) pair = prev_pair;
    prev_pair = pair;
    tau += 2 * pair;
  }
  if (stopped || P >= N / 2) {
    tau -= 1;
    if (tau < 1.0 / log10((double)(C * N))) tau = 1.0 / log10((double)(C * N));
    *ess = (double)(C * N) / tau;
    return;
  }

  *fallback = 1;
  for (int64_t c = 0; c < C; c++)
    for (int64_t k = 0; k < nb; k++) y[c * nb + k] = (double)S[c].bm[k * D + j] + S[c].shift[j];
  double vpy;
  double tauy = geyer_tau(y, C, nb, 1, cm, cv, &vpy);
  double cap = (double)(C * N) * log10((double)(C * N));  // as the draw-level estimate's lower bound on tau
  if (!(vpy > 0)) {
    *ess = NAN;
  } else if (!(tauy > 0)) {
    *ess = cap;
  } else {
    double e = (double)(C * nb) * vp / (vpy * tauy);
    *ess = e < cap ? e : cap;
  }
}

// How many rows of a parameter the summary prints with all its columns:
// every entry of a parameter with at most SUMMARY_ALL entries, otherwise the
// first SUMMARY_HEAD.
#define SUMMARY_ALL 12
#define SUMMARY_HEAD 3
static int64_t summary_rows(int64_t len) { return len > SUMMARY_ALL ? SUMMARY_HEAD : len; }

MintPosterior *mint_sample(mint_logp_fn f, mint_constrain_fn constrain, int64_t D, int64_t draws,
                           int64_t warmup, int64_t chains, int64_t seed, int64_t nparams,
                           const char **names, const int64_t *sizes) {
  if (draws < 4) mint_panic("sample: draws must be at least 4");
  if (chains < 1) mint_panic("sample: chains must be at least 1");
  if (warmup < 0) mint_panic("sample: warmup must be non-negative");

  const char *bench = getenv("MINT_BENCH_GRAD");
  kernel_nt = kernel_threads(1);
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
  const char *ltest = getenv("MINT_LEAP_TEST");
  if (ltest) leap_test(f, D, atoll(ltest));
  kernel_nt = 1;
  // The model's fused leapfrog, when it has one (only a program built with
  // mintc --fused-leapfrog does): MINT_FUSED_LEAPFROG=0 turns it off, and
  // =exact keeps the sums in the runtime's own order; unset or any other
  // value is on.
  const char *lenv = getenv("MINT_FUSED_LEAPFROG");
  int leap_mode = !lenv ? 2 : strcmp(lenv, "0") == 0 ? 0 : strcmp(lenv, "exact") == 0 ? 3 : 2;

  // With collapsed parameters each draw has more values than NUTS's D.
  int64_t D_out = model_collapse ? model_d_out : D;
  MintPosterior *post = calloc(1, sizeof *post);
  post->D = D_out, post->draws = draws, post->chains = chains, post->nparams = nparams;
  post->D_sampled = D, post->n_latent = model_collapse ? model_n_latent : 0;
  post->labels = calloc((size_t)D_out, sizeof(char *));
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
      if (k < D_out) post->labels[k] = strdup(names[j]);
      k++;
      continue;
    }
    for (int64_t i = 0; i < sizes[j]; i++) {
      char buf[128];
      snprintf(buf, sizeof buf, "%s[%lld]", names[j], (long long)(i + 1));
      if (k < D_out) post->labels[k] = strdup(buf);
      k++;
    }
  }
  if (k != D_out) mint_panic("sample: parameter sizes do not add up to the model dimension");

  // Which parameters keep every draw: the rows the summary prints (with
  // their quantiles), every parameter named in MINT_KEEP_DRAWS (a comma
  // separated list), or all of them with MINT_KEEP_DRAWS=all (the full-draw
  // path: the summary is then computed from the draws alone, as before
  // streaming summaries). The others are summarised as they are drawn.
  // (over every value of a draw: the collapsed parameters, drawn by FFBS,
  // are kept and summarised like the others)
  post->keep_pos = calloc((size_t)D_out, sizeof(int64_t));
  for (int64_t i = 0; i < D_out; i++) post->keep_pos[i] = -1;
  const char *kenv = getenv("MINT_KEEP_DRAWS");
  int keep_all = kenv && strcmp(kenv, "all") == 0;
  // MINT_DRAWS: every constrained draw, u64 chains, draws, D_out, then
  // chains x draws x D_out f64 (D_out = D without collapsed parameters). To a regular file (or a new one), each chain
  // writes its draws into its own part of FILE.partial as they are drawn,
  // and the file is renamed to FILE when sampling has finished, so a run
  // that dies leaves FILE as it was. Anything else (a pipe, a terminal)
  // cannot be written out of order: then every draw is kept in memory and
  // written in order after sampling, as before streaming summaries.
  const char *dump = getenv("MINT_DRAWS");
  int draws_fd = -1;
  char *draws_tmp = NULL;
  if (dump) {
    struct stat sb;
    if (stat(dump, &sb) != 0 || S_ISREG(sb.st_mode)) {
      draws_tmp = malloc(strlen(dump) + 9);
      if (!draws_tmp) mint_panic("out of memory");
      strcpy(draws_tmp, dump);
      strcat(draws_tmp, ".partial");
      draws_fd = open(draws_tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (draws_fd < 0) mint_panic("cannot open MINT_DRAWS file");
      uint64_t hdr[3] = {(uint64_t)chains, (uint64_t)draws, (uint64_t)D_out};
      write_at(draws_fd, hdr, sizeof hdr, 0);
    } else {
      keep_all = 1;
    }
  }
  for (int64_t j = 0; j < nparams; j++) {
    int64_t s0 = post->block_start[j], len = post->block_len[j];
    int64_t rows = keep_all ? len : summary_rows(len);
    for (int64_t i = s0; i < s0 + rows; i++) post->keep_pos[i] = 0;
  }
  if (kenv && !keep_all) {
    char *list = strdup(kenv), *save = NULL;
    for (char *tok = strtok_r(list, ", ", &save); tok; tok = strtok_r(NULL, ", ", &save)) {
      int64_t j = 0;
      while (j < nparams && strcmp(post->block_name[j], tok) != 0) j++;
      if (j == nparams) {
        fprintf(stderr, "warning: MINT_KEEP_DRAWS names %s, which is not a parameter of this model\n", tok);
        continue;
      }
      for (int64_t i = post->block_start[j]; i < post->block_start[j] + post->block_len[j]; i++) post->keep_pos[i] = 0;
    }
    free(list);
  }
  post->keep_idx = calloc((size_t)D_out, sizeof(int64_t));
  for (int64_t i = 0; i < D_out; i++)
    if (post->keep_pos[i] == 0) post->keep_pos[i] = post->nkeep, post->keep_idx[post->nkeep++] = i;
  post->draw = mint_alloc(chains * draws * post->nkeep);
  post->stats = calloc((size_t)chains, sizeof(ChainStats));

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
  cpu_set_t l3[MAX_L3];
  int n_l3 = 0;
  const char *aenv = getenv("MINT_CHAIN_AFFINITY");
  if (tpc > 1 && !(aenv && strcmp(aenv, "0") == 0)) n_l3 = l3_groups(l3, MAX_L3);
  ChainJob *jobs = calloc((size_t)chains, sizeof *jobs);
  pthread_t *th = calloc((size_t)chains, sizeof *th);
  WarmupCfg cfg = warmup_cfg(warmup);
  const char *metric_env = getenv("MINT_METRIC");
  post->grad_metric = metric_env && strcmp(metric_env, "grad") == 0;
  post->lowrank = metric_env && strcmp(metric_env, "lowrank") == 0;
  Pool pool, *pool_p = NULL;
  memset(&pool, 0, sizeof pool);
  if (cfg.pool && chains > 1 && !post->grad_metric) {
    pool.chains = chains;
    pool.n = calloc((size_t)chains, sizeof *pool.n);
    pool.mean = calloc((size_t)chains, sizeof *pool.mean);
    pool.m2 = calloc((size_t)chains, sizeof *pool.m2);
    if (pthread_barrier_init(&pool.bar, NULL, (unsigned)chains) != 0) mint_panic("could not create a barrier");
    pool_p = &pool;
    // the low-rank metric's window rows (see Pool); a chain's team is at most
    // tpc threads, so it keeps at most lowrank_k(D, tpc) directions
    int kmax = lowrank_k(D, tpc);
    if (kmax > D) kmax = (int)D;
    int nmax = post->lowrank ? lowrank_nmax(D, cfg.iters) : 0;
    if (nmax && kmax) {
      pool.lr_nmax = nmax;
      pool.lr_q = mint_alloc((int64_t)chains * nmax * D);
      pool.lr_g = mint_alloc((int64_t)chains * nmax * D);
      pool.lr_u = mint_alloc((int64_t)kmax * D);
      pool.lr_rows = calloc((size_t)chains, sizeof *pool.lr_rows);
      pool.lr_kmax = calloc((size_t)chains, sizeof *pool.lr_kmax);
    }
  }
  post->warmup_iters = cfg.iters;
  post->warmup_fast = cfg.fast;
  double t0 = mint_clock();
  for (int64_t c = 0; c < chains; c++) {
    jobs[c] = (ChainJob){.f = f,
                         .leap = model_leap,
                         .leap_blocks = model_leap_blocks,
                         .leap_mode = leap_mode,
                         .constrain = constrain,
                         .collapse = model_collapse,
                         .D = D,
                         .d_out = D_out,
                         .draws = draws,
                         .warmup = warmup,
                         .seed = (uint64_t)seed,
                         .chain = (int)c,
                         .threads_per_chain = tpc,
                         .l3 = n_l3 > 1 ? &l3[c % n_l3] : NULL,
                         .cfg = &cfg,
                         .pool = pool_p,
                         .keep_idx = post->keep_idx,
                         .nkeep = post->nkeep,
                         .draws_fd = draws_fd,
                         .out = post->draw + c * draws * post->nkeep,
                         .stats = &post->stats[c]};
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
  if (pool_p) {
    pthread_barrier_destroy(&pool.bar);
    free(pool.n), free(pool.mean), free(pool.m2);
    free(pool.lr_q), free(pool.lr_g), free(pool.lr_u), free(pool.lr_rows), free(pool.lr_kmax);
  }
  post->leapfrog = model_leap && leap_mode >= 2 ? (leap_mode == 3 ? 2 : 1) : 0;
  post->rank_min = jobs[0].rank, post->rank_max = jobs[0].rank;
  for (int64_t c = 0; c < chains; c++) {
    if (jobs[c].rank < post->rank_min) post->rank_min = jobs[c].rank;
    if (jobs[c].rank > post->rank_max) post->rank_max = jobs[c].rank;
    if (jobs[c].team_min < post->team_min) post->team_min = jobs[c].team_min;
    post->n_grad += jobs[c].n_grad;
    post->n_fused += jobs[c].n_fused;
    post->warmup_grad += jobs[c].warmup_grad;
    post->divergent += jobs[c].divergent;
    post->step_size[c] = jobs[c].step_size;
    post->mean_leapfrog[c] = jobs[c].mean_leapfrog;
    post->stats_seconds += jobs[c].stats_seconds;
  }
  free(jobs);
  free(th);
  if (draws_fd >= 0) {
    if (close(draws_fd) != 0 || rename(draws_tmp, dump) != 0) mint_panic("cannot write the MINT_DRAWS file");
    free(draws_tmp);
  } else if (dump) {  // not a regular file: every draw was kept, written in order
    FILE *f = fopen(dump, "wb");
    if (!f) mint_panic("cannot open MINT_DRAWS file");
    uint64_t hdr[3] = {(uint64_t)chains, (uint64_t)draws, (uint64_t)D_out};
    fwrite(hdr, sizeof hdr[0], 3, f);
    fwrite(post->draw, sizeof(double), (size_t)(chains * draws * D_out), f);
    if (fclose(f) != 0) mint_panic("cannot write the MINT_DRAWS file");
  }
  return post;
}

typedef struct {
  double mean, sd, q5, q50, q95, ess, rhat;
  int fallback;  // streaming ESS from batch means (see stream_stats)
} ColStat;

typedef struct {
  MintPosterior *p;
  ColStat *out;
  ColStat *stream;  // with MINT_STATS_DUMP: the streaming statistics of every parameter, or NULL
  int64_t lo, hi;
} StatJob;

// Statistics of parameters lo..hi-1: from the draws when they are kept (the
// quantiles need them; the ESS and R-hat then use every draw), otherwise
// from the streaming summaries, without quantiles (the summary never prints
// those rows).
static void *stat_worker(void *arg) {
  StatJob *j = arg;
  MintPosterior *p = j->p;
  int64_t N = p->draws, C = p->chains, K = p->nkeep, M = N * C;
  double *col = mint_alloc(M);
  double *scratch = mint_alloc(6 * C + C * p->stats[0].nb + p->stats[0].P);
  for (int64_t c = j->lo; c < j->hi; c++) {
    ColStat *o = &j->out[c];
    int64_t pos = p->keep_pos[c];
    if (j->stream) {
      ColStat *z = &j->stream[c];
      z->q5 = z->q50 = z->q95 = NAN;
      stream_stats(p->stats, C, c, scratch, &z->mean, &z->sd, &z->rhat, &z->ess, &z->fallback);
    }
    if (pos < 0) {
      o->q5 = o->q50 = o->q95 = NAN;
      stream_stats(p->stats, C, c, scratch, &o->mean, &o->sd, &o->rhat, &o->ess, &o->fallback);
      continue;
    }
    double s = 0;
    for (int64_t i = 0; i < M; i++) col[i] = p->draw[i * K + pos], s += col[i];
    o->mean = s / (double)M;
    double v = 0;
    for (int64_t i = 0; i < M; i++) v += (col[i] - o->mean) * (col[i] - o->mean);
    o->sd = sqrt(v / (double)(M - 1));
    rhat_ess(p->draw + pos, C, N, K, &o->rhat, &o->ess);
    qsort(col, (size_t)M, sizeof(double), cmp_double);
    o->q5 = quantile_sorted(col, M, 0.05);
    o->q50 = quantile_sorted(col, M, 0.5);
    o->q95 = quantile_sorted(col, M, 0.95);
  }
  free(col);
  free(scratch);
  return NULL;
}

void mint_print_posterior(MintPosterior *p) {
  int64_t N = p->draws, C = p->chains, D = p->D;
  // MINT_STATS_DUMP=FILE writes, for every parameter, the statistics the
  // summary used and the streaming ones (for comparing the two on the same
  // draws; with MINT_KEEP_DRAWS=all every parameter has both).
  const char *sdump = getenv("MINT_STATS_DUMP");
  // per-column statistics, computed on up to 24 threads
  ColStat *st = calloc((size_t)D, sizeof *st);
  ColStat *ss = sdump ? calloc((size_t)D, sizeof *ss) : NULL;
  int64_t nt = D < 24 ? D : 24;
  pthread_t th[24];
  StatJob jobs[24];
  for (int64_t t = 0; t < nt; t++) {
    jobs[t] = (StatJob){p, st, ss, D * t / nt, D * (t + 1) / nt};
    pthread_create(&th[t], NULL, stat_worker, &jobs[t]);
  }
  for (int64_t t = 0; t < nt; t++) pthread_join(th[t], NULL);
  if (sdump) {
    FILE *f = fopen(sdump, "w");
    if (!f) mint_panic("cannot open MINT_STATS_DUMP file");
    fprintf(f, "parameter\tkept\tmean\tsd\trhat\tess\tstream_mean\tstream_sd\tstream_rhat\tstream_ess\tstream_fallback\n");
    for (int64_t c = 0; c < D; c++)
      fprintf(f, "%s\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%d\n", p->labels[c],
              p->keep_pos[c] >= 0, st[c].mean, st[c].sd, st[c].rhat, st[c].ess, ss[c].mean, ss[c].sd, ss[c].rhat,
              ss[c].ess, ss[c].fallback);
    if (fclose(f) != 0) mint_panic("cannot write the MINT_STATS_DUMP file");
    free(ss);
  }

  printf("%-14s %12s %12s %12s %12s %12s %8s %6s\n", "parameter", "mean", "sd", "5%", "50%", "95%",
         "ess", "rhat");
  int64_t worst_r = 0, worst_e = 0, nfall = 0;
  for (int64_t c = 0; c < D; c++) {
    nfall += st[c].fallback;
    if (st[c].rhat > st[worst_r].rhat || isnan(st[c].rhat)) worst_r = c;
    if (st[c].ess < st[worst_e].ess || isnan(st[c].ess)) worst_e = c;
  }
  for (int64_t b = 0; b < p->nparams; b++) {
    int64_t s0 = p->block_start[b], len = p->block_len[b];
    int64_t shown = summary_rows(len);
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
  fprintf(stderr, "gradients: warmup=%lld sampling=%lld (warmup=%s, %lld iterations)\n", (long long)p->warmup_grad,
          (long long)(p->n_grad - p->warmup_grad), p->warmup_fast ? "fast" : "stan", (long long)p->warmup_iters);
  fprintf(stderr, "sampler: threads per chain=%d (smallest team that ran=%d) metric=%s", p->threads_per_chain,
          p->team_min, p->grad_metric ? "grad" : p->lowrank ? "lowrank" : "stan");
  if (p->lowrank) fprintf(stderr, " (rank %d to %d)", p->rank_min, p->rank_max);
  // the leapfrog as it ran: "fused" only when leaves went through it
  fprintf(stderr, " leapfrog=%s", !p->n_fused ? "runtime" : p->leapfrog == 2 ? "fused-exact" : "fused");
  if (p->n_fused) fprintf(stderr, " (%lld of %lld gradients)", (long long)p->n_fused, (long long)p->n_grad);
  fprintf(stderr, "\n");
  if (p->D_sampled != p->D)
    fprintf(stderr,
            "collapsed: NUTS sampled %lld of the %lld parameters; %lld latent scalars were integrated out by a Kalman "
            "filter and drawn for each kept draw by forward filtering, backward sampling\n",
            (long long)p->D_sampled, (long long)p->D, (long long)p->n_latent);
  const ChainStats *S = &p->stats[0];
  fprintf(stderr, "summary: draws kept for %lld of %lld parameters (constraining%s, keeping and summarising draws took %.3f s summed over chains)",
          (long long)p->nkeep, (long long)D, p->n_latent ? " with the FFBS draws" : "", p->stats_seconds);
  if (p->nkeep < D)
    fprintf(stderr, "; ess of the others from autocovariances at lags below %lld, %lld of them from %lld batch means of %lld draws per chain",
            (long long)S->L, (long long)nfall, (long long)S->nb, (long long)S->b);
  fprintf(stderr, "\n");
}

// Posterior mean of flat component j (0-based), used by generated code.
double mint_posterior_mean(MintPosterior *p, int64_t j) {
  int64_t pos = p->keep_pos[j];
  if (pos < 0) {
    double g = 0;
    for (int64_t c = 0; c < p->chains; c++) g += p->stats[c].mt[j];
    return g / (double)p->chains;
  }
  double s = 0;
  int64_t n = p->draws * p->chains;
  for (int64_t i = 0; i < n; i++) s += p->draw[i * p->nkeep + pos];
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

// Narrow data. A model's vector kernels can read a copy of a data buffer in a
// narrower type when that type holds every value exactly: converting back
// gives the same double, bit for bit, so the results are unchanged. Kinds:
// 1 int8, 2 int16, 3 float. An integer kind rejects -0.0 (it would come back
// as +0.0); float rejects NaN (its payload might not survive) and keeps
// infinities, -0.0 and values that are float subnormals.
static int narrow_exact(double v, int kind) {
  uint64_t a, b;
  double back;
  switch (kind) {
    case 1:
      if (!(v >= -128.0 && v <= 127.0)) return 0;
      back = (double)(int8_t)v;
      break;
    case 2:
      if (!(v >= -32768.0 && v <= 32767.0)) return 0;
      back = (double)(int16_t)v;
      break;
    case 3:
      if (isnan(v) || (isfinite(v) && fabs(v) > 3.4028234663852886e38)) return 0;
      back = (double)(float)v;
      break;
    default:
      return 0;
  }
  memcpy(&a, &v, sizeof a);
  memcpy(&b, &back, sizeof b);
  return a == b;
}

// The narrowest kind among `allowed` (bit k-1 for kind k) that holds every
// value of p[0..n) exactly: returns a new copy in that kind and stores the
// kind in *kind, or returns NULL with *kind = 0 when none does or
// MINT_NARROW=0. MINT_NARROW=int16 or =float skips the narrower kinds (for
// tests and measurements). MINT_NARROW_REPORT=1 prints the choice to stderr.
void *mint_narrow(const double *p, int64_t n, int64_t allowed, int64_t *kind, const char *what) {
  static const char *names[] = {"double", "int8", "int16", "float"};
  const char *env = getenv("MINT_NARROW");
  int k = 0, first = 1;
  if (env && strcmp(env, "0") == 0) first = 4;
  else if (env && strcmp(env, "int16") == 0) first = 2;
  else if (env && strcmp(env, "float") == 0) first = 3;
  {
    for (int c = first; c <= 3 && !k; c++) {
      if (!(allowed & (1 << (c - 1)))) continue;
      int64_t i = 0;
      while (i < n && narrow_exact(p[i], c)) i++;
      if (i == n) k = c;
    }
  }
  *kind = k;
  void *q = NULL;
  if (k) {
    static const int bytes[] = {0, 1, 2, 4};
    q = mint_alloc((n * bytes[k] + 7) / 8);
    for (int64_t i = 0; i < n; i++) {
      if (k == 1) ((int8_t *)q)[i] = (int8_t)p[i];
      else if (k == 2) ((int16_t *)q)[i] = (int16_t)p[i];
      else ((float *)q)[i] = (float)p[i];
    }
  }
  const char *rep = getenv("MINT_NARROW_REPORT");
  if (rep && strcmp(rep, "0") != 0) fprintf(stderr, "narrow: %s: %s\n", what, names[k]);
  return q;
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
