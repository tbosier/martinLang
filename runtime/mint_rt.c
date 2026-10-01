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

// ---------------------------------------------------------------- NUTS
// Multinomial NUTS with the generalised no-U-turn criterion, diagonal metric
// and Stan's windowed warmup (step-size dual averaging plus metric windows).
// This is a port of the structure of Stan's base_nuts.hpp.

typedef double (*mint_logp_fn)(const double *theta, double *grad);
typedef void (*mint_constrain_fn)(const double *unc, double *out);

#define MAX_DEPTH 10

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
    x->p = mint_alloc(s->D);
    x->g = mint_alloc(s->D);
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

// Leaf step of the tree: leapfrog, then in one pass the final momentum and
// the kinetic energy. The sampler's sums are vectorised (`omp simd
// reduction`): a fixed order, so draws are reproducible for a given build,
// but not the strict left-to-right order of scalar code. Returns the Hamiltonian. (A leaf's momentum sum is its
// own momentum, so nothing is copied for it; see build_tree.)
static double leaf_into(Nuts *s, const St *z, St *n, double eps) {
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
static int merge_checks(Nuts *s, double *rho, const double *ra, const double *rb, const double *beg_p,
                        const double *end_p, const double *mid2_p, const double *mid1_p) {
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
  return (v[0] > 0 && v[1] > 0) & (v[2] > 0 && v[3] > 0) & (v[4] > 0 && v[5] > 0);
}

static void sample_momentum(Nuts *s, St *z) {
  for (int64_t i = 0; i < s->D; i++) z->p[i] = rng_normal(&s->rng) / sqrt(s->inv_m[i]);
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
  vcopy(t->rho, z0->p, D);

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

// Stan's heuristic: double or halve the step size until the acceptance
// probability of a single leapfrog step crosses 0.8.
static void init_stepsize(Nuts *s) {
  St *a = st_clone_position(s, s->cur), *b = st_acquire(s);
  sample_momentum(s, a);
  double H0 = hamiltonian(s, a);
  leapfrog_into(s, a, b, s->eps);
  double h = hamiltonian(s, b);
  if (isnan(h)) h = INFINITY;
  int direction = (H0 - h) > log(0.8) ? 1 : -1;
  for (;;) {
    sample_momentum(s, a);
    H0 = hamiltonian(s, a);
    leapfrog_into(s, a, b, s->eps);
    h = hamiltonian(s, b);
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
    L->rho_init = mint_alloc(D);
    L->rho_final = mint_alloc(D);
  }
  Traj t;
  double **tv[] = {&t.rho, &t.rho_fwd, &t.rho_bck};
  for (size_t k = 0; k < sizeof tv / sizeof tv[0]; k++) *tv[k] = mint_alloc(D);

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
        }
        if (end_window(&w)) {
          next_window(&w);
          double n = (double)wn;
          for (int64_t i = 0; i < D; i++) {
            double var = wn > 1 ? wm2[i] / (n - 1.0) : 1.0;
            if (grad_metric && wn > 1 && gm2[i] > 0) var = sqrt(var / (gm2[i] / (n - 1.0)));
            s->inv_m[i] = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0));
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
  for (int64_t c = 0; c < chains; c++) {
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
  fprintf(stderr, "sampler: threads per chain=%d (smallest team that ran=%d) metric=%s\n", p->threads_per_chain,
          p->team_min, p->grad_metric ? "grad" : "stan");
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
