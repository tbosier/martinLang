// Mint runtime: memory, binary I/O, printing, Cholesky solve, RNG and a NUTS sampler.
//
// Generated programs call into this file for everything that is not model or
// numeric-kernel code. The Rust baselines link the same object so that the
// sampler is identical on both sides of the benchmark.

#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
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
#define NPART (1 + 6 * (MAX_DEPTH + 2))

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
  // merges waiting for a leaf: pend[pend_lo .. npend-1], innermost last
  Merge pend[MAX_DEPTH + 2];
  int npend, pend_lo;
  // the next leaf's first half-step, computed ahead by the previous leaf
  St *spec;
  const St *spec_from;
  double spec_eps;
  double part[MAX_NT][NPART];  // per-thread partial sums of a leaf pass
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
// the number of threads.
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

// Second half-step over [lo, hi) (lo a multiple of LN): the final momentum
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
  LevelJob lev[MAX_DEPTH + 2];
} LeafJob;

static void leaf_work(LeafJob *J, int t, int T) {
  Nuts *s = J->s;
  St *n = J->n, *nx = J->nx;
  const double *im = s->inv_m;
  int64_t lo, hi;
  split_range(s->D, t, T, &lo, &hi);
  double k[LN] = {0};
  double acc[MAX_DEPTH + 2][6][LN];
  memset(acc, 0, (size_t)J->nlev * sizeof acc[0]);
  double buf[2][CHUNK];
  for (int64_t c0 = lo; c0 < hi; c0 += CHUNK) {
    int64_t c1 = c0 + CHUNK < hi ? c0 + CHUNK : hi, m = c1 - c0;
    kern_half2(c0, c1, J->eps, im, n->p, n->g, n->q, nx ? nx->p : NULL, nx ? nx->q : NULL, k);
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
  double *part = s->part[t];
  part[0] = lanes_total(k);
  for (int v = 0; v < J->nlev; v++)
    for (int j = 0; j < 6; j++) part[1 + 6 * v + j] = lanes_total(acc[v][j]);
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
  int nt = s->nt;
  if (nt == 1) {
    kern_half1(0, D, eps, s->inv_m, z->p, z->g, z->q, n->p, n->q);
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
  }
  note_team(s, used);
}

// Second half of leaf n's leapfrog and the merges waiting for it (see above).
// Returns the Hamiltonian; each completed merge's result is in its Merge.
static double leaf_finish(Nuts *s, St *n, double eps) {
  int lo = s->pend_lo, hi = s->npend;
  LeafJob J = {.s = s, .n = n, .eps = eps, .nlev = hi - lo};
  // When the merges reach the transition's own (pend[0]), this leaf ends the
  // new subtree and the next direction is not yet drawn: no next leaf.
  if (!(lo == 0 && hi > 0)) J.nx = st_acquire(s);
  for (int v = 0; v < J.nlev; v++) {
    const Merge *m = &s->pend[hi - 1 - v];
    LevelJob *L = &J.lev[v];
    L->ra = m->ra, L->rb = m->rb;
    L->bp = (*m->beg)->p, L->ep = (*m->end)->p, L->m2p = (*m->mid2)->p, L->m1p = (*m->mid1)->p;
    L->out = v == J.nlev - 1 ? m->rho : NULL;
    if (v == J.nlev - 1 && !L->out) mint_panic("internal error: outermost merge has no output");
    // two single leaves: the halves are the earlier leaf and this one
    L->leaves = v == 0 && m->ra && !m->rb && m->ra == L->bp && m->ra == L->m1p && L->ep == n->p &&
                L->m2p == n->p;
  }
  if (s->nt == 1) {
    leaf_work(&J, 0, 1);
    note_team(s, 1);
  } else {
    int used = 1;
#pragma omp parallel num_threads(s->nt)
    {
      int t = omp_get_thread_num(), T = omp_get_num_threads();
      if (t == 0) used = T;
      leaf_work(&J, t, T);
    }
    note_team(s, used);
    for (int t = 1; t < used; t++)
      for (int j = 0; j < 1 + 6 * J.nlev; j++) s->part[0][j] += s->part[t][j];
  }
  const double *v = s->part[0];
  for (int l = 0; l < J.nlev; l++) {
    const double *x = v + 1 + 6 * l;
    int ok = J.lev[l].leaves ? (x[0] > 0 && x[1] > 0)
                             : (x[0] > 0 && x[1] > 0) & (x[2] > 0 && x[3] > 0) & (x[4] > 0 && x[5] > 0);
    s->pend[hi - 1 - l].persist = ok;
  }
  if (J.nx) {
    s->spec = J.nx;
    s->spec_from = n;
    s->spec_eps = eps;
  }
  return -n->lp + 0.5 * v[0];
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
    eval(s, n);
    st_release(s, s->edge);
    s->edge = n;
    s->n_leapfrog++;
    st_set(s, prop, n);
    st_set(s, beg, n);
    st_set(s, end, n);
    *rho_out = n->p;
    double h = leaf_finish(s, n, eps);
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

typedef struct {
  // inputs
  mint_logp_fn f;
  mint_constrain_fn constrain;
  int64_t D, draws, warmup;
  uint64_t seed;
  int chain;
  int threads_per_chain;
  const cpu_set_t *l3;  // CPUs for this chain's threads, or NULL
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
  int64_t saved_knt = kernel_nt;
  kernel_nt = kernel_threads(s->nt);
  int bound = job->l3 && s->nt > 1 && bind_team(s->nt, job->l3);
  s->f = job->f;
  s->inv_m = mint_alloc(D);
  for (int64_t i = 0; i < D; i++) s->inv_m[i] = 1.0;
  rng_seed(&s->rng, job->seed * 0x9E3779B97F4A7C15ull + (uint64_t)job->chain + 1);
  for (int d = 0; d <= MAX_DEPTH; d++) {
    Level *L = &s->lv[d];
    L->rho_init = mint_alloc(D);
  }
  Traj t;
  double **tv[] = {&t.rho, &t.rho_next};
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
  if (bound) restore_team(s->nt);
  job->mean_leapfrog = job->draws ? (double)total_leapfrog / (double)job->draws : 0;

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
    printf("exact log density: %.17g\n", lp);
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
  kernel_nt = 1;

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
  cpu_set_t l3[MAX_L3];
  int n_l3 = 0;
  const char *aenv = getenv("MINT_CHAIN_AFFINITY");
  if (tpc > 1 && !(aenv && strcmp(aenv, "0") == 0)) n_l3 = l3_groups(l3, MAX_L3);
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
                         .l3 = n_l3 > 1 ? &l3[c % n_l3] : NULL,
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
