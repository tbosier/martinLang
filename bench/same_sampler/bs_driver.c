// Runs a Stan model, compiled to a shared library by BridgeStan, under the
// Mint runtime's NUTS sampler (runtime/mint_rt.c, mint_sample): the same
// sampler, warmup, metric adaptation and random number stream that Mint's
// compiled programs and the Rust baselines use. Only the log density and
// gradient differ: here they come from bs_log_density_gradient.
//
// usage: bs_driver MODEL_model.so DATA.json DRAWS WARMUP CHAINS SEED
//
// The log density is bs_log_density_gradient(propto = true, jacobian = true):
// Stan drops constant terms (Mint keeps each Normal's -log(scale), so the two
// differ by a constant, which the sampler never sees) and adds the log
// Jacobian of its constraining transforms, as Mint does. Draws are
// constrained with bs_param_constrain (no transformed parameters or
// generated quantities). Parameter blocks for the runtime's summary are read
// from bs_param_names ("beta.3" belongs to block "beta").
//
// Every runtime switch works unchanged: MINT_BENCH_GRAD=K times K gradients
// at the runtime's benchmark point, MINT_PRINT_GRAD=1 also prints that
// gradient, MINT_GRADCHECK=1 checks it against finite differences,
// MINT_DRAWS=FILE dumps the draws.
//
// Threads. Mint's runtime runs each chain on its own thread and calls the
// log density from it. BridgeStan documents that a model built with
// STAN_THREADS=true may be called concurrently from several threads (Stan's
// autodiff tape is then thread-local, and the model object is read-only).
// The build uses STAN_THREADS=true. By default the chains share one model
// object; BS_MODEL_PER_CHAIN=1 instead constructs one per chain and gives each
// calling thread its own (the draws must be identical either way, which
// run_checks.py verifies).
//
// A Stan exception (for example an overflowing Poisson rate) returns a log
// density of -inf with a zero gradient, which the sampler treats as a
// divergence, as Stan's own sampler does with a rejection.
#include <dlfcn.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct bs_model bs_model;
typedef struct bs_rng bs_rng;

// The runtime's entry points (runtime/mint_rt.c).
typedef double (*mint_logp_fn)(const double *theta, double *grad);
typedef void (*mint_constrain_fn)(const double *unc, double *out);
void *mint_sample(mint_logp_fn f, mint_constrain_fn constrain, int64_t D, int64_t draws, int64_t warmup,
                  int64_t chains, int64_t seed, int64_t nparams, const char **names, const int64_t *sizes);
void mint_print_posterior(void *p);
void mint_set_prep_seconds(double s);
double mint_clock(void);

// BridgeStan's C API, resolved with dlsym (src/bridgestan.h of BridgeStan 2.9).
static bs_model *(*bs_model_construct)(const char *data, unsigned int seed, char **err);
static int (*bs_param_unc_num)(const bs_model *m);
static int (*bs_param_num)(const bs_model *m, bool include_tp, bool include_gq);
static const char *(*bs_param_names)(const bs_model *m, bool include_tp, bool include_gq);
static int (*bs_log_density_gradient)(const bs_model *m, bool propto, bool jacobian, const double *theta_unc,
                                      double *val, double *grad, char **err);
static int (*bs_param_constrain)(const bs_model *m, bool include_tp, bool include_gq, const double *theta_unc,
                                 double *theta, bs_rng *rng, char **err);
static void (*bs_free_error_msg)(char *err);

#define MAX_MODELS 64
static bs_model *models[MAX_MODELS];
static int n_models;
static atomic_int next_model;
static int64_t D;
static __thread bs_model *mine;

static inline const bs_model *model_here(void) {
  if (!mine) mine = models[atomic_fetch_add(&next_model, 1) % n_models];
  return mine;
}

static double logp(const double *theta, double *grad) {
  double lp;
  if (bs_log_density_gradient(model_here(), true, true, theta, &lp, grad, NULL) != 0) {
    memset(grad, 0, (size_t)D * sizeof(double));
    return -INFINITY;
  }
  return lp;
}

static void constrain(const double *unc, double *out) {
  if (bs_param_constrain(model_here(), false, false, unc, out, NULL, NULL) != 0) {
    for (int64_t i = 0; i < D; i++) out[i] = NAN;
  }
}

static void *sym(void *h, const char *name) {
  void *p = dlsym(h, name);
  if (!p) {
    fprintf(stderr, "bs_driver: %s not found: %s\n", name, dlerror());
    exit(2);
  }
  return p;
}

int main(int argc, char **argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: %s MODEL_model.so DATA.json DRAWS WARMUP CHAINS SEED\n", argv[0]);
    return 2;
  }
  int64_t draws = atoll(argv[3]), warmup = atoll(argv[4]), chains = atoll(argv[5]), seed = atoll(argv[6]);
  double t0 = mint_clock();
  void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    fprintf(stderr, "bs_driver: cannot load %s: %s\n", argv[1], dlerror());
    return 2;
  }
  *(void **)&bs_model_construct = sym(h, "bs_model_construct");
  *(void **)&bs_param_unc_num = sym(h, "bs_param_unc_num");
  *(void **)&bs_param_num = sym(h, "bs_param_num");
  *(void **)&bs_param_names = sym(h, "bs_param_names");
  *(void **)&bs_log_density_gradient = sym(h, "bs_log_density_gradient");
  *(void **)&bs_param_constrain = sym(h, "bs_param_constrain");
  *(void **)&bs_free_error_msg = sym(h, "bs_free_error_msg");

  const char *per = getenv("BS_MODEL_PER_CHAIN");
  n_models = per && strcmp(per, "1") == 0 ? (int)(chains < MAX_MODELS ? chains : MAX_MODELS) : 1;
  for (int i = 0; i < n_models; i++) {
    char *err = NULL;
    models[i] = bs_model_construct(argv[2], 0, &err);
    if (!models[i]) {
      fprintf(stderr, "bs_driver: model construction failed: %s\n", err ? err : "(no message)");
      return 2;
    }
  }
  D = bs_param_unc_num(models[0]);
  if (bs_param_num(models[0], false, false) != D) {
    fprintf(stderr, "bs_driver: constrained and unconstrained sizes differ; not supported\n");
    return 2;
  }

  // parameter blocks from the names: "pop", "beta.1", "innov.3.17"
  char *all = strdup(bs_param_names(models[0], false, false));
  const char **names = calloc((size_t)D, sizeof *names);
  int64_t *sizes = calloc((size_t)D, sizeof *sizes);
  int64_t nb = 0;
  char *save = NULL;
  for (char *tok = strtok_r(all, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
    char *dot = strchr(tok, '.');
    if (dot) *dot = 0;
    if (nb > 0 && dot && sizes[nb - 1] >= 0 && strcmp(names[nb - 1], tok) == 0) {
      sizes[nb - 1]++;
    } else {
      names[nb] = tok;
      sizes[nb] = dot ? 1 : -1;
      nb++;
    }
  }
  fprintf(stderr, "bs_driver: D=%lld blocks=%lld models=%d (propto, jacobian)\n", (long long)D, (long long)nb,
          n_models);
  mint_set_prep_seconds(mint_clock() - t0);
  void *post = mint_sample(logp, constrain, D, draws, warmup, chains, seed, nb, names, sizes);
  mint_print_posterior(post);
  return 0;
}
