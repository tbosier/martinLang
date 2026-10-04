// bench/dynpois/dynpois.stan with the per-series likelihood (and the series'
// innovation prior) summed by reduce_sum, so one chain can use several
// threads (compile with STAN_THREADS, run with threads_per_chain). The
// sliced argument is innov itself, so each task copies only beta and shared.
functions {
  real partial_lpdf(array[] vector innov_slice, int start, int end,
                    array[,] int y, vector beta, vector shared) {
    real lp = 0;
    for (i in 1:size(innov_slice)) {
      int g = start + i - 1;
      lp += normal_lupdf(innov_slice[i] | 0, 0.08);
      lp += poisson_log_lupmf(y[g] | beta[g] + cumulative_sum(shared + innov_slice[i]));
    }
    return lp;
  }
}
data {
  int<lower=1> G;
  int<lower=1> T;
  array[G, T] int<lower=0> y;
}
parameters {
  real pop;
  vector[G] beta;
  vector[T] shared;
  array[G] vector[T] innov;
}
model {
  pop ~ normal(0, 1);
  beta ~ normal(pop, 0.4);
  shared ~ normal(0, 0.05);
  target += reduce_sum(partial_lupdf, innov, 1, y, beta, shared);
}
generated quantities {
  vector[G] terminal;
  {
    real shared_total = sum(shared);
    for (g in 1:G) {
      terminal[g] = shared_total + sum(innov[g]);
    }
  }
}
