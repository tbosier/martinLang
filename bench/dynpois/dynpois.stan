// Hierarchical dynamic Poisson panel, natural (centred) form of SPEC.md.
// Unconstrained order: pop, beta[1:G], shared[1:T], innov[1][1:T], ..., innov[G][1:T]
// which is SPEC's theta layout (innov row-major g*T + t).
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
  for (g in 1:G) {
    innov[g] ~ normal(0, 0.08);
    // state[g, t] = sum_{s <= t} (shared[s] + innov[g, s])
    y[g] ~ poisson_log(beta[g] + cumulative_sum(shared + innov[g]));
  }
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
