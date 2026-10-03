// Hierarchical dynamic Poisson panel (bench/dynpois/SPEC.md).
// The model block is bench/dynpois/dynpois.stan's, unchanged; its generated
// quantities are left out because nothing here runs them (BridgeStan's
// log_density_gradient never does, and draws are taken with include_gq =
// false).
// Unconstrained order: pop, beta[1:G], shared[1:T], innov[1][1:T], ...,
// innov[G][1:T], which is Martin's user order (innov row-major g*T + t).
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
