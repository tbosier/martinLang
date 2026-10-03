// Eight schools, non-centred, examples/eight_schools.mint written directly.
// tau is sampled as log(tau) with its Jacobian, as Martin does for a Positive
// parameter. Unconstrained order: mu, log(tau), eta[1:J] (Martin's order).
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] s;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] eta;
}
model {
  mu ~ normal(0, 5);
  tau ~ normal(0, 5);
  eta ~ normal(0, 1);
  y ~ normal(mu + tau * eta, s);
}
