// Bayesian logistic regression, examples/logistic_bayes.mint written
// directly. Unconstrained order: alpha, beta[1:p] (Martin's order).
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n, p] X;
  array[n] int<lower=0, upper=1> y;
}
parameters {
  real alpha;
  vector[p] beta;
}
model {
  alpha ~ normal(0, 2.5);
  beta ~ normal(0, 1);
  y ~ bernoulli_logit(alpha + X * beta);
}
