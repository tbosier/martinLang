// The same model as logistic.stan, with the likelihood written as Stan's
// bernoulli_logit_glm, which is what a Stan expert would use: the same
// mathematics, with an analytic gradient for the whole likelihood instead
// of an autodiff node per observation.
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
  y ~ bernoulli_logit_glm(X, alpha, beta);
}
