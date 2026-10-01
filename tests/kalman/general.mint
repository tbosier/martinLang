// Every input of the filter depends on parameters, by series, by time or
// both: a time-indexed data covariate in the mean and inside the running
// sum, a prior mean for the innovations, per-series scales (so the filter's
// variances differ between series), a series-indexed coefficient on the
// innovations, a scalar one (tau, so q = (tau sigma_w)^2) and a coefficient
// other than 1 on the running sum. tau is weakly identified (only the
// product tau sigma_w is), which only the exact checks use.
model General {
    data y: Matrix[G, T]
    data x: Vector[T]
    param beta: Vector[G]
    param gamma: Real
    param mu_w: Real
    param sigma_w: Positive[G]
    param sigma_y: Positive[G]
    param tau: Positive
    param innov: Matrix[G, T]
    beta    ~ Normal(0, 1)
    gamma   ~ Normal(0, 1)
    mu_w    ~ Normal(0, 0.1)
    sigma_w ~ Normal(0, 0.5)
    sigma_y ~ Normal(0, 1)
    tau     ~ Normal(0, 1)
    innov   ~ Normal(mu_w, sigma_w)
    y ~ Normal(gamma * x - 0.5 * cumsum(0.1 * x + 2 * tau * innov, T) + beta, sigma_y)
}

fn main() {
    let y: Matrix[G, T] = read("DATA/y.f64")
    let x: Vector[T] = read("DATA/x.f64")
    let post = sample(General(y, x), draws = DRAWS, warmup = WARMUP, chains = 4, seed = SEED)
    print(post)
}
