// A random-walk scale per series, so the filter's variances differ between
// series (the general kernel, not the shared one), multiplying the
// innovations (non-centred), a prior mean for the innovations, a data
// covariate inside and outside the running sum, and a coefficient other
// than 1 on the running sum.
model Series {
    data y: Matrix[G, T]
    data x: Vector[T]
    param beta: Vector[G]
    param gamma: Real
    param mu_w: Real
    param sigma_w: Positive[G]
    param sigma_y: Positive
    param innov: Matrix[G, T]
    beta    ~ Normal(0, 1)
    gamma   ~ Normal(0, 1)
    mu_w    ~ Normal(0, 0.1)
    sigma_w ~ Normal(0, 0.5)
    sigma_y ~ Normal(0, 1)
    innov   ~ Normal(mu_w, 1)
    y ~ Normal(gamma * x - 0.5 * cumsum(0.1 * x + sigma_w .* innov, T) + beta, sigma_y)
}

fn main() {
    let y: Matrix[G, T] = read("DATA/y.f64")
    let x: Vector[T] = read("DATA/x.f64")
    let post = sample(Series(y, x), draws = DRAWS, warmup = WARMUP, chains = 4, seed = SEED)
    print(post)
}
