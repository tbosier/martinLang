// Bayesian linear regression with an unknown noise scale, sampled with NUTS.
//
// sigma is declared Positive, so Mint samples log(sigma), adds the Jacobian,
// and the Normal(0, 5) prior becomes a half-normal automatically.
//
// Because y and X are data, the scale is one scalar, and the mean is linear
// in the parameters, the compiler rewrites the likelihood to use X'X, X'y and
// y'y, computed once before sampling (disable with --no-suffstats).

model Linear {
    data X: Matrix[n, p]
    data y: Vector[n]

    param alpha: Real
    param beta: Vector[p]
    param sigma: Positive

    alpha ~ Normal(0, 10)
    beta  ~ Normal(0, 10)
    sigma ~ Normal(0, 5)
    y     ~ Normal(alpha + X * beta, sigma)
}

fn main() {
    let X: Matrix[n, p] = read("data/linear_X.f64")
    let y: Vector[n]    = read("data/linear_y.f64")
    let post = sample(Linear(X, y), draws = 1000, warmup = 1000, chains = 1, seed = 7)
    print(post)
}
