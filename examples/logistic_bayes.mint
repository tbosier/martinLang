// Bayesian logistic regression, sampled with NUTS.
// The gradient is derived by the compiler; nothing here is hand-written calculus.

model Logistic {
    data X: Matrix[n, p]
    data y: Vector[n]

    param alpha: Real
    param beta: Vector[p]

    alpha ~ Normal(0, 2.5)
    beta  ~ Normal(0, 1)
    y     ~ BernoulliLogit(alpha + X * beta)
}

fn main() {
    let X: Matrix[n, p] = read("data/logit_X.f64")
    let y: Vector[n]    = read("data/logit_y.f64")
    let post = sample(Logistic(X, y), draws = 1000, warmup = 1000, chains = 1, seed = 7)
    print(post)
}
