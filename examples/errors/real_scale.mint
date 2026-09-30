// A Normal scale must be positive; a Real parameter could go negative.
model Linear {
    data X: Matrix[n, p]
    data y: Vector[n]
    param beta: Vector[p]
    param sigma: Real

    beta ~ Normal(0, 10)
    y    ~ Normal(X * beta, sigma)
}

fn main() {
    let X: Matrix[n, p] = read("data/linear_X.f64")
    let y: Vector[n]    = read("data/linear_y.f64")
    print(sample(Linear(X, y)))
}
