// BernoulliLogit with a matrix-vector product; two rows reach |eta| = 1000.
model L {
    data X: Matrix[n, p]
    data y: Vector[n]
    param alpha: Real
    param beta: Vector[p]
    alpha ~ Normal(0, 2.5)
    beta  ~ Normal(0, 1)
    y ~ BernoulliLogit(alpha + X * beta)
}

fn main() {
    let X: Matrix[n, p] = read("build/fis_Xbig.f64")
    let y: Vector[n] = read("build/fis_y.f64")
    let post = sample(L(X, y), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
