// A product of data only (no gradient pass) next to a parameter product.
model D {
    data X: Matrix[n, p]
    data y: Vector[n]
    data w0: Vector[p]
    param a: Real
    param b: Vector[p]
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    y ~ BernoulliLogit(a * (X * w0) + X * b)
}

fn main() {
    let X: Matrix[n, p] = read("build/fis_X.f64")
    let y: Vector[n] = read("build/fis_y.f64")
    let w0: Vector[p] = read("build/fis_w0.f64")
    let post = sample(D(X, y, w0), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
