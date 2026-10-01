// Exponential with a rate from a matrix-vector product.
model E {
    data X: Matrix[n, p]
    data w: Positive[n]
    param b: Vector[p]
    b ~ Normal(0, 1)
    w ~ Exponential(exp(X * b))
}

fn main() {
    let X: Matrix[n, p] = read("build/fis_X.f64")
    let w: Positive[n] = read("build/fis_w.f64")
    let post = sample(E(X, w), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
