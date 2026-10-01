// PoissonLog with a matrix-vector product; two rows reach eta = -1000.
model P {
    data X: Matrix[n, p]
    data c: Vector[n]
    param a: Real
    param b: Vector[p]
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    c ~ PoissonLog(a + X * b)
}

fn main() {
    let X: Matrix[n, p] = read("build/fis_Xbig.f64")
    let c: Vector[n] = read("build/fis_c.f64")
    let post = sample(P(X, c), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
