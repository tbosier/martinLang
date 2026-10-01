// Normal with an indexed scale: two matrix-vector products of one matrix,
// and log of the scale inside the density.
model N {
    data X: Matrix[n, p]
    data z: Vector[n]
    param b: Vector[p]
    param s: Vector[p]
    b ~ Normal(0, 1)
    s ~ Normal(0, 0.3)
    z ~ Normal(X * b, exp(X * s))
}

fn main() {
    let X: Matrix[n, p] = read("build/fis_X.f64")
    let z: Vector[n] = read("build/fis_z.f64")
    let post = sample(N(X, z), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
