// A Gaussian posterior with a known mean and covariance that is much
// narrower than the prior along a few dense directions (tests/metric/make_gauss.py).
model Gauss {
    data A: Matrix[m, d]
    data b: Vector[m]

    param x: Vector[d]

    x ~ Normal(0, 1)
    b ~ Normal(A * x, 1)
}

fn main() {
    let A: Matrix[m, d] = read("build/gauss_A.f64")
    let b: Vector[m]    = read("build/gauss_b.f64")
    let post = sample(Gauss(A, b), draws = 1000, warmup = 1000, chains = 4, seed = 5)
    print(post)
}
