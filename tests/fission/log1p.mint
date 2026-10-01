// log1p has no vector form: the statement keeps the three-pass fission.
model G {
    data X: Matrix[n, p]
    data y: Vector[n]
    param b: Vector[p]
    b ~ Normal(0, 1)
    y ~ BernoulliLogit(log1p(exp(X * b)) - 1)
}

fn main() {
    let X: Matrix[n, p] = read("build/fis_X.f64")
    let y: Vector[n] = read("build/fis_y.f64")
    let post = sample(G(X, y), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
