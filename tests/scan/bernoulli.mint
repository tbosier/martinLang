// BernoulliLogit over a running sum: the kernel's one-lane path.
model Bern {
    data y: Matrix[G, T]
    param beta: Vector[G]
    param innov: Matrix[G, T]
    beta  ~ Normal(0, 1)
    innov ~ Normal(0, 0.3)
    y ~ BernoulliLogit(beta + cumsum(innov, T))
}

fn main() {
    let y: Matrix[G, T] = read("build/scan_binary_NG.f64")
    let post = sample(Bern(y), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
