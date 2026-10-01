// One series: the random walk is a Vector[T] (the local-level model).
model Single {
    data y: Vector[T]
    param level0: Real
    param sigma_w: Positive
    param sigma_y: Positive
    param innov: Vector[T]
    level0  ~ Normal(0, 1)
    sigma_w ~ Normal(0, 0.5)
    sigma_y ~ Normal(0, 1)
    innov   ~ Normal(0, sigma_w)
    y ~ Normal(level0 + cumsum(innov), sigma_y)
}

fn main() {
    let y: Vector[T] = read("DATA/yv.f64")
    let post = sample(Single(y), draws = DRAWS, warmup = WARMUP, chains = 4, seed = SEED)
    print(post)
}
