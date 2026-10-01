// Non-centred random walk: the scale multiplies the innovation inside the
// running sum. Same marginal density as basic.mint.
model NonCentred {
    data y: Matrix[G, T]
    param pop: Real
    param beta: Vector[G]
    param sigma_w: Positive
    param sigma_y: Positive
    param innov: Matrix[G, T]
    pop     ~ Normal(0, 1)
    beta    ~ Normal(pop, 0.4)
    sigma_w ~ Normal(0, 0.5)
    sigma_y ~ Normal(0, 1)
    innov   ~ Normal(0, 1)
    y ~ Normal(beta + cumsum(sigma_w * innov, T), sigma_y)
}

fn main() {
    let y: Matrix[G, T] = read("DATA/y.f64")
    let post = sample(NonCentred(y), draws = DRAWS, warmup = WARMUP, chains = 4, seed = SEED)
    print(post)
}
