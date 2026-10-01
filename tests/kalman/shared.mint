// A drift shared by every series inside the running sum (as in
// examples/dynamic_poisson.mint): it makes the series dependent, so it stays
// a NUTS parameter; given it, each series is filtered on its own.
// Non-centred (the scale multiplies the innovations).
model Shared {
    data y: Matrix[G, T]
    param pop: Real
    param beta: Vector[G]
    param shared: Vector[T]
    param sigma_w: Positive
    param sigma_y: Positive
    param innov: Matrix[G, T]
    pop     ~ Normal(0, 1)
    beta    ~ Normal(pop, 0.4)
    shared  ~ Normal(0, 0.05)
    sigma_w ~ Normal(0, 0.5)
    sigma_y ~ Normal(0, 1)
    innov   ~ Normal(0, 1)
    let state = cumsum(shared + sigma_w * innov, T)
    y ~ Normal(beta + state, sigma_y)
}

fn main() {
    let y: Matrix[G, T] = read("DATA/y.f64")
    let post = sample(Shared(y), draws = DRAWS, warmup = WARMUP, chains = 4, seed = SEED)
    print(post)
}
