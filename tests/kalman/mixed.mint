// A collapsed random walk next to one that is not (Poisson counts): the
// other's running sum puts [G, T] quantities in the scan layout, so the
// filter reads the observations through it, and NUTS samples a matrix.
model Mixed {
    data y: Matrix[G, T]
    data n: Matrix[G, T]
    param beta: Vector[G]
    param sigma_y: Positive
    param innov: Matrix[G, T]
    param other: Matrix[G, T]
    beta    ~ Normal(0, 1)
    sigma_y ~ Normal(0, 1)
    innov   ~ Normal(0, 0.3)
    other   ~ Normal(0, 0.1)
    n ~ PoissonLog(beta + cumsum(other, T))
    y ~ Normal(beta + cumsum(innov, T), sigma_y)
}

fn main() {
    let y: Matrix[G, T] = read("DATA/y.f64")
    let n: Matrix[G, T] = read("DATA/n.f64")
    let post = sample(Mixed(y, n), draws = DRAWS, warmup = WARMUP, chains = 4, seed = SEED)
    print(post)
}
