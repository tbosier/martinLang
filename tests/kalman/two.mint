// Two random walks integrated out in one model, over different shapes, one
// sharing its intercepts with the other's observation.
model Two {
    data y: Matrix[G, T]
    data z: Matrix[G, S]
    param beta: Vector[G]
    param sigma_y: Positive
    param sigma_z: Positive
    param innov: Matrix[G, T]
    param walk: Matrix[G, S]
    beta    ~ Normal(0, 1)
    sigma_y ~ Normal(0, 1)
    sigma_z ~ Normal(0, 1)
    innov   ~ Normal(0, 0.3)
    walk    ~ Normal(0.1, sigma_z)
    y ~ Normal(beta + cumsum(innov, T), sigma_y)
    z ~ Normal(beta + 2 * cumsum(walk, S), 0.5)
}

fn main() {
    let y: Matrix[G, T] = read("DATA/y.f64")
    let z: Matrix[G, S] = read("DATA/z.f64")
    let post = sample(Two(y, z), draws = DRAWS, warmup = WARMUP, chains = 4, seed = SEED)
    print(post)
}
