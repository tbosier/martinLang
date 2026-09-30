// PoissonLog with a row-indexed parameter and a scalar inside the running sum
// and a column-indexed parameter outside it.
model Mixed {
    data y: Matrix[G, T]
    param s: Positive
    param beta: Vector[G]
    param drift: Vector[G]
    param season: Vector[T]
    param innov: Matrix[G, T]
    s      ~ Exponential(10)
    beta   ~ Normal(0, 1)
    drift  ~ Normal(0, 0.05)
    season ~ Normal(0, 0.2)
    innov  ~ Normal(0, 0.05)
    y ~ PoissonLog(beta + (season + cumsum(drift + s * innov, T)))
}

fn main() {
    let y: Matrix[G, T] = read("build/scan_count_NG.f64")
    let post = sample(Mixed(y), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
