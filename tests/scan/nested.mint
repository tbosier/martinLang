// Two nested running sums with a scalar-scaled shared term (Normal outcome).
model Nested {
    data y: Matrix[G, T]
    param a: Real
    param beta: Vector[G]
    param shared: Vector[T]
    param innov: Matrix[G, T]
    a      ~ Normal(0, 1)
    beta   ~ Normal(0, 1)
    shared ~ Normal(0, 0.1)
    innov  ~ Normal(0, 0.1)
    let level = cumsum(a * shared + innov, T)
    let trend = cumsum(level, T)
    y ~ Normal(beta + 0.1 * trend, 1.5)
}

fn main() {
    let y: Matrix[G, T] = read("build/scan_normal_NG.f64")
    let post = sample(Nested(y), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
