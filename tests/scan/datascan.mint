// A running sum of data only (no adjoint), next to a parameter.
model DataScan {
    data y: Matrix[G, T]
    param a: Real
    param z: Matrix[G, T]
    a ~ Normal(0, 1)
    z ~ Normal(0, 1)
    y ~ Normal(a + 0.1 * cumsum(y, T) + z, 1)
}

fn main() {
    let y: Matrix[G, T] = read("build/scan_normal_NG.f64")
    let post = sample(DataScan(y), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
