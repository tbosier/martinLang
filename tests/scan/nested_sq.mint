// A nonlinear nested running sum: the outer sum's derivative needs the inner
// sum's value in the reverse pass.
model NestedSq {
    data y: Matrix[G, T]
    param z: Matrix[G, T]
    z ~ Normal(0, 1)
    let level = cumsum(z, T)
    y ~ Normal(0.05 * cumsum(level .* level, T), 1.5)
}

fn main() {
    let y: Matrix[G, T] = read("build/scan_normal_NG.f64")
    let post = sample(NestedSq(y), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
