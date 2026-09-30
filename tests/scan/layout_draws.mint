// Draws of a matrix parameter stored column-major inside the compiled model
// must come back in the user's (row-major) order: innov is pinned to the data
// by a tight prior, so each posterior mean must match its own entry of y.
model LayoutDraws {
    data y: Matrix[G, T]
    param innov: Matrix[G, T]
    innov ~ Normal(y, 0.01)
    y ~ Normal(cumsum(innov, T), 1000)
}

fn main() {
    let y: Matrix[G, T] = read("build/scan_normal_NG.f64")
    let post = sample(LayoutDraws(y), draws = 200, warmup = 200, chains = 1, seed = 3)
    print(post)
}
