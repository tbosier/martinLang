// Two scan statements over the same shape sharing a matrix parameter, and a
// second matrix parameter of that shape.
model TwoHosts {
    data y: Matrix[G, T]
    data c: Matrix[G, T]
    param u: Matrix[G, T]
    param v: Matrix[G, T]
    u ~ Normal(0, 0.2)
    v ~ Normal(0, 0.2)
    c ~ PoissonLog(cumsum(u, T))
    y ~ Normal(cumsum(u + v, T), 1.5)
}

fn main() {
    let y: Matrix[G, T] = read("build/scan_normal_NG.f64")
    let c: Matrix[G, T] = read("build/scan_count_NG.f64")
    let post = sample(TwoHosts(y, c), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
