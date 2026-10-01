// Two scan statements over the same shape, each owning its matrix
// parameters (u and w the first, v the second), with a vector parameter
// between them in theta: three covered blocks, two kernels. (The first
// kernel absorbs every prior over the shape, so v has none: a prior on v
// would make it the first kernel's guest, and neither kernel would own it.)
model TwoOwned {
    data y: Matrix[G, T]
    data c: Matrix[G, T]
    param u: Matrix[G, T]
    param b: Vector[G]
    param v: Matrix[G, T]
    param w: Matrix[G, T]
    u ~ Normal(0, 0.2)
    b ~ Normal(0, 1)
    w ~ Normal(0, 0.2)
    y ~ Normal(cumsum(u + 0.5 * w, T), 1.5)
    c ~ PoissonLog(b + cumsum(v, T))
}

fn main() {
    let y: Matrix[G, T] = read("build/scan_normal_NG.f64")
    let c: Matrix[G, T] = read("build/scan_count_NG.f64")
    let post = sample(TwoOwned(y, c), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
