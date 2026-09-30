// A column-indexed parameter used in the fused scan and again, before and
// after it, in statements over another shape.
model ColReuse {
    data y: Matrix[G, T]
    data q: Matrix[H, T]
    param p: Vector[T]
    param z: Matrix[G, T]
    q ~ Normal(p, 1.3)
    p ~ Normal(0, 1)
    z ~ Normal(0, 1)
    y ~ Normal(cumsum(z + p, T), 2)
    q ~ Normal(p, 1)
}

fn main() {
    let y: Matrix[G, T] = read("build/scan_normal_NG.f64")
    let q: Matrix[H, T] = read("build/scan_normal_7.f64")
    let post = sample(ColReuse(y, q), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
