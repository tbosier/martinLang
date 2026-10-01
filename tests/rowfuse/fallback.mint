// A fused row loop whose producer uses log1p (no vector form, so the per-row
// values run one row at a time), a Gram weight other than Newton's, and a
// result that combines both consumers.
fn f(X: Matrix[n, p], y: Vector[n]) -> Vector[p] {
    let w = 0.05 * ones(p)
    let mut out = zeros(p)
    repeat 1 {
        let s = log1p(exp(X * w))
        let g = X' * (s - y)
        let H = X' * diag(s .* s) * X
        out = g + H * w
    }
    out
}

fn main() {
    let X: Matrix[n, p] = read("build/rowfuse_X.f64")
    let y: Vector[n]    = read("build/rowfuse_y.f64")
    print("g", f(X, y))
}
