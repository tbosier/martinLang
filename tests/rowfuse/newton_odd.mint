// Newton on a row count that is not a multiple of the 32-row chunk.
fn fit(X: Matrix[n, p], y: Vector[n], lambda: Positive) -> Vector[p] {
    let mut w = zeros(p)
    repeat 8 {
        let mu = sigmoid(X * w)
        let g  = X' * (mu - y) + lambda * w
        let H  = X' * diag(mu .* (1 - mu)) * X + lambda * I(p)
        w = w - solve(H, g)
    }
    w
}

fn main() {
    let X: Matrix[n, p] = read("build/rowfuse_X.f64")
    let y: Vector[n]    = read("build/rowfuse_y.f64")
    print("w", fit(X, y, 0.5))
}
