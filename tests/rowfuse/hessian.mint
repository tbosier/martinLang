// One Newton step's Hessian, printed: the fused, tiled Gram kernel must give
// the same matrix as the unfused, untiled build. (Newton's coefficients alone
// do not show a wrong Hessian: the iteration still converges to the same w.)
fn hess(X: Matrix[n, p], y: Vector[n]) -> Matrix[p, p] {
    let w = 0.05 * ones(p)
    let mu = sigmoid(X * w)
    let g  = X' * (mu - y)
    let H  = X' * diag(mu .* (1 - mu)) * X + 0.5 * I(p)
    H
}

fn main() {
    let X: Matrix[n, p] = read("build/rowfuse_X.f64")
    let y: Vector[n]    = read("build/rowfuse_y.f64")
    print("H", hess(X, y))
}
