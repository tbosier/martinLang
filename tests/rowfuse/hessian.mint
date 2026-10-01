// One Newton step's gradient and Hessian, printed: the fused, tiled Gram
// kernel must give the same values as the unfused, untiled build. (Newton's
// coefficients alone do not show a wrong Hessian: the iteration still
// converges to the same w.) Row fusion applies inside repeat bodies.
fn grad(X: Matrix[n, p], y: Vector[n]) -> Vector[p] {
    let w = 0.05 * ones(p)
    let mut out = zeros(p)
    repeat 1 {
        let mu = sigmoid(X * w)
        let g  = X' * (mu - y) + 0.5 * w
        let H  = X' * diag(mu .* (1 - mu)) * X + 0.5 * I(p)
        out = g
    }
    out
}

fn hess(X: Matrix[n, p], y: Vector[n]) -> Matrix[p, p] {
    let w = 0.05 * ones(p)
    let mut out = I(p)
    repeat 1 {
        let mu = sigmoid(X * w)
        let g  = X' * (mu - y)
        let H  = X' * diag(mu .* (1 - mu)) * X + 0.5 * I(p)
        out = H
    }
    out
}

fn main() {
    let X: Matrix[n, p] = read("build/rowfuse_X.f64")
    let y: Vector[n]    = read("build/rowfuse_y.f64")
    print("g", grad(X, y))
    print("H", hess(X, y))
}
