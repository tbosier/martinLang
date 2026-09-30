// L2-regularised logistic regression, fitted by Newton's method.
//
// The type checker proves that the Hessian H is symmetric positive definite:
//   sigmoid(X * w)       every entry is in (0, 1), so mu is a Prob vector
//   mu .* (1 - mu)       a product of two Prob vectors is Prob, so the weights are positive
//   X' * diag(..) * X    is PSD for any non-negative weights
//   ... + lambda * I(p)  is SPD because lambda is declared Positive
// so solve(H, g) compiles to a Cholesky solve. diag(..) is never built as an
// n-by-n matrix, and only one triangle of the Gram product is computed.

fn fit(X: Matrix[n, p], y: Vector[n], lambda: Positive) -> Vector[p] {
    let mut w = zeros(p)
    repeat 10 {
        let mu = sigmoid(X * w)
        let g  = X' * (mu - y) + lambda * w
        let H  = X' * diag(mu .* (1 - mu)) * X + lambda * I(p)
        w = w - solve(H, g)
    }
    w
}

fn main() {
    let X: Matrix[n, p] = read("data/newton_X.f64")
    let y: Vector[n]    = read("data/newton_y.f64")
    let t = clock()
    let w = fit(X, y, 1.0)
    print("fit_seconds", clock() - t)
    print("w", w)
}
