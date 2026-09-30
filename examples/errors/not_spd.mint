// X' * X is only PSD: it is singular when X has dependent columns.
fn least_squares(X: Matrix[n, p], y: Vector[n]) -> Vector[p] {
    solve(X' * X, X' * y)
}

fn main() {
    let X: Matrix[n, p] = read("data/logit_X.f64")
    let y: Vector[n]    = read("data/logit_y.f64")
    print(least_squares(X, y))
}
