// A shape error: X is n-by-p, so X * v needs v of length p.
fn predict(X: Matrix[n, p], y: Vector[n]) -> Vector[n] {
    X * y
}

fn main() {
    let X: Matrix[n, p] = read("data/logit_X.f64")
    let y: Vector[n]    = read("data/logit_y.f64")
    print(predict(X, y))
}
