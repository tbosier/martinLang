// The annotation promises SPD, but the compiler can only prove PSD.
fn hessian(X: Matrix[n, p]) -> Matrix[p, p] {
    let H: SPD[p] = X' * X
    H
}

fn main() {
    let X: Matrix[n, p] = read("data/logit_X.f64")
    print(hessian(X))
}
