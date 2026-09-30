// Shapes are checked once, when data enters the program: these two files
// have different numbers of rows, so this stops before any computation.
fn main() {
    let X: Matrix[n, p] = read("data/logit_X.f64")
    let y: Vector[n]    = read("data/linear_y.f64")
    print(X' * y)
}
