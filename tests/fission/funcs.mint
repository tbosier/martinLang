// Functions of the products inside the linear predictor (abs, sigmoid,
// powers, log of a Positive vector parameter), a data vector times a scalar
// parameter, and a vector parameter indexed by observation.
model F {
    data X: Matrix[n, p]
    data y: Vector[n]
    data off: Vector[n]
    param a: Real
    param b: Vector[p]
    param v: Vector[n]
    param u: Positive[n]
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    v ~ Normal(0, 1)
    u ~ Normal(0, 1)
    y ~ BernoulliLogit(abs(X * b) - sigmoid(X * b) + (X * b)^2 + (X * b)^3 + off .* a + v + log(u))
}

fn main() {
    let X: Matrix[n, p] = read("build/fis_X.f64")
    let y: Vector[n] = read("build/fis_y.f64")
    let off: Vector[n] = read("build/fis_off.f64")
    let post = sample(F(X, y, off), draws = 4, warmup = 0, chains = 1, seed = 1)
    print(post)
}
