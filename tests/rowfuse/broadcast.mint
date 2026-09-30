// A transposed product plus a term of another shape: the result broadcasts,
// so row fusion must leave it alone.
fn main() {
    let X = ones(3, 5)
    let w = ones(5)
    repeat 1 {
        let mu = X * w
        let g = X' * mu + ones(5, 7)
        let h = X' * mu + 1
        print(g)
        print(h)
    }
}
