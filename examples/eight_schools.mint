// The eight schools model (Rubin 1981), non-centred.
// A standard correctness check: the posterior is well known.

model EightSchools {
    data y: Vector[J]
    data s: Positive[J]

    param mu: Real
    param tau: Positive
    param eta: Vector[J]

    mu  ~ Normal(0, 5)
    tau ~ Normal(0, 5)
    eta ~ Normal(0, 1)
    y   ~ Normal(mu + tau * eta, s)
}

fn main() {
    let y = [28, 8, -3, 7, -1, 1, 18, 12]
    let s = [15, 10, 16, 11, 9, 11, 10, 18]
    let post = sample(EightSchools(y, s), draws = 4000, warmup = 1000, chains = 4, seed = 3)
    print(post)
}
