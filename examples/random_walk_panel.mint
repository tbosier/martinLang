// A panel of G series observed at T times, each a level around a pooled
// intercept plus its own Gaussian random walk, observed with Gaussian noise.
// Both scales are unknown.
//
// mintc integrates the G x T innovations out with a Kalman filter per series
// (see docs/architecture.md, "Kalman collapse"): NUTS samples pop, beta and
// the two scales, G + 3 parameters, and the innovations are drawn afterwards
// for each kept draw. `mintc build --no-collapse` samples all of them.

model RandomWalkPanel {
    data y: Matrix[G, T]

    param pop: Real
    param beta: Vector[G]
    param sigma_w: Positive
    param sigma_y: Positive
    param innov: Matrix[G, T]

    pop     ~ Normal(0, 1)
    beta    ~ Normal(pop, 0.4)
    sigma_w ~ Normal(0, 0.5)
    sigma_y ~ Normal(0, 1)
    innov   ~ Normal(0, sigma_w)

    y ~ Normal(beta + cumsum(innov, T), sigma_y)
}

fn main() {
    let y: Matrix[G, T] = read("bench/kalman/data_small/y.f64")
    let post = sample(RandomWalkPanel(y), draws = 1000, warmup = 1000, chains = 4, seed = 5)
    print(post)
}
