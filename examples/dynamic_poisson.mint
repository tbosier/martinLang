// Hierarchical dynamic Poisson panel: G series observed at T times.
// Each series has its own intercept around a population mean and a latent
// random walk made of a shared innovation and its own; counts are Poisson.
// This is the model rustmc fits with BayesianDynamicPoisson (fixed scales).

model DynamicPoisson {
    data y: Matrix[G, T]

    param pop: Real
    param beta: Vector[G]
    param shared: Vector[T]
    param innov: Matrix[G, T]

    pop    ~ Normal(0, 1)
    beta   ~ Normal(pop, 0.4)
    shared ~ Normal(0, 0.05)
    innov  ~ Normal(0, 0.08)

    let state = cumsum(shared + innov, T)
    y ~ PoissonLog(beta + state)
}

fn main() {
    let y: Matrix[G, T] = read("bench/dynpois/data_small/y.f64")
    let post = sample(DynamicPoisson(y), draws = 1000, warmup = 1000, chains = 4, seed = 11)
    print(post)
}
