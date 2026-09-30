//! Hand-tuned Rust: as logistic_newton.rs, but with the gradient and Hessian
//! fused into one pass over X, symmetry exploited (upper triangle only) and
//! four independent accumulators in the dot products so they vectorise.
use std::time::Instant;

#[path = "common.rs"]
mod common;

fn dot4(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let chunks = a.len() / 4;
    for c in 0..chunks {
        for l in 0..4 {
            acc[l] += a[4 * c + l] * b[4 * c + l];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for k in 4 * chunks..a.len() {
        s += a[k] * b[k];
    }
    s
}

fn cholesky_solve(h: &[f64], p: usize, g: &[f64]) -> Vec<f64> {
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        for i in j..p {
            let s = h[i * p + j] - dot4(&l[i * p..i * p + j], &l[j * p..j * p + j]);
            if i == j {
                assert!(s > 0.0, "Hessian is not positive definite");
                l[j * p + j] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }
    let mut x = vec![0.0; p];
    for i in 0..p {
        x[i] = (g[i] - dot4(&l[i * p..i * p + i], &x[..i])) / l[i * p + i];
    }
    for i in (0..p).rev() {
        let mut s = x[i];
        for k in i + 1..p {
            s -= l[k * p + i] * x[k];
        }
        x[i] = s / l[i * p + i];
    }
    x
}

fn fit(x: &[f64], n: usize, p: usize, y: &[f64], lambda: f64) -> Vec<f64> {
    let mut w = vec![0.0; p];
    let mut g = vec![0.0; p];
    let mut h = vec![0.0; p * p];
    for _ in 0..10 {
        for j in 0..p {
            g[j] = lambda * w[j];
        }
        h.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..n {
            let row = &x[i * p..(i + 1) * p];
            let mu = 1.0 / (1.0 + (-dot4(row, &w)).exp());
            let r = mu - y[i];
            let s = mu * (1.0 - mu);
            for j in 0..p {
                g[j] += row[j] * r;
                let t = s * row[j];
                let hrow = &mut h[j * p..(j + 1) * p];
                for k in j..p {
                    hrow[k] += t * row[k];
                }
            }
        }
        for j in 0..p {
            for k in 0..j {
                h[j * p + k] = h[k * p + j];
            }
            h[j * p + j] += lambda;
        }
        let step = cholesky_solve(&h, p, &g);
        for j in 0..p {
            w[j] -= step[j];
        }
    }
    w
}

fn main() {
    let (n, p, x) = common::read_f64("data/newton_X.f64");
    let (ny, _, y) = common::read_f64("data/newton_y.f64");
    assert_eq!(n, ny);
    let t = Instant::now();
    let w = fit(&x, n, p, &y, 1.0);
    println!("fit_seconds {}", t.elapsed().as_secs_f64());
    let s: Vec<String> = w.iter().map(|v| format!("{v:.10}")).collect();
    println!("w [{}]", s.join(", "));
}
