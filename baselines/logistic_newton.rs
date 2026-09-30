//! Straightforward Rust: L2-regularised logistic regression by Newton's method.
use std::time::Instant;

#[path = "common.rs"]
mod common;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn cholesky_solve(h: &[f64], p: usize, g: &[f64]) -> Vec<f64> {
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        for i in j..p {
            let mut s = h[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
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
        let mut s = g[i];
        for k in 0..i {
            s -= l[i * p + k] * x[k];
        }
        x[i] = s / l[i * p + i];
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
    for _ in 0..10 {
        let mu: Vec<f64> = (0..n)
            .map(|i| sigmoid(x[i * p..(i + 1) * p].iter().zip(&w).map(|(a, b)| a * b).sum()))
            .collect();
        let mut g: Vec<f64> = w.iter().map(|wj| lambda * wj).collect();
        let mut h = vec![0.0; p * p];
        for i in 0..n {
            let row = &x[i * p..(i + 1) * p];
            let r = mu[i] - y[i];
            let s = mu[i] * (1.0 - mu[i]);
            for j in 0..p {
                g[j] += row[j] * r;
                for k in 0..p {
                    h[j * p + k] += s * row[j] * row[k];
                }
            }
        }
        for j in 0..p {
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
