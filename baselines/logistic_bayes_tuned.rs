//! Hand-tuned Rust: as logistic_bayes.rs, but one fused pass over X per
//! gradient (dot product and gradient update per row), no allocation, and a
//! four-accumulator dot product so it vectorises.
use std::sync::OnceLock;

#[path = "common.rs"]
mod common;

struct Data {
    n: usize,
    p: usize,
    x: Vec<f64>,
    y: Vec<f64>,
}
static DATA: OnceLock<Data> = OnceLock::new();

#[inline]
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

extern "C" fn logp(theta: *const f64, grad: *mut f64) -> f64 {
    let d = DATA.get().unwrap();
    let (n, p) = (d.n, d.p);
    let theta = unsafe { std::slice::from_raw_parts(theta, p + 1) };
    let grad = unsafe { std::slice::from_raw_parts_mut(grad, p + 1) };
    let alpha = theta[0];
    let beta = &theta[1..];
    let mut lp = -0.5 * (alpha / 2.5).powi(2) - 2.5f64.ln();
    grad[0] = -alpha / (2.5 * 2.5);
    for j in 0..p {
        lp -= 0.5 * beta[j] * beta[j];
        grad[1 + j] = -beta[j];
    }
    let (g0, gb) = grad.split_at_mut(1);
    let mut ga = 0.0;
    for i in 0..n {
        let row = &d.x[i * p..(i + 1) * p];
        let eta = alpha + dot4(row, beta);
        let e = (-eta.abs()).exp();
        lp += d.y[i] * eta - (eta.max(0.0) + e.ln_1p());
        let sig = if eta >= 0.0 { 1.0 / (1.0 + e) } else { e / (1.0 + e) };
        let r = d.y[i] - sig;
        ga += r;
        for j in 0..p {
            gb[j] += row[j] * r;
        }
    }
    g0[0] += ga;
    lp
}

extern "C" fn constrain(unc: *const f64, out: *mut f64) {
    let p = DATA.get().unwrap().p;
    unsafe { std::ptr::copy_nonoverlapping(unc, out, p + 1) };
}

fn main() {
    let (n, p, x) = common::read_f64("data/logit_X.f64");
    let (ny, _, y) = common::read_f64("data/logit_y.f64");
    assert_eq!(n, ny);
    DATA.set(Data { n, p, x, y }).ok();
    common::sample_and_print(logp, constrain, p + 1, &["alpha", "beta"], &[-1, p as i64], 1000, 1000, 1, 7);
}
