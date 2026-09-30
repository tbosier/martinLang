//! Straightforward Rust: Bayesian logistic regression. The log density and its
//! gradient are written by hand; the sampler is the Mint runtime's NUTS.
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

fn softplus(z: f64) -> f64 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

// theta = [alpha, beta_1..beta_p]
extern "C" fn logp(theta: *const f64, grad: *mut f64) -> f64 {
    let d = DATA.get().unwrap();
    let (n, p) = (d.n, d.p);
    let theta = unsafe { std::slice::from_raw_parts(theta, p + 1) };
    let grad = unsafe { std::slice::from_raw_parts_mut(grad, p + 1) };
    let alpha = theta[0];
    let beta = &theta[1..];

    // priors: alpha ~ Normal(0, 2.5), beta ~ Normal(0, 1)
    let mut lp = -0.5 * (alpha / 2.5).powi(2) - 2.5f64.ln();
    lp += beta.iter().map(|b| -0.5 * b * b).sum::<f64>();
    grad[0] = -alpha / (2.5 * 2.5);
    for j in 0..p {
        grad[1 + j] = -beta[j];
    }

    // likelihood: y ~ BernoulliLogit(alpha + X beta)
    let eta: Vec<f64> = (0..n)
        .map(|i| alpha + d.x[i * p..(i + 1) * p].iter().zip(beta).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    let mut resid = vec![0.0; n];
    for i in 0..n {
        lp += d.y[i] * eta[i] - softplus(eta[i]);
        resid[i] = d.y[i] - sigmoid(eta[i]);
    }
    grad[0] += resid.iter().sum::<f64>();
    for i in 0..n {
        for j in 0..p {
            grad[1 + j] += d.x[i * p + j] * resid[i];
        }
    }
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
