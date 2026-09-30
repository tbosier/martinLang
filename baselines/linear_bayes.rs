//! Straightforward Rust: Bayesian linear regression with unknown noise scale.
//! sigma is sampled on the log scale by hand, including the Jacobian term.
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

// theta = [alpha, beta_1..beta_p, log_sigma]
extern "C" fn logp(theta: *const f64, grad: *mut f64) -> f64 {
    let d = DATA.get().unwrap();
    let (n, p) = (d.n, d.p);
    let theta = unsafe { std::slice::from_raw_parts(theta, p + 2) };
    let grad = unsafe { std::slice::from_raw_parts_mut(grad, p + 2) };
    let alpha = theta[0];
    let beta = &theta[1..p + 1];
    let log_sigma = theta[p + 1];
    let sigma = log_sigma.exp();

    // priors: alpha, beta ~ Normal(0, 10); sigma ~ half-Normal(0, 5)
    let mut lp = -0.5 * (alpha / 10.0).powi(2) - 10f64.ln();
    grad[0] = -alpha / 100.0;
    for j in 0..p {
        lp += -0.5 * (beta[j] / 10.0).powi(2) - 10f64.ln();
        grad[1 + j] = -beta[j] / 100.0;
    }
    lp += -0.5 * (sigma / 5.0).powi(2) - 5f64.ln();
    let mut dsigma = -sigma / 25.0;

    // likelihood: y ~ Normal(alpha + X beta, sigma)
    let resid: Vec<f64> = (0..n)
        .map(|i| d.y[i] - alpha - d.x[i * p..(i + 1) * p].iter().zip(beta).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    let rss: f64 = resid.iter().map(|r| r * r).sum();
    let inv_s2 = 1.0 / (sigma * sigma);
    lp += -0.5 * rss * inv_s2 - n as f64 * sigma.ln();
    grad[0] += resid.iter().sum::<f64>() * inv_s2;
    for i in 0..n {
        let w = resid[i] * inv_s2;
        for j in 0..p {
            grad[1 + j] += d.x[i * p + j] * w;
        }
    }
    dsigma += rss / (sigma * sigma * sigma) - n as f64 / sigma;

    // Jacobian of sigma = exp(log_sigma)
    lp += log_sigma;
    grad[p + 1] = dsigma * sigma + 1.0;
    lp
}

extern "C" fn constrain(unc: *const f64, out: *mut f64) {
    let p = DATA.get().unwrap().p;
    unsafe {
        std::ptr::copy_nonoverlapping(unc, out, p + 2);
        *out.add(p + 1) = (*unc.add(p + 1)).exp();
    }
}

fn main() {
    let (n, p, x) = common::read_f64("data/linear_X.f64");
    let (ny, _, y) = common::read_f64("data/linear_y.f64");
    assert_eq!(n, ny);
    DATA.set(Data { n, p, x, y }).ok();
    common::sample_and_print(logp, constrain, p + 2, &["alpha", "beta", "sigma"], &[-1, p as i64, -1], 1000, 1000, 1, 7);
}
