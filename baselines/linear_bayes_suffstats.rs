//! Hand-optimised Rust: linear regression using sufficient statistics
//! Z'Z, Z'y and y'y (Z = [1, X]) computed once, so each gradient is O(p^2).
//! This is the rewrite Martin's compiler performs automatically. The gradient
//! does not allocate and its dot products use four accumulators.
use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

#[path = "common.rs"]
mod common;

struct Stats {
    n: usize,
    q: usize,
    g: Vec<f64>, // Z'Z, q x q
    b: Vec<f64>, // Z'y
    c: f64,      // y'y
}
static STATS: OnceLock<Stats> = OnceLock::new();
thread_local! {
    static GT: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

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

// theta = [alpha, beta_1..beta_p, log_sigma]; theta[..q] is the coefficient vector
extern "C" fn logp(theta: *const f64, grad: *mut f64) -> f64 {
    let s = STATS.get().unwrap();
    let q = s.q;
    let theta = unsafe { std::slice::from_raw_parts(theta, q + 1) };
    let grad = unsafe { std::slice::from_raw_parts_mut(grad, q + 1) };
    let coef = &theta[..q];
    let log_sigma = theta[q];
    let sigma = log_sigma.exp();
    let mut lp = 0.0;
    for j in 0..q {
        lp += -0.5 * (coef[j] / 10.0).powi(2) - 10f64.ln();
        grad[j] = -coef[j] / 100.0;
    }
    lp += -0.5 * (sigma / 5.0).powi(2) - 5f64.ln();
    let mut dsigma = -sigma / 25.0;
    let inv_s2 = 1.0 / (sigma * sigma);
    let rss = GT.with(|gt| {
        let mut gt = gt.borrow_mut();
        gt.resize(q, 0.0);
        for j in 0..q {
            gt[j] = dot4(&s.g[j * q..(j + 1) * q], coef);
        }
        let rss = s.c - 2.0 * dot4(coef, &s.b) + dot4(coef, &gt);
        for j in 0..q {
            grad[j] += (s.b[j] - gt[j]) * inv_s2;
        }
        rss
    });
    lp += -0.5 * rss * inv_s2 - s.n as f64 * sigma.ln();
    dsigma += rss * inv_s2 / sigma - s.n as f64 / sigma;
    lp += log_sigma;
    grad[q] = dsigma * sigma + 1.0;
    lp
}

extern "C" fn constrain(unc: *const f64, out: *mut f64) {
    let q = STATS.get().unwrap().q;
    unsafe {
        std::ptr::copy_nonoverlapping(unc, out, q + 1);
        *out.add(q) = (*unc.add(q)).exp();
    }
}

fn main() {
    let (n, p, x) = common::read_f64("data/linear_X.f64");
    let (ny, _, y) = common::read_f64("data/linear_y.f64");
    assert_eq!(n, ny);
    let t = Instant::now();
    let q = p + 1;
    let mut g = vec![0.0; q * q];
    let mut b = vec![0.0; q];
    let mut c = 0.0;
    let mut z = vec![0.0; q];
    for i in 0..n {
        z[0] = 1.0;
        z[1..].copy_from_slice(&x[i * p..(i + 1) * p]);
        for j in 0..q {
            b[j] += z[j] * y[i];
            for k in 0..q {
                g[j * q + k] += z[j] * z[k];
            }
        }
        c += y[i] * y[i];
    }
    STATS.set(Stats { n, q, g, b, c }).ok();
    common::set_prep_seconds(t.elapsed().as_secs_f64());
    common::sample_and_print(logp, constrain, p + 2, &["alpha", "beta", "sigma"], &[-1, p as i64, -1], 1000, 1000, 1, 7);
}
