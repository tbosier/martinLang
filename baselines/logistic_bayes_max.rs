//! Max-effort Rust (nightly), three-pass variant: the same loop structure Martin
//! generates, written by hand with AVX2/FMA intrinsics and glibc's vector exp/log.
#![feature(simd_ffi)]
use std::sync::OnceLock;

#[path = "common.rs"]
mod common;
#[path = "simd.rs"]
mod simd;
use simd::*;

struct Data {
    n: usize,
    p: usize,
    x: Vec<f64>,
    y: Vec<f64>,
}
static DATA: OnceLock<Data> = OnceLock::new();
thread_local! {
    static WS: std::cell::RefCell<Vec<f64>> = std::cell::RefCell::new(Vec::new());
}

/// Log-likelihood terms and residuals y - sigmoid(eta) for four observations.
#[inline(always)]
unsafe fn four(eta: __m256d, y: __m256d) -> (__m256d, __m256d) {
    let one = _mm256_set1_pd(1.0);
    let zero = _mm256_setzero_pd();
    let sign = _mm256_set1_pd(-0.0);
    let neg_abs = _mm256_or_pd(eta, sign); // -|eta|
    let e = _ZGVdN4v_exp(neg_abs);
    let onep = _mm256_add_pd(one, e);
    let softplus = _mm256_add_pd(_mm256_max_pd(eta, zero), _ZGVdN4v_log(onep));
    let lp = _mm256_fmsub_pd(y, eta, softplus);
    let ge = _mm256_cmp_pd(eta, zero, _CMP_GE_OQ);
    let num = _mm256_blendv_pd(e, one, ge);
    let r = _mm256_sub_pd(y, _mm256_div_pd(num, onep));
    (lp, r)
}

extern "C" fn logp(theta: *const f64, grad: *mut f64) -> f64 {
    let d = DATA.get().unwrap();
    let (n, p) = (d.n, d.p);
    let theta = unsafe { std::slice::from_raw_parts(theta, p + 1) };
    let grad = unsafe { std::slice::from_raw_parts_mut(grad, p + 1) };
    let alpha = theta[0];
    let beta = &theta[1..];
    let mut lp = -0.5 * (alpha / 2.5) * (alpha / 2.5) - 2.5f64.ln();
    grad[0] = -alpha / (2.5 * 2.5);
    for j in 0..p {
        lp -= 0.5 * beta[j] * beta[j];
        grad[1 + j] = -beta[j];
    }
    let (g0, gb) = grad.split_at_mut(1);
    WS.with(|ws| unsafe {
        let mut ws = ws.borrow_mut();
        ws.resize(n, 0.0);
        let r = &mut ws[..];
        // pass 1: eta for four rows at a time (shared loads of beta)
        let mut i = 0;
        while i + 4 <= n {
            let rows: [&[f64]; 4] = std::array::from_fn(|l| &d.x[(i + l) * p..(i + l + 1) * p]);
            _mm256_storeu_pd(r.as_mut_ptr().add(i), _mm256_add_pd(_mm256_set1_pd(alpha), dot4rows(rows, beta)));
            i += 4;
        }
        while i < n {
            r[i] = alpha + dot(&d.x[i * p..(i + 1) * p], beta);
            i += 1;
        }
        // pass 2: vectorised density and residuals, overwriting eta with r
        let mut lp_acc = _mm256_setzero_pd();
        let mut ga_acc = _mm256_setzero_pd();
        let mut i = 0;
        while i + 4 <= n {
            let (l, rv) = four(_mm256_loadu_pd(r.as_ptr().add(i)), _mm256_loadu_pd(d.y.as_ptr().add(i)));
            lp_acc = _mm256_add_pd(lp_acc, l);
            ga_acc = _mm256_add_pd(ga_acc, rv);
            _mm256_storeu_pd(r.as_mut_ptr().add(i), rv);
            i += 4;
        }
        lp += hsum(lp_acc);
        let mut ga = hsum(ga_acc);
        while i < n {
            let eta = r[i];
            let e = (-eta.abs()).exp();
            lp += d.y[i] * eta - (eta.max(0.0) + (1.0 + e).ln());
            r[i] = d.y[i] - (if eta >= 0.0 { 1.0 } else { e }) / (1.0 + e);
            ga += r[i];
            i += 1;
        }
        g0[0] += ga;
        // pass 3: gradient, four rows per pass over it
        let mut i = 0;
        while i + 4 <= n {
            let rows: [&[f64]; 4] = std::array::from_fn(|l| &d.x[(i + l) * p..(i + l + 1) * p]);
            axpy4(gb, [r[i], r[i + 1], r[i + 2], r[i + 3]], rows);
            i += 4;
        }
        while i < n {
            axpy(gb, r[i], &d.x[i * p..(i + 1) * p]);
            i += 1;
        }
    });
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
