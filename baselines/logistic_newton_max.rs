//! Max-effort Rust (nightly): Newton's method for L2-regularised logistic
//! regression with AVX2/FMA intrinsics and glibc's vector exp. One pass over X
//! per iteration, four rows at a time: shared-load dot products, a vector
//! sigmoid, a register-blocked gradient update and a register-blocked update
//! of the upper triangle of the Hessian. No allocation inside the iterations.
#![feature(simd_ffi)]
use std::time::Instant;

#[path = "common.rs"]
mod common;
#[path = "simd.rs"]
mod simd;
use simd::*;

fn cholesky_solve(h: &[f64], p: usize, g: &[f64], l: &mut [f64], x: &mut [f64]) {
    for j in 0..p {
        for i in j..p {
            let s = h[i * p + j] - unsafe { dot(&l[i * p..i * p + j], &l[j * p..j * p + j]) };
            if i == j {
                assert!(s > 0.0, "Hessian is not positive definite");
                l[j * p + j] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }
    for i in 0..p {
        x[i] = (g[i] - unsafe { dot(&l[i * p..i * p + i], &x[..i]) }) / l[i * p + i];
    }
    for i in (0..p).rev() {
        let mut s = x[i];
        for k in i + 1..p {
            s -= l[k * p + i] * x[k];
        }
        x[i] = s / l[i * p + i];
    }
}

fn fit(x: &[f64], n: usize, p: usize, y: &[f64], lambda: f64) -> Vec<f64> {
    let mut w = vec![0.0; p];
    let mut g = vec![0.0; p];
    let mut h = vec![0.0; p * p];
    let mut l = vec![0.0; p * p];
    let mut step = vec![0.0; p];
    for _ in 0..10 {
        for j in 0..p {
            g[j] = lambda * w[j];
        }
        h.iter_mut().for_each(|v| *v = 0.0);
        unsafe {
            let one = _mm256_set1_pd(1.0);
            let mut i = 0;
            while i + 4 <= n {
                let rows: [&[f64]; 4] = std::array::from_fn(|k| &x[(i + k) * p..(i + k + 1) * p]);
                let eta = dot4rows(rows, &w);
                let mu = _mm256_div_pd(one, _mm256_add_pd(one, _ZGVdN4v_exp(_mm256_sub_pd(_mm256_setzero_pd(), eta))));
                let r: [f64; 4] = std::mem::transmute(_mm256_sub_pd(mu, _mm256_loadu_pd(y.as_ptr().add(i))));
                let s: [f64; 4] = std::mem::transmute(_mm256_mul_pd(mu, _mm256_sub_pd(one, mu)));
                axpy4(&mut g, r, rows);
                for j in 0..p {
                    let t = [s[0] * rows[0][j], s[1] * rows[1][j], s[2] * rows[2][j], s[3] * rows[3][j]];
                    let sub: [&[f64]; 4] = std::array::from_fn(|k| &rows[k][j..]);
                    axpy4(&mut h[j * p + j..(j + 1) * p], t, sub);
                }
                i += 4;
            }
            while i < n {
                let row = &x[i * p..(i + 1) * p];
                let mu = 1.0 / (1.0 + (-dot(row, &w)).exp());
                let s = mu * (1.0 - mu);
                axpy(&mut g, mu - y[i], row);
                for j in 0..p {
                    axpy(&mut h[j * p + j..(j + 1) * p], s * row[j], &row[j..]);
                }
                i += 1;
            }
        }
        for j in 0..p {
            for k in 0..j {
                h[j * p + k] = h[k * p + j];
            }
            h[j * p + j] += lambda;
        }
        cholesky_solve(&h, p, &g, &mut l, &mut step);
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
