//! Hand-written SIMD helpers for the "max effort" Rust baselines (nightly
//! Rust): AVX2/FMA intrinsics and glibc's vector math library (the same
//! `exp`/`log` that Martin's generated code calls).
#![allow(dead_code)]

pub use std::arch::x86_64::*;

#[allow(improper_ctypes)]
extern "C" {
    pub fn _ZGVdN4v_exp(x: __m256d) -> __m256d;
    pub fn _ZGVdN4v_log(x: __m256d) -> __m256d;
}

#[inline(always)]
pub unsafe fn hsum(v: __m256d) -> f64 {
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd(v, 1);
    let s = _mm_add_pd(lo, hi);
    let h = _mm_unpackhi_pd(s, s);
    _mm_cvtsd_f64(_mm_add_sd(s, h))
}

/// Dot product with four independent 4-lane FMA accumulators.
#[inline(always)]
pub unsafe fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm256_setzero_pd(); 4];
    let mut k = 0;
    while k + 16 <= n {
        for l in 0..4 {
            acc[l] = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(k + 4 * l)), _mm256_loadu_pd(pb.add(k + 4 * l)), acc[l]);
        }
        k += 16;
    }
    while k + 4 <= n {
        acc[0] = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(k)), _mm256_loadu_pd(pb.add(k)), acc[0]);
        k += 4;
    }
    let mut s = hsum(_mm256_add_pd(_mm256_add_pd(acc[0], acc[1]), _mm256_add_pd(acc[2], acc[3])));
    while k < n {
        s += a[k] * b[k];
        k += 1;
    }
    s
}

/// y += a * x
#[inline(always)]
pub unsafe fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    let n = y.len();
    let (py, px) = (y.as_mut_ptr(), x.as_ptr());
    let va = _mm256_set1_pd(a);
    let mut k = 0;
    while k + 4 <= n {
        _mm256_storeu_pd(py.add(k), _mm256_fmadd_pd(va, _mm256_loadu_pd(px.add(k)), _mm256_loadu_pd(py.add(k))));
        k += 4;
    }
    while k < n {
        *py.add(k) += a * *px.add(k);
        k += 1;
    }
}

/// y += a0*x0 + a1*x1 + a2*x2 + a3*x3 (four rows per pass over y)
#[inline(always)]
pub unsafe fn axpy4(y: &mut [f64], a: [f64; 4], x: [&[f64]; 4]) {
    let n = y.len();
    let py = y.as_mut_ptr();
    let va = a.map(|v| _mm256_set1_pd(v));
    let px = x.map(|s| s.as_ptr());
    let mut k = 0;
    while k + 4 <= n {
        let mut t = _mm256_loadu_pd(py.add(k));
        for l in 0..4 {
            t = _mm256_fmadd_pd(va[l], _mm256_loadu_pd(px[l].add(k)), t);
        }
        _mm256_storeu_pd(py.add(k), t);
        k += 4;
    }
    while k < n {
        *py.add(k) += a[0] * *px[0].add(k) + a[1] * *px[1].add(k) + a[2] * *px[2].add(k) + a[3] * *px[3].add(k);
        k += 1;
    }
}

/// Four dot products against the same vector b, sharing the loads of b.
#[inline(always)]
pub unsafe fn dot4rows(r: [&[f64]; 4], b: &[f64]) -> __m256d {
    let n = b.len();
    let pb = b.as_ptr();
    let pr = r.map(|s| s.as_ptr());
    let mut acc = [_mm256_setzero_pd(); 4];
    let mut k = 0;
    while k + 4 <= n {
        let vb = _mm256_loadu_pd(pb.add(k));
        for l in 0..4 {
            acc[l] = _mm256_fmadd_pd(_mm256_loadu_pd(pr[l].add(k)), vb, acc[l]);
        }
        k += 4;
    }
    // transpose-and-add the four accumulators into one vector of four sums
    let t0 = _mm256_hadd_pd(acc[0], acc[1]);
    let t1 = _mm256_hadd_pd(acc[2], acc[3]);
    let lo = _mm256_permute2f128_pd(t0, t1, 0x20);
    let hi = _mm256_permute2f128_pd(t0, t1, 0x31);
    let mut s: [f64; 4] = std::mem::transmute(_mm256_add_pd(lo, hi));
    while k < n {
        for l in 0..4 {
            s[l] += *pr[l].add(k) * *pb.add(k);
        }
        k += 1;
    }
    std::mem::transmute(s)
}
