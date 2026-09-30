//! Max-effort Rust (nightly): hierarchical dynamic Poisson panel
//! (bench/dynpois/SPEC.md), log density and gradient written by hand with
//! AVX2/FMA intrinsics and glibc's vector exp.
//!
//! theta = [pop, beta[G], shared[T], innov[G*T] (row-major g*T + t)].
//!
//! Layout of the work: groups are processed eight at a time (two AVX2
//! vectors of four groups), so the vector lanes run across groups and the
//! sequential prefix sum over t is a plain vector add, with two independent
//! dependency chains. Rows of `innov` are brought into "column" form (one
//! vector = four groups at one t) with half-width loads plus one in-lane
//! unpack per vector; gradient columns go back to rows with a 4x4 transpose
//! and full-width stores. `y` is transposed into the same column form once,
//! when the data is loaded. Per block of groups there are three passes over
//! a 9.6 KB L1-resident scratch array:
//!   (a) forward: state += shared[t] + innov[g,t]; eta = beta + state;
//!       accumulate y*eta and innov^2 (tree-summed per four steps),
//!   (b) eta <- exp(eta), a tight loop of independent glibc vector calls
//!       (every ymm register is caller-saved across the call, so a separate
//!       loop costs no more than fusing it and spilling),
//!   (c) backward: r = y - exp(eta); c += r (reverse cumsum);
//!       d/d innov = c - innov/0.08^2 (innov re-read in column form rather
//!       than stored in (a): stores are the scarcer resource);
//!       shared accumulator[t] += c.
//! The shared gradient is accumulated in a T x 4 array (one partial sum per
//! lane) and reduced horizontally once per call. The log density is summed
//! per block and the large constant added last, which keeps its rounding
//! noise near ulp(lp) (the runtime's finite-difference gradcheck sees it).
//! Scratch is thread-local and allocated on a thread's first call only, so
//! the chains' threads share nothing mutable.
//!
//! `DYNPOIS_SELFTEST=1` compares this against a plain scalar reference
//! implementation on random shapes and points and exits.
#![feature(simd_ffi)]
use std::cell::RefCell;
use std::sync::OnceLock;

#[path = "common.rs"]
mod common;
#[path = "simd.rs"]
mod simd;
use simd::*;

const SD_BETA: f64 = 0.4;
const SD_SHARED: f64 = 0.05;
const SD_INNOV: f64 = 0.08;
// 1/sd^2, all exactly representable.
const PREC_BETA: f64 = 6.25;
const PREC_SHARED: f64 = 400.0;
const PREC_INNOV: f64 = 156.25;

struct Data {
    g: usize,
    t: usize,
    /// y, row-major G x T (used by the reference implementation).
    y: Vec<f64>,
    /// y in column form: quad q (groups 4q..4q+3), time t is the vector
    /// yt[q*T + t]; groups beyond G are zero.
    yt: Vec<__m256d>,
    /// Sum of the kept -log(scale) constants.
    lp_const: f64,
}

impl Data {
    fn new(g: usize, t: usize, y: Vec<f64>) -> Data {
        assert_eq!(y.len(), g * t);
        let nq = g.div_ceil(4);
        let mut yt = vec![unsafe { _mm256_setzero_pd() }; nq * t];
        for q in 0..nq {
            for tt in 0..t {
                let mut v = [0.0f64; 4];
                for l in 0..4 {
                    let gg = 4 * q + l;
                    if gg < g {
                        v[l] = y[gg * t + tt];
                    }
                }
                yt[q * t + tt] = unsafe { std::mem::transmute::<[f64; 4], __m256d>(v) };
            }
        }
        let lp_const = -(1.0f64).ln() - (g as f64) * SD_BETA.ln() - (t as f64) * SD_SHARED.ln()
            - ((g * t) as f64) * SD_INNOV.ln();
        Data { g, t, y, yt, lp_const }
    }
    fn dim(&self) -> usize {
        1 + self.g + self.t + self.g * self.t
    }
}

static DATA: OnceLock<Data> = OnceLock::new();

/// Per-thread scratch, all L1/L2-resident for T = 150.
struct Ws {
    /// eta, then exp(eta): index t*V + v for the block being processed.
    e: Vec<__m256d>,
    /// Per-lane partial sums of the shared-gradient reverse cumsums.
    acc: Vec<__m256d>,
    /// A zero row (input for padding lanes) and a sink row (their output).
    zero: Vec<f64>,
    sink: Vec<f64>,
}

impl Ws {
    fn fit(&mut self, t: usize) {
        if self.acc.len() != t {
            let z = unsafe { _mm256_setzero_pd() };
            self.e = vec![z; 2 * t];
            self.acc = vec![z; t];
            self.zero = vec![0.0; t];
            self.sink = vec![0.0; t];
        }
    }
}

thread_local! {
    static WS: RefCell<Ws> = const { RefCell::new(Ws { e: Vec::new(), acc: Vec::new(), zero: Vec::new(), sink: Vec::new() }) };
}

/// Columns t..t+3 of four rows, as four vectors (one per t) across the rows.
#[inline(always)]
unsafe fn load_cols(r: &[*const f64; 4], t: usize) -> [__m256d; 4] {
    let a = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(r[0].add(t))), _mm_loadu_pd(r[2].add(t)), 1);
    let b = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(r[1].add(t))), _mm_loadu_pd(r[3].add(t)), 1);
    let c = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(r[0].add(t + 2))), _mm_loadu_pd(r[2].add(t + 2)), 1);
    let d = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(r[1].add(t + 2))), _mm_loadu_pd(r[3].add(t + 2)), 1);
    [_mm256_unpacklo_pd(a, b), _mm256_unpackhi_pd(a, b), _mm256_unpacklo_pd(c, d), _mm256_unpackhi_pd(c, d)]
}

/// Inverse of load_cols: writes four column vectors to columns t..t+3 of
/// four rows (full 4x4 transpose, then one full-width store per row).
#[inline(always)]
unsafe fn store_cols(r: &[*mut f64; 4], t: usize, c: [__m256d; 4]) {
    let a = _mm256_unpacklo_pd(c[0], c[1]); // r0[t..t+2] | r2[t..t+2]
    let b = _mm256_unpackhi_pd(c[0], c[1]); // r1 | r3
    let e = _mm256_unpacklo_pd(c[2], c[3]); // r0[t+2..t+4] | r2
    let f = _mm256_unpackhi_pd(c[2], c[3]); // r1 | r3
    _mm256_storeu_pd(r[0].add(t), _mm256_permute2f128_pd(a, e, 0x20));
    _mm256_storeu_pd(r[1].add(t), _mm256_permute2f128_pd(b, f, 0x20));
    _mm256_storeu_pd(r[2].add(t), _mm256_permute2f128_pd(a, e, 0x31));
    _mm256_storeu_pd(r[3].add(t), _mm256_permute2f128_pd(b, f, 0x31));
}

#[inline(always)]
unsafe fn load_col1(r: &[*const f64; 4], t: usize) -> __m256d {
    _mm256_setr_pd(*r[0].add(t), *r[1].add(t), *r[2].add(t), *r[3].add(t))
}

#[inline(always)]
unsafe fn store_col1(r: &[*mut f64; 4], t: usize, c: __m256d) {
    let v: [f64; 4] = std::mem::transmute(c);
    for l in 0..4 {
        *r[l].add(t) = v[l];
    }
}

/// One block of 4*V groups. Writes d/d innov for its rows, adds its
/// reverse cumsums into ws.acc, and returns (per lane: sum over t of
/// r = y - exp(eta); the block's sum over t of
/// y*eta - exp(eta) - 0.5*innov^2/0.08^2).
#[inline(always)]
unsafe fn block<const V: usize, const MASKED: bool>(
    t_len: usize,
    shared: *const f64,
    inn: [[*const f64; 4]; V],
    gout: [[*mut f64; 4]; V],
    beta: [__m256d; V],
    mask: [__m256d; V],
    yq: [*const __m256d; V],
    e: *mut __m256d,
    acc: *mut __m256d,
) -> ([__m256d; V], __m256d) {
    let z = _mm256_setzero_pd();
    let t4 = t_len & !3;

    // (a) forward: eta into scratch, y*eta and innov^2 into accumulators
    let mut state = [z; V];
    let mut sq = [z; V];
    let mut yeta = [z; V];
    macro_rules! fwd {
        ($v:expr, $t:expr, $x:expr) => {{
            let (v, t, x) = ($v, $t, $x);
            let sh = _mm256_broadcast_sd(&*shared.add(t));
            state[v] = _mm256_add_pd(state[v], _mm256_add_pd(sh, x));
            let mut eta = _mm256_add_pd(beta[v], state[v]);
            if MASKED {
                eta = _mm256_and_pd(eta, mask[v]); // padding lanes: eta = 0
            }
            *e.add(t * V + v) = eta;
            sq[v] = _mm256_fmadd_pd(x, x, sq[v]);
            yeta[v] = _mm256_fmadd_pd(*yq[v].add(t), eta, yeta[v]);
        }};
    }
    let mut t = 0;
    while t < t4 {
        for v in 0..V {
            let x = load_cols(&inn[v], t);
            // Only the state recurrence is sequential; the two sums are
            // formed as a small tree per chunk of four steps, so their
            // loop-carried chains are one add per chunk instead of four FMAs.
            let mut eta = [z; 4];
            for k in 0..4 {
                let sh = _mm256_broadcast_sd(&*shared.add(t + k));
                state[v] = _mm256_add_pd(state[v], _mm256_add_pd(sh, x[k]));
                eta[k] = _mm256_add_pd(beta[v], state[v]);
                if MASKED {
                    eta[k] = _mm256_and_pd(eta[k], mask[v]); // padding lanes: eta = 0
                }
                *e.add((t + k) * V + v) = eta[k];
            }
            let ye01 = _mm256_fmadd_pd(*yq[v].add(t + 1), eta[1], _mm256_mul_pd(*yq[v].add(t), eta[0]));
            let ye23 = _mm256_fmadd_pd(*yq[v].add(t + 3), eta[3], _mm256_mul_pd(*yq[v].add(t + 2), eta[2]));
            yeta[v] = _mm256_add_pd(yeta[v], _mm256_add_pd(ye01, ye23));
            let sq01 = _mm256_fmadd_pd(x[1], x[1], _mm256_mul_pd(x[0], x[0]));
            let sq23 = _mm256_fmadd_pd(x[3], x[3], _mm256_mul_pd(x[2], x[2]));
            sq[v] = _mm256_add_pd(sq[v], _mm256_add_pd(sq01, sq23));
        }
        t += 4;
    }
    while t < t_len {
        for v in 0..V {
            fwd!(v, t, load_col1(&inn[v], t));
        }
        t += 1;
    }

    // (b) exp, in place: a tight loop of independent calls
    let n = t_len * V;
    let mut j = 0;
    while j + 2 <= n {
        let a = _ZGVdN4v_exp(*e.add(j));
        let b = _ZGVdN4v_exp(*e.add(j + 1));
        *e.add(j) = a;
        *e.add(j + 1) = b;
        j += 2;
    }
    if j < n {
        *e.add(j) = _ZGVdN4v_exp(*e.add(j));
    }

    // (c) backward: reverse cumsum c of r = y - exp(eta);
    // d/d innov[g,t] = c - innov/0.08^2 (innov reloaded in column form)
    let prec = _mm256_set1_pd(PREC_INNOV);
    let mut c = [z; V];
    let mut esum = [z; V];
    macro_rules! bwd {
        ($t:expr, $x:expr) => {{
            let (t, x): (usize, [__m256d; V]) = ($t, $x);
            let mut out = [z; V];
            let mut tot = *acc.add(t);
            for v in 0..V {
                let mut ev = *e.add(t * V + v);
                if MASKED {
                    ev = _mm256_and_pd(ev, mask[v]);
                }
                let r = _mm256_sub_pd(*yq[v].add(t), ev);
                esum[v] = _mm256_add_pd(esum[v], ev);
                c[v] = _mm256_add_pd(c[v], r);
                out[v] = _mm256_fnmadd_pd(x[v], prec, c[v]);
                tot = _mm256_add_pd(tot, c[v]);
            }
            *acc.add(t) = tot;
            out
        }};
    }
    let mut t = t_len;
    while t > t4 {
        t -= 1;
        let out = bwd!(t, std::array::from_fn(|v| load_col1(&inn[v], t)));
        for v in 0..V {
            store_col1(&gout[v], t, out[v]);
        }
    }
    while t > 0 {
        t -= 4;
        let x: [[__m256d; 4]; V] = std::array::from_fn(|v| load_cols(&inn[v], t));
        let o3 = bwd!(t + 3, std::array::from_fn(|v| x[v][3]));
        let o2 = bwd!(t + 2, std::array::from_fn(|v| x[v][2]));
        let o1 = bwd!(t + 1, std::array::from_fn(|v| x[v][1]));
        let o0 = bwd!(t, std::array::from_fn(|v| x[v][0]));
        for v in 0..V {
            store_cols(&gout[v], t, [o0[v], o1[v], o2[v], o3[v]]);
        }
    }

    let mut lpv = z;
    let half_prec = _mm256_set1_pd(-0.5 * PREC_INNOV);
    for v in 0..V {
        // kept per block so the magnitudes summed stay small (the runtime's
        // finite-difference gradcheck sees rounding noise in lp directly)
        lpv = _mm256_add_pd(lpv, _mm256_sub_pd(yeta[v], esum[v]));
        lpv = _mm256_fmadd_pd(half_prec, sq[v], lpv);
    }
    (c, lpv)
}

#[inline(always)]
unsafe fn logp_impl(d: &Data, theta: *const f64, grad: *mut f64) -> f64 {
    let (gn, tn) = (d.g, d.t);
    let pop = *theta;
    let beta = theta.add(1);
    let shared = theta.add(1 + gn);
    let innov = theta.add(1 + gn + tn);
    let gbeta = grad.add(1);
    let gshared = grad.add(1 + gn);
    let ginnov = grad.add(1 + gn + tn);

    // priors on pop and beta
    let mut lp = -0.5 * pop * pop;
    let mut gpop = -pop;
    let mut sb = 0.0;
    for g in 0..gn {
        let diff = *beta.add(g) - pop;
        sb += diff * diff;
        let gb = -diff * PREC_BETA;
        *gbeta.add(g) = gb;
        gpop -= gb;
    }
    *grad = gpop;
    lp -= 0.5 * PREC_BETA * sb;

    WS.with(|ws| {
        let mut ws = ws.borrow_mut();
        ws.fit(tn);
        let ws = &mut *ws;
        let z = _mm256_setzero_pd();
        for a in ws.acc.iter_mut() {
            *a = z;
        }
        let (e, acc) = (ws.e.as_mut_ptr(), ws.acc.as_mut_ptr());
        let yt = d.yt.as_ptr();
        let ones = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
        let mut lpv = z;
        let rows_in = |g: usize| -> [*const f64; 4] { std::array::from_fn(|l| innov.add((g + l) * tn) as *const f64) };
        let rows_out = |g: usize| -> [*mut f64; 4] { std::array::from_fn(|l| ginnov.add((g + l) * tn)) };
        let add_beta = |g: usize, c: __m256d, lanes: usize| {
            let cv: [f64; 4] = std::mem::transmute(c);
            for l in 0..lanes {
                *gbeta.add(g + l) += cv[l];
            }
        };

        let nfull = gn / 4;
        let mut q = 0;
        while q + 2 <= nfull {
            let g = 4 * q;
            let (c, l) = block::<2, false>(
                tn,
                shared,
                [rows_in(g), rows_in(g + 4)],
                [rows_out(g), rows_out(g + 4)],
                [_mm256_loadu_pd(beta.add(g)), _mm256_loadu_pd(beta.add(g + 4))],
                [ones, ones],
                [yt.add(q * tn), yt.add((q + 1) * tn)],
                e,
                acc,
            );
            add_beta(g, c[0], 4);
            add_beta(g + 4, c[1], 4);
            lpv = _mm256_add_pd(lpv, l);
            q += 2;
        }
        if q < nfull {
            let g = 4 * q;
            let (c, l) = block::<1, false>(tn, shared, [rows_in(g)], [rows_out(g)], [_mm256_loadu_pd(beta.add(g))], [ones], [yt.add(q * tn)], e, acc);
            add_beta(g, c[0], 4);
            lpv = _mm256_add_pd(lpv, l);
            q += 1;
        }
        let rem = gn - 4 * nfull;
        if rem > 0 {
            let g = 4 * q;
            let zero = ws.zero.as_ptr();
            let sink = ws.sink.as_mut_ptr();
            let ri: [*const f64; 4] = std::array::from_fn(|l| if l < rem { innov.add((g + l) * tn) as *const f64 } else { zero });
            let ro: [*mut f64; 4] = std::array::from_fn(|l| if l < rem { ginnov.add((g + l) * tn) } else { sink });
            let mut bv = [0.0f64; 4];
            let mut mv = [0i64; 4];
            for l in 0..rem {
                bv[l] = *beta.add(g + l);
                mv[l] = -1;
            }
            let (c, l) = block::<1, true>(
                tn,
                shared,
                [ri],
                [ro],
                [std::mem::transmute::<[f64; 4], __m256d>(bv)],
                [std::mem::transmute::<[i64; 4], __m256d>(mv)],
                [yt.add(q * tn)],
                e,
                acc,
            );
            add_beta(g, c[0], rem);
            lpv = _mm256_add_pd(lpv, l);
        }
        lp += hsum(lpv);

        // shared: reduce the per-lane partial sums, add the prior
        let mut ss = 0.0;
        for t in 0..tn {
            let s = *shared.add(t);
            ss += s * s;
            *gshared.add(t) = hsum(*acc.add(t)) - s * PREC_SHARED;
        }
        lp -= 0.5 * PREC_SHARED * ss;
    });
    // the large constant last, so it does not magnify rounding in the sums
    lp + d.lp_const
}

extern "C" fn logp(theta: *const f64, grad: *mut f64) -> f64 {
    unsafe { logp_impl(DATA.get().unwrap(), theta, grad) }
}

extern "C" fn constrain(unc: *const f64, out: *mut f64) {
    let d = DATA.get().unwrap().dim();
    unsafe { std::ptr::copy_nonoverlapping(unc, out, d) };
}

// ------------------------------------------------------------------ reference

/// Straightforward scalar implementation of the same density, written
/// directly from the spec, used only to check the fast version.
fn logp_reference(d: &Data, theta: &[f64], grad: &mut [f64]) -> f64 {
    let (gn, tn) = (d.g, d.t);
    let pop = theta[0];
    let beta = &theta[1..1 + gn];
    let shared = &theta[1 + gn..1 + gn + tn];
    let innov = &theta[1 + gn + tn..];
    grad.fill(0.0);
    let mut lp = -0.5 * pop * pop - (1.0f64).ln();
    grad[0] -= pop;
    for g in 0..gn {
        let z = (beta[g] - pop) / SD_BETA;
        lp += -0.5 * z * z - SD_BETA.ln();
        grad[1 + g] -= z / SD_BETA;
        grad[0] += z / SD_BETA;
    }
    for t in 0..tn {
        let z = shared[t] / SD_SHARED;
        lp += -0.5 * z * z - SD_SHARED.ln();
        grad[1 + gn + t] -= z / SD_SHARED;
    }
    for i in 0..gn * tn {
        let z = innov[i] / SD_INNOV;
        lp += -0.5 * z * z - SD_INNOV.ln();
        grad[1 + gn + tn + i] -= z / SD_INNOV;
    }
    let mut r = vec![0.0; tn];
    for g in 0..gn {
        let mut state = 0.0;
        for t in 0..tn {
            state += shared[t] + innov[g * tn + t];
            let eta = beta[g] + state;
            let y = d.y[g * tn + t];
            lp += y * eta - eta.exp();
            r[t] = y - eta.exp();
        }
        // d eta[g,t] / d beta[g] = 1; d eta[g,t] / d (shared[s], innov[g,s]) = [s <= t]
        let mut c = 0.0;
        for s in (0..tn).rev() {
            c += r[s];
            grad[1 + gn + tn + g * tn + s] += c;
            grad[1 + gn + s] += c;
        }
        grad[1 + g] += c;
    }
    lp
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unif().max(1e-300), self.unif());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// The runtime's random gradcheck point (see gradcheck in runtime/mint_rt.c).
fn runtime_gradcheck_point(dim: usize, seed: u64) -> Vec<f64> {
    let mut sm = Rng(seed);
    let mut st = [0u64; 4];
    for x in st.iter_mut() {
        *x = sm.next();
    }
    (0..dim)
        .map(|_| {
            let r = st[0].wrapping_add(st[3]).rotate_left(23).wrapping_add(st[0]);
            let t = st[1] << 17;
            st[2] ^= st[0];
            st[3] ^= st[1];
            st[1] ^= st[2];
            st[0] ^= st[3];
            st[2] ^= t;
            st[3] = st[3].rotate_left(45);
            4.0 * ((r >> 11) as f64 * (1.0 / (1u64 << 53) as f64)) - 2.0
        })
        .collect()
}

/// Worst finite-difference error of the fast gradient, with the runtime's
/// metric |fd - g| / max(|fd|, 1) but a five-point stencil and a larger step
/// (h = 1e-3 * max(|q|, 1)): truncation error is O(h^4) and roundoff is
/// 100x smaller than with the runtime's h = 1e-5 central differences.
fn fd_worst(d: &Data, q: &mut [f64]) -> f64 {
    let dim = d.dim();
    let mut g = vec![0.0; dim];
    let mut g2 = vec![0.0; dim];
    unsafe { logp_impl(d, q.as_ptr(), g.as_mut_ptr()) };
    let mut worst: f64 = 0.0;
    for i in 0..dim {
        let h = 1e-3 * q[i].abs().max(1.0);
        let orig = q[i];
        let mut f = |x: f64| {
            q[i] = x;
            unsafe { logp_impl(d, q.as_ptr(), g2.as_mut_ptr()) }
        };
        let (p2, p1, m1, m2) = (f(orig + 2.0 * h), f(orig + h), f(orig - h), f(orig - 2.0 * h));
        q[i] = orig;
        let fd = (8.0 * (p1 - m1) - (p2 - m2)) / (12.0 * h);
        let err = (fd - g[i]).abs() / fd.abs().max(1.0);
        worst = if err.is_nan() { f64::INFINITY } else { worst.max(err) };
    }
    worst
}

/// Compares the fast implementation with the reference on random shapes
/// (including G and T not multiples of 4) and on the real data, at random
/// points. Returns the worst errors seen.
fn selftest(real: Option<&Data>) {
    let mut rng = Rng(12345);
    let mut worst_lp: f64 = 0.0;
    let mut worst_g: f64 = 0.0;
    let mut worst_norm: f64 = 0.0;
    let mut cases = 0;
    let check = |d: &Data, theta: &[f64], worst_lp: &mut f64, worst_g: &mut f64, worst_norm: &mut f64| {
        let dim = d.dim();
        let mut g1 = vec![f64::NAN; dim];
        let mut g2 = vec![0.0; dim];
        let l1 = unsafe { logp_impl(d, theta.as_ptr(), g1.as_mut_ptr()) };
        let l2 = logp_reference(d, theta, &mut g2);
        let el = (l1 - l2).abs() / l2.abs().max(1.0);
        let (mut num, mut den, mut eg): (f64, f64, f64) = (0.0, 0.0, 0.0);
        for i in 0..dim {
            let diff = (g1[i] - g2[i]).abs();
            num += diff * diff;
            den += g2[i] * g2[i];
            let e = diff / g2[i].abs().max(1.0);
            eg = if e.is_nan() { f64::INFINITY } else { eg.max(e) };
        }
        if !l1.is_finite() || !l2.is_finite() {
            eg = f64::INFINITY;
        }
        *worst_lp = worst_lp.max(el);
        *worst_g = worst_g.max(eg);
        *worst_norm = worst_norm.max((num / den.max(1e-300)).sqrt());
    };
    let gs = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 20, 31];
    let ts = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 150];
    for &g in &gs {
        for &t in &ts {
            let y: Vec<f64> = (0..g * t).map(|_| (rng.unif() * 12.0).floor()).collect();
            let d = Data::new(g, t, y);
            for rep in 0..3 {
                let scale = [0.05, 0.3, 1.0][rep];
                let theta: Vec<f64> = (0..d.dim()).map(|_| scale * rng.normal()).collect();
                check(&d, &theta, &mut worst_lp, &mut worst_g, &mut worst_norm);
                cases += 1;
            }
        }
    }
    if let Some(d) = real {
        for rep in 0..20 {
            let scale = [0.02, 0.1, 0.5, 1.0][rep % 4];
            let mut theta: Vec<f64> = (0..d.dim()).map(|_| scale * rng.normal()).collect();
            theta[0] += 1.5;
            for g in 0..d.g {
                theta[1 + g] += 1.5;
            }
            check(d, &theta, &mut worst_lp, &mut worst_g, &mut worst_norm);
            cases += 1;
        }
        let mut theta = vec![0.0; d.dim()];
        for (i, x) in theta.iter_mut().enumerate() {
            *x = 0.05 * (((i * 37) % 11) as f64 - 5.0) / 5.0; // the runtime's bench point
        }
        check(d, &theta, &mut worst_lp, &mut worst_g, &mut worst_norm);
        cases += 1;
    }
    // Large counts (y up to 1e17) and a padded-lane overflow case: the real
    // state stays finite while shared + 0 would overflow in padding lanes.
    for &(g, t) in &[(1usize, 1usize), (3, 5), (6, 9), (13, 150)] {
        let y: Vec<f64> = (0..g * t).map(|_| (rng.unif() * 1e17).floor()).collect();
        let d = Data::new(g, t, y);
        for offset in [18.0, 0.0, 5.0] {
            // offset 0: theta = 0 exactly, so lp is its constant minus G*T
            // and y - exp(0) must not absorb the -1 per term
            let sd = if offset == 0.0 { 0.0 } else { 0.3 };
            let mut theta: Vec<f64> = (0..d.dim()).map(|_| sd * rng.normal()).collect();
            theta[0] += offset;
            for gg in 0..g {
                theta[1 + gg] += offset;
            }
            check(&d, &theta, &mut worst_lp, &mut worst_g, &mut worst_norm);
            cases += 1;
        }
    }
    {
        let d = Data::new(1, 2, vec![0.0, 0.0]);
        let theta = [0.0, 0.0, 1e308, 1e308, -1e308, -1e308];
        let mut g = [0.0; 6];
        let lp = unsafe { logp_impl(&d, theta.as_ptr(), g.as_mut_ptr()) };
        let lr = logp_reference(&d, &theta, &mut [0.0; 6]);
        println!("selftest: padded-lane overflow case: lp = {lp}, reference = {lr}");
        if lp != lr {
            println!("selftest: FAIL (padded-lane overflow)");
            std::process::exit(1);
        }
    }
    if let Some(d) = real {
        // The runtime's second MINT_GRADCHECK point (xoshiro256++ seeded by
        // splitmix64(seed = 7), theta_i = 4u - 2). At it |lp| is ~1e23 for the
        // large data, so central differences carry roundoff of order
        // |lp| * eps / h; the fast gradient is compared to the reference here.
        let theta = runtime_gradcheck_point(d.dim(), 7);
        let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
        check(d, &theta, &mut a, &mut b, &mut c);
        let mut g = vec![0.0; d.dim()];
        let lp = unsafe { logp_impl(d, theta.as_ptr(), g.as_mut_ptr()) };
        println!(
            "selftest: runtime gradcheck point 2: logp = {lp:.17e}; FD roundoff scale |lp|*eps/h = {:.3e}; fast vs reference: lp {a:.3e}, per-component {b:.3e}, norm {c:.3e}",
            lp.abs() * f64::EPSILON / 2e-5
        );
        worst_lp = worst_lp.max(a);
        worst_g = worst_g.max(b);
        worst_norm = worst_norm.max(c);
        // Finite differences (gated at 1e-6) at points drawn from the prior
        // (pop, beta near 1.5) and at twice its innovation scale. Roundoff
        // grows with sum(exp(eta)); at |lp| ~ 1e23 it is why the runtime's
        // uniform(-2, 2) gradcheck point cannot pass for any implementation.
        for (k, scale) in [0.5, 1.0, 1.0, 2.0].into_iter().enumerate() {
            let mut theta = vec![0.0; d.dim()];
            theta[0] = 1.5 + 0.2 * rng.normal();
            for g in 0..d.g {
                theta[1 + g] = theta[0] + SD_BETA * rng.normal();
            }
            for t in 0..d.t {
                theta[1 + d.g + t] = scale * SD_SHARED * rng.normal();
            }
            for i in 0..d.g * d.t {
                theta[1 + d.g + d.t + i] = scale * SD_INNOV * rng.normal();
            }
            let mut ref_g = vec![0.0; d.dim()];
            logp_reference(d, &theta, &mut ref_g);
            let sum_exp: f64 = {
                let mut s = 0.0;
                for g in 0..d.g {
                    let mut st = 0.0;
                    for t in 0..d.t {
                        st += theta[1 + d.g + t] + theta[1 + d.g + d.t + g * d.t + t];
                        s += (theta[1 + g] + st).exp();
                    }
                }
                s
            };
            let fd = fd_worst(d, &mut theta);
            println!("selftest: finite differences (5-point), point {k} (innovations x{scale} prior sd): sum exp(eta) = {sum_exp:.3e}, worst relative error {fd:.3e}");
            if !(fd < 1e-6) {
                println!("selftest: FAIL (finite differences)");
                std::process::exit(1);
            }
        }
    }
    println!(
        "selftest: {cases} cases; worst |lp - ref|/max(|ref|,1) = {worst_lp:.3e}; worst per-component |g - ref|/max(|ref|,1) = {worst_g:.3e}; worst ||g - ref||/||ref|| = {worst_norm:.3e}"
    );
    let ok = worst_lp < 1e-10 && worst_g < 1e-10 && worst_norm < 1e-10;
    println!("selftest: {}", if ok { "PASS (< 1e-10)" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "bench/dynpois/data_small/y.f64".to_string());
    let (g, t, y) = common::read_f64(&path);
    let t0 = std::time::Instant::now();
    let d = Data::new(g, t, y);
    common::set_prep_seconds(t0.elapsed().as_secs_f64());
    if std::env::var_os("DYNPOIS_SELFTEST").is_some() {
        selftest(Some(&d));
        return;
    }
    let dim = d.dim();
    DATA.set(d).ok();
    common::sample_and_print(logp, constrain, dim, &["pop", "beta", "shared", "innov"], &[-1, g as i64, t as i64, (g * t) as i64], 1000, 1000, 4, 7);
}
