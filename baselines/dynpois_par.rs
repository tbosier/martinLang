//! Max-effort Rust (nightly), second version: hierarchical dynamic Poisson
//! panel (bench/dynpois/SPEC.md), log density and gradient written by hand
//! with AVX2/FMA intrinsics, brought to the algorithmic level of Mint's
//! compiled kernel where the shared sampler allows it.
//!
//! theta = [pop, beta[G], shared[T], innov[G*T] (row-major g*T + t)]: the
//! user's order, which the shared sampler fixes. (Mint's compiled program
//! stores innov column-major internally and converts at the sampler's
//! boundary; that layout is not available here, see bench/same_sampler/README.md.)
//!
//! Against baselines/dynpois_max.rs (kept unchanged) this version adds the
//! two tricks Mint's compiler uses that the first version did not:
//!
//! 1. Mint's table-driven exp (compiler/src/ir.rs, mint_exp_fast): x =
//!    (256 k + j) ln2/256 + r, e^x = 2^k * T[j] * (1 + q(r)) with the same
//!    256-entry table (EXP_TAB, copied bit for bit), the same degree-4
//!    polynomial and the same AVX2 gather, inline. Inputs with |x| > 708 or
//!    NaN take a cold path (libm's exp per lane). Where to put it was
//!    measured (DYNPOIS_EXP selects; bench/same_sampler/results/grad.json):
//!      table  (the default) a separate tight pass over the block's eta,
//!             replacing dynpois_max.rs's glibc calls: the fastest;
//!      fused  inside the forward pass (eta, exp, the density terms and
//!             r = y - exp(eta) per step, storing r), as Mint does with its
//!             column-major layout: about 20% slower here;
//!      back   inside the backward pass: slower than table;
//!      glibc  glibc's _ZGVdN4v_exp in the separate pass, i.e. dynpois_max.rs's
//!             kernel with only the threading added.
//! 2. The gradient is split across the chain's threads like Mint's parallel
//!    fused scan kernel (compiler/src/model.rs, gen_fused_scan and
//!    par_kernel_call): blocks of eight rows (two AVX2 vectors of four
//!    groups) are handed to the runtime's mint_par_groups with the thread
//!    count mint_par_threads() reports for the calling chain (the runtime
//!    sets it to the chain's threads per chain during sampling, and to 1, or
//!    MINT_KERNEL_THREADS, in MINT_BENCH_GRAD). Thread k writes the partial
//!    sums of the shared gradient into its own T x 4 slot and its log density
//!    into its own cell; the caller adds them in thread order, so the result
//!    is deterministic for a given thread count. Rows left over after the
//!    blocks of eight run on the calling thread.
//!
//! What it still does not do (not available under the shared sampler's
//! parameter order): Mint's column-major storage of innov, which makes four
//! groups at one time step one contiguous load. Here rows are brought into
//! column form with half-width loads and unpacks, and the gradient goes back
//! with 4x4 transposes, as in dynpois_max.rs.
//!
//! `DYNPOIS_SELFTEST=1` compares this against a plain scalar reference on
//! random shapes and points, for thread counts 1, 2, 3, 4 and 7, and exits.
#![feature(simd_ffi)]
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicI64, Ordering};
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
/// The runtime's limit on threads per kernel (MAX_NT in runtime/mint_rt.c).
const MAX_NT: usize = 64;

extern "C" {
    fn mint_par_groups(f: extern "C" fn(*mut c_void, i64, i64, i64), ctx: *mut c_void, ngroups: i64, nt: i64) -> i64;
    fn mint_par_threads() -> i64;
}

/// Thread count forced by the self-test (0: ask the runtime).
static FORCE_NT: AtomicI64 = AtomicI64::new(0);

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

// ------------------------------------------------------------------ exp

#[inline(always)]
unsafe fn exp_tab_ptr() -> *const f64 {
    EXP_TAB.as_ptr() as *const f64
}

/// exp of four lanes outside the fast path's range (|x| > 708 or NaN).
#[cold]
#[inline(never)]
unsafe fn exp4_slow(x: __m256d, fast: __m256d, out: __m256d) -> __m256d {
    let xv: [f64; 4] = std::mem::transmute(x);
    let mut fv: [f64; 4] = std::mem::transmute(fast);
    let ov: [i64; 4] = std::mem::transmute(out);
    for l in 0..4 {
        if ov[l] != 0 {
            fv[l] = xv[l].exp();
        }
    }
    std::mem::transmute(fv)
}

/// Mint's fast exp (compiler/src/ir.rs, mint_exp_fast), four lanes.
#[inline(always)]
unsafe fn exp4(x: __m256d) -> __m256d {
    let shift = _mm256_set1_pd(6755399441055744.0); // 0x1.8p52
    let kd0 = _mm256_fmadd_pd(x, _mm256_set1_pd(std::f64::consts::LOG2_E * 256.0), shift);
    let kb = _mm256_castpd_si256(kd0);
    let kd = _mm256_sub_pd(kd0, shift);
    let r0 = _mm256_fmadd_pd(kd, _mm256_set1_pd(-f64::from_bits(0x3FE62E42FEFA39EF) / 256.0), x);
    let r = _mm256_fmadd_pd(kd, _mm256_set1_pd(-f64::from_bits(0x3C7ABC9E3B39803F) / 256.0), r0);
    let r2 = _mm256_mul_pd(r, r);
    let qa = _mm256_fmadd_pd(r, _mm256_set1_pd(1.0 / 6.0), _mm256_set1_pd(0.5));
    let qb = _mm256_fmadd_pd(r2, _mm256_set1_pd(1.0 / 24.0), qa);
    let q = _mm256_fmadd_pd(r2, qb, r);
    let j = _mm256_and_si256(kb, _mm256_set1_epi64x(255));
    let t = _mm256_i64gather_pd::<8>(exp_tab_ptr(), j);
    let m = _mm256_fmadd_pd(t, q, t);
    let sc = _mm256_slli_epi64::<44>(_mm256_and_si256(kb, _mm256_set1_epi64x(-256)));
    let fast = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(m), sc));
    let ax = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    let out = _mm256_cmp_pd::<_CMP_NLE_UQ>(ax, _mm256_set1_pd(708.0)); // ugt: also NaN
    if _mm256_movemask_pd(out) != 0 {
        return exp4_slow(x, fast, out);
    }
    fast
}

// ------------------------------------------------------------------ kernel

/// Per-thread scratch: r = y - exp(eta) for the block being processed
/// (index t*V + v), and a zero row and a sink row for padding lanes.
struct Ws {
    r: Vec<__m256d>,
    zero: Vec<f64>,
    sink: Vec<f64>,
}

impl Ws {
    fn fit(&mut self, t: usize) {
        if self.zero.len() != t {
            self.r = vec![unsafe { _mm256_setzero_pd() }; 2 * t];
            self.zero = vec![0.0; t];
            self.sink = vec![0.0; t];
        }
    }
}

thread_local! {
    static WS: RefCell<Ws> = const { RefCell::new(Ws { r: Vec::new(), zero: Vec::new(), sink: Vec::new() }) };
    /// The calling chain's per-thread slots: shared-gradient partial sums
    /// (T vectors per slot) and log densities (one cache line per slot).
    static SLOTS: RefCell<(Vec<__m256d>, Vec<[f64; 8]>)> = const { RefCell::new((Vec::new(), Vec::new())) };
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
    let a = _mm256_unpacklo_pd(c[0], c[1]);
    let b = _mm256_unpackhi_pd(c[0], c[1]);
    let e = _mm256_unpacklo_pd(c[2], c[3]);
    let f = _mm256_unpackhi_pd(c[2], c[3]);
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

/// How exp(eta) is computed (DYNPOIS_EXP, read once; see the file header).
const EXP_FUSED: u8 = 0; // Mint's table exp inside the forward pass
const EXP_TABLE: u8 = 1; // Mint's table exp in a separate pass over the block
const EXP_GLIBC: u8 = 2; // glibc's _ZGVdN4v_exp in a separate pass (dynpois_max.rs)
const EXP_BACK: u8 = 3; // Mint's table exp inside the backward pass

/// One block of 4*V groups. Writes d/d innov for its rows, adds its
/// reverse cumsums into acc (T vectors), and returns (per lane: sum over t
/// of r = y - exp(eta); the block's sum of y*eta - exp(eta) -
/// 0.5*innov^2/0.08^2).
///
/// EXP_FUSED: the forward pass computes eta, exp(eta), the density terms
/// and r = y - exp(eta) per step and stores r; the backward pass reads r.
/// EXP_BACK computes exp(eta) in the backward pass instead.
/// EXP_TABLE / EXP_GLIBC: the forward pass stores eta, a separate tight
/// loop replaces it by exp(eta), and the backward pass forms r (the
/// structure of dynpois_max.rs).
#[inline(always)]
unsafe fn block<const V: usize, const MASKED: bool, const MODE: u8>(
    t_len: usize,
    shared: *const f64,
    inn: [[*const f64; 4]; V],
    gout: [[*mut f64; 4]; V],
    beta: [__m256d; V],
    mask: [__m256d; V],
    yq: [*const __m256d; V],
    rs: *mut __m256d,
    acc: *mut __m256d,
) -> ([__m256d; V], __m256d) {
    let z = _mm256_setzero_pd();
    let t4 = t_len & !3;
    let fused = MODE == EXP_FUSED;

    // forward
    let mut state = [z; V];
    let mut sq = [z; V];
    let mut lpv = [z; V]; // fused: y*eta - exp(eta); otherwise y*eta
    macro_rules! eta_at {
        ($v:expr, $t:expr, $x:expr) => {{
            let sh = _mm256_broadcast_sd(&*shared.add($t));
            state[$v] = _mm256_add_pd(state[$v], _mm256_add_pd(sh, $x));
            let eta = _mm256_add_pd(beta[$v], state[$v]);
            if MASKED {
                _mm256_and_pd(eta, mask[$v]) // padding lanes: eta = 0
            } else {
                eta
            }
        }};
    }
    // one fused step: r and the density terms
    macro_rules! fused_step {
        ($v:expr, $t:expr, $eta:expr) => {{
            let (v, t, eta) = ($v, $t, $eta);
            let mut e = exp4(eta);
            if MASKED {
                e = _mm256_and_pd(e, mask[v]); // exp(eta) = 0 there, so r = 0
            }
            let y = *yq[v].add(t);
            *rs.add(t * V + v) = _mm256_sub_pd(y, e);
            lpv[v] = _mm256_add_pd(lpv[v], _mm256_fmsub_pd(y, eta, e));
        }};
    }
    let mut t = 0;
    while t < t4 {
        for v in 0..V {
            let x = load_cols(&inn[v], t);
            let mut eta = [z; 4];
            for k in 0..4 {
                eta[k] = eta_at!(v, t + k, x[k]);
                if fused {
                    fused_step!(v, t + k, eta[k]);
                } else {
                    *rs.add((t + k) * V + v) = eta[k];
                }
            }
            if !fused {
                // the two sums as a small tree per chunk of four steps
                let ye01 = _mm256_fmadd_pd(*yq[v].add(t + 1), eta[1], _mm256_mul_pd(*yq[v].add(t), eta[0]));
                let ye23 = _mm256_fmadd_pd(*yq[v].add(t + 3), eta[3], _mm256_mul_pd(*yq[v].add(t + 2), eta[2]));
                lpv[v] = _mm256_add_pd(lpv[v], _mm256_add_pd(ye01, ye23));
            }
            let sq01 = _mm256_fmadd_pd(x[1], x[1], _mm256_mul_pd(x[0], x[0]));
            let sq23 = _mm256_fmadd_pd(x[3], x[3], _mm256_mul_pd(x[2], x[2]));
            sq[v] = _mm256_add_pd(sq[v], _mm256_add_pd(sq01, sq23));
        }
        t += 4;
    }
    while t < t_len {
        for v in 0..V {
            let x = load_col1(&inn[v], t);
            let eta = eta_at!(v, t, x);
            if fused {
                fused_step!(v, t, eta);
            } else {
                *rs.add(t * V + v) = eta;
                lpv[v] = _mm256_fmadd_pd(*yq[v].add(t), eta, lpv[v]);
            }
            sq[v] = _mm256_fmadd_pd(x, x, sq[v]);
        }
        t += 1;
    }

    // separate exp pass, in place: a tight loop of independent calls
    if !fused && MODE != EXP_BACK {
        let n = t_len * V;
        let mut j = 0;
        while j + 2 <= n {
            let (a, b) = if MODE == EXP_GLIBC {
                (_ZGVdN4v_exp(*rs.add(j)), _ZGVdN4v_exp(*rs.add(j + 1)))
            } else {
                (exp4(*rs.add(j)), exp4(*rs.add(j + 1)))
            };
            *rs.add(j) = a;
            *rs.add(j + 1) = b;
            j += 2;
        }
        if j < n {
            *rs.add(j) = if MODE == EXP_GLIBC { _ZGVdN4v_exp(*rs.add(j)) } else { exp4(*rs.add(j)) };
        }
    }

    // backward: reverse cumsum c of r; d/d innov[g,t] = c - innov/0.08^2
    // (innov reloaded in column form)
    let prec = _mm256_set1_pd(PREC_INNOV);
    let mut c = [z; V];
    let mut esum = [z; V];
    macro_rules! bwd {
        ($t:expr, $x:expr) => {{
            let (t, x): (usize, [__m256d; V]) = ($t, $x);
            let mut out = [z; V];
            let mut tot = *acc.add(t);
            for v in 0..V {
                let r = if fused {
                    *rs.add(t * V + v)
                } else {
                    let mut ev = *rs.add(t * V + v);
                    if MODE == EXP_BACK {
                        ev = exp4(ev);
                    }
                    if MASKED {
                        ev = _mm256_and_pd(ev, mask[v]);
                    }
                    esum[v] = _mm256_add_pd(esum[v], ev);
                    _mm256_sub_pd(*yq[v].add(t), ev)
                };
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

    let mut lp = z;
    let half_prec = _mm256_set1_pd(-0.5 * PREC_INNOV);
    for v in 0..V {
        lp = _mm256_add_pd(lp, _mm256_sub_pd(lpv[v], esum[v]));
        lp = _mm256_fmadd_pd(half_prec, sq[v], lp);
    }
    (c, lp)
}

/// What the threads of one gradient share (read-only apart from their own
/// rows of the gradient and their own slots).
struct Ctx {
    d: *const Data,
    theta: *const f64,
    grad: *mut f64,
    /// slot k's shared-gradient partial sums: acc[k*T .. (k+1)*T]
    acc: *mut __m256d,
    /// slot k's log density: lp[k][0]
    lp: *mut [f64; 8],
}

#[inline(always)]
unsafe fn rows_in(innov: *const f64, tn: usize, g: usize) -> [*const f64; 4] {
    std::array::from_fn(|l| innov.add((g + l) * tn))
}

#[inline(always)]
unsafe fn rows_out(ginnov: *mut f64, tn: usize, g: usize) -> [*mut f64; 4] {
    std::array::from_fn(|l| ginnov.add((g + l) * tn))
}

#[inline(always)]
unsafe fn add_beta(gbeta: *mut f64, g: usize, c: __m256d, lanes: usize) {
    let cv: [f64; 4] = std::mem::transmute(c);
    for l in 0..lanes {
        *gbeta.add(g + l) += cv[l];
    }
}

/// Blocks q0..q1 of eight rows, on slot `tid` (mint_group_fn).
extern "C" fn kernel<const MODE: u8>(ctx: *mut c_void, q0: i64, q1: i64, tid: i64) {
    unsafe {
        let cx = &*(ctx as *const Ctx);
        let d = &*cx.d;
        let (gn, tn) = (d.g, d.t);
        let (theta, grad) = (cx.theta, cx.grad);
        let beta = theta.add(1);
        let shared = theta.add(1 + gn);
        let innov = theta.add(1 + gn + tn);
        let gbeta = grad.add(1);
        let ginnov = grad.add(1 + gn + tn);
        let acc = cx.acc.add(tid as usize * tn);
        let z = _mm256_setzero_pd();
        for k in 0..tn {
            *acc.add(k) = z;
        }
        let ones = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
        let yt = d.yt.as_ptr();
        let mut lpv = z;
        WS.with(|ws| {
            let mut ws = ws.borrow_mut();
            ws.fit(tn);
            let rs = ws.r.as_mut_ptr();
            for q in q0 as usize..q1 as usize {
                let g = 8 * q;
                let (c, l) = block::<2, false, MODE>(
                    tn,
                    shared,
                    [rows_in(innov, tn, g), rows_in(innov, tn, g + 4)],
                    [rows_out(ginnov, tn, g), rows_out(ginnov, tn, g + 4)],
                    [_mm256_loadu_pd(beta.add(g)), _mm256_loadu_pd(beta.add(g + 4))],
                    [ones, ones],
                    [yt.add(2 * q * tn), yt.add((2 * q + 1) * tn)],
                    rs,
                    acc,
                );
                add_beta(gbeta, g, c[0], 4);
                add_beta(gbeta, g + 4, c[1], 4);
                lpv = _mm256_add_pd(lpv, l);
            }
        });
        (*cx.lp.add(tid as usize))[0] = hsum(lpv);
    }
}

#[inline(always)]
unsafe fn logp_mode<const MODE: u8>(d: &Data, theta: *const f64, grad: *mut f64) -> f64 {
    let forced = FORCE_NT.load(Ordering::Relaxed);
    let nt = if forced > 0 { forced } else { mint_par_threads() }.clamp(1, MAX_NT as i64);
    let (gn, tn) = (d.g, d.t);
    let pop = *theta;
    let beta = theta.add(1);
    let shared = theta.add(1 + gn);
    let innov = theta.add(1 + gn + tn);
    let gbeta = grad.add(1);
    let gshared = grad.add(1 + gn);
    let ginnov = grad.add(1 + gn + tn);

    // priors on pop and beta (the kernel adds the likelihood's part of
    // d/d beta into gbeta afterwards)
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

    SLOTS.with(|slots| {
        let mut slots = slots.borrow_mut();
        let (acc_v, lp_v) = &mut *slots;
        if acc_v.len() < MAX_NT * tn {
            *acc_v = vec![_mm256_setzero_pd(); MAX_NT * tn];
            *lp_v = vec![[0.0; 8]; MAX_NT];
        }
        let cx = Ctx { d, theta, grad, acc: acc_v.as_mut_ptr(), lp: lp_v.as_mut_ptr() };
        let nblocks = gn / 8;
        // (with no blocks, the runtime still calls kernel once, which zeroes slot 0)
        let used = mint_par_groups(kernel::<MODE>, &cx as *const Ctx as *mut c_void, nblocks as i64, nt) as usize;

        // rows left over after the blocks of eight, on this thread, into slot 0
        let acc0 = acc_v.as_mut_ptr();
        let mut rest = _mm256_setzero_pd();
        WS.with(|ws| {
            let mut ws = ws.borrow_mut();
            ws.fit(tn);
            let rs = ws.r.as_mut_ptr();
            let ones = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
            let yt = d.yt.as_ptr();
            let mut g = 8 * nblocks;
            if gn - g >= 4 {
                let q = g / 4;
                let (c, l) = block::<1, false, MODE>(tn, shared, [rows_in(innov, tn, g)], [rows_out(ginnov, tn, g)], [_mm256_loadu_pd(beta.add(g))], [ones], [yt.add(q * tn)], rs, acc0);
                add_beta(gbeta, g, c[0], 4);
                rest = _mm256_add_pd(rest, l);
                g += 4;
            }
            let rem = gn - g;
            if rem > 0 {
                let q = g / 4;
                let zero = ws.zero.as_ptr();
                let sink = ws.sink.as_mut_ptr();
                let ri: [*const f64; 4] = std::array::from_fn(|l| if l < rem { innov.add((g + l) * tn) } else { zero });
                let ro: [*mut f64; 4] = std::array::from_fn(|l| if l < rem { ginnov.add((g + l) * tn) } else { sink });
                let mut bv = [0.0f64; 4];
                let mut mv = [0i64; 4];
                for l in 0..rem {
                    bv[l] = *beta.add(g + l);
                    mv[l] = -1;
                }
                let (c, l) = block::<1, true, MODE>(
                    tn,
                    shared,
                    [ri],
                    [ro],
                    [std::mem::transmute::<[f64; 4], __m256d>(bv)],
                    [std::mem::transmute::<[i64; 4], __m256d>(mv)],
                    [yt.add(q * tn)],
                    rs,
                    acc0,
                );
                add_beta(gbeta, g, c[0], rem);
                rest = _mm256_add_pd(rest, l);
            }
        });

        // the threads' results, in thread order
        let mut lk = 0.0;
        for k in 0..used {
            lk += lp_v[k][0];
        }
        lp += lk + hsum(rest);
        let mut ss = 0.0;
        for t in 0..tn {
            let s = *shared.add(t);
            ss += s * s;
            let mut tot = acc_v[t];
            for k in 1..used {
                tot = _mm256_add_pd(tot, acc_v[k * tn + t]);
            }
            *gshared.add(t) = hsum(tot) - s * PREC_SHARED;
        }
        lp -= 0.5 * PREC_SHARED * ss;
    });
    // the large constant last, so it does not magnify rounding in the sums
    lp + d.lp_const
}

/// DYNPOIS_EXP: table (the default), fused, back or glibc.
fn exp_mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("DYNPOIS_EXP").as_deref() {
        Err(_) | Ok("table") => EXP_TABLE,
        Ok("fused") => EXP_FUSED,
        Ok("glibc") => EXP_GLIBC,
        Ok("back") => EXP_BACK,
        Ok(v) => panic!("DYNPOIS_EXP must be table, fused, back or glibc, got {v:?}"),
    })
}

#[inline(always)]
unsafe fn logp_impl(d: &Data, theta: *const f64, grad: *mut f64) -> f64 {
    match exp_mode() {
        EXP_FUSED => logp_mode::<EXP_FUSED>(d, theta, grad),
        EXP_TABLE => logp_mode::<EXP_TABLE>(d, theta, grad),
        EXP_BACK => logp_mode::<EXP_BACK>(d, theta, grad),
        _ => logp_mode::<EXP_GLIBC>(d, theta, grad),
    }
}

extern "C" fn logp(theta: *const f64, grad: *mut f64) -> f64 {
    unsafe { logp_impl(DATA.get().unwrap(), theta, grad) }
}

extern "C" fn constrain(unc: *const f64, out: *mut f64) {
    let d = DATA.get().unwrap().dim();
    unsafe { std::ptr::copy_nonoverlapping(unc, out, d) };
}

// ------------------------------------------------------------------ the exp table

/// 2^(j/256) for j = 0..255, each correctly rounded: Mint's table
/// (compiler/src/ir.rs, EXP_TAB), copied bit for bit.
#[repr(align(64))]
struct Tab([u64; 256]);
static EXP_TAB_A: Tab = Tab([
    0x3FF0000000000000, 0x3FF00B1AFA5ABCBF, 0x3FF0163DA9FB3335, 0x3FF02168143B0281,
    0x3FF02C9A3E778061, 0x3FF037D42E11BBCC, 0x3FF04315E86E7F85, 0x3FF04E5F72F654B1,
    0x3FF059B0D3158574, 0x3FF0650A0E3C1F89, 0x3FF0706B29DDF6DE, 0x3FF07BD42B72A836,
    0x3FF0874518759BC8, 0x3FF092BDF66607E0, 0x3FF09E3ECAC6F383, 0x3FF0A9C79B1F3919,
    0x3FF0B5586CF9890F, 0x3FF0C0F145E46C85, 0x3FF0CC922B7247F7, 0x3FF0D83B23395DEC,
    0x3FF0E3EC32D3D1A2, 0x3FF0EFA55FDFA9C5, 0x3FF0FB66AFFED31B, 0x3FF1073028D7233E,
    0x3FF11301D0125B51, 0x3FF11EDBAB5E2AB6, 0x3FF12ABDC06C31CC, 0x3FF136A814F204AB,
    0x3FF1429AAEA92DE0, 0x3FF14E95934F312E, 0x3FF15A98C8A58E51, 0x3FF166A45471C3C2,
    0x3FF172B83C7D517B, 0x3FF17ED48695BBC0, 0x3FF18AF9388C8DEA, 0x3FF1972658375D2F,
    0x3FF1A35BEB6FCB75, 0x3FF1AF99F8138A1C, 0x3FF1BBE084045CD4, 0x3FF1C82F95281C6B,
    0x3FF1D4873168B9AA, 0x3FF1E0E75EB44027, 0x3FF1ED5022FCD91D, 0x3FF1F9C18438CE4D,
    0x3FF2063B88628CD6, 0x3FF212BE3578A819, 0x3FF21F49917DDC96, 0x3FF22BDDA27912D1,
    0x3FF2387A6E756238, 0x3FF2451FFB82140A, 0x3FF251CE4FB2A63F, 0x3FF25E85711ECE75,
    0x3FF26B4565E27CDD, 0x3FF2780E341DDF29, 0x3FF284DFE1F56381, 0x3FF291BA7591BB70,
    0x3FF29E9DF51FDEE1, 0x3FF2AB8A66D10F13, 0x3FF2B87FD0DAD990, 0x3FF2C57E39771B2F,
    0x3FF2D285A6E4030B, 0x3FF2DF961F641589, 0x3FF2ECAFA93E2F56, 0x3FF2F9D24ABD886B,
    0x3FF306FE0A31B715, 0x3FF31432EDEEB2FD, 0x3FF32170FC4CD831, 0x3FF32EB83BA8EA32,
    0x3FF33C08B26416FF, 0x3FF3496266E3FA2D, 0x3FF356C55F929FF1, 0x3FF36431A2DE883B,
    0x3FF371A7373AA9CB, 0x3FF37F26231E754A, 0x3FF38CAE6D05D866, 0x3FF39A401B7140EF,
    0x3FF3A7DB34E59FF7, 0x3FF3B57FBFEC6CF4, 0x3FF3C32DC313A8E5, 0x3FF3D0E544EDE173,
    0x3FF3DEA64C123422, 0x3FF3EC70DF1C5175, 0x3FF3FA4504AC801C, 0x3FF40822C367A024,
    0x3FF4160A21F72E2A, 0x3FF423FB2709468A, 0x3FF431F5D950A897, 0x3FF43FFA3F84B9D4,
    0x3FF44E086061892D, 0x3FF45C2042A7D232, 0x3FF46A41ED1D0057, 0x3FF4786D668B3237,
    0x3FF486A2B5C13CD0, 0x3FF494E1E192AED2, 0x3FF4A32AF0D7D3DE, 0x3FF4B17DEA6DB7D7,
    0x3FF4BFDAD5362A27, 0x3FF4CE41B817C114, 0x3FF4DCB299FDDD0D, 0x3FF4EB2D81D8ABFF,
    0x3FF4F9B2769D2CA7, 0x3FF508417F4531EE, 0x3FF516DAA2CF6642, 0x3FF5257DE83F4EEF,
    0x3FF5342B569D4F82, 0x3FF542E2F4F6AD27, 0x3FF551A4CA5D920F, 0x3FF56070DDE910D2,
    0x3FF56F4736B527DA, 0x3FF57E27DBE2C4CF, 0x3FF58D12D497C7FD, 0x3FF59C0827FF07CC,
    0x3FF5AB07DD485429, 0x3FF5BA11FBA87A03, 0x3FF5C9268A5946B7, 0x3FF5D84590998B93,
    0x3FF5E76F15AD2148, 0x3FF5F6A320DCEB71, 0x3FF605E1B976DC09, 0x3FF6152AE6CDF6F4,
    0x3FF6247EB03A5585, 0x3FF633DD1D1929FD, 0x3FF6434634CCC320, 0x3FF652B9FEBC8FB7,
    0x3FF6623882552225, 0x3FF671C1C70833F6, 0x3FF68155D44CA973, 0x3FF690F4B19E9538,
    0x3FF6A09E667F3BCD, 0x3FF6B052FA75173E, 0x3FF6C012750BDABF, 0x3FF6CFDCDDD47645,
    0x3FF6DFB23C651A2F, 0x3FF6EF9298593AE5, 0x3FF6FF7DF9519484, 0x3FF70F7466F42E87,
    0x3FF71F75E8EC5F74, 0x3FF72F8286EAD08A, 0x3FF73F9A48A58174, 0x3FF74FBD35D7CBFD,
    0x3FF75FEB564267C9, 0x3FF77024B1AB6E09, 0x3FF780694FDE5D3F, 0x3FF790B938AC1CF6,
    0x3FF7A11473EB0187, 0x3FF7B17B0976CFDB, 0x3FF7C1ED0130C132, 0x3FF7D26A62FF86F0,
    0x3FF7E2F336CF4E62, 0x3FF7F3878491C491, 0x3FF80427543E1A12, 0x3FF814D2ADD106D9,
    0x3FF82589994CCE13, 0x3FF8364C1EB941F7, 0x3FF8471A4623C7AD, 0x3FF857F4179F5B21,
    0x3FF868D99B4492ED, 0x3FF879CAD931A436, 0x3FF88AC7D98A6699, 0x3FF89BD0A478580F,
    0x3FF8ACE5422AA0DB, 0x3FF8BE05BAD61778, 0x3FF8CF3216B5448C, 0x3FF8E06A5E0866D9,
    0x3FF8F1AE99157736, 0x3FF902FED0282C8A, 0x3FF9145B0B91FFC6, 0x3FF925C353AA2FE2,
    0x3FF93737B0CDC5E5, 0x3FF948B82B5F98E5, 0x3FF95A44CBC8520F, 0x3FF96BDD9A7670B3,
    0x3FF97D829FDE4E50, 0x3FF98F33E47A22A2, 0x3FF9A0F170CA07BA, 0x3FF9B2BB4D53FE0D,
    0x3FF9C49182A3F090, 0x3FF9D674194BB8D5, 0x3FF9E86319E32323, 0x3FF9FA5E8D07F29E,
    0x3FFA0C667B5DE565, 0x3FFA1E7AED8EB8BB, 0x3FFA309BEC4A2D33, 0x3FFA42C980460AD8,
    0x3FFA5503B23E255D, 0x3FFA674A8AF46052, 0x3FFA799E1330B358, 0x3FFA8BFE53C12E59,
    0x3FFA9E6B5579FDBF, 0x3FFAB0E521356EBA, 0x3FFAC36BBFD3F37A, 0x3FFAD5FF3A3C2774,
    0x3FFAE89F995AD3AD, 0x3FFAFB4CE622F2FF, 0x3FFB0E07298DB666, 0x3FFB20CE6C9A8952,
    0x3FFB33A2B84F15FB, 0x3FFB468415B749B1, 0x3FFB59728DE5593A, 0x3FFB6C6E29F1C52A,
    0x3FFB7F76F2FB5E47, 0x3FFB928CF22749E4, 0x3FFBA5B030A1064A, 0x3FFBB8E0B79A6F1F,
    0x3FFBCC1E904BC1D2, 0x3FFBDF69C3F3A207, 0x3FFBF2C25BD71E09, 0x3FFC06286141B33D,
    0x3FFC199BDD85529C, 0x3FFC2D1CD9FA652C, 0x3FFC40AB5FFFD07A, 0x3FFC544778FAFB22,
    0x3FFC67F12E57D14B, 0x3FFC7BA88988C933, 0x3FFC8F6D9406E7B5, 0x3FFCA3405751C4DB,
    0x3FFCB720DCEF9069, 0x3FFCCB0F2E6D1675, 0x3FFCDF0B555DC3FA, 0x3FFCF3155B5BAB74,
    0x3FFD072D4A07897C, 0x3FFD1B532B08C968, 0x3FFD2F87080D89F2, 0x3FFD43C8EACAA1D6,
    0x3FFD5818DCFBA487, 0x3FFD6C76E862E6D3, 0x3FFD80E316C98398, 0x3FFD955D71FF6075,
    0x3FFDA9E603DB3285, 0x3FFDBE7CD63A8315, 0x3FFDD321F301B460, 0x3FFDE7D5641C0658,
    0x3FFDFC97337B9B5F, 0x3FFE11676B197D17, 0x3FFE264614F5A129, 0x3FFE3B333B16EE12,
    0x3FFE502EE78B3FF6, 0x3FFE653924676D76, 0x3FFE7A51FBC74C83, 0x3FFE8F7977CDB740,
    0x3FFEA4AFA2A490DA, 0x3FFEB9F4867CCA6E, 0x3FFECF482D8E67F1, 0x3FFEE4AAA2188510,
    0x3FFEFA1BEE615A27, 0x3FFF0F9C1CB6412A, 0x3FFF252B376BBA97, 0x3FFF3AC948DD7274,
    0x3FFF50765B6E4540, 0x3FFF6632798844F8, 0x3FFF7BFDAD9CBE14, 0x3FFF91D802243C89,
    0x3FFFA7C1819E90D8, 0x3FFFBDBA3692D514, 0x3FFFD3C22B8F71F1, 0x3FFFE9D96B2A23D9,
]);
static EXP_TAB: &[u64; 256] = &EXP_TAB_A.0;

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
    if let Ok(v) = std::env::var("DYNPOIS_SELFTEST") {
        // DYNPOIS_SELFTEST=1: kernel thread counts 1, 2, 3, 4 and 7;
        // =threads:A,B,...: those thread counts
        let nts: Vec<i64> = match v.strip_prefix("threads:") {
            Some(list) => list.split(',').map(|x| x.parse().expect("DYNPOIS_SELFTEST=threads:A,B,...")).collect(),
            None => vec![1, 2, 3, 4, 7],
        };
        for nt in nts {
            FORCE_NT.store(nt, Ordering::Relaxed);
            println!("selftest: kernel threads = {nt}");
            selftest(Some(&d));
        }
        return;
    }
    let dim = d.dim();
    DATA.set(d).ok();
    // optional second argument: the seed (default 7)
    let seed = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    common::sample_and_print(logp, constrain, dim, &["pop", "beta", "shared", "innov"], &[-1, g as i64, t as i64, (g * t) as i64], 1000, 1000, 4, seed);
}

