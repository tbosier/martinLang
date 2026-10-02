//! Shared helpers for the Rust baselines: .f64 reading and the FFI surface of
//! the Mint runtime's NUTS sampler (which stands in for a sampler crate, so
//! that both sides of the benchmark run the identical sampler).
#![allow(dead_code)]

use std::ffi::{c_char, c_void, CString};

pub fn read_f64(path: &str) -> (usize, usize, Vec<f64>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let rows = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let cols = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let data = bytes[16..]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(data.len(), rows * cols, "{path} is truncated");
    (rows, cols, data)
}

pub type LogpFn = extern "C" fn(*const f64, *mut f64) -> f64;
pub type ConstrainFn = extern "C" fn(*const f64, *mut f64);

extern "C" {
    fn mint_sample(
        f: LogpFn,
        c: ConstrainFn,
        d: i64,
        draws: i64,
        warmup: i64,
        chains: i64,
        seed: i64,
        nparams: i64,
        names: *const *const c_char,
        sizes: *const i64,
    ) -> *mut c_void;
    fn mint_print_posterior(p: *mut c_void);
    fn mint_set_prep_seconds(s: f64);
}

/// Reports time spent preparing before sampling (printed with the sampling time).
pub fn set_prep_seconds(s: f64) {
    unsafe { mint_set_prep_seconds(s) }
}

/// The value of environment variable `name` as an integer, or `dflt` when it
/// is unset. bench/same_sampler sets MINT_BASELINE_{DRAWS,WARMUP,CHAINS,SEED}
/// so that every baseline runs with the same sampler settings as the Mint
/// and Stan programs it is compared with; unset, each baseline keeps its own.
fn env_or(name: &str, dflt: i64) -> i64 {
    match std::env::var(name) {
        Ok(v) => v.parse().unwrap_or_else(|_| panic!("{name} must be an integer, got {v:?}")),
        Err(_) => dflt,
    }
}

/// sizes[j] < 0 marks a scalar parameter.
pub fn sample_and_print(f: LogpFn, c: ConstrainFn, d: usize, names: &[&str], sizes: &[i64], draws: i64, warmup: i64, chains: i64, seed: i64) {
    let draws = env_or("MINT_BASELINE_DRAWS", draws);
    let warmup = env_or("MINT_BASELINE_WARMUP", warmup);
    let chains = env_or("MINT_BASELINE_CHAINS", chains);
    let seed = env_or("MINT_BASELINE_SEED", seed);
    if std::env::vars().any(|(k, _)| k.starts_with("MINT_BASELINE_")) {
        eprintln!("baseline: draws={draws} warmup={warmup} chains={chains} seed={seed}");
    }
    let cnames: Vec<CString> = names.iter().map(|n| CString::new(*n).unwrap()).collect();
    let ptrs: Vec<*const c_char> = cnames.iter().map(|c| c.as_ptr()).collect();
    unsafe {
        let p = mint_sample(f, c, d as i64, draws, warmup, chains, seed, names.len() as i64, ptrs.as_ptr(), sizes.as_ptr());
        mint_print_posterior(p);
    }
}
