//! Hierarchical dynamic Poisson panel (bench/dynpois/SPEC.md), all in Rust:
//! the hand-tuned gradient of baselines/dynpois_par.rs (src/dynpois.rs,
//! generated from it) sampled by nuts-rs, the NUTS library nutpie is built
//! on. Each chain runs on its own thread and splits every gradient across a
//! team of DYNPOIS_THREADS threads (default 3; src/team.rs).
//!
//! usage:
//!   rust_nuts sample DATA.f64 SEED diag|lowrank OUT.bin   4 chains, 1000 tune + 1000 draws
//!   rust_nuts grad DATA.f64 THETA.f64                       log density and gradient at a point
//!   rust_nuts selftest DATA.f64                             fast kernel against a scalar reference
//!
//! OUT.bin: u64 chains, draws, G, then chains x draws x (1 + 2G) f64:
//! pop, beta[G], terminal[G] (terminal = state[g, T], without beta). A JSON
//! report goes to stdout.
mod dynpois;
mod team;

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use dynpois::{logp_impl, selftest, Data};
use nuts_rs::rand::rngs::ChaCha8Rng;
use nuts_rs::rand::{Rng, RngExt, SeedableRng};
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, LowRankNutsSettings, Settings,
};

const CHAINS: usize = 4;

thread_local! {
    /// time spent in the log density on this chain's thread (ns), and calls
    static LOGP_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static LOGP_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
const TUNE: u64 = 1000;
const DRAWS: u64 = 1000;

fn read_f64(path: &str) -> (usize, usize, Vec<f64>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let rows = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let cols = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let data: Vec<f64> = bytes[16..].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(data.len(), rows * cols, "{path} is truncated");
    (rows, cols, data)
}

#[derive(Debug, thiserror::Error)]
enum DensityError {
    #[error("log density is not finite")]
    NotFinite,
}
impl LogpError for DensityError {
    fn is_recoverable(&self) -> bool {
        true // treated as a divergence, like a Stan rejection
    }
}

struct Density {
    data: Arc<Data>,
    threads: i64,
    ready: bool,
}

impl HasDims for Density {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([
            ("unconstrained_parameter".to_string(), self.data.dim() as u64),
            ("dim".to_string(), self.data.dim() as u64),
        ])
    }
}

impl CpuLogpFunc for Density {
    type LogpError = DensityError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.data.dim()
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, DensityError> {
        if !self.ready {
            team::set_threads(self.threads); // builds this chain thread's team on first use
            self.ready = true;
        }
        let t0 = Instant::now();
        let lp = unsafe { logp_impl(&self.data, position.as_ptr(), gradient.as_mut_ptr()) };
        LOGP_NS.with(|c| c.set(c.get() + t0.elapsed().as_nanos() as u64));
        LOGP_CALLS.with(|c| c.set(c.get() + 1));
        if lp.is_finite() && gradient.iter().all(|g| g.is_finite()) { Ok(lp) } else { Err(DensityError::NotFinite) }
    }

    fn expand_vector<R: Rng + ?Sized>(&mut self, _rng: &mut R, array: &[f64]) -> Result<Vec<f64>, CpuMathError> {
        Ok(array.to_vec())
    }
}

struct ChainOut {
    draws: Vec<f64>, // DRAWS x (1 + 2G)
    gradients: u64,
    warmup_gradients: u64,
    divergences: u64,
    step_size: f64,
    mean_steps: f64,
    seconds: f64,
    logp_seconds: f64,
    logp_calls: u64,
}

fn run_chain<S: Settings>(settings: S, data: Arc<Data>, threads: i64, seed: u64, chain: u64) -> ChainOut {
    let t0 = Instant::now();
    let (g, t) = (data.g, data.t);
    // nuts-rs's own seeding of chain `chain` (sampler.rs, ChainRunner::new)
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(chain + 1);
    let math = CpuMath::new(Density { data: Arc::clone(&data), threads, ready: false });
    let mut sampler = settings.new_chain(chain, math, &mut rng).expect("cannot create chain");
    // initial point: uniform(-2, 2) on every unconstrained coordinate (Stan's default)
    let dim = data.dim();
    let mut init = vec![0.0; dim];
    let mut gradients = 0u64;
    let mut ok = false;
    for _ in 0..500 {
        for x in init.iter_mut() {
            *x = rng.random_range(-2.0..2.0);
        }
        gradients += 1;
        if sampler.set_position(&init).is_ok() {
            ok = true;
            break;
        }
    }
    assert!(ok, "no valid initial point");
    let w = 1 + 2 * g;
    let mut draws = Vec::with_capacity(DRAWS as usize * w);
    let (mut warmup_gradients, mut divergences, mut steps_sampling) = (0u64, 0u64, 0u64);
    let mut step_size = f64::NAN;
    for i in 0..TUNE + DRAWS {
        let (pos, prog) = sampler.draw().expect("unrecoverable error while sampling");
        gradients += prog.num_steps;
        if i < TUNE {
            assert!(prog.tuning || i + 1 == TUNE);
            warmup_gradients += prog.num_steps;
            continue;
        }
        steps_sampling += prog.num_steps;
        divergences += prog.diverging as u64;
        step_size = prog.step_size;
        draws.push(pos[0]);
        draws.extend_from_slice(&pos[1..1 + g]);
        let shared_total: f64 = pos[1 + g..1 + g + t].iter().sum();
        let innov = &pos[1 + g + t..];
        for gg in 0..g {
            draws.push(shared_total + innov[gg * t..(gg + 1) * t].iter().sum::<f64>());
        }
    }
    ChainOut {
        draws,
        gradients,
        warmup_gradients,
        divergences,
        step_size,
        mean_steps: steps_sampling as f64 / DRAWS as f64,
        seconds: t0.elapsed().as_secs_f64(),
        logp_seconds: LOGP_NS.with(|c| c.get()) as f64 * 1e-9,
        logp_calls: LOGP_CALLS.with(|c| c.get()),
    }
}

fn sample<S: Settings>(settings: S, data: Arc<Data>, seed: u64, out: &str, name: &str) {
    let threads: i64 = std::env::var("DYNPOIS_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let t0 = Instant::now();
    let results: Vec<ChainOut> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..CHAINS as u64)
            .map(|c| {
                let data = Arc::clone(&data);
                s.spawn(move || run_chain(settings, data, threads, seed, c))
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("chain panicked")).collect()
    });
    let wall = t0.elapsed().as_secs_f64();
    let g = data.g;
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("cannot create output"));
    for v in [CHAINS as u64, DRAWS, g as u64] {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    for r in &results {
        for x in &r.draws {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }
    f.flush().unwrap();
    let list = |v: Vec<String>| v.join(", ");
    println!(
        "{{\"sampler\": \"nuts-rs 0.19.0\", \"adaptation\": \"{name}\", \"seed\": {seed}, \"chains\": {CHAINS}, \"tune\": {TUNE}, \"draws\": {DRAWS}, \
\"threads_per_chain\": {threads}, \"sampling_seconds\": {wall}, \"chain_seconds\": [{}], \"gradients\": {}, \"warmup_gradients\": {}, \
\"divergences\": {}, \"step_size\": [{}], \"mean_steps\": [{}], \"maxdepth\": {}, \"logp_seconds\": [{}], \"logp_calls\": [{}]}}",
        list(results.iter().map(|r| format!("{:.3}", r.seconds)).collect()),
        results.iter().map(|r| r.gradients).sum::<u64>(),
        results.iter().map(|r| r.warmup_gradients).sum::<u64>(),
        results.iter().map(|r| r.divergences).sum::<u64>(),
        list(results.iter().map(|r| format!("{}", r.step_size)).collect()),
        list(results.iter().map(|r| format!("{:.2}", r.mean_steps)).collect()),
        10,
        list(results.iter().map(|r| format!("{:.3}", r.logp_seconds)).collect()),
        list(results.iter().map(|r| format!("{}", r.logp_calls)).collect()),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (g, t, y) = read_f64(&args[2]);
    let data = Arc::new(Data::new(g, t, y));
    match args[1].as_str() {
        "sample" => {
            let seed: u64 = args[3].parse().expect("seed");
            match args[4].as_str() {
                "diag" => {
                    let mut s = DiagNutsSettings::default();
                    s.num_tune = TUNE;
                    s.num_draws = DRAWS;
                    s.num_chains = CHAINS;
                    s.seed = seed;
                    assert_eq!(s.maxdepth, 10);
                    sample(s, data, seed, &args[5], "diag (DiagNutsSettings::default)")
                }
                "lowrank" => {
                    let mut s = LowRankNutsSettings::default();
                    s.num_tune = TUNE;
                    s.num_draws = DRAWS;
                    s.num_chains = CHAINS;
                    s.seed = seed;
                    assert_eq!(s.maxdepth, 10);
                    sample(s, data, seed, &args[5], "low rank (LowRankNutsSettings::default)")
                }
                other => panic!("adaptation must be diag or lowrank, got {other}"),
            }
        }
        "grad" => {
            let threads: i64 = std::env::var("DYNPOIS_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
            team::set_threads(threads);
            let (_, _, theta) = read_f64(&args[3]);
            assert_eq!(theta.len(), data.dim());
            let mut grad = vec![0.0; data.dim()];
            let lp = unsafe { logp_impl(&data, theta.as_ptr(), grad.as_mut_ptr()) };
            let reps = 2000;
            let t0 = Instant::now();
            for _ in 0..reps {
                unsafe { logp_impl(&data, theta.as_ptr(), grad.as_mut_ptr()) };
            }
            let us = 1e6 * t0.elapsed().as_secs_f64() / reps as f64;
            println!("us_per_gradient {us}");
            println!("logp {lp:.17e}");
            print!("grad:");
            for x in &grad {
                print!(" {x:.17e}");
            }
            println!();
        }
        "selftest" => {
            for nt in [1i64, 2, 3, 4, 7] {
                dynpois::FORCE_NT.store(nt, std::sync::atomic::Ordering::Relaxed);
                println!("selftest: kernel threads = {nt}");
                selftest(Some(&data));
            }
        }
        other => panic!("unknown command {other}"),
    }
}
