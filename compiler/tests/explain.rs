//! `mintc explain` on every example and test program: it must compile
//! exactly what `emit` compiles (byte-identical IR under every flag set
//! tried), report every `~` statement and function, and report the key
//! decisions of the programs the compiler work targets.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn mintc(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_mintc")).current_dir(root()).args(args).output().expect("run mintc")
}

fn explain(file: &str, flags: &[&str]) -> String {
    let mut args = vec!["explain", file];
    args.extend(flags);
    let out = mintc(&args);
    assert!(out.status.success(), "explain {file} {flags:?} failed:\n{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).expect("utf-8 report")
}

#[track_caller]
fn has(out: &str, needle: &str) {
    assert!(out.contains(needle), "expected {needle:?} in:\n{out}");
}

#[track_caller]
fn lacks(out: &str, needle: &str) {
    assert!(!out.contains(needle), "did not expect {needle:?} in:\n{out}");
}

/// examples/*.mint and tests/*/*.mint (not the programs that must fail).
/// The Kalman test models (tests/kalman) are templates that their scripts
/// fill in (DATA, DRAWS, WARMUP, SEED); they are instantiated once into a
/// temporary directory, the lines unchanged, and that copy is compiled.
fn programs() -> Vec<String> {
    static ALL: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    ALL.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("mint-explain-templates-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        programs_as_written()
            .into_iter()
            .map(|p| {
                let src = std::fs::read_to_string(root().join(&p)).unwrap();
                if !src.contains("DRAWS") {
                    return p;
                }
                let filled = src.replace("DATA", "data").replace("DRAWS", "4").replace("WARMUP", "0").replace("SEED", "1");
                let path = dir.join(p.replace('/', "_"));
                std::fs::write(&path, filled).unwrap();
                path.to_str().unwrap().to_string()
            })
            .collect()
    })
    .clone()
}

fn programs_as_written() -> Vec<String> {
    let mut out = Vec::new();
    let mut add = |dir: &Path, rel: &str| {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| {
                let n = e.unwrap().file_name().into_string().unwrap();
                n.ends_with(".mint").then(|| format!("{rel}/{n}"))
            })
            .collect();
        v.sort();
        out.extend(v);
    };
    add(&root().join("examples"), "examples");
    let mut dirs: Vec<String> = std::fs::read_dir(root().join("tests")).unwrap().filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).map(|e| e.file_name().into_string().unwrap()).collect();
    dirs.sort();
    for d in dirs {
        add(&root().join("tests").join(&d), &format!("tests/{d}"));
    }
    assert!(out.len() >= 20, "found only {} programs", out.len());
    out
}

const FLAG_SETS: [&[&str]; 8] = [
    &[],
    &["--strict-fp"],
    &["--no-scan-fusion", "--no-parallel-kernel"],
    &["--no-scan-layout", "--no-narrow-data"],
    &["--no-row-fusion", "--no-gram-blocking", "--no-inline-exp", "--no-inline-log"],
    &["--no-suffstats", "--no-fission-kernel", "--fused-leapfrog"],
    &["--no-fission", "--no-negzero-sums", "--no-vecmath"],
    // the scan, fused-leapfrog and narrow-data paths of the models that
    // would otherwise be collapsed
    &["--no-collapse", "--fused-leapfrog"],
];

/// Both commands come from the same binary, so this shows that explain
/// compiles what emit (and build) compile, not that the code generators
/// are unchanged. It assumes the IR is deterministic: gen_fission_kernel
/// orders its scalar accumulators by a HashMap, which can differ between
/// runs for a fission kernel with two or more scalar parameters (no program
/// here has one).
#[test]
fn explain_compiles_what_emit_compiles() {
    let dir = std::env::temp_dir().join(format!("mint-explain-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (a, b) = (dir.join("emit.ll"), dir.join("explain.ll"));
    let (a, b) = (a.to_str().unwrap(), b.to_str().unwrap());
    for p in programs() {
        for flags in FLAG_SETS {
            let mut e = vec!["emit", p.as_str(), "-o", a];
            e.extend(flags);
            assert!(mintc(&e).status.success(), "emit {p} {flags:?}");
            let mut x = vec!["explain", p.as_str(), "-o", b];
            x.extend(flags);
            assert!(mintc(&x).status.success(), "explain {p} {flags:?}");
            assert!(std::fs::read(a).unwrap() == std::fs::read(b).unwrap(), "{p} {flags:?}: explain compiled different IR from emit");
        }
    }
    for f in ["emit.ll", "explain.ll"] {
        let _ = std::fs::remove_file(dir.join(f));
    }
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn every_statement_and_function_is_reported() {
    for p in programs() {
        let src = std::fs::read_to_string(root().join(&p)).unwrap();
        for flags in FLAG_SETS {
            let out = explain(&p, flags);
            has(&out, "switches: ");
            let lines: Vec<&str> = out.lines().collect();
            for (k, l) in src.lines().enumerate() {
                let code = l.split("//").next().unwrap().trim();
                if let Some(rest) = code.strip_prefix("model ") {
                    has(&out, &format!("model {}", rest.trim_end_matches('{').trim()));
                }
                if let Some(rest) = code.strip_prefix("fn ") {
                    has(&out, &format!("fn {}(", rest.split('(').next().unwrap()));
                }
                if code.contains(" ~ ") {
                    // reported, with at least one line about how it was compiled
                    let head = format!("    line {}: ", k + 1);
                    let at = lines.iter().position(|x| x.starts_with(&head)).unwrap_or_else(|| panic!("{p} {flags:?}: no report for line {}:\n{out}", k + 1));
                    assert!(lines.get(at + 1).is_some_and(|n| n.starts_with("      ")), "{p} {flags:?}: line {} has no details:\n{out}", k + 1);
                }
            }
        }
    }
}

#[test]
fn dynamic_poisson() {
    let f = "examples/dynamic_poisson.mint";
    let out = explain(f, &[]);
    has(&out, "parameters: NUTS samples G x T + G + T + 1 unconstrained values");
    has(&out, "innov   Matrix[G, T]  G x T  unconstrained; scan layout inside the sampler");
    has(&out, "Matrix[G, T]: scan layout, because a running sum runs along T: rows in blocks of 8");
    has(&out, "data y: copied into the scan layout when sample() starts");
    has(&out, "line 17: innov ~ Normal(0, 0.08)\n      absorbed into the fused scan kernel of line 20");
    has(&out, "line 20: y ~ PoissonLog(beta + state)\n      with state = cumsum(shared + innov, T)");
    has(&out, "fused scan kernel over Matrix[G, T]: 1 running sum along T");
    has(&out, "vector code: 4 lanes (<4 x double>), one row per lane, in groups of 8 rows (2 vectors)");
    has(&out, "B, exp over the scratch with nothing else live (the exp split)");
    has(&out, "absorbed: line 17 (innov ~ Normal(0, 0.08)), in the reverse loop");
    has(&out, "owned gradient: innov");
    has(&out, "gradients of beta (indexed by G): summed in registers");
    has(&out, "gradients of shared (indexed by T): per-lane partial sums");
    has(&out, "threads: the groups of 8 rows are split across the chain's threads");
    has(&out, "math: Mint's exp (4 lanes)");
    has(&out, "narrow data (checked when sample() starts):\n    y: tries int8, then int16, then float");
    has(&out, "4 variants of logp are compiled");
    has(&out, "sample() at line 25 in fn main: G, T known at run time");

    let off = explain(f, &["--no-scan-fusion"]);
    has(&off, "switches: --no-scan-fusion");
    lacks(&off, "fused scan kernel over");
    lacks(&off, "absorbed");
    lacks(&off, "owned gradient");
    has(&off, "fused scan kernel: off (--no-scan-fusion)");
    has(&off, "running sum of (shared + innov) materialised before the loop (column by column in the scan layout)");
    has(&off, "line 17: innov ~ Normal(0, 0.08)\n      one flat loop over the G x T elements");
    // the vector kernels are gone, so nothing is read narrow
    lacks(&off, "narrow data (checked");
    has(&off, "narrow data: none, no data is read by Mint's own vector kernels");

    let row_major = explain(f, &["--no-scan-layout"]);
    has(&row_major, "Matrix[G, T]: row-major (--no-scan-layout)");
    has(&row_major, "not a fused scan kernel: its shape is not in the scan layout");
    has(&row_major, "four rows interleaved");

    has(&explain(f, &["--no-parallel-kernel"]), "threads: one (parallel kernel off, --no-parallel-kernel)");
    has(&explain(f, &["--strict-fp"]), "threads: one (parallel kernel off, --strict-fp)");
    has(&explain(f, &["--no-narrow-data"]), "off (--no-narrow-data); the vector kernels read y as doubles");
    has(&explain(f, &["--no-inline-exp"]), "math: llvm.exp (4 lanes)");
    has(&explain(f, &["--fused-leapfrog"]), "fused leapfrog (--fused-leapfrog): the leap entry point runs the sampler's leaf work on innov from inside the fused scan kernel");
    // without the parallel kernel the hook runs on the calling thread
    let lf = explain(f, &["--fused-leapfrog", "--no-parallel-kernel"]);
    has(&lf, "threads: one (parallel kernel off, --no-parallel-kernel)");
    lacks(&lf, "to the kernel threads");
}

#[test]
fn logistic_newton() {
    let f = "examples/logistic_newton.mint";
    let out = explain(f, &[]);
    has(&out, "fn fit(X: Matrix[n, p], y: Vector[n], lambda: Positive) -> Vector[p]");
    has(&out, "row fusion: lines 14 to 16 run as one loop over chunks of 32 rows of X, so each chunk of X is read from memory once");
    has(&out, "line 14: mu = sigmoid(X * w): one value per row");
    has(&out, "line 14: mu = sigmoid(X * w): one value per row; the chunk's dot products X * w first");
    has(&out, "line 15: g = X' * (mu - y) + lambda * w: X' * (mu - y) by row updates");
    has(&out, "when p is not a multiple of 4 (checked at run time), computed in the Gram kernel's first padding column");
    has(&out, "line 16: H = X' * diag(mu .* (1 - mu)) * X + lambda * I(p): tiled Gram kernel: chunks of 32 rows");
    has(&out, "4 x 12 and 4 x 8 tiles");
    has(&out, "solve(H, g): Cholesky solve (mint_chol_solve), allowed because H is proved SPD:");
    // the reasoning chain, each step below the one it supports
    has(
        &out,
        "        H = X' * diag(mu .* (1 - mu)) * X + lambda * I(p) is SPD: PSD + SPD
          X' * diag(mu .* (1 - mu)) * X is PSD: a Gram product with weights >= 0 (the weights are Prob)
            mu .* (1 - mu) is Prob: Prob .* Prob
              mu = sigmoid(X * w) is Prob: sigmoid is always in (0, 1)
              1 - mu is Prob: 1 - Prob
                mu is Prob (shown above)
          lambda * I(p) is SPD: Positive * SPD
            lambda is Positive: declared Positive
            I(p) is SPD: the identity",
    );

    let off = explain(f, &["--no-row-fusion"]);
    has(&off, "row fusion: off (--no-row-fusion); lines 14 to 16 each stream the rows of X separately");
    lacks(&off, "run as one loop over chunks");
    has(&off, "X * w: row dot products, 4 rows at a time");
    has(&off, "X' * diag(mu .* (1 - mu)) * X: tiled Gram kernel");

    let nb = explain(f, &["--no-gram-blocking"]);
    has(&nb, "row fusion: lines 14 to 15 run as one loop");
    has(&nb, "Gram kernel, 1 row(s) of X per pass, upper triangle only, mirrored at the end, the weights computed per row and never stored (--no-gram-blocking)");

    let strict = explain(f, &["--strict-fp"]);
    has(&strict, "row fusion: off (--strict-fp)");
    has(&strict, "Gram kernel, 4 row(s) of X per pass");
    has(&strict, "(the tiled kernel is off under --strict-fp)");
    has(&strict, "is proved SPD:");
}

#[test]
fn logistic_bayes_and_linear() {
    let out = explain("examples/logistic_bayes.mint", &[]);
    has(&out, "parameters: NUTS samples p + 1 unconstrained values");
    has(&out, "loop fission: BernoulliLogit needs exp or log");
    has(&out, "fission kernel: one loop over chunks of 32 rows; per chunk, the row dot products of X * beta (4 rows at a time)");
    has(&out, "the density's exp(-|eta|) runs first, in a loop of its own over the chunk");
    has(&out, "rows left over: the last n mod 32 in groups of 4");
    has(&out, "math: Mint's exp (4 lanes), Mint's log1p (4 lanes)");
    has(&out, "the outcome is checked to be 0 or 1 when sample() starts");
    has(&out, "y: tries int8; ");
    has(&out, "(a BernoulliLogit outcome, checked to be 0 or 1, so int8 only)");
    has(&out, "X: tries float; ");
    let nk = explain("examples/logistic_bayes.mint", &["--no-fission-kernel"]);
    has(&nk, "fission kernel: off (--no-fission-kernel)");
    has(&nk, "separate passes over the n elements: the row dot products of X * beta (4 rows at a time), then an elementwise loop for the density and its derivatives (left to LLVM), then the gradient row updates of X * beta");
    has(&explain("examples/logistic_bayes.mint", &["--no-fission"]), "loop fission: off (--no-fission)");

    let lin = explain("examples/linear_bayes.mint", &[]);
    has(&lin, "sufficient statistics: Normal with a data outcome");
    has(&lin, "Z = [1 for alpha, X (p columns) for beta], q = p + 1");
    has(&lin, "init computes Z'Z ((p + 1) x (p + 1), one triangle, mirrored)");
    has(&lin, "each gradient: O(q^2) = O((p + 1)^2)");
    has(&lin, "sigma  Positive   1  sampled as log; exp maps it back, log-Jacobian added");
    has(&explain("examples/linear_bayes.mint", &["--no-suffstats"]), "sufficient statistics: off (--no-suffstats)");

    let es = explain("examples/eight_schools.mint", &[]);
    has(&es, "sample() at line 21 in fn main: J = 8, so NUTS samples 10 values");
    has(&es, "sufficient statistics: no, the scale is not one scalar");
    has(&es, "scale s is Positive: data declared Positive[J]");
    has(&explain("tests/fission/normal.mint", &[]), "scale exp(X * s) is Positive: exp is always Positive");
}

/// Runs explain on a program written to a temporary file.
fn explain_src(name: &str, src: &str, flags: &[&str]) -> String {
    let dir = std::env::temp_dir().join(format!("mint-explain-src-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.mint"));
    std::fs::write(&path, src).unwrap();
    let out = explain(path.to_str().unwrap(), flags);
    let _ = std::fs::remove_file(&path);
    out
}

/// Report lines that once described code the compiler does not emit.
#[test]
fn reports_only_what_is_emitted() {
    // a product of data only has no gradient pass, in either fission path
    let src = "model D {
    data X: Matrix[n, p]
    data y: Vector[n]
    data w0: Vector[p]
    param a: Real
    a ~ Normal(0, 1)
    y ~ BernoulliLogit(a * (X * w0))
}
fn main() {
    let X: Matrix[n, p] = read(\"x.f64\")
    let y: Vector[n] = read(\"y.f64\")
    let w0: Vector[p] = read(\"w.f64\")
    print(sample(D(X, y, w0), draws = 4, warmup = 0, chains = 1, seed = 1))
}
";
    let k = explain_src("dataprod", src, &[]);
    has(&k, "fission kernel: one loop over chunks of 32 rows; per chunk, the row dot products of X * w0 (4 rows at a time), then the density and its derivatives on 4 rows per vector; all in");
    lacks(&k, "gradient updates");
    let p = explain_src("dataprod", src, &["--no-fission-kernel"]);
    has(&p, "the row dot products of X * w0");
    lacks(&p, "gradient row updates");

    // a running sum of a vector is one sequential pass
    // (a Gaussian random walk, so only without the Kalman collapse)
    let vsrc = "model V {
    data y: Vector[n]
    param z: Vector[n]
    param s: Positive
    z ~ Normal(0, 1)
    s ~ Normal(0, 1)
    y ~ Normal(cumsum(z), s)
}
fn main() {
    let y: Vector[n] = read(\"y.f64\")
    print(sample(V(y), draws = 4, warmup = 0, chains = 1, seed = 1))
}
";
    let v = explain_src("vcumsum", vsrc, &["--no-collapse"]);
    has(&v, "running sum of z materialised before the loop (one sequential pass)");
    has(&v, "scale s is Positive: param declared Positive");
    // collapsed: one series
    let vc = explain_src("vcumsum", vsrc, &[]);
    has(&vc, "Kalman collapse: z (Vector[n], n latent scalars) integrated out: given the other parameters, the one series is a local-level model");
    has(&vc, "one series, so the filter's variance recursion runs once (mint_kalman_ll_shared)");
    has(&vc, "mint_kalman_ll_shared filters the series,");
    has(&vc, "parameters: NUTS samples 1 unconstrained values");
    lacks(&vc, "running sum of z");

    // a row group without exp reports no exp; a norm is >= 0 by being a norm
    let r = explain_src(
        "rownoexp",
        "fn f(X: Matrix[n, p], y: Vector[n], v: Vector[p]) -> Vector[p] {
    let w = ones(p)
    let mut out = zeros(p)
    repeat 1 {
        let mu = X * w
        let g = X' * (mu - y)
        let H = X' * X + (1 + norm(v)) * I(p)
        out = solve(H, g)
    }
    out
}
fn s(A: SPD[p], z: Vector[p]) -> Vector[p] {
    solve(A, z)
}
fn main() {
    let X: Matrix[n, p] = read(\"x.f64\")
    let y: Vector[n] = read(\"y.f64\")
    print(f(X, y, ones(p)))
    print(s(I(p), ones(p)))
}
",
        &[],
    );
    has(&r, "row fusion: lines 5 to 7 run as one loop");
    lacks(&r, "math: Mint's exp");
    lacks(&r, "exp inline");
    has(&r, "X' * X is PSD: a Gram product");
    has(&r, "norm(v) is NonNeg: a norm is always >= 0");
    has(&r, "  line 13: returns solve(A, z)\n    solve(A, z): Cholesky solve (mint_chol_solve), allowed because A is proved SPD:\n      A is SPD: declared SPD[p]");
}

#[test]
fn scan_and_fission_test_models() {
    // (v is a Gaussian random walk, which the Kalman collapse would
    // integrate out; the scan kernels as written need --no-collapse)
    let th = explain("tests/scan/twohosts.mint", &["--no-collapse"]);
    has(&th, "gradient of u accumulated in memory, not owned: line 11 also uses it");
    has(&th, "gradient of u accumulated in memory, not owned: lines 8, 10 also use it");
    has(&th, "Kalman collapse: off (--no-collapse)");
    // collapsed, the filter adds to u's gradient after its kernel
    let tc = explain("tests/scan/twohosts.mint", &[]);
    has(&tc, "gradient of u accumulated in memory, not owned: the Kalman filter that integrates out v (line 11) also adds to it, after the kernel");
    has(&tc, "parameters: NUTS samples G x T unconstrained values");
    has(&explain("tests/scan/datascan.mint", &[]), "not a fused scan kernel: a running sum of data only");
    let b = explain("tests/scan/bernoulli.mint", &[]);
    has(&b, "scalar code, one row at a time: BernoulliLogit has no vector form in this kernel");
    has(&b, "threads: one (only vector kernels are split)");
    has(&b, "gradients of beta (indexed by G): summed in registers per group of rows");
    has(&explain("tests/scan/nested.mint", &[]), "2 running sums along T");
    has(&explain("tests/fission/log1p.mint", &[]), "not a fission kernel: log1p or a running sum has no vector form in it");
    has(&explain("tests/fission/funcs.mint", &[]), "X * b (4 times, once per occurrence)");
    has(&explain("tests/rowfuse/fallback.mint", &[]), "per-row values one row at a time: log1p has no vector form");
}

/// The Kalman collapse: what was integrated out, by which filter, what NUTS
/// samples, and why a candidate was refused, on the statements where
/// detect_kalman decided.
#[test]
fn kalman_collapse() {
    let f = "examples/random_walk_panel.mint";
    let out = explain(f, &[]);
    has(&out, "switches: defaults");
    has(&out, "parameters: NUTS samples G + 3 unconstrained values");
    has(&out, "innov    Matrix[G, T]  G x T  not sampled by NUTS: integrated out by a Kalman filter (line 25), drawn by FFBS for each kept draw");
    has(&out, "line 23: innov ~ Normal(0, sigma_w)\n      scale sigma_w is Positive: param declared Positive\n      integrated out with line 25 (Kalman collapse of innov)");
    has(&out, "line 25: y ~ Normal(beta + cumsum(innov, T), sigma_y)");
    has(&out, "Kalman collapse: innov (Matrix[G, T], G x T latent scalars) integrated out: given the other parameters, each of the G series (rows) is a local-level model, state x[t] = x[t-1] + e[t] with e[t] ~ Normal(0, sigma_w) independent, observed as y ~ Normal(beta + x[t], sigma_y)");
    has(&out, "so they are the same for every series: the variance recursion runs once per time step and only the means are filtered per series (mint_kalman_ll_shared)");
    has(&out, "then mint_kalman_ll_shared filters every series, vectorised across series, and returns the log density and the adjoints of its inputs; a second loop pushes those back through the expressions to beta, sigma_w, sigma_y");
    has(&out, "the collapse function draws innov from its posterior given that draw of the other parameters, by forward filtering, backward sampling (mint_kalman_ffbs)");
    has(&out, "draws: G x T + G + 3 values each, every parameter as written: the G + 3 values NUTS samples, and innov (G x T) drawn for each kept draw");
    lacks(&out, "not integrated out");
    // as written: every parameter is NUTS's, and the report says so
    let off = explain(f, &["--no-collapse"]);
    has(&off, "switches: --no-collapse");
    has(&off, "parameters: NUTS samples G x T + G + 3 unconstrained values");
    has(&off, "Kalman collapse: off (--no-collapse); the parameters inside running sums are sampled by NUTS as written");
    lacks(&off, "integrated out");
    lacks(&off, "draws: ");

    // refused: the observation is PoissonLog (both walks: shared and innov)
    let dp = explain("examples/dynamic_poisson.mint", &[]);
    has(&dp, "Kalman collapse: innov not integrated out: its observation is PoissonLog, not Normal (that needs a Laplace approximation, which is not implemented)");
    has(&dp, "Kalman collapse: shared not integrated out: its observation is PoissonLog, not Normal");
    has(&dp, "parameters: NUTS samples G x T + G + T + 1 unconstrained values");
    lacks(&dp, "draws: ");

    // per-series scales: the general filter
    let g = explain_src(
        "kalgeneral",
        "model K {
    data y: Matrix[G, T]
    param s: Vector[G]
    param innov: Matrix[G, T]
    s ~ Normal(0, 1)
    innov ~ Normal(0, 0.1)
    y ~ Normal(cumsum(innov, T), exp(s))
}
fn main() {
    let y: Matrix[G, T] = read(\"y.f64\")
    print(sample(K(y), draws = 4, warmup = 0, chains = 1, seed = 1))
}
",
        &[],
    );
    has(&g, "the innovation or observation variance has a term indexed by element or by G: one full filter per series, vectorised across series (mint_kalman_ll)");
    has(&g, "parameters: NUTS samples G unconstrained values");
    // refused: no parameter would be left for NUTS
    let z = explain_src(
        "kalnothing",
        "model Z {
    data y: Matrix[G, T]
    param innov: Matrix[G, T]
    innov ~ Normal(0, 0.3)
    y ~ Normal(cumsum(innov, T), 0.5)
}
fn main() {
    let y: Matrix[G, T] = read(\"y.f64\")
    print(sample(Z(y), draws = 4, warmup = 0, chains = 1, seed = 1))
}
",
        &[],
    );
    has(&z, "line 5: y ~ Normal(cumsum(innov, T), 0.5)");
    has(&z, "Kalman collapse: innov not integrated out: no other parameter would be left for NUTS to sample");
    has(&z, "parameters: NUTS samples G x T unconstrained values");

    // with the fused leapfrog, the line about it says what the runtime decides
    let lf = explain("examples/dynamic_poisson.mint", &["--fused-leapfrog"]);
    has(&lf, "at run time the sampler uses it unless MINT_FUSED_LEAPFROG=0, and with MINT_METRIC=lowrank the low-rank part of each leaf runs in a pass after the kernel");
}
