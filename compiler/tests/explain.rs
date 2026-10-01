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
fn programs() -> Vec<String> {
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

const FLAG_SETS: [&[&str]; 7] = [
    &[],
    &["--strict-fp"],
    &["--no-scan-fusion", "--no-parallel-kernel"],
    &["--no-scan-layout", "--no-narrow-data"],
    &["--no-row-fusion", "--no-gram-blocking", "--no-inline-exp", "--no-inline-log"],
    &["--no-suffstats", "--no-fission-kernel", "--fused-leapfrog"],
    &["--no-fission", "--no-negzero-sums", "--no-vecmath"],
];

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
    has(&explain(f, &["--fused-leapfrog"]), "fused leapfrog (--fused-leapfrog): the leap entry point hands the sampler's leaf work on innov");
}

#[test]
fn logistic_newton() {
    let f = "examples/logistic_newton.mint";
    let out = explain(f, &[]);
    has(&out, "fn fit(X: Matrix[n, p], y: Vector[n], lambda: Positive) -> Vector[p]");
    has(&out, "row fusion: lines 14 to 16 run as one loop over chunks of 32 rows of X, so each chunk of X is read from memory once");
    has(&out, "line 14: mu = sigmoid(X * w): one value per row");
    has(&out, "line 15: g = X' * (mu - y) + lambda * w: X' * f by row updates");
    has(&out, "when p is not a multiple of 4 (checked at run time), computed in the Gram kernel's first padding column");
    has(&out, "line 16: H = X' * diag(mu .* (1 - mu)) * X + lambda * I(p): tiled Gram kernel: chunks of 32 rows");
    has(&out, "4 x 12 and 4 x 8 tiles");
    has(&out, "solve(H, g): Cholesky solve (mint_chol_solve), allowed because H is proved SPD:");
    // the reasoning chain, each step below the one it supports
    has(
        &out,
        "        H = X' * diag(mu .* (1 - mu)) * X + lambda * I(p) is SPD: PSD + SPD
          X' * diag(mu .* (1 - mu)) * X is PSD: A' * diag(w) * A with weights w >= 0 (here Prob)
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
    has(&nk, "three passes over the n elements: the row dot products of X * beta");
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

#[test]
fn scan_and_fission_test_models() {
    let th = explain("tests/scan/twohosts.mint", &[]);
    has(&th, "gradient of u accumulated in memory, not owned: line 11 also uses it");
    has(&th, "gradient of u accumulated in memory, not owned: lines 8, 10 also use it");
    has(&explain("tests/scan/datascan.mint", &[]), "not a fused scan kernel: a running sum of data only");
    let b = explain("tests/scan/bernoulli.mint", &[]);
    has(&b, "scalar code, one row at a time: BernoulliLogit has no vector form in this kernel");
    has(&b, "threads: one (only vector kernels are split)");
    has(&explain("tests/scan/nested.mint", &[]), "2 running sums along T");
    has(&explain("tests/fission/log1p.mint", &[]), "not a fission kernel: log1p or a running sum has no vector form in it");
    has(&explain("tests/fission/funcs.mint", &[]), "X * b (4 times, once per occurrence)");
    has(&explain("tests/rowfuse/fallback.mint", &[]), "per-row values one row at a time: log1p has no vector form");
}
