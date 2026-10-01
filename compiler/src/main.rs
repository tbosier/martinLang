//! mintc: compiler driver.
//!
//!   mintc check  prog.mint
//!   mintc emit   prog.mint [-o prog.ll] [flags]
//!   mintc build  prog.mint [-o prog]    [flags]
//!
//! flags: --strict-fp      no reassociation, FMA contraction or vector math
//!        --no-suffstats   disable the sufficient-statistics rewrite
//!        --no-fission     keep each likelihood in one fused loop
//!        --no-vecmath     do not use glibc's vector math library
//!        --no-gram-blocking  one row of A per pass in the Gram kernel
//!        --no-scan-layout    keep scanned matrices row-major
//!        --no-inline-exp     call the vector math library's exp
//!        --no-scan-fusion    materialise running sums in separate passes
//!        --no-row-fusion     run statements that stream one matrix separately
//!        --no-fission-kernel  split likelihoods as three whole passes, the
//!                            elementwise one vectorised by LLVM
//!        --no-inline-log     call the vector math library's log in the
//!                            fission kernel
//!        --no-parallel-kernel  run fused scan kernels on the calling thread only

mod ast;
mod check;
mod codegen;
mod diag;
mod ir;
mod lexer;
mod model;
mod parser;
mod types;

use std::path::{Path, PathBuf};
use std::process::{exit, Command};

pub fn frontend(src: &str) -> Result<check::TProgram, diag::Diag> {
    let toks = lexer::lex(src)?;
    let prog = parser::Parser::new(toks).program()?;
    check::Checker::new().program(&prog)
}

fn runtime_path() -> PathBuf {
    if let Ok(p) = std::env::var("MINT_RUNTIME") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../runtime/mint_rt.c")
}

/// The runtime is compiled once and cached, keyed by a hash of its source,
/// so a program build only compiles the program's own IR.
fn runtime_object() -> PathBuf {
    use std::hash::{Hash, Hasher};
    let src_path = runtime_path();
    let src = std::fs::read(&src_path).unwrap_or_else(|e| {
        eprintln!("error: cannot read runtime {}: {e}", src_path.display());
        exit(1);
    });
    // The cache key covers the compile flags as well as the source.
    const RT_FLAGS: [&str; 4] = ["-O3", "-march=native", "-fopenmp", "-c"];
    let mut h = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut h);
    RT_FLAGS.hash(&mut h);
    let dir = match std::env::var("MINT_CACHE") {
        Ok(d) => PathBuf::from(d),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR")).join("target/runtime-cache"),
    };
    let obj = dir.join(format!("mint_rt-{:016x}.o", h.finish()));
    if obj.exists() {
        return obj;
    }
    std::fs::create_dir_all(&dir).expect("create runtime cache directory");
    let tmp = dir.join(format!("mint_rt-{}.o.tmp", std::process::id()));
    let status = Command::new("clang")
        .args(RT_FLAGS)
        .arg(&src_path)
        .arg("-o")
        .arg(&tmp)
        .status()
        .expect("run clang");
    if !status.success() {
        eprintln!("error: clang failed to compile the runtime");
        exit(1);
    }
    std::fs::rename(&tmp, &obj).expect("install cached runtime");
    obj
}

fn usage() -> ! {
    eprintln!("usage: mintc (check|emit|build) FILE.mint [-o OUT] [--strict-fp] [--no-suffstats] [--no-fission] [--no-vecmath] [--no-gram-blocking] [--no-scan-layout] [--no-inline-exp] [--no-scan-fusion] [--no-row-fusion] [--no-fission-kernel] [--no-inline-log] [--no-parallel-kernel] [--no-narrow-data]");
    exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        usage();
    }
    let cmd = args[0].as_str();
    let path = &args[1];
    let mut out: Option<String> = None;
    let mut opts = codegen::Opts { strict_fp: false, suffstats: true, fission: true, vecmath: true, gram_block: 4, scan_layout: true, inline_exp: true, scan_fusion: true, row_fusion: true, fission_kernel: true, inline_log: true, parallel_kernel: true, narrow_data: true };
    let mut k = 2;
    while k < args.len() {
        match args[k].as_str() {
            "-o" => {
                k += 1;
                out = args.get(k).cloned();
            }
            "--strict-fp" => opts.strict_fp = true,
            "--no-suffstats" => opts.suffstats = false,
            "--no-fission" => opts.fission = false,
            "--no-vecmath" => opts.vecmath = false,
            "--no-gram-blocking" => opts.gram_block = 1,
            "--no-scan-layout" => opts.scan_layout = false,
            "--no-inline-exp" => opts.inline_exp = false,
            "--no-scan-fusion" => opts.scan_fusion = false,
            "--no-row-fusion" => opts.row_fusion = false,
            "--no-fission-kernel" => opts.fission_kernel = false,
            "--no-inline-log" => opts.inline_log = false,
            "--no-parallel-kernel" => opts.parallel_kernel = false,
            "--no-narrow-data" => opts.narrow_data = false,
            _ => usage(),
        }
        k += 1;
    }
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            exit(1);
        }
    };
    let prog = match frontend(&src) {
        Ok(p) => p,
        Err(d) => {
            eprint!("{}", d.render(path, &src));
            exit(1);
        }
    };
    if cmd == "check" {
        println!("ok");
        return;
    }
    let ll = codegen::compile(&prog, &opts);
    let stem = Path::new(path).with_extension("");
    match cmd {
        "emit" => {
            let o = out.unwrap_or_else(|| format!("{}.ll", stem.display()));
            std::fs::write(&o, ll).expect("write .ll");
        }
        "build" => {
            let o = out.unwrap_or_else(|| stem.display().to_string());
            let ll_path = format!("{o}.ll");
            std::fs::write(&ll_path, ll).expect("write .ll");
            let rt = runtime_object();
            let mut cmd = Command::new("clang");
            cmd.args(["-O3", "-march=native", "-Wno-override-module"]);
            if opts.vecmath && !opts.strict_fp {
                // glibc's vector math library (exp, log, ... on 4 lanes, <= 4 ulp)
                cmd.arg("-fveclib=libmvec");
            }
            cmd.arg(&ll_path).arg(&rt).args(["-o", &o, "-fopenmp", "-lm", "-lpthread"]);
            if opts.vecmath && !opts.strict_fp {
                cmd.arg("-lmvec");
            }
            let status = cmd.status().expect("run clang");
            if !status.success() {
                eprintln!("error: clang failed on generated IR ({ll_path})");
                exit(1);
            }
        }
        _ => usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::frontend;
    use crate::check::{TStmt, TK};
    use crate::types::{Struct, Ty};

    fn let_ty(src: &str, fn_name: &str, var: &str) -> Ty {
        let p = frontend(src).unwrap_or_else(|d| panic!("{}", d.render("t", src)));
        let f = p.fns.iter().find(|f| f.name == fn_name).unwrap();
        for s in &f.body {
            if let TStmt::Let { name, value } = s {
                if name.starts_with(&format!("{var}.")) {
                    return value.ty.clone();
                }
            }
        }
        panic!("no let {var}");
    }

    const HDR: &str = "fn main() {}\n";

    #[test]
    fn newton_hessian_is_proved_spd() {
        let src = format!("{HDR}fn f(X: Matrix[n, p], w: Vector[p], lambda: Positive) -> Vector[p] {{
            let mu = sigmoid(X * w)
            let H = X' * diag(mu .* (1 - mu)) * X + lambda * I(p)
            w
        }}");
        assert!(matches!(let_ty(&src, "f", "H"), Ty::Matrix(_, _, Struct::Spd)));
    }

    #[test]
    fn gram_alone_is_only_psd_and_real_ridge_is_not_spd() {
        let src = format!("{HDR}fn f(X: Matrix[n, p], c: Real) -> Real {{
            let G = X' * X
            let R = X' * X + c * I(p)
            let S = X' * X + 2 * I(p)
            0
        }}");
        assert!(matches!(let_ty(&src, "f", "G"), Ty::Matrix(_, _, Struct::Psd)));
        assert!(matches!(let_ty(&src, "f", "R"), Ty::Matrix(_, _, Struct::Sym)));
        assert!(matches!(let_ty(&src, "f", "S"), Ty::Matrix(_, _, Struct::Spd)));
    }

    #[test]
    fn negative_weights_break_psd() {
        let src = format!("{HDR}fn f(X: Matrix[n, p], v: Vector[n]) -> Real {{
            let G = X' * diag(v) * X
            0
        }}");
        assert!(matches!(let_ty(&src, "f", "G"), Ty::Matrix(_, _, Struct::Sym)));
    }

    #[test]
    fn products_become_kernels() {
        let src = format!("{HDR}fn f(X: Matrix[n, p], w: Vector[p], y: Vector[n]) -> Real {{
            let a = X' * (y - X * w)
            let q = w' * X' * X * w
            0
        }}");
        let p = frontend(&src).unwrap();
        let f = p.fns.iter().find(|f| f.name == "f").unwrap();
        match &f.body[0] {
            TStmt::Let { value, .. } => assert!(matches!(value.kind, TK::MatVec { trans: true, .. })),
            _ => panic!(),
        }
        match &f.body[1] {
            TStmt::Let { value, .. } => assert!(matches!(value.kind, TK::Dot(..))),
            _ => panic!(),
        }
    }

    #[test]
    fn errors_point_at_the_problem() {
        for (src, needle) in [
            ("fn main() { let a = [1, 2]\n let b = [1, 2, 3]\n print(a + b) }", "same length"),
            ("fn main() { let x = 1\n x = 2 }", "not mutable"),
            ("fn main() { let X: Matrix[n, p] = read(\"x\")\n print(solve(X, zeros(p))) }", "square"),
            ("fn main() { print(zeros(q)) }", "expected a dimension"),
            ("fn main() { let X = read(\"x\") }", "needs a type annotation"),
            ("fn main() { repeat 3 { let X: Matrix[n, p] = read(\"x\") } }", "inside `repeat`"),
        ] {
            let e = frontend(src).err().unwrap_or_else(|| panic!("accepted: {src}"));
            assert!(e.msg.contains(needle), "{src}: got '{}'", e.msg);
        }
    }
}
