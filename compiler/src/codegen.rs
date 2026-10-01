//! Lowers typed functions to LLVM IR.
//!
//! Key decisions:
//! * Every vector or matrix value lives in a buffer whose size is known from
//!   its symbolic type. Buffers are allocated once per top-level statement
//!   (hoisted out of `repeat` loops) and freed when the function returns, so
//!   loops never allocate.
//! * Elementwise expression trees are fused into a single loop; only
//!   non-elementwise subterms (products, solves, calls) are materialised.
//! * `X' * v`, `X' * diag(w) * X` and `v' * w` never materialise a transpose
//!   or a diagonal matrix, and Gram products compute one triangle only.

use std::collections::HashMap;

use crate::ast::BinOp;
use crate::check::{bcast_axis, Axis, Func, TExpr, TFn, TProgram, TStmt, TK};
use crate::ir::{fconst, for_range, rows_axpy_blocked, rows_dot_blocked, Fb, HasFb, Module};
use crate::model;
use crate::types::{Dim, Dom, Ty};

pub struct Opts {
    pub strict_fp: bool,
    pub suffstats: bool,
    pub fission: bool,
    pub vecmath: bool,
    /// rows of A per pass in the Gram kernel (1 disables register blocking)
    pub gram_block: usize,
    /// store matrices that are scanned along their last dimension column-major
    pub scan_layout: bool,
    /// Mint's own inlinable exp instead of the vector math library's
    pub inline_exp: bool,
    /// fuse consecutive statements that stream the rows of one matrix
    pub row_fusion: bool,
    /// run scan statements as one fused, row-blocked loop nest
    pub scan_fusion: bool,
    /// run a split likelihood (loop fission) as one loop over chunks of rows,
    /// with the elementwise pass emitted as Mint's own vector code
    pub fission_kernel: bool,
    /// Mint's own vector log in the fission kernel instead of the vector
    /// math library's
    pub inline_log: bool,
}

#[derive(Clone)]
enum Loc {
    Scalar(String),
    Int(String),
    Buf(String),
    Post(String),
}

/// One statement of a fused row group (see `Cg::stmt_list`).
enum RowStmt<'e> {
    /// let name = elementwise(..., X * v, ...), one value per row of X
    Prod { name: String, e: &'e TExpr, mv: &'e TExpr },
    /// let name = X' * f + rest
    Trans { name: String, f: &'e TExpr, rest: Option<&'e TExpr>, x: &'e TExpr, ty: &'e Ty },
    /// let name = X' * diag(w) * X + rest
    Gram { name: String, w: Option<&'e TExpr>, rest: Option<&'e TExpr>, x: &'e TExpr, ty: &'e Ty },
}

impl RowStmt<'_> {
    fn name(&self) -> &str {
        match self {
            RowStmt::Prod { name, .. } | RowStmt::Trans { name, .. } | RowStmt::Gram { name, .. } => name,
        }
    }
    fn ty(&self) -> &Ty {
        match self {
            RowStmt::Prod { e, .. } => &e.ty,
            RowStmt::Trans { ty, .. } | RowStmt::Gram { ty, .. } => ty,
        }
    }
    fn matrix(&self) -> &str {
        let m: &TExpr = match self {
            RowStmt::Prod { mv, .. } => match &mv.kind {
                TK::MatVec { m, .. } => m,
                _ => unreachable!(),
            },
            RowStmt::Trans { x, .. } | RowStmt::Gram { x, .. } => x,
        };
        match &m.kind {
            TK::Var(n) => n.as_str(),
            _ => unreachable!(),
        }
    }
    fn matrix_ty(&self) -> Option<&Ty> {
        match self {
            RowStmt::Prod { mv, .. } => match &mv.kind {
                TK::MatVec { m, .. } => Some(&m.ty),
                _ => None,
            },
            RowStmt::Trans { x, .. } | RowStmt::Gram { x, .. } => Some(&x.ty),
        }
    }
}

fn children(e: &TExpr) -> Vec<&TExpr> {
    match &e.kind {
        TK::Bin(_, a, b) | TK::Dot(a, b) => vec![a, b],
        TK::Neg(a) | TK::Func(_, a) | TK::Transpose(a) | TK::AssumeSpd(a) | TK::Sum(a) | TK::Cumsum(a) | TK::Norm(a) => vec![a],
        TK::MatVec { m, v, .. } => vec![m, v],
        TK::Gram { a, w } => {
            let mut c = vec![&**a];
            if let Some(w) = w {
                c.push(w);
            }
            c
        }
        TK::MatMul { a, b, .. } => vec![a, b],
        TK::Solve { h, g } => vec![h, g],
        TK::VecLit(xs) => xs.iter().collect(),
        TK::Call { args, .. } | TK::ModelInst { data: args, .. } => args.iter().collect(),
        TK::Sample { inst, .. } => vec![inst],
        _ => vec![],
    }
}

fn mentions_any(e: &TExpr, names: &[&str]) -> bool {
    if let TK::Var(n) = &e.kind {
        if names.contains(&n.as_str()) {
            return true;
        }
    }
    children(e).into_iter().any(|c| mentions_any(c, names))
}

/// An elementwise tree over vectors of the row dimension that can be
/// evaluated one row at a time. `mv` receives its one X * v node when it is
/// a producer; group names may appear only as earlier producers.
fn ew_rows_ok<'e>(e: &'e TExpr, group: &[&str], prods: &[&str], x: &str, mv: &mut Option<Option<&'e TExpr>>) -> bool {
    if e.ty.is_scalar() {
        return !mentions_any(e, group);
    }
    match &e.kind {
        TK::Bin(_, a, b) => ew_rows_ok(a, group, prods, x, mv) && ew_rows_ok(b, group, prods, x, mv),
        TK::Neg(a) | TK::Func(_, a) => ew_rows_ok(a, group, prods, x, mv),
        TK::Fill(_) => true,
        TK::Var(n) => !group.contains(&n.as_str()) || prods.contains(&n.as_str()),
        TK::MatVec { m, trans: false, v } => {
            let is_x = matches!(&m.kind, TK::Var(n) if n == x);
            match mv {
                Some(slot @ None) if is_x && !mentions_any(v, group) => {
                    *slot = Some(e);
                    true
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn classify_row_stmt<'e>(name: &str, value: &'e TExpr, group: &[&str], prods: &[&str], x: Option<&str>, vars: &HashMap<String, Loc>) -> Option<RowStmt<'e>> {
    let is_buf_var = |m: &TExpr| match &m.kind {
        TK::Var(n) => matches!(vars.get(n), Some(Loc::Buf(_))) && x.is_none_or(|x| x == n),
        _ => false,
    };
    // X' * f  or  X' diag(w) X, possibly plus a term that does not stream X
    let (core, rest) = match &value.kind {
        TK::Bin(BinOp::Add, a, b) if matches!(a.kind, TK::MatVec { trans: true, .. } | TK::Gram { .. }) => (&**a, Some(&**b)),
        TK::Bin(BinOp::Add, a, b) if matches!(b.kind, TK::MatVec { trans: true, .. } | TK::Gram { .. }) => (&**b, Some(&**a)),
        _ => (value, None),
    };
    if let Some(r) = rest {
        // the fused loop builds the product in the result buffer and adds the
        // other term element by element: no broadcasting, no scalars
        if mentions_any(r, &[name]) || !same_shape(&r.ty, &core.ty) || !same_shape(&value.ty, &core.ty) {
            return None;
        }
    }
    match &core.kind {
        TK::MatVec { m, trans: true, v } if is_buf_var(m) => {
            let xn = match &m.kind {
                TK::Var(n) => n.as_str(),
                _ => unreachable!(),
            };
            let mut none = None;
            if !ew_rows_ok(v, group, prods, xn, &mut none) {
                return None;
            }
            Some(RowStmt::Trans { name: name.to_string(), f: v, rest, x: m, ty: &value.ty })
        }
        TK::Gram { a, w } if is_buf_var(a) => {
            let xn = match &a.kind {
                TK::Var(n) => n.as_str(),
                _ => unreachable!(),
            };
            if let Some(w) = w {
                let mut none = None;
                if !ew_rows_ok(w, group, prods, xn, &mut none) {
                    return None;
                }
            }
            Some(RowStmt::Gram { name: name.to_string(), w: w.as_deref(), rest, x: a, ty: &value.ty })
        }
        _ if rest.is_none() && matches!(value.ty, Ty::Vector(..)) => {
            // a producer: find its X
            let mut slot: Option<Option<&TExpr>> = Some(None);
            let xn = x.map(str::to_string).or_else(|| find_matvec_x(value, vars))?;
            if !ew_rows_ok(value, group, prods, &xn, &mut slot) {
                return None;
            }
            let mv = slot.flatten()?;
            Some(RowStmt::Prod { name: name.to_string(), e: value, mv })
        }
        _ => None,
    }
}

/// Same vector length or matrix dimensions (domains and structure aside).
fn same_shape(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        (Ty::Vector(n, _), Ty::Vector(m, _)) => n == m,
        (Ty::Matrix(r, c, _), Ty::Matrix(r2, c2, _)) => r == r2 && c == c2,
        _ => false,
    }
}

fn find_matvec_x(e: &TExpr, vars: &HashMap<String, Loc>) -> Option<String> {
    if let TK::MatVec { m, trans: false, .. } = &e.kind {
        if let TK::Var(n) = &m.kind {
            if matches!(vars.get(n), Some(Loc::Buf(_))) {
                return Some(n.clone());
            }
        }
    }
    children(e).into_iter().find_map(|c| find_matvec_x(c, vars))
}

/// Rows per chunk of the tiled Gram kernel (the chunk's two scratch copies
/// then fit in L1 for up to about 64 columns).
const GRAM_CHUNK: u32 = 32;

enum Prep {
    Scalar(String),
    Buf(String),
    /// row i of matrix m (c columns) dotted with vector v, computed per element
    RowDot { m: String, v: String, c: String },
}

pub struct Cg<'a> {
    pub m: &'a mut Module,
    pub f: Fb,
    vars: HashMap<String, Loc>,
    dims: HashMap<String, String>,
    prog: &'a TProgram,
    gram_block: usize,
    /// Shape of the matrix loop currently being emitted, so vectors inside it
    /// can be broadcast along the dimension with the matching name.
    mat_shape: Option<(Dim, Dim)>,
    /// fuse consecutive statements that stream the rows of one matrix
    row_fusion: bool,
}

impl HasFb for Cg<'_> {
    fn fb(&mut self) -> &mut Fb {
        &mut self.f
    }
}

pub fn declare_runtime(m: &mut Module) {
    for d in [
        // fresh memory, like malloc: LLVM may assume it overlaps nothing else
        "declare noalias ptr @mint_alloc(i64)",
        "declare void @mint_free(ptr)",
        "declare double @mint_clock()",
        "declare void @mint_check_dim(i64, i64, ptr)",
        "declare void @mint_check_domain(ptr, i64, i64, ptr)",
        "declare ptr @mint_read_matrix(ptr, ptr, ptr)",
        "declare ptr @mint_read_vector(ptr, ptr)",
        "declare void @mint_print_str(ptr)",
        "declare void @mint_print_sep()",
        "declare void @mint_print_newline()",
        "declare void @mint_print_f64(double)",
        "declare void @mint_print_vec(ptr, i64)",
        "declare void @mint_print_mat(ptr, i64, i64)",
        "declare void @mint_print_posterior(ptr)",
        "declare void @mint_chol_solve(ptr, i64, ptr, ptr, ptr)",
        "declare void @mint_check_spd(ptr, i64, ptr)",
        "declare void @mint_set_prep_seconds(double)",
        "declare void @mint_check_binary(double, i64, ptr)",
        "declare void @mint_check_count(double, i64, ptr)",
        "declare ptr @mint_sample(ptr, ptr, i64, i64, i64, i64, i64, i64, ptr, ptr)",
    ] {
        m.declare(d);
    }
}

pub fn compile(p: &TProgram, opts: &Opts) -> String {
    let mut m = Module::default();
    m.inline_exp = opts.inline_exp && !opts.strict_fp;
    m.inline_log = opts.inline_log && !opts.strict_fp;
    #[cfg(target_arch = "x86_64")]
    {
        m.avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    }
    declare_runtime(&mut m);
    for tm in &p.models {
        model::gen_model(&mut m, tm, opts);
    }
    for f in &p.fns {
        gen_fn(&mut m, p, f, opts);
    }
    m.funcs.push("define i32 @main() {\nentry:\n  call void @mint_fn_main()\n  ret i32 0\n}\n".into());
    m.finish()
}

fn gen_fn(m: &mut Module, prog: &TProgram, f: &TFn, opts: &Opts) {
    let mut cg = Cg { m, f: Fb::new(opts.strict_fp), vars: HashMap::new(), dims: HashMap::new(), prog, gram_block: opts.gram_block, mat_shape: None, row_fusion: opts.row_fusion && !opts.strict_fp };
    let mut params = Vec::new();
    for (k, (name, ty)) in f.params.iter().enumerate() {
        let a = format!("%a{k}");
        match ty {
            Ty::Scalar(_) => {
                params.push(format!("double {a}"));
                let slot = cg.f.alloca("double");
                cg.f.emit(format!("store double {a}, ptr {slot}"));
                cg.vars.insert(name.clone(), Loc::Scalar(slot));
            }
            Ty::Int => {
                params.push(format!("i64 {a}"));
                let slot = cg.f.alloca("i64");
                cg.f.emit(format!("store i64 {a}, ptr {slot}"));
                cg.vars.insert(name.clone(), Loc::Int(slot));
            }
            _ => {
                params.push(format!("ptr {a}"));
                cg.vars.insert(name.clone(), Loc::Buf(a));
            }
        }
    }
    for d in &f.dim_params {
        params.push(format!("i64 %dim.{d}"));
        cg.dims.insert(d.clone(), format!("%dim.{d}"));
    }
    let ret_ty = match &f.ret {
        Some(t) if t.is_buffer() => {
            params.push("ptr %out".into());
            "void"
        }
        Some(_) => "double",
        None => "void",
    };
    for st in &f.body {
        cg.f.begin_hoist();
        cg.stmt(st);
        cg.f.end_hoist();
    }
    let mut ret_val = None;
    if let Some(t) = &f.tail {
        cg.f.begin_hoist();
        if t.ty.is_buffer() {
            cg.gen_into(t, "%out");
        } else {
            ret_val = Some(cg.gen_scalar(t));
        }
        cg.f.end_hoist();
    }
    let mut epi: Vec<String> = cg.f.frees.iter().map(|p| format!("call void @mint_free(ptr {p})")).collect();
    epi.push(match ret_val {
        Some(v) => format!("ret double {v}"),
        None => "ret void".into(),
    });
    let header = format!("define {ret_ty} @mint_fn_{}({})", f.name, params.join(", "));
    let fb = std::mem::replace(&mut cg.f, Fb::new(true));
    cg.m.funcs.push(fb.finish(&header, &epi));
}

impl Cg<'_> {
    fn dim(&self, d: &Dim) -> String {
        match d {
            Dim::Const(c) => c.to_string(),
            Dim::Sym(s) => self.dims[s].clone(),
        }
    }

    fn size(&mut self, ty: &Ty) -> String {
        match ty {
            Ty::Vector(n, _) => self.dim(n),
            Ty::Matrix(r, c, _) => {
                let (r, c) = (self.dim(r), self.dim(c));
                self.f.imul(&r, &c)
            }
            _ => unreachable!("size of non-buffer"),
        }
    }

    /// Allocates a buffer for `ty` at the start of the current top-level
    /// statement; it is freed when the function returns.
    fn alloc_buf(&mut self, ty: &Ty) -> String {
        let dims: Vec<String> = match ty {
            Ty::Vector(n, _) => vec![self.dim(n)],
            Ty::Matrix(r, c, _) => vec![self.dim(r), self.dim(c)],
            _ => unreachable!(),
        };
        let p = self.f.hoisted(|f| {
            let n = if dims.len() == 2 { f.imul(&dims[0], &dims[1]) } else { dims[0].clone() };
            let p = f.reg();
            f.emit(format!("{p} = call ptr @mint_alloc(i64 {n})"));
            p
        });
        self.f.frees.push(p.clone());
        p
    }

    fn stmt(&mut self, s: &TStmt) {
        match s {
            TStmt::Let { name, value } => {
                let loc = match (&value.kind, &value.ty) {
                    (TK::Read(path), ty) => Loc::Buf(self.read(path, ty)),
                    (_, Ty::Scalar(_)) => {
                        let slot = self.f.alloca("double");
                        let v = self.gen_scalar(value);
                        self.f.emit(format!("store double {v}, ptr {slot}"));
                        Loc::Scalar(slot)
                    }
                    (_, Ty::Int) => {
                        let slot = self.f.alloca("i64");
                        let v = self.gen_int(value);
                        self.f.emit(format!("store i64 {v}, ptr {slot}"));
                        Loc::Int(slot)
                    }
                    (_, Ty::Posterior(_)) => {
                        let slot = self.f.alloca("ptr");
                        let v = self.gen_sample(value);
                        self.f.emit(format!("store ptr {v}, ptr {slot}"));
                        Loc::Post(slot)
                    }
                    (_, ty) if ty.is_buffer() => {
                        let b = self.alloc_buf(ty);
                        self.gen_into(value, &b);
                        Loc::Buf(b)
                    }
                    (_, ty) => unreachable!("let of {ty}"),
                };
                self.vars.insert(name.clone(), loc);
            }
            TStmt::Assign { name, value } => match self.vars[name].clone() {
                Loc::Scalar(slot) => {
                    let v = self.gen_scalar(value);
                    self.f.emit(format!("store double {v}, ptr {slot}"));
                }
                Loc::Int(slot) => {
                    let v = self.gen_int(value);
                    self.f.emit(format!("store i64 {v}, ptr {slot}"));
                }
                Loc::Buf(b) => {
                    // Evaluate into a temporary first: the right-hand side may read `name`.
                    let tmp = self.alloc_buf(&value.ty);
                    self.gen_into(value, &tmp);
                    let n = self.size(&value.ty);
                    self.f.memcpy(self.m, &b, &tmp, &n);
                }
                Loc::Post(_) => unreachable!(),
            },
            TStmt::Repeat { count, body } => {
                let n = self.gen_int(count);
                for_range(self, "0", &n, |cg, _| cg.stmt_list(body));
            }
            TStmt::Print(vals) => {
                for (k, v) in vals.iter().enumerate() {
                    if k > 0 {
                        self.f.emit("call void @mint_print_sep()");
                    }
                    self.print(v);
                }
                self.f.emit("call void @mint_print_newline()");
            }
            TStmt::Expr(e) => match &e.ty {
                Ty::Void => {
                    if let TK::Call { name, args, dims } = &e.kind {
                        self.gen_call(name, args, dims, None);
                    }
                }
                Ty::Scalar(_) => {
                    self.gen_scalar(e);
                }
                Ty::Int => {
                    self.gen_int(e);
                }
                Ty::Posterior(_) => {
                    self.gen_sample(e);
                }
                ty if ty.is_buffer() => {
                    let b = self.alloc_buf(ty);
                    self.gen_into(e, &b);
                }
                _ => {}
            },
        }
    }

    fn read(&mut self, path: &str, ty: &Ty) -> String {
        let s = self.m.string(path);
        let r = self.f.reg();
        let (got, want): (Vec<String>, Vec<Dim>) = match ty {
            Ty::Vector(n, _) => {
                let a = self.f.alloca("i64");
                self.f.emit(format!("{r} = call ptr @mint_read_vector(ptr {s}, ptr {a})"));
                (vec![self.f.load_i64(&a)], vec![n.clone()])
            }
            Ty::Matrix(rows, cols, _) => {
                let a = self.f.alloca("i64");
                let b = self.f.alloca("i64");
                self.f.emit(format!("{r} = call ptr @mint_read_matrix(ptr {s}, ptr {a}, ptr {b})"));
                (vec![self.f.load_i64(&a), self.f.load_i64(&b)], vec![rows.clone(), cols.clone()])
            }
            _ => unreachable!(),
        };
        for (g, d) in got.iter().zip(want) {
            let bound = match &d {
                Dim::Const(c) => Some(c.to_string()),
                Dim::Sym(sym) => self.dims.get(sym).cloned(),
            };
            match bound {
                Some(b) => {
                    let msg = self.m.string(&format!("{path}: dimension {d}"));
                    self.f.emit(format!("call void @mint_check_dim(i64 {g}, i64 {b}, ptr {msg})"));
                }
                None => {
                    if let Dim::Sym(sym) = d {
                        self.dims.insert(sym, g.clone());
                    }
                }
            }
        }
        if let Ty::Vector(n, dom) = ty {
            if *dom != Dom::Real {
                let code = match dom {
                    Dom::Prob => 0,
                    Dom::Positive => 1,
                    Dom::NonNeg => 2,
                    Dom::Real => unreachable!(),
                };
                let n = self.dim(n);
                let msg = self.m.string(path);
                self.f.emit(format!("call void @mint_check_domain(ptr {r}, i64 {n}, i64 {code}, ptr {msg})"));
            }
        }
        self.f.frees.push(r.clone());
        r
    }

    fn print(&mut self, v: &TExpr) {
        match &v.ty {
            Ty::Str => {
                let s = match &v.kind {
                    TK::Str(s) => self.m.string(s),
                    _ => unreachable!(),
                };
                self.f.emit(format!("call void @mint_print_str(ptr {s})"));
            }
            Ty::Scalar(_) | Ty::Int => {
                let x = self.gen_scalar(v);
                self.f.emit(format!("call void @mint_print_f64(double {x})"));
            }
            Ty::Vector(n, _) => {
                let b = self.gen_buf(v);
                let n = self.dim(n);
                self.f.emit(format!("call void @mint_print_vec(ptr {b}, i64 {n})"));
            }
            Ty::Matrix(r, c, _) => {
                let b = self.gen_buf(v);
                let (r, c) = (self.dim(r), self.dim(c));
                self.f.emit(format!("call void @mint_print_mat(ptr {b}, i64 {r}, i64 {c})"));
            }
            Ty::Posterior(_) => {
                let p = match &v.kind {
                    TK::Var(n) => match &self.vars[n] {
                        Loc::Post(slot) => {
                            let slot = slot.clone();
                            self.f.load_ptr(&slot)
                        }
                        _ => unreachable!(),
                    },
                    _ => self.gen_sample(v),
                };
                self.f.emit(format!("call void @mint_print_posterior(ptr {p})"));
            }
            _ => unreachable!(),
        }
    }

    fn gen_sample(&mut self, e: &TExpr) -> String {
        let (inst, draws, warmup, chains, seed) = match &e.kind {
            TK::Sample { inst, draws, warmup, chains, seed } => (inst, *draws, *warmup, *chains, *seed),
            TK::Var(n) => {
                if let Loc::Post(slot) = self.vars[n].clone() {
                    return self.f.load_ptr(&slot);
                }
                unreachable!()
            }
            _ => unreachable!(),
        };
        let (mname, data, dims) = match &inst.kind {
            TK::ModelInst { model, data, dims } => (model, data, dims),
            _ => unreachable!(),
        };
        let tm = self.prog.models.iter().find(|m| &m.name == mname).unwrap();
        for ((dname, dty), arg) in tm.data.iter().zip(data) {
            let g = model::data_global(mname, dname);
            if dty.is_buffer() {
                let p = self.gen_buf(arg);
                self.f.emit(format!("store ptr {p}, ptr {g}"));
            } else {
                let v = self.gen_scalar(arg);
                self.f.emit(format!("store double {v}, ptr {g}"));
            }
        }
        for (sym, d) in tm.dims.iter().zip(dims) {
            let v = self.dim(d);
            let g = model::dim_global(mname, sym);
            self.f.emit(format!("store i64 {v}, ptr {g}"));
        }
        let t0 = self.f.reg();
        self.f.emit(format!("{t0} = call double @mint_clock()"));
        self.f.emit(format!("call void @mint_model_{mname}_init()"));
        let t1 = self.f.reg();
        self.f.emit(format!("{t1} = call double @mint_clock()"));
        let dt = self.f.fsub(&t1, &t0);
        self.f.emit(format!("call void @mint_set_prep_seconds(double {dt})"));
        let r = self.f.reg();
        self.f.emit(format!(
            "{r} = call ptr @mint_model_{mname}_sample(i64 {draws}, i64 {warmup}, i64 {chains}, i64 {seed})"
        ));
        r
    }

    // ------------------------------------------------------------ scalars

    pub fn gen_int(&mut self, e: &TExpr) -> String {
        match &e.kind {
            TK::Num(v) => (*v as i64).to_string(),
            TK::DimVal(d) => self.dim(d),
            TK::Var(n) => match self.vars[n].clone() {
                Loc::Int(slot) => self.f.load_i64(&slot),
                _ => unreachable!(),
            },
            _ => unreachable!("integer expression"),
        }
    }

    pub fn gen_scalar(&mut self, e: &TExpr) -> String {
        match &e.kind {
            TK::Num(v) => fconst(*v),
            TK::DimVal(d) => {
                let d = self.dim(d);
                self.f.sitofp(&d)
            }
            TK::Var(n) => match self.vars[n].clone() {
                Loc::Scalar(slot) => self.f.load_f64(&slot),
                Loc::Int(slot) => {
                    let i = self.f.load_i64(&slot);
                    self.f.sitofp(&i)
                }
                _ => unreachable!(),
            },
            TK::Bin(op, a, b) => {
                let x = self.gen_scalar(a);
                let y = self.gen_scalar(b);
                self.scalar_bin(*op, &x, &y, const_of(b))
            }
            TK::Neg(a) => {
                let x = self.gen_scalar(a);
                self.f.fneg(&x)
            }
            TK::Func(func, a) => {
                let x = self.gen_scalar(a);
                self.scalar_func(*func, &x)
            }
            TK::Sum(v) => self.reduce(v, None, false),
            TK::Dot(a, b) => self.reduce(a, Some(b), false),
            TK::Norm(v) => {
                let s = self.reduce(v, None, true);
                self.f.intrinsic1(self.m, "llvm.sqrt.f64", &s)
            }
            TK::Call { name, args, dims } => self.gen_call(name, args, dims, None).unwrap(),
            TK::Clock => {
                let r = self.f.reg();
                self.f.emit(format!("{r} = call double @mint_clock()"));
                r
            }
            k => unreachable!("scalar codegen for {k:?}"),
        }
    }

    pub fn scalar_bin(&mut self, op: BinOp, a: &str, b: &str, b_const: Option<f64>) -> String {
        match op {
            BinOp::Add => self.f.fadd(a, b),
            BinOp::Sub => self.f.fsub(a, b),
            BinOp::Mul | BinOp::EMul => self.f.fmul(a, b),
            BinOp::Div | BinOp::EDiv => self.f.fdiv(a, b),
            BinOp::Pow => match b_const {
                Some(k) if k == 2.0 => self.f.fmul(a, a),
                Some(k) if k == 3.0 => {
                    let t = self.f.fmul(a, a);
                    self.f.fmul(&t, a)
                }
                Some(k) if k == 1.0 => a.to_string(),
                Some(k) if k == 0.5 => self.f.intrinsic1(self.m, "llvm.sqrt.f64", a),
                _ => self.f.intrinsic2(self.m, "llvm.pow.f64", a, b),
            },
        }
    }

    pub fn scalar_func(&mut self, func: Func, x: &str) -> String {
        scalar_func(&mut self.f, self.m, func, x)
    }

    /// sum(v), dot(a, b) or (with `square`) sum(v .* v), fused with the
    /// elementwise expressions that produce the operands.
    fn reduce(&mut self, a: &TExpr, b: Option<&TExpr>, square: bool) -> String {
        let mut prep = HashMap::new();
        self.ew_prepare(a, &mut prep);
        if let Some(b) = b {
            self.ew_prepare(b, &mut prep);
        }
        let acc = self.f.acc_new(&fconst(0.0));
        match &a.ty {
            Ty::Vector(n, _) => {
                let n = self.dim(n);
                for_range(self, "0", &n, |cg, i| {
                    let x = cg.ew_elem(a, &prep, i, None);
                    let t = match (b, square) {
                        (Some(b), _) => {
                            let y = cg.ew_elem(b, &prep, i, None);
                            cg.f.fmul(&x, &y)
                        }
                        (None, true) => cg.f.fmul(&x, &x),
                        (None, false) => x,
                    };
                    cg.f.acc_add(&acc, &t);
                });
            }
            Ty::Matrix(rd, cd, _) => {
                let (r, c) = (self.dim(rd), self.dim(cd));
                let saved = self.mat_shape.replace((rd.clone(), cd.clone()));
                for_range(self, "0", &r, |cg, i| {
                    for_range(cg, "0", &c, |cg, j| {
                        let row = cg.f.imul(i, &c);
                        let idx = cg.f.iadd(&row, j);
                        let x = cg.ew_elem(a, &prep, &idx, Some((i, j)));
                        cg.f.acc_add(&acc, &x);
                    });
                });
                self.mat_shape = saved;
            }
            _ => unreachable!(),
        }
        self.f.acc_get(&acc)
    }

    fn gen_call(&mut self, name: &str, args: &[TExpr], dims: &[Dim], dest: Option<&str>) -> Option<String> {
        let callee = self.prog.fns.iter().find(|f| f.name == name).unwrap();
        let mut a = Vec::new();
        for (k, arg) in args.iter().enumerate() {
            match &callee.params[k].1 {
                Ty::Scalar(_) => {
                    let v = self.gen_scalar(arg);
                    a.push(format!("double {v}"));
                }
                Ty::Int => {
                    let v = self.gen_int(arg);
                    a.push(format!("i64 {v}"));
                }
                _ => {
                    let p = self.gen_buf(arg);
                    a.push(format!("ptr {p}"));
                }
            }
        }
        for d in dims {
            a.push(format!("i64 {}", self.dim(d)));
        }
        match &callee.ret {
            Some(t) if t.is_buffer() => {
                a.push(format!("ptr {}", dest.unwrap()));
                self.f.emit(format!("call void @mint_fn_{name}({})", a.join(", ")));
                None
            }
            Some(_) => {
                let r = self.f.reg();
                self.f.emit(format!("{r} = call double @mint_fn_{name}({})", a.join(", ")));
                Some(r)
            }
            None => {
                self.f.emit(format!("call void @mint_fn_{name}({})", a.join(", ")));
                None
            }
        }
    }

    // ------------------------------------------------------------ buffers

    /// A pointer to a buffer holding `e`; variables are used in place.
    fn gen_buf(&mut self, e: &TExpr) -> String {
        if let TK::Var(n) = &e.kind {
            if let Loc::Buf(b) = &self.vars[n] {
                return b.clone();
            }
        }
        if let TK::AssumeSpd(inner) = &e.kind {
            let b = self.gen_buf(inner);
            self.check_spd(e, &b);
            return b;
        }
        let b = self.alloc_buf(&e.ty);
        self.gen_into(e, &b);
        b
    }

    fn gen_into(&mut self, e: &TExpr, dest: &str) {
        match &e.kind {
            TK::Var(_) => {
                let src = self.gen_buf(e);
                let n = self.size(&e.ty);
                self.f.memcpy(self.m, dest, &src, &n);
            }
            TK::Fill(v) if *v == 0.0 => {
                let n = self.size(&e.ty);
                self.f.memzero(self.m, dest, &n);
            }
            TK::Fill(_) | TK::Identity(_) | TK::Bin(..) | TK::Neg(_) | TK::Func(..) => self.ew(e, dest),
            TK::AssumeSpd(a) => {
                self.gen_into(a, dest);
                self.check_spd(e, dest);
            }
            TK::Cumsum(a) => {
                // running sum along the last dimension, in place
                self.gen_into(a, dest);
                let (rows, cols) = match &e.ty {
                    Ty::Vector(n, _) => ("1".to_string(), self.dim(n)),
                    Ty::Matrix(r, c, _) => (self.dim(r), self.dim(c)),
                    _ => unreachable!(),
                };
                let dest = dest.to_string();
                for_range(self, "0", &rows, |cg, r| {
                    let base = cg.f.imul(r, &cols);
                    let acc = cg.f.acc_new(&fconst(0.0));
                    for_range(cg, "0", &cols, |cg, k| {
                        let idx = cg.f.iadd(&base, k);
                        let x = cg.f.load(&dest, &idx);
                        let old = cg.f.acc_get(&acc);
                        let s = cg.f.fadd(&old, &x);
                        cg.f.emit(format!("store double {s}, ptr {acc}"));
                        cg.f.store(&s, &dest, &idx);
                    });
                });
            }
            TK::VecLit(items) => {
                for (k, it) in items.iter().enumerate() {
                    let v = self.gen_scalar(it);
                    self.f.store(&v, dest, &k.to_string());
                }
            }
            TK::MatVec { m, trans, v } => self.matvec(m, *trans, v, dest),
            TK::Gram { a, w } => self.gram(a, w.as_deref(), dest),
            TK::MatMul { a, ta, b, tb } => self.matmul(a, *ta, b, *tb, dest),
            TK::Transpose(a) => {
                let src = self.gen_buf(a);
                let (r, c) = match &a.ty {
                    Ty::Matrix(r, c, _) => (self.dim(r), self.dim(c)),
                    _ => unreachable!(),
                };
                let dest = dest.to_string();
                for_range(self, "0", &r, |cg, i| {
                    for_range(cg, "0", &c, |cg, j| {
                        let si = cg.f.imul(i, &c);
                        let si = cg.f.iadd(&si, j);
                        let di = cg.f.imul(j, &r);
                        let di = cg.f.iadd(&di, i);
                        let x = cg.f.load(&src, &si);
                        cg.f.store(&x, &dest, &di);
                    });
                });
            }
            TK::Solve { h, g } => {
                let hb = self.gen_buf(h);
                let gb = self.gen_buf(g);
                let p = match &g.ty {
                    Ty::Vector(n, _) => self.dim(n),
                    _ => unreachable!(),
                };
                let what = if matches!(h.kind, TK::AssumeSpd(_)) { "solve (matrix passed to assume_spd)" } else { "solve" };
                let msg = self.m.string(what);
                self.f.emit(format!("call void @mint_chol_solve(ptr {hb}, i64 {p}, ptr {gb}, ptr {dest}, ptr {msg})"));
            }
            TK::Call { name, args, dims } => {
                self.gen_call(name, args, dims, Some(dest));
            }
            k => unreachable!("buffer codegen for {k:?}"),
        }
    }

    /// Computes scalar subterms and materialises non-elementwise subterms of
    /// an elementwise tree, before the fused loop runs.
    fn ew_prepare(&mut self, e: &TExpr, prep: &mut HashMap<usize, Prep>) {
        let key = e as *const TExpr as usize;
        if prep.contains_key(&key) {
            return; // prepared by the caller (a row dot product in a fused group)
        }
        if e.ty.is_scalar() {
            let v = self.gen_scalar(e);
            prep.insert(key, Prep::Scalar(v));
            return;
        }
        match &e.kind {
            TK::Bin(_, a, b) => {
                self.ew_prepare(a, prep);
                self.ew_prepare(b, prep);
            }
            TK::Neg(a) | TK::Func(_, a) => self.ew_prepare(a, prep),
            TK::Identity(_) | TK::Fill(_) => {}
            _ => {
                let b = self.gen_buf(e);
                prep.insert(key, Prep::Buf(b));
            }
        }
    }

    fn ew_elem(&mut self, e: &TExpr, prep: &HashMap<usize, Prep>, idx: &str, ij: Option<(&str, &str)>) -> String {
        let key = e as *const TExpr as usize;
        if let Some(p) = prep.get(&key) {
            return match p {
                Prep::Scalar(v) => {
                    let v = v.clone();
                    self.f.splat(&v)
                }
                Prep::Buf(b) => {
                    // a vector inside a matrix loop repeats along the other dimension
                    let at = match (&e.ty, ij, &self.mat_shape) {
                        (Ty::Vector(n, _), Some((i, j)), Some((r, c))) => match bcast_axis(n, r, c) {
                            Ok(Axis::Row) => i.to_string(),
                            Ok(Axis::Col) => j.to_string(),
                            Err(_) => unreachable!("checked"),
                        },
                        _ => idx.to_string(),
                    };
                    self.f.load(b, &at)
                }
                Prep::RowDot { m, v, c } => {
                    let (m, v, c) = (m.clone(), v.clone(), c.clone());
                    let row = self.f.imul(idx, &c);
                    let acc = self.f.acc_new(&fconst(0.0));
                    for_range(self, "0", &c, |cg, k| {
                        let a = cg.f.iadd(&row, k);
                        let x = cg.f.load(&m, &a);
                        let y = cg.f.load(&v, k);
                        let t = cg.f.fmul(&x, &y);
                        cg.f.acc_add(&acc, &t);
                    });
                    self.f.acc_get(&acc)
                }
            };
        }
        match &e.kind {
            TK::Bin(op, a, b) => {
                let x = self.ew_elem(a, prep, idx, ij);
                let y = self.ew_elem(b, prep, idx, ij);
                self.scalar_bin(*op, &x, &y, const_of(b))
            }
            TK::Neg(a) => {
                let x = self.ew_elem(a, prep, idx, ij);
                self.f.fneg(&x)
            }
            TK::Func(func, a) => {
                let x = self.ew_elem(a, prep, idx, ij);
                self.scalar_func(*func, &x)
            }
            TK::Fill(v) => fconst(*v),
            TK::Identity(_) => {
                let (i, j) = ij.expect("identity outside a matrix loop");
                let c = self.f.reg();
                self.f.emit(format!("{c} = icmp eq i64 {i}, {j}"));
                let r = self.f.reg();
                self.f.emit(format!("{r} = select i1 {c}, double {}, double {}", fconst(1.0), fconst(0.0)));
                r
            }
            k => unreachable!("elementwise codegen for {k:?}"),
        }
    }

    fn ew(&mut self, e: &TExpr, dest: &str) {
        let mut prep = HashMap::new();
        self.ew_prepare(e, &mut prep);
        let dest = dest.to_string();
        match &e.ty {
            Ty::Vector(n, _) => {
                let n = self.dim(n);
                for_range(self, "0", &n, |cg, i| {
                    let v = cg.ew_elem(e, &prep, i, None);
                    cg.f.store(&v, &dest, i);
                });
            }
            Ty::Matrix(rd, cd, _) => {
                let (r, c) = (self.dim(rd), self.dim(cd));
                let saved = self.mat_shape.replace((rd.clone(), cd.clone()));
                for_range(self, "0", &r, |cg, i| {
                    let row = cg.f.imul(i, &c);
                    for_range(cg, "0", &c, |cg, j| {
                        let idx = cg.f.iadd(&row, j);
                        let v = cg.ew_elem(e, &prep, &idx, Some((i, j)));
                        cg.f.store(&v, &dest, &idx);
                    });
                });
                self.mat_shape = saved;
            }
            _ => unreachable!(),
        }
    }

    fn check_spd(&mut self, e: &TExpr, buf: &str) {
        let p = match &e.ty {
            Ty::Matrix(r, _, _) => self.dim(r),
            _ => unreachable!(),
        };
        let msg = self.m.string(&format!("assume_spd at line {}", e.span.line));
        self.f.emit(format!("call void @mint_check_spd(ptr {buf}, i64 {p}, ptr {msg})"));
    }

    fn mat_dims(&self, m: &TExpr) -> (String, String) {
        match &m.ty {
            Ty::Matrix(r, c, _) => (self.dim(r), self.dim(c)),
            _ => unreachable!(),
        }
    }

    fn matvec(&mut self, m: &TExpr, trans: bool, v: &TExpr, dest: &str) {
        let mb = self.gen_buf(m);
        let (r, c) = self.mat_dims(m);
        let dest = dest.to_string();
        if !trans {
            // dest[i] = sum_k M[i,k] v[k], four rows at a time sharing loads of v
            let vb = self.gen_buf(v);
            let d = dest.clone();
            rows_dot_blocked(self, &mb, &vb, &c, "0", &r, &move |cg: &mut Cg, i: &str, s: &str| cg.f.store(s, &d, i));
        } else {
            // dest[k] = sum_i M[i,k] v[i], row by row (four rows per pass over
            // dest) so M is read in storage order; v's elementwise expression is
            // fused in.
            self.f.memzero(self.m, &dest, &c);
            let mut prep = HashMap::new();
            self.ew_prepare(v, &mut prep);
            let coef = |cg: &mut Cg, i: &str| cg.ew_elem(v, &prep, i, None);
            rows_axpy_blocked(self, &mb, &c, "0", &r, &coef, &dest);
        }
    }

    /// dest = A' diag(w) A. Only the upper triangle is accumulated and it is
    /// mirrored at the end. Rows of A are processed four at a time, so each
    /// load/store of a destination row feeds four FMAs (register blocking);
    /// w's elementwise expression is fused in, never materialised.
    fn gram(&mut self, a: &TExpr, w: Option<&TExpr>, dest: &str) {
        if self.gram_block > 1 && !self.f.strict {
            let ab = self.gen_buf(a);
            let (r, c) = self.mat_dims(a);
            let mut prep = HashMap::new();
            if let Some(w) = w {
                self.ew_prepare(w, &mut prep);
            }
            let st = self.gram_start(&c);
            let ab2 = ab.clone();
            let chunks = self.f.iop("add nsw", &r, &(GRAM_CHUNK - 1).to_string());
            let chunks = self.f.iop("sdiv", &chunks, &GRAM_CHUNK.to_string());
            for_range(self, "0", &chunks, |cg, ch| {
                let i0 = cg.f.imul(ch, &GRAM_CHUNK.to_string());
                let i1 = cg.f.iadd(&i0, &GRAM_CHUNK.to_string());
                let over = cg.f.reg();
                cg.f.emit(format!("{over} = icmp sgt i64 {i1}, {r}"));
                let i1c = cg.f.reg();
                cg.f.emit(format!("{i1c} = select i1 {over}, i64 {r}, i64 {i1}"));
                let nrows = cg.f.iop("sub nsw", &i1c, &i0);
                for_range(cg, "0", &nrows, |cg, ii| {
                    let i = cg.f.iadd(&i0, ii);
                    let wi = match w {
                        Some(w) => cg.ew_elem(w, &prep, &i, None),
                        None => fconst(1.0),
                    };
                    cg.gram_row(&st, &ab2, &c, &i, ii, &wi);
                });
                cg.gram_chunk(&st, &nrows);
            });
            self.gram_finish(&st, &c, dest);
            return;
        }
        let b_rows = self.gram_block;
        let ab = self.gen_buf(a);
        let (r, c) = self.mat_dims(a);
        let dest = dest.to_string();
        let cc = self.f.imul(&c, &c);
        self.f.memzero(self.m, &dest, &cc);
        let mut prep = HashMap::new();
        if let Some(w) = w {
            self.ew_prepare(w, &mut prep);
        }
        // Accumulates rows [i0, i0 + width) into the upper triangle.
        let block = |cg: &mut Cg, i0: &str, width: usize| {
            let mut rows = Vec::new();
            let mut ws = Vec::new();
            for l in 0..width {
                let i = cg.f.iadd(i0, &l.to_string());
                ws.push(match w {
                    Some(w) => cg.ew_elem(w, &prep, &i, None),
                    None => fconst(1.0),
                });
                rows.push(cg.f.imul(&i, &c));
            }
            for_range(cg, "0", &c, |cg, j| {
                let mut ts = Vec::new();
                for l in 0..width {
                    let ij = cg.f.iadd(&rows[l], j);
                    let aij = cg.f.load(&ab, &ij);
                    ts.push(cg.f.fmul(&ws[l], &aij));
                }
                let drow = cg.f.imul(j, &c);
                for_range(cg, j, &c, |cg, k| {
                    let mut sum: Option<String> = None;
                    for l in 0..width {
                        let ik = cg.f.iadd(&rows[l], k);
                        let aik = cg.f.load(&ab, &ik);
                        let p = cg.f.fmul(&ts[l], &aik);
                        sum = Some(match sum {
                            None => p,
                            Some(s) => cg.f.fadd(&s, &p),
                        });
                    }
                    let jk = cg.f.iadd(&drow, k);
                    cg.f.add_to(&dest, &jk, &sum.unwrap());
                });
            });
        };
        let nblocks = self.f.iop("sdiv", &r, &b_rows.to_string());
        for_range(self, "0", &nblocks, |cg, b| {
            let i0 = cg.f.imul(b, &b_rows.to_string());
            block(cg, &i0, b_rows);
        });
        let done = self.f.imul(&nblocks, &b_rows.to_string());
        for_range(self, &done, &r, |cg, i| block(cg, i, 1));
        for_range(self, "0", &c, |cg, j| {
            for_range(cg, "0", j, |cg, k| {
                let kj = cg.f.imul(k, &c);
                let kj = cg.f.iadd(&kj, j);
                let x = cg.f.load(&dest, &kj);
                let jk = cg.f.imul(j, &c);
                let jk = cg.f.iadd(&jk, k);
                cg.f.store(&x, &dest, &jk);
            });
        });
    }

    // ---- row fusion
    //
    // Newton and IRLS iterations compute, one statement after another,
    //   let mu = f(X * w)                      (a producer: one value per row)
    //   let g  = X' * r(mu, ...) + ...         (a transposed product)
    //   let H  = X' * diag(s(mu, ...)) * X + ...  (a weighted Gram product)
    // and each statement streams all of X. When consecutive statements only
    // stream the rows of the same X and only use earlier producers of the
    // group elementwise, they run as one loop over chunks of rows: each
    // chunk of X is read from memory once and used by all of them from L1.

    fn stmt_list(&mut self, body: &[TStmt]) {
        let mut k = 0;
        while k < body.len() {
            if self.row_fusion {
                let n = self.try_row_fusion(&body[k..]);
                if n > 0 {
                    k += n;
                    continue;
                }
            }
            self.stmt(&body[k]);
            k += 1;
        }
    }

    /// Fuses the longest run of fusible statements at the start of `body`;
    /// returns how many it consumed (0: none).
    fn try_row_fusion(&mut self, body: &[TStmt]) -> usize {
        let mut group: Vec<RowStmt> = Vec::new();
        let mut x: Option<String> = None;
        for st in body {
            let TStmt::Let { name, value } = st else { break };
            let names: Vec<&str> = group.iter().map(|r| r.name()).collect();
            let prods: Vec<&str> = group.iter().filter_map(|r| if let RowStmt::Prod { name, .. } = r { Some(name.as_str()) } else { None }).collect();
            let Some(r) = classify_row_stmt(name, value, &names, &prods, x.as_deref(), &self.vars) else { break };
            // the fused loop uses the tiled Gram kernel, so --no-gram-blocking
            // keeps Gram products out of it
            if matches!(r, RowStmt::Gram { .. }) && self.gram_block <= 1 {
                break;
            }
            if x.is_none() {
                x = Some(r.matrix().to_string());
            }
            group.push(r);
        }
        // at least two statements, one of which consumes rows
        while group.len() >= 2 && matches!(group.last(), Some(RowStmt::Prod { .. })) {
            group.pop();
        }
        if group.len() < 2 {
            return 0;
        }
        self.emit_row_group(&group, x.as_deref().unwrap());
        group.len()
    }

    fn emit_row_group(&mut self, group: &[RowStmt], xname: &str) {
        let xb = match &self.vars[xname] {
            Loc::Buf(b) => b.clone(),
            _ => unreachable!(),
        };
        let xty = self.var_ty_of(group, xname);
        let (rd, cd) = match &xty {
            Ty::Matrix(r, c, _) => (r.clone(), c.clone()),
            _ => unreachable!(),
        };
        let (n, c) = (self.dim(&rd), self.dim(&cd));
        // destination buffers, visible to later statements of the group
        let mut dests = Vec::new();
        for r in group {
            let b = self.alloc_buf(r.ty());
            self.vars.insert(r.name().to_string(), Loc::Buf(b.clone()));
            dests.push(b);
        }
        let mut prep = HashMap::new();
        let mut coef_bufs: Vec<Option<String>> = Vec::new();
        let mut grams: Vec<Option<(String, String, String, String)>> = Vec::new();
        let chunk_buf = |cg: &mut Cg| {
            let p = cg.f.hoisted(|f| {
                let p = f.reg();
                f.emit(format!("{p} = call ptr @mint_alloc(i64 {GRAM_CHUNK})"));
                p
            });
            cg.f.frees.push(p.clone());
            p
        };
        for (r, d) in group.iter().zip(&dests) {
            let (mut cbuf, mut gst) = (None, None);
            match r {
                RowStmt::Prod { e, mv, .. } => {
                    // X * v is one dot product per row, computed in the row loop
                    let TK::MatVec { v, .. } = &mv.kind else { unreachable!() };
                    let vb = self.gen_buf(v);
                    prep.insert(*mv as *const TExpr as usize, Prep::RowDot { m: xb.clone(), v: vb, c: c.clone() });
                    self.ew_prepare(e, &mut prep);
                }
                RowStmt::Trans { f, .. } => {
                    self.ew_prepare(f, &mut prep);
                    self.f.memzero(self.m, d, &c);
                    cbuf = Some(chunk_buf(self));
                }
                RowStmt::Gram { w, .. } => {
                    if let Some(w) = w {
                        self.ew_prepare(w, &mut prep);
                    }
                    gst = Some(self.gram_start(&c));
                }
            }
            coef_bufs.push(cbuf);
            grams.push(gst);
        }
        let chunks = self.f.iop("add nsw", &n, &(GRAM_CHUNK - 1).to_string());
        let chunks = self.f.iop("sdiv", &chunks, &GRAM_CHUNK.to_string());
        for_range(self, "0", &chunks, |cg, ch| {
            let i0 = cg.f.imul(ch, &GRAM_CHUNK.to_string());
            let i1 = cg.f.iadd(&i0, &GRAM_CHUNK.to_string());
            let over = cg.f.reg();
            cg.f.emit(format!("{over} = icmp sgt i64 {i1}, {n}"));
            let i1c = cg.f.reg();
            cg.f.emit(format!("{i1c} = select i1 {over}, i64 {n}, i64 {i1}"));
            let nrows = cg.f.iop("sub nsw", &i1c, &i0);
            let base = cg.f.imul(&i0, &c);
            let xc = cg.f.gep(&xb, &base);
            // 1: per row, the producers (a dot product each), then the
            // consumers' coefficients and weights. Each row of X is read from
            // memory here and is in L1 for everything after. This loop has
            // inner loops, so LLVM does not vectorise it and Mint's own
            // (inline) exp is the fast choice in it.
            let per_row = |cg: &mut Cg, ii: &str| {
                let i = cg.f.iadd(&i0, ii);
                for (k, r) in group.iter().enumerate() {
                    match r {
                        RowStmt::Prod { e, .. } => {
                            let v = cg.ew_elem(e, &prep, &i, None);
                            cg.f.store(&v, &dests[k], &i);
                        }
                        RowStmt::Trans { f, .. } => {
                            let v = cg.ew_elem(f, &prep, &i, None);
                            cg.f.store(&v, coef_bufs[k].as_ref().unwrap(), ii);
                        }
                        RowStmt::Gram { w, .. } => {
                            let v = match w {
                                Some(w) => cg.ew_elem(w, &prep, &i, None),
                                None => fconst(1.0),
                            };
                            cg.gram_row(grams[k].as_ref().unwrap(), &xb, &c, &i, ii, &v);
                        }
                    }
                }
            };
            cg.f.scalar_inline_exp = true;
            for_range(cg, "0", &nrows, |cg, ii| per_row(cg, ii));
            cg.f.scalar_inline_exp = false;
            // 2: the consumers' updates
            for (k, r) in group.iter().enumerate() {
                match r {
                    RowStmt::Trans { .. } => {
                        let cb = coef_bufs[k].clone().unwrap();
                        rows_axpy_blocked(cg, &xc, &c, "0", &nrows, &move |cg: &mut Cg, ii: &str| cg.f.load(&cb, ii), &dests[k]);
                    }
                    RowStmt::Gram { .. } => cg.gram_chunk(grams[k].as_ref().unwrap(), &nrows),
                    RowStmt::Prod { .. } => {}
                }
            }
        });
        for (k, r) in group.iter().enumerate() {
            if let RowStmt::Gram { .. } = r {
                self.gram_finish(grams[k].as_ref().unwrap(), &c, &dests[k]);
            }
        }
        // the terms outside X, in statement order
        for (k, r) in group.iter().enumerate() {
            let rest = match r {
                RowStmt::Trans { rest, .. } | RowStmt::Gram { rest, .. } => *rest,
                RowStmt::Prod { .. } => None,
            };
            let Some(rest) = rest else { continue };
            let rb = self.gen_buf(rest);
            let size = self.size(r.ty());
            let d = dests[k].clone();
            for_range(self, "0", &size, |cg, j| {
                let a = cg.f.load(&d, j);
                let b = cg.f.load(&rb, j);
                let s = cg.f.fadd(&a, &b);
                cg.f.store(&s, &d, j);
            });
        }
    }

    /// The type of variable `name` as the group's statements see it.
    fn var_ty_of(&self, group: &[RowStmt], name: &str) -> Ty {
        for r in group {
            if let Some(t) = r.matrix_ty() {
                return t.clone();
            }
        }
        unreachable!("row group without its matrix ({name})")
    }

    // ---- tiled Gram kernel: H = A' diag(w) A, in chunks of rows
    //
    // Each chunk of GRAM_CHUNK rows is copied into L1 scratch twice, as X
    // (the rows) and W (the rows scaled by their weights), with the columns
    // padded with zeros to a multiple of 8 (pp). The upper triangle of the
    // padded H is then updated tile by tile: a 4 x 8 tile lives in eight
    // vector registers while the chunk streams through it, so each step is
    // six loads for eight vector FMAs (the row-by-row form needs a load and a
    // store of H for every four). Padding makes every tile full-size.

    /// Allocates and zeroes the scratch: (X chunk, W chunk, padded H, pp).
    fn gram_start(&mut self, c: &str) -> (String, String, String, String) {
        let c = c.to_string();
        let (xs, ws, hs, pp) = self.f.hoisted(|f| {
            let pp = f.iop("add nsw", &c, "7");
            let pp = f.iop("sdiv", &pp, "8");
            let pp = f.imul(&pp, "8");
            let n = f.imul(&pp, &GRAM_CHUNK.to_string());
            let hh = f.imul(&pp, &pp);
            let mut al = |n: &str| {
                let p = f.reg();
                f.emit(format!("{p} = call ptr @mint_alloc(i64 {n})"));
                p
            };
            let (xs, ws, hs) = (al(&n), al(&n), al(&hh));
            (xs, ws, hs, pp)
        });
        for b in [&xs, &ws, &hs] {
            self.f.frees.push(b.clone());
        }
        let n = self.f.imul(&pp, &GRAM_CHUNK.to_string());
        let hh = self.f.imul(&pp, &pp);
        self.f.memzero(self.m, &xs, &n);
        self.f.memzero(self.m, &ws, &n);
        self.f.memzero(self.m, &hs, &hh);
        (xs, ws, hs, pp)
    }

    /// Copies row i of A (row ii of the chunk) and its weighted copy into the scratch.
    fn gram_row(&mut self, st: &(String, String, String, String), ab: &str, c: &str, i: &str, ii: &str, wi: &str) {
        let (xs, ws, _, pp) = st;
        let src = self.f.imul(i, c);
        let dst = self.f.imul(ii, pp);
        let wi = wi.to_string();
        for_range(self, "0", c, |cg, k| {
            let si = cg.f.iadd(&src, k);
            let x = cg.f.load(ab, &si);
            let di = cg.f.iadd(&dst, k);
            cg.f.store(&x, xs, &di);
            let y = cg.f.fmul(&wi, &x);
            cg.f.store(&y, ws, &di);
        });
    }

    /// Adds the chunk's rows [0, nrows) to the upper triangle of the padded H.
    fn gram_chunk(&mut self, st: &(String, String, String, String), nrows: &str) {
        let (xs, ws, hs, pp) = st.clone();
        let nj = self.f.iop("sdiv", &pp, "4");
        for_range(self, "0", &nj, |cg, jb| {
            let j0 = cg.f.imul(jb, "4");
            let k_start = cg.f.iop("and", &j0, "-8");
            let nk = cg.f.iop("sub nsw", &pp, &k_start);
            let nk = cg.f.iop("sdiv", &nk, "8");
            for_range(cg, "0", &nk, |cg, kb| {
                let k0 = cg.f.imul(kb, "8");
                let k0 = cg.f.iadd(&k0, &k_start);
                let k1 = cg.f.iadd(&k0, "4");
                cg.f.lanes = 4;
                let acc: Vec<String> = (0..8).map(|_| cg.f.acc_new(&fconst(0.0))).collect();
                for_range(cg, "0", nrows, |cg, ii| {
                    let row = cg.f.imul(ii, &pp);
                    let a0 = cg.f.iadd(&row, &k0);
                    let a1 = cg.f.iadd(&row, &k1);
                    let xa = cg.f.load(&xs, &a0);
                    let xb = cg.f.load(&xs, &a1);
                    for jj in 0..4 {
                        let j = cg.f.iadd(&j0, &jj.to_string());
                        let wi = cg.f.iadd(&row, &j);
                        let b = cg.f.load_scalar(&ws, &wi);
                        let b = cg.f.splat(&b);
                        let t0 = cg.f.fmul(&b, &xa);
                        cg.f.acc_add(&acc[2 * jj], &t0);
                        let t1 = cg.f.fmul(&b, &xb);
                        cg.f.acc_add(&acc[2 * jj + 1], &t1);
                    }
                });
                for jj in 0..4 {
                    let j = cg.f.iadd(&j0, &jj.to_string());
                    let hrow = cg.f.imul(&j, &pp);
                    for (h, k) in [(0, &k0), (1, &k1)] {
                        let v = cg.f.acc_get(&acc[2 * jj + h]);
                        let at = cg.f.iadd(&hrow, k);
                        cg.f.add_to(&hs, &at, &v);
                    }
                }
                cg.f.lanes = 1;
            });
        });
    }

    /// dest (c x c) = the upper triangle of the padded H, mirrored.
    fn gram_finish(&mut self, st: &(String, String, String, String), c: &str, dest: &str) {
        let (_, _, hs, pp) = st.clone();
        let dest = dest.to_string();
        for_range(self, "0", c, |cg, j| {
            for_range(cg, j, c, |cg, k| {
                let hj = cg.f.imul(j, &pp);
                let hj = cg.f.iadd(&hj, k);
                let x = cg.f.load(&hs, &hj);
                let a = cg.f.imul(j, c);
                let a = cg.f.iadd(&a, k);
                cg.f.store(&x, &dest, &a);
                let b = cg.f.imul(k, c);
                let b = cg.f.iadd(&b, j);
                cg.f.store(&x, &dest, &b);
            });
        });
    }

    fn matmul(&mut self, a: &TExpr, ta: bool, b: &TExpr, tb: bool, dest: &str) {
        let abuf = self.gen_buf(a);
        let bbuf = self.gen_buf(b);
        let (ar, ac) = self.mat_dims(a);
        let (br, bc) = self.mat_dims(b);
        let (rows, inner) = if ta { (ac.clone(), ar.clone()) } else { (ar.clone(), ac.clone()) };
        let cols = if tb { br.clone() } else { bc.clone() };
        let dest = dest.to_string();
        for_range(self, "0", &rows, |cg, i| {
            for_range(cg, "0", &cols, |cg, j| {
                let acc = cg.f.acc_new(&fconst(0.0));
                for_range(cg, "0", &inner, |cg, k| {
                    let ai = if ta { cg.f.imul(k, &ac) } else { cg.f.imul(i, &ac) };
                    let ai = cg.f.iadd(&ai, if ta { i } else { k });
                    let bi = if tb { cg.f.imul(j, &bc) } else { cg.f.imul(k, &bc) };
                    let bi = cg.f.iadd(&bi, if tb { k } else { j });
                    let x = cg.f.load(&abuf, &ai);
                    let y = cg.f.load(&bbuf, &bi);
                    let t = cg.f.fmul(&x, &y);
                    cg.f.acc_add(&acc, &t);
                });
                let s = cg.f.acc_get(&acc);
                let di = cg.f.imul(i, &cols);
                let di = cg.f.iadd(&di, j);
                cg.f.store(&s, &dest, &di);
            });
        });
    }
}

pub fn const_of(e: &TExpr) -> Option<f64> {
    match &e.kind {
        TK::Num(v) => Some(*v),
        _ => None,
    }
}

pub fn scalar_func(f: &mut Fb, m: &mut Module, func: Func, x: &str) -> String {
    match func {
        Func::Exp => f.intrinsic1(m, "llvm.exp.f64", x),
        Func::Log => f.intrinsic1(m, "llvm.log.f64", x),
        Func::Sqrt => f.intrinsic1(m, "llvm.sqrt.f64", x),
        Func::Abs => f.intrinsic1(m, "llvm.fabs.f64", x),
        Func::Log1p => f.intrinsic1(m, "log1p", x),
        Func::Sigmoid => {
            let nx = f.fneg(x);
            let e = f.intrinsic1(m, "llvm.exp.f64", &nx);
            let d = f.fadd(&fconst(1.0), &e);
            f.fdiv(&fconst(1.0), &d)
        }
    }
}
