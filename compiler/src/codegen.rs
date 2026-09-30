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
}

#[derive(Clone)]
enum Loc {
    Scalar(String),
    Int(String),
    Buf(String),
    Post(String),
}

enum Prep {
    Scalar(String),
    Buf(String),
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
}

impl HasFb for Cg<'_> {
    fn fb(&mut self) -> &mut Fb {
        &mut self.f
    }
}

pub fn declare_runtime(m: &mut Module) {
    for d in [
        "declare ptr @mint_alloc(i64)",
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
    let mut cg = Cg { m, f: Fb::new(opts.strict_fp), vars: HashMap::new(), dims: HashMap::new(), prog, gram_block: opts.gram_block, mat_shape: None };
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
                for_range(self, "0", &n, |cg, _| {
                    for st in body {
                        cg.stmt(st);
                    }
                });
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
                Prep::Scalar(v) => v.clone(),
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
            rows_dot_blocked(self, &mb, &vb, &c, &r, &move |cg: &mut Cg, i: &str, s: &str| cg.f.store(s, &d, i));
        } else {
            // dest[k] = sum_i M[i,k] v[i], row by row (four rows per pass over
            // dest) so M is read in storage order; v's elementwise expression is
            // fused in.
            self.f.memzero(self.m, &dest, &c);
            let mut prep = HashMap::new();
            self.ew_prepare(v, &mut prep);
            let coef = |cg: &mut Cg, i: &str| cg.ew_elem(v, &prep, i, None);
            rows_axpy_blocked(self, &mb, &c, &r, &coef, &dest);
        }
    }

    /// dest = A' diag(w) A. Only the upper triangle is accumulated and it is
    /// mirrored at the end. Rows of A are processed four at a time, so each
    /// load/store of a destination row feeds four FMAs (register blocking);
    /// w's elementwise expression is fused in, never materialised.
    fn gram(&mut self, a: &TExpr, w: Option<&TExpr>, dest: &str) {
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
