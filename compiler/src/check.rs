//! Type checker: resolves names, checks shapes, infers domains and matrix
//! structure, and rewrites operator syntax into explicit kernels (matrix-vector
//! product, Gram product, dot product, ...). Output is a typed tree for codegen.

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::diag::{err, err_help, Diag, Res};
use crate::types::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Func {
    Exp,
    Log,
    Log1p,
    Sqrt,
    Sigmoid,
    Abs,
}

impl Func {
    pub fn name(self) -> &'static str {
        match self {
            Func::Exp => "exp",
            Func::Log => "log",
            Func::Log1p => "log1p",
            Func::Sqrt => "sqrt",
            Func::Sigmoid => "sigmoid",
            Func::Abs => "abs",
        }
    }
    fn from_name(s: &str) -> Option<Func> {
        Some(match s {
            "exp" => Func::Exp,
            "log" => Func::Log,
            "log1p" => Func::Log1p,
            "sqrt" => Func::Sqrt,
            "sigmoid" => Func::Sigmoid,
            "abs" => Func::Abs,
            _ => return None,
        })
    }
    fn dom(self, a: Dom) -> Dom {
        match self {
            Func::Exp => Dom::Positive,
            Func::Sigmoid => Dom::Prob,
            Func::Abs => Dom::NonNeg,
            Func::Sqrt | Func::Log1p => {
                if a <= Dom::Positive {
                    Dom::Positive
                } else if a <= Dom::NonNeg {
                    Dom::NonNeg
                } else {
                    Dom::Real
                }
            }
            Func::Log => Dom::Real,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dist {
    Normal,
    BernoulliLogit,
    PoissonLog,
    Exponential,
}

#[derive(Clone, Debug)]
pub struct TExpr {
    pub kind: TK,
    pub ty: Ty,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TK {
    Num(f64),
    Str(String),
    Var(String),
    VecLit(Vec<TExpr>),
    DimVal(Dim),
    /// Scalar arithmetic, or elementwise arithmetic with scalar broadcasting.
    Bin(BinOp, Box<TExpr>, Box<TExpr>),
    Neg(Box<TExpr>),
    Func(Func, Box<TExpr>),
    MatVec { m: Box<TExpr>, trans: bool, v: Box<TExpr> },
    /// a' * diag(w) * a  (w absent means a' * a)
    Gram { a: Box<TExpr>, w: Option<Box<TExpr>> },
    MatMul { a: Box<TExpr>, ta: bool, b: Box<TExpr>, tb: bool },
    Transpose(Box<TExpr>),
    Identity(Dim),
    Fill(f64),
    Solve { h: Box<TExpr>, g: Box<TExpr> },
    AssumeSpd(Box<TExpr>),
    Sum(Box<TExpr>),
    /// Running sum along the last dimension (time): per row for a matrix.
    Cumsum(Box<TExpr>),
    Dot(Box<TExpr>, Box<TExpr>),
    Norm(Box<TExpr>),
    Call { name: String, args: Vec<TExpr>, dims: Vec<Dim> },
    Read(String),
    ModelInst { model: String, data: Vec<TExpr>, dims: Vec<Dim> },
    Sample { inst: Box<TExpr>, draws: i64, warmup: i64, chains: i64, seed: i64 },
    Clock,
}

#[derive(Clone, Debug)]
pub enum TStmt {
    Let { name: String, value: TExpr },
    Assign { name: String, value: TExpr },
    Repeat { count: TExpr, body: Vec<TStmt> },
    Print(Vec<TExpr>),
    Expr(TExpr),
}

#[derive(Clone, Debug)]
pub struct TFn {
    pub name: String,
    pub params: Vec<(String, Ty)>,
    pub dim_params: Vec<String>,
    pub ret: Option<Ty>,
    pub body: Vec<TStmt>,
    pub tail: Option<TExpr>,
}

#[derive(Clone, Debug)]
pub enum TModelStmt {
    Let { name: String, value: TExpr },
    Tilde { lhs: TExpr, dist: Dist, args: Vec<TExpr>, shape: SShape, span: Span },
}

/// The index space of a `~` statement (or of a scan inside one).
#[derive(Clone, Debug, PartialEq)]
pub enum SShape {
    Scalar,
    Vec(Dim),
    Mat(Dim, Dim),
}

/// How a vector lines up with a matrix of shape [rows, cols], by dimension
/// name: a Vector[rows] repeats across columns, a Vector[cols] across rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Axis {
    Row,
    Col,
}

pub fn bcast_axis(n: &Dim, rows: &Dim, cols: &Dim) -> Result<Axis, String> {
    match (n == rows, n == cols) {
        (true, false) => Ok(Axis::Row),
        (false, true) => Ok(Axis::Col),
        (true, true) => Err(format!(
            "a Vector[{n}] against a Matrix[{rows}, {cols}] is ambiguous: both dimensions are {n}; give them different names"
        )),
        (false, false) => Err(format!("a Vector[{n}] does not line up with either dimension of a Matrix[{rows}, {cols}]")),
    }
}

#[derive(Clone, Debug)]
pub struct TModel {
    pub name: String,
    pub dims: Vec<String>,
    pub data: Vec<(String, Ty)>,
    pub params: Vec<(String, Ty)>,
    pub body: Vec<TModelStmt>,
}

#[derive(Clone, Debug)]
pub struct TProgram {
    pub fns: Vec<TFn>,
    pub models: Vec<TModel>,
}

#[derive(Clone, Debug)]
struct FnSig {
    params: Vec<(String, Ty)>,
    dim_params: Vec<String>,
    ret: Option<Ty>,
}

#[derive(Clone, Debug)]
struct VarInfo {
    uniq: String,
    ty: Ty,
    mutable: bool,
}

pub struct Checker {
    fns: HashMap<String, FnSig>,
    models: HashMap<String, TModel>,
    scopes: Vec<HashMap<String, VarInfo>>,
    dims: HashSet<String>,
    counter: usize,
    /// Inside a model body: names of data and params (for the model checker).
    in_model: bool,
    loop_depth: usize,
}

fn ann_ty(a: &TypeAnn) -> Ty {
    let d = |d: &DimAnn| match d {
        DimAnn::Sym(s) => Dim::Sym(s.clone()),
        DimAnn::Const(c) => Dim::Const(*c),
    };
    match a {
        TypeAnn::Real => Ty::Scalar(Dom::Real),
        TypeAnn::Positive => Ty::Scalar(Dom::Positive),
        TypeAnn::Prob => Ty::Scalar(Dom::Prob),
        TypeAnn::Int => Ty::Int,
        TypeAnn::Vector(n) => Ty::Vector(d(n), Dom::Real),
        TypeAnn::VecOf(n, dom) => Ty::Vector(
            d(n),
            match dom.as_str() {
                "Positive" => Dom::Positive,
                "Prob" => Dom::Prob,
                _ => Dom::Real,
            },
        ),
        TypeAnn::Matrix(r, c) => Ty::Matrix(d(r), d(c), Struct::General),
        TypeAnn::Psd(n) => Ty::Matrix(d(n), d(n), Struct::Psd),
        TypeAnn::Spd(n) => Ty::Matrix(d(n), d(n), Struct::Spd),
    }
}

fn ty_dims(t: &Ty) -> Vec<Dim> {
    match t {
        Ty::Vector(n, _) => vec![n.clone()],
        Ty::Matrix(r, c, _) => vec![r.clone(), c.clone()],
        _ => vec![],
    }
}

fn sym_dims(t: &Ty, out: &mut Vec<String>) {
    for d in ty_dims(t) {
        if let Dim::Sym(s) = d {
            if !out.contains(&s) {
                out.push(s);
            }
        }
    }
}

/// Explains why `actual` does not fit `declared`, or None if it does.
fn misfit(actual: &Ty, declared: &Ty) -> Option<String> {
    match (actual, declared) {
        (Ty::Scalar(a), Ty::Scalar(d)) if a <= d => None,
        (Ty::Int, Ty::Scalar(d)) if Dom::NonNeg <= *d => None,
        (Ty::Int, Ty::Int) => None,
        (Ty::Scalar(a), Ty::Scalar(d)) => Some(format!("a {a} value cannot be used where {d} is required")),
        (Ty::Vector(n1, a), Ty::Vector(n2, d)) => {
            if n1 != n2 {
                Some(format!("length {n1} does not match length {n2}"))
            } else if a > d {
                Some(format!("elements are {a}, but {d} is required"))
            } else {
                None
            }
        }
        (Ty::Matrix(r1, c1, s1), Ty::Matrix(r2, c2, s2)) => {
            if r1 != r2 || c1 != c2 {
                Some(format!("shape [{r1}, {c1}] does not match [{r2}, {c2}]"))
            } else if s1 > s2 {
                Some(format!("the compiler can only prove this matrix is {s1}, not {s2}"))
            } else {
                None
            }
        }
        (a, d) if a == d => None,
        (a, d) => Some(format!("expected {d}, found {a}")),
    }
}

enum Factor {
    Vec(TExpr, bool),
    Mat(TExpr, bool),
    Diag(TExpr),
    Ident(Dim),
}

fn flatten_mul<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match &e.kind {
        ExprKind::Bin(BinOp::Mul, l, r) => {
            flatten_mul(l, out);
            flatten_mul(r, out);
        }
        _ => out.push(e),
    }
}

fn const_of(e: &TExpr) -> Option<f64> {
    match &e.kind {
        TK::Num(v) => Some(*v),
        _ => None,
    }
}

impl Checker {
    pub fn new() -> Self {
        Checker {
            fns: HashMap::new(),
            models: HashMap::new(),
            scopes: vec![],
            dims: HashSet::new(),
            counter: 0,
            in_model: false,
            loop_depth: 0,
        }
    }

    pub fn program(&mut self, p: &Program) -> Res<TProgram> {
        let mut models = Vec::new();
        for m in &p.models {
            let tm = self.model(m)?;
            self.models.insert(m.name.clone(), tm.clone());
            models.push(tm);
        }
        for f in &p.fns {
            if self.fns.contains_key(&f.name) {
                return err(f.span, format!("function `{}` is defined twice", f.name));
            }
            let params: Vec<(String, Ty)> = f.params.iter().map(|p| (p.name.clone(), ann_ty(&p.ann))).collect();
            let mut dim_params = Vec::new();
            for (_, t) in &params {
                sym_dims(t, &mut dim_params);
            }
            let ret = f.ret.as_ref().map(ann_ty);
            if let Some(r) = &ret {
                let mut rd = Vec::new();
                sym_dims(r, &mut rd);
                for d in rd {
                    if !dim_params.contains(&d) {
                        return err(f.span, format!("return type of `{}` uses dimension `{d}`, which no parameter defines", f.name));
                    }
                }
                if matches!(r, Ty::Matrix(..)) || r.is_buffer() || r.is_scalar() {
                } else {
                    return err(f.span, "functions may return scalars, vectors or matrices");
                }
            }
            self.fns.insert(f.name.clone(), FnSig { params, dim_params, ret });
        }
        if !self.fns.contains_key("main") {
            return err(Span { line: 1, col: 1 }, "no `fn main()` found");
        }
        let mut fns = Vec::new();
        for f in &p.fns {
            fns.push(self.function(f)?);
        }
        Ok(TProgram { fns, models })
    }

    fn fresh(&mut self, name: &str) -> String {
        self.counter += 1;
        format!("{name}.{}", self.counter)
    }

    fn lookup(&self, name: &str) -> Option<&VarInfo> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }

    fn bind(&mut self, name: &str, ty: Ty, mutable: bool) -> String {
        let uniq = if self.in_model { name.to_string() } else { self.fresh(name) };
        self.scopes.last_mut().unwrap().insert(name.to_string(), VarInfo { uniq: uniq.clone(), ty, mutable });
        uniq
    }

    fn check_dims_bound(&self, t: &Ty, span: Span) -> Res<()> {
        for d in ty_dims(t) {
            if let Dim::Sym(s) = &d {
                if !self.dims.contains(s) {
                    return err_help(
                        span,
                        format!("unknown dimension `{s}`"),
                        "dimensions are introduced by function parameter types or by `let X: Matrix[n, p] = read(...)`",
                    );
                }
            }
        }
        Ok(())
    }

    fn function(&mut self, f: &FnDecl) -> Res<TFn> {
        let sig = self.fns[&f.name].clone();
        self.dims = sig.dim_params.iter().cloned().collect();
        self.scopes = vec![HashMap::new()];
        let mut params = Vec::new();
        for (name, ty) in &sig.params {
            if matches!(ty, Ty::Matrix(..)) || ty.is_buffer() || ty.is_scalar() {
                let u = self.bind(name, ty.clone(), false);
                params.push((u, ty.clone()));
            }
        }
        let (body, tail) = self.block(&f.body)?;
        let mut tail = tail;
        match (&sig.ret, &tail) {
            (Some(r), Some(t)) => {
                if let Some(why) = misfit(&t.ty, r) {
                    return err(t.span, format!("`{}` must return {r}, but this is {}: {why}", f.name, t.ty));
                }
            }
            (Some(r), None) => return err(f.span, format!("`{}` must end with an expression of type {r}", f.name)),
            (None, Some(_)) => {}
            (None, None) => {}
        }
        let mut body = body;
        if sig.ret.is_none() {
            if let Some(t) = tail.take() {
                body.push(TStmt::Expr(t));
            }
        }
        Ok(TFn { name: f.name.clone(), params, dim_params: sig.dim_params, ret: sig.ret, body, tail })
    }

    fn block(&mut self, b: &Block) -> Res<(Vec<TStmt>, Option<TExpr>)> {
        let mut out = Vec::new();
        for s in &b.stmts {
            out.push(self.stmt(s)?);
        }
        let tail = match &b.tail {
            Some(e) => {
                if let Some(p) = self.print_stmt(e)? {
                    out.push(p);
                    None
                } else {
                    Some(self.expr(e)?)
                }
            }
            None => None,
        };
        Ok((out, tail))
    }

    fn print_stmt(&mut self, e: &Expr) -> Res<Option<TStmt>> {
        if let ExprKind::Call(name, args) = &e.kind {
            if name == "print" {
                let mut vals = Vec::new();
                for a in args {
                    let v = self.expr(&a.value)?;
                    if matches!(v.ty, Ty::Void | Ty::Model(_)) {
                        return err(a.value.span, format!("cannot print {}", v.ty));
                    }
                    vals.push(v);
                }
                return Ok(Some(TStmt::Print(vals)));
            }
        }
        Ok(None)
    }

    fn stmt(&mut self, s: &Stmt) -> Res<TStmt> {
        match s {
            Stmt::Let { name, mutable, ann, value, span } => {
                if let ExprKind::Call(f, args) = &value.kind {
                    if f == "read" {
                        return self.read_let(name, *mutable, ann.as_ref(), args, *span, value.span);
                    }
                }
                let v = self.expr(value)?;
                if matches!(v.ty, Ty::Void) {
                    return err(value.span, "this expression has no value");
                }
                if matches!(v.ty, Ty::Model(_)) {
                    return err_help(value.span, "a model applied to data can only be passed straight to sample", "write sample(MyModel(X, y), ...)");
                }
                let ty = match ann {
                    Some(a) => {
                        let t = ann_ty(a);
                        self.check_dims_bound(&t, *span)?;
                        if let Some(why) = misfit(&v.ty, &t) {
                            return self.annotation_error(&v, &t, &why, *span);
                        }
                        t
                    }
                    None if *mutable => widen(&v.ty),
                    None => v.ty.clone(),
                };
                let u = self.bind(name, ty.clone(), *mutable);
                Ok(TStmt::Let { name: u, value: TExpr { ty, ..v } })
            }
            Stmt::Assign { name, value, span } => {
                let info = match self.lookup(name) {
                    Some(i) => i.clone(),
                    None => return err(*span, format!("unknown variable `{name}`")),
                };
                if !info.mutable {
                    return err_help(*span, format!("`{name}` is not mutable"), format!("declare it with `let mut {name} = ...`"));
                }
                let v = self.expr(value)?;
                if let Some(why) = misfit(&v.ty, &info.ty) {
                    return err(value.span, format!("cannot assign {} to `{name}` of type {}: {why}", v.ty, info.ty));
                }
                Ok(TStmt::Assign { name: info.uniq, value: v })
            }
            Stmt::Repeat { count, body, span } => {
                let c = self.expr(count)?;
                let ok = match (&c.kind, &c.ty) {
                    (TK::Num(v), _) => v.fract() == 0.0 && *v >= 0.0,
                    (_, Ty::Int) => true,
                    _ => false,
                };
                if !ok {
                    return err(*span, "`repeat` needs a whole number of iterations");
                }
                self.scopes.push(HashMap::new());
                self.loop_depth += 1;
                let (mut b, tail) = self.block(body)?;
                self.loop_depth -= 1;
                if let Some(t) = tail {
                    b.push(TStmt::Expr(t));
                }
                self.scopes.pop();
                Ok(TStmt::Repeat { count: c, body: b })
            }
            Stmt::Expr(e) => {
                if let Some(p) = self.print_stmt(e)? {
                    return Ok(p);
                }
                Ok(TStmt::Expr(self.expr(e)?))
            }
        }
    }

    fn annotation_error(&self, v: &TExpr, t: &Ty, why: &str, span: Span) -> Res<TStmt> {
        let help = match (&v.ty, t) {
            (Ty::Matrix(_, _, Struct::Psd), Ty::Matrix(_, _, Struct::Spd)) => {
                "a Gram matrix like X' * X is only PSD (it is singular when columns are dependent); add a positive multiple of the identity, e.g. `+ lambda * I(p)` with `lambda: Positive`, or wrap it in assume_spd(...) to check at run time".to_string()
            }
            (Ty::Scalar(_), Ty::Scalar(_)) | (Ty::Vector(..), Ty::Vector(..)) => {
                "the compiler tracks value ranges through arithmetic: exp and sigmoid give positive values, sums of positives stay positive".to_string()
            }
            _ => String::new(),
        };
        let msg = format!("this value is {}, which does not fit the annotation {t}: {why}", v.ty);
        if help.is_empty() {
            err(span, msg)
        } else {
            err_help(span, msg, help)
        }
    }

    fn read_let(&mut self, name: &str, mutable: bool, ann: Option<&TypeAnn>, args: &[Arg], span: Span, vspan: Span) -> Res<TStmt> {
        if self.loop_depth > 0 {
            return err(span, "read cannot be used inside `repeat`: data sizes must be known before any loop");
        }
        let path = match args {
            [Arg { name: None, value: Expr { kind: ExprKind::Str(s), .. } }] => s.clone(),
            _ => return err(vspan, "read takes one file path, e.g. read(\"data/X.f64\")"),
        };
        let t = match ann {
            Some(a @ (TypeAnn::Vector(_) | TypeAnn::VecOf(..) | TypeAnn::Matrix(..))) => ann_ty(a),
            Some(_) => return err(span, "read can load a Vector[n], Positive[n], Prob[n] or Matrix[m, n]"),
            None => {
                return err_help(
                    span,
                    "read needs a type annotation",
                    format!("write `let {name}: Matrix[n, p] = read(...)`; the dimension names are bound to the file's sizes and checked everywhere after"),
                )
            }
        };
        for d in ty_dims(&t) {
            if let Dim::Sym(s) = d {
                self.dims.insert(s);
            }
        }
        let u = self.bind(name, t.clone(), mutable);
        Ok(TStmt::Let { name: u, value: TExpr { kind: TK::Read(path), ty: t, span: vspan } })
    }

    pub fn expr(&mut self, e: &Expr) -> Res<TExpr> {
        let span = e.span;
        let mk = |kind: TK, ty: Ty| Ok(TExpr { kind, ty, span });
        match &e.kind {
            ExprKind::Num(v, _) => mk(TK::Num(*v), Ty::Scalar(const_dom(*v))),
            ExprKind::Str(s) => mk(TK::Str(s.clone()), Ty::Str),
            ExprKind::VecLit(items) => {
                if items.is_empty() {
                    return err(span, "empty vector literal");
                }
                let mut vals = Vec::new();
                let mut dom = Dom::Prob;
                for it in items {
                    let v = self.expr(it)?;
                    if !v.ty.is_scalar() {
                        return err(it.span, format!("vector literal entries must be numbers, got {}", v.ty));
                    }
                    dom = dom.max(v.ty.dom());
                    vals.push(v);
                }
                let n = vals.len() as i64;
                mk(TK::VecLit(vals), Ty::Vector(Dim::Const(n), dom))
            }
            ExprKind::Var(name) => {
                if let Some(info) = self.lookup(name) {
                    return mk(TK::Var(info.uniq.clone()), info.ty.clone());
                }
                if self.dims.contains(name) {
                    return mk(TK::DimVal(Dim::Sym(name.clone())), Ty::Int);
                }
                if self.models.contains_key(name) {
                    return err_help(span, format!("`{name}` is a model"), format!("apply it to data: {name}(X, y)"));
                }
                err(span, format!("unknown variable `{name}`"))
            }
            ExprKind::Neg(a) => {
                let a = self.expr(a)?;
                let ty = match &a.ty {
                    Ty::Scalar(_) | Ty::Int => Ty::Scalar(Dom::Real),
                    Ty::Vector(n, _) => Ty::Vector(n.clone(), Dom::Real),
                    Ty::Matrix(r, c, s) => Ty::Matrix(r.clone(), c.clone(), if *s <= Struct::Sym { Struct::Sym } else { Struct::General }),
                    t => return err(span, format!("cannot negate {t}")),
                };
                mk(TK::Neg(Box::new(a)), ty)
            }
            ExprKind::Transpose(inner) => {
                let a = self.expr(inner)?;
                match &a.ty {
                    Ty::Matrix(r, c, s) => {
                        if *s <= Struct::Sym {
                            return Ok(a); // symmetric: A' == A
                        }
                        let ty = Ty::Matrix(c.clone(), r.clone(), *s);
                        mk(TK::Transpose(Box::new(a)), ty)
                    }
                    Ty::Vector(..) => err_help(span, "a transposed vector can only appear in a product", "write v' * w for a dot product, or v' * A * w for a quadratic form"),
                    Ty::Scalar(_) => Ok(a),
                    t => err(span, format!("cannot transpose {t}")),
                }
            }
            ExprKind::Bin(BinOp::Mul, _, _) => self.product(e),
            ExprKind::Bin(op, l, r) => {
                let l = self.expr(l)?;
                let r = self.expr(r)?;
                self.elementwise(*op, l, r, span)
            }
            ExprKind::Call(name, args) => self.call(name, args, span),
        }
    }

    fn elementwise(&mut self, op: BinOp, l: TExpr, r: TExpr, span: Span) -> Res<TExpr> {
        let op_dom = |a: Dom, b: Dom, lc: Option<f64>, rc: Option<f64>| match op {
            BinOp::Add => dom_add(a, b),
            BinOp::Sub => dom_sub(lc, b),
            BinOp::Mul | BinOp::EMul => dom_mul(a, b),
            BinOp::Div | BinOp::EDiv => dom_div(a, b),
            BinOp::Pow => dom_pow(a, rc),
        };
        let (lc, rc) = (const_of(&l), const_of(&r));
        let s = op.symbol();
        let ty = match (&l.ty, &r.ty) {
            (a, b) if a.is_scalar() && b.is_scalar() => Ty::Scalar(op_dom(a.dom(), b.dom(), lc, rc)),
            (Ty::Vector(n, a), b) if b.is_scalar() => {
                if op == BinOp::Pow && rc.is_none() {
                    return err(span, "vector powers need a literal exponent, e.g. v ^ 2");
                }
                Ty::Vector(n.clone(), op_dom(*a, b.dom(), lc, rc))
            }
            (a, Ty::Vector(n, b)) if a.is_scalar() => {
                if op == BinOp::Pow {
                    return err(span, "a scalar raised to a vector power is not supported");
                }
                Ty::Vector(n.clone(), op_dom(a.dom(), *b, lc, rc))
            }
            (Ty::Vector(n1, a), Ty::Vector(n2, b)) => {
                if n1 != n2 {
                    return err(span, format!("`{s}` needs vectors of the same length, got {} and {}", l.ty, r.ty));
                }
                match op {
                    BinOp::Div => return err_help(span, "`/` between two vectors is ambiguous", "use ./ for elementwise division"),
                    BinOp::Pow => return err(span, "elementwise powers of two vectors are not supported"),
                    _ => {}
                }
                Ty::Vector(n1.clone(), op_dom(*a, *b, lc, rc))
            }
            (Ty::Matrix(r1, c1, s1), Ty::Matrix(r2, c2, s2)) => {
                if r1 != r2 || c1 != c2 {
                    return err(span, format!("`{s}` needs matrices of the same shape, got {} and {}", l.ty, r.ty));
                }
                let st = match op {
                    BinOp::Add => struct_add(*s1, *s2),
                    BinOp::Sub if *s1 <= Struct::Sym && *s2 <= Struct::Sym => Struct::Sym,
                    BinOp::EMul => struct_hadamard(*s1, *s2),
                    BinOp::Sub | BinOp::EDiv => Struct::General,
                    BinOp::Div => return err_help(span, "`/` between two matrices is not division", "use solve(A, b) for linear systems, or ./ for elementwise division"),
                    _ => return err(span, format!("`{s}` is not supported between matrices")),
                };
                Ty::Matrix(r1.clone(), c1.clone(), st)
            }
            (Ty::Matrix(r1, c1, _), Ty::Vector(n, _)) | (Ty::Vector(n, _), Ty::Matrix(r1, c1, _)) => {
                if let Err(why) = bcast_axis(n, r1, c1) {
                    return err_help(span, format!("`{s}` between {} and {}: {why}", l.ty, r.ty), "a vector is matched to the matrix dimension with the same name");
                }
                if op == BinOp::Pow {
                    return err(span, format!("`{s}` is not supported between a matrix and a vector"));
                }
                Ty::Matrix(r1.clone(), c1.clone(), Struct::General)
            }
            (Ty::Matrix(r1, c1, s1), b) if b.is_scalar() => {
                let st = match op {
                    BinOp::Div => struct_scale(dom_div(Dom::Positive, b.dom()), *s1),
                    _ => Struct::General,
                };
                Ty::Matrix(r1.clone(), c1.clone(), st)
            }
            (a, Ty::Matrix(r1, c1, _)) if a.is_scalar() => Ty::Matrix(r1.clone(), c1.clone(), Struct::General),
            (a, b) => return err(span, format!("`{s}` is not defined between {a} and {b}")),
        };
        Ok(TExpr { kind: TK::Bin(op, Box::new(l), Box::new(r)), ty, span })
    }

    /// Resolves a chain of `*` into scalar scaling, matrix-vector products,
    /// Gram products, dot products and matrix products.
    fn product(&mut self, e: &Expr) -> Res<TExpr> {
        let mut parts = Vec::new();
        flatten_mul(e, &mut parts);
        let mut scalars: Vec<TExpr> = Vec::new();
        let mut rest: Vec<Factor> = Vec::new();
        for p in parts {
            let (base, trans) = match &p.kind {
                ExprKind::Transpose(b) => (&**b, true),
                _ => (p, false),
            };
            if let ExprKind::Call(n, args) = &base.kind {
                if n == "diag" {
                    if args.len() != 1 {
                        return err(base.span, "diag takes one vector");
                    }
                    let v = self.expr(&args[0].value)?;
                    if !matches!(v.ty, Ty::Vector(..)) {
                        return err(base.span, format!("diag takes a vector, got {}", v.ty));
                    }
                    rest.push(Factor::Diag(v));
                    continue;
                }
                if n == "I" {
                    let d = self.dim_arg(args, 0, base.span)?;
                    rest.push(Factor::Ident(d));
                    continue;
                }
            }
            let t = self.expr(base)?;
            match &t.ty {
                ty if ty.is_scalar() => scalars.push(t),
                Ty::Vector(..) => rest.push(Factor::Vec(t, trans)),
                Ty::Matrix(_, _, s) => {
                    let trans = trans && *s > Struct::Sym;
                    rest.push(Factor::Mat(t, trans))
                }
                ty => return err(base.span, format!("cannot multiply {ty}")),
            }
        }
        let span = e.span;
        let mut scalar: Option<TExpr> = None;
        for s in scalars {
            scalar = Some(match scalar {
                None => s,
                Some(acc) => {
                    let ty = Ty::Scalar(dom_mul(acc.ty.dom(), s.ty.dom()));
                    TExpr { kind: TK::Bin(BinOp::Mul, Box::new(acc), Box::new(s)), ty, span }
                }
            });
        }
        if rest.is_empty() {
            return Ok(scalar.unwrap());
        }
        let body = self.chain(rest, span)?;
        Ok(match scalar {
            None => body,
            Some(s) => {
                let ty = match &body.ty {
                    Ty::Vector(n, d) => Ty::Vector(n.clone(), dom_mul(s.ty.dom(), *d)),
                    Ty::Matrix(r, c, st) => Ty::Matrix(r.clone(), c.clone(), struct_scale(s.ty.dom(), *st)),
                    t => Ty::Scalar(dom_mul(s.ty.dom(), t.dom())),
                };
                TExpr { kind: TK::Bin(BinOp::Mul, Box::new(s), Box::new(body)), ty, span }
            }
        })
    }

    fn chain(&mut self, mut f: Vec<Factor>, span: Span) -> Res<TExpr> {
        let shape = |x: &Factor| -> (Dim, Dim) {
            match x {
                Factor::Mat(t, tr) => match &t.ty {
                    Ty::Matrix(r, c, _) if *tr => (c.clone(), r.clone()),
                    Ty::Matrix(r, c, _) => (r.clone(), c.clone()),
                    _ => unreachable!(),
                },
                Factor::Diag(t) | Factor::Vec(t, _) => match &t.ty {
                    Ty::Vector(n, _) => (n.clone(), n.clone()),
                    _ => unreachable!(),
                },
                Factor::Ident(d) => (d.clone(), d.clone()),
            }
        };
        let name = |x: &Factor| -> String {
            match x {
                Factor::Mat(t, tr) => format!("{}{}", t.ty, if *tr { " (transposed)" } else { "" }),
                Factor::Vec(t, _) | Factor::Diag(t) => format!("{}", t.ty),
                Factor::Ident(d) => format!("I({d})"),
            }
        };
        let inner_check = |a: &Factor, b: &Factor| -> Res<()> {
            let (_, ac) = shape(a);
            let (br, _) = shape(b);
            if ac != br {
                return err_help(
                    span,
                    format!("matrix product dimensions do not match: {} times {}", name(a), name(b)),
                    format!("the inner dimensions {ac} and {br} must be equal"),
                );
            }
            Ok(())
        };
        for w in f.windows(2) {
            inner_check(&w[0], &w[1])?;
        }
        if f.len() == 1 {
            return match f.pop().unwrap() {
                Factor::Vec(v, false) => Ok(v),
                Factor::Vec(_, true) => err_help(span, "a transposed vector can only appear in a product", "write v' * w for a dot product"),
                Factor::Mat(m, false) => Ok(m),
                Factor::Mat(m, true) => {
                    let ty = match &m.ty {
                        Ty::Matrix(r, c, s) => Ty::Matrix(c.clone(), r.clone(), *s),
                        _ => unreachable!(),
                    };
                    Ok(TExpr { kind: TK::Transpose(Box::new(m)), ty, span })
                }
                Factor::Ident(d) => Ok(TExpr { kind: TK::Identity(d.clone()), ty: Ty::Matrix(d.clone(), d, Struct::Spd), span }),
                Factor::Diag(_) => err_help(span, "diag(w) can only appear inside a product", "for example X' * diag(w) * X"),
            };
        }
        // Drop identities: I * A == A (dimensions were checked above).
        let ident_dim = match &f[0] {
            Factor::Ident(d) => Some(d.clone()),
            _ => None,
        };
        f.retain(|x| !matches!(x, Factor::Ident(..)));
        if f.is_empty() {
            let d = ident_dim.unwrap();
            return Ok(TExpr { kind: TK::Identity(d.clone()), ty: Ty::Matrix(d.clone(), d, Struct::Spd), span });
        }
        if f.len() == 1 {
            return self.chain(f, span);
        }
        let n = f.len();
        // v' * ... * w  → dot product
        if let (Factor::Vec(_, true), Factor::Vec(_, false)) = (&f[0], &f[n - 1]) {
            let first = f.remove(0);
            let v = match first {
                Factor::Vec(v, _) => v,
                _ => unreachable!(),
            };
            let w = self.chain(f, span)?;
            return Ok(TExpr { kind: TK::Dot(Box::new(v), Box::new(w)), ty: Ty::Scalar(Dom::Real), span });
        }
        for (k, x) in f.iter().enumerate() {
            if let Factor::Vec(_, tr) = x {
                if *tr || k != n - 1 {
                    return err_help(
                        span,
                        "`*` with a vector on the left of other factors is ambiguous",
                        "use dot(a, b) or a' * b for inner products and a .* b for elementwise products",
                    );
                }
            }
        }
        // ... * w  → fold matrix-vector products from the right
        if let Factor::Vec(..) = f[n - 1] {
            let mut acc = match f.pop().unwrap() {
                Factor::Vec(v, _) => v,
                _ => unreachable!(),
            };
            while let Some(x) = f.pop() {
                acc = match x {
                    Factor::Mat(m, trans) => {
                        let rows = shape(&Factor::Mat(m.clone(), trans)).0;
                        TExpr { kind: TK::MatVec { m: Box::new(m), trans, v: Box::new(acc) }, ty: Ty::Vector(rows, Dom::Real), span }
                    }
                    Factor::Diag(d) => {
                        let ty = Ty::Vector(shape(&Factor::Diag(d.clone())).0, dom_mul(d.ty.dom(), acc.ty.dom()));
                        TExpr { kind: TK::Bin(BinOp::EMul, Box::new(d), Box::new(acc)), ty, span }
                    }
                    _ => unreachable!(),
                };
            }
            return Ok(acc);
        }
        // Gram products: A' * diag(w) * A and A' * A
        let same = |a: &TExpr, b: &TExpr| matches!((&a.kind, &b.kind), (TK::Var(x), TK::Var(y)) if x == y);
        if n == 3 {
            if let (Factor::Mat(a, true), Factor::Diag(w), Factor::Mat(b, false)) = (&f[0], &f[1], &f[2]) {
                if same(a, b) {
                    let c = shape(&f[0]).0;
                    let st = if w.ty.dom() <= Dom::NonNeg { Struct::Psd } else { Struct::Sym };
                    return Ok(TExpr {
                        kind: TK::Gram { a: Box::new(a.clone()), w: Some(Box::new(w.clone())) },
                        ty: Ty::Matrix(c.clone(), c, st),
                        span,
                    });
                }
            }
        }
        if n == 2 {
            if let (Factor::Mat(a, true), Factor::Mat(b, false)) = (&f[0], &f[1]) {
                if same(a, b) {
                    let c = shape(&f[0]).0;
                    return Ok(TExpr { kind: TK::Gram { a: Box::new(a.clone()), w: None }, ty: Ty::Matrix(c.clone(), c, Struct::Psd), span });
                }
            }
        }
        // General matrix products, folded left.
        let mut it = f.into_iter();
        let (mut acc, mut acc_t) = match it.next().unwrap() {
            Factor::Mat(m, t) => (m, t),
            Factor::Diag(_) => return err_help(span, "diag(w) is only supported in the form A' * diag(w) * A", "materialising an n-by-n diagonal matrix is almost never what you want"),
            _ => unreachable!(),
        };
        for x in it {
            match x {
                Factor::Mat(b, tb) => {
                    let (r, _) = shape(&Factor::Mat(acc.clone(), acc_t));
                    let (_, c) = shape(&Factor::Mat(b.clone(), tb));
                    acc = TExpr { kind: TK::MatMul { a: Box::new(acc), ta: acc_t, b: Box::new(b), tb }, ty: Ty::Matrix(r, c, Struct::General), span };
                    acc_t = false;
                }
                _ => return err_help(span, "diag(w) is only supported in the form A' * diag(w) * A", "materialising an n-by-n diagonal matrix is almost never what you want"),
            }
        }
        Ok(acc)
    }

    fn dim_arg(&mut self, args: &[Arg], k: usize, span: Span) -> Res<Dim> {
        let a = match args.get(k) {
            Some(a) => a,
            None => return err(span, "expected a dimension argument"),
        };
        match &a.value.kind {
            ExprKind::Num(v, true) if *v >= 0.0 => Ok(Dim::Const(*v as i64)),
            ExprKind::Var(n) if self.dims.contains(n) && self.lookup(n).is_none() => Ok(Dim::Sym(n.clone())),
            _ => err_help(a.value.span, "expected a dimension", "use a dimension name such as `p` or a whole number"),
        }
    }

    fn call(&mut self, name: &str, args: &[Arg], span: Span) -> Res<TExpr> {
        let mk = |kind: TK, ty: Ty| Ok(TExpr { kind, ty, span });
        let arity = |n: usize| -> Res<()> {
            if args.len() != n {
                return err(span, format!("`{name}` takes {n} argument(s), got {}", args.len()));
            }
            if let Some(a) = args.iter().find(|a| a.name.is_some()) {
                return err(a.value.span, format!("`{name}` does not take named arguments"));
            }
            Ok(())
        };
        if let Some(func) = Func::from_name(name) {
            arity(1)?;
            let a = self.expr(&args[0].value)?;
            let ty = match &a.ty {
                t if t.is_scalar() => Ty::Scalar(func.dom(t.dom())),
                Ty::Vector(n, d) => Ty::Vector(n.clone(), func.dom(*d)),
                Ty::Matrix(r, c, _) => Ty::Matrix(r.clone(), c.clone(), Struct::General),
                t => return err(span, format!("`{name}` cannot be applied to {t}")),
            };
            return mk(TK::Func(func, Box::new(a)), ty);
        }
        match name {
            "zeros" | "ones" => {
                let fill = if name == "zeros" { 0.0 } else { 1.0 };
                let d = if fill == 0.0 { Dom::NonNeg } else { Dom::Positive };
                match args.len() {
                    1 => {
                        let n = self.dim_arg(args, 0, span)?;
                        mk(TK::Fill(fill), Ty::Vector(n, d))
                    }
                    2 => {
                        let r = self.dim_arg(args, 0, span)?;
                        let c = self.dim_arg(args, 1, span)?;
                        mk(TK::Fill(fill), Ty::Matrix(r, c, Struct::General))
                    }
                    _ => err(span, format!("`{name}` takes one or two dimensions")),
                }
            }
            "I" => {
                arity(1)?;
                let d = self.dim_arg(args, 0, span)?;
                mk(TK::Identity(d.clone()), Ty::Matrix(d.clone(), d, Struct::Spd))
            }
            "diag" => err_help(span, "diag(w) can only appear inside a product", "for example X' * diag(w) * X"),
            "solve" => {
                arity(2)?;
                let h = self.expr(&args[0].value)?;
                let g = self.expr(&args[1].value)?;
                let (r, c, s) = match &h.ty {
                    Ty::Matrix(r, c, s) => (r.clone(), c.clone(), *s),
                    t => return err(args[0].value.span, format!("solve needs a matrix, got {t}")),
                };
                if r != c {
                    return err(args[0].value.span, format!("solve needs a square matrix, got {}", h.ty));
                }
                if s != Struct::Spd {
                    let why = match s {
                        Struct::Psd => "it is only known to be PSD, which allows it to be singular",
                        Struct::Sym => "it is only known to be symmetric",
                        _ => "nothing is known about its structure",
                    };
                    return err_help(
                        args[0].value.span,
                        format!("solve needs a matrix the compiler can prove is SPD, but {why}"),
                        "SPD comes from I(p), from PSD + (Positive * I(p)), from sums and positive multiples of SPD matrices, or from assume_spd(A), which is checked at run time",
                    );
                }
                match &g.ty {
                    Ty::Vector(n, _) if *n == r => {}
                    t => return err(args[1].value.span, format!("solve's right-hand side must be Vector[{r}], got {t}")),
                }
                mk(TK::Solve { h: Box::new(h), g: Box::new(g) }, Ty::Vector(r, Dom::Real))
            }
            "assume_spd" => {
                arity(1)?;
                let h = self.expr(&args[0].value)?;
                match &h.ty.clone() {
                    Ty::Matrix(r, c, _) if r == c => mk(TK::AssumeSpd(Box::new(h)), Ty::Matrix(r.clone(), c.clone(), Struct::Spd)),
                    t => err(span, format!("assume_spd needs a square matrix, got {t}")),
                }
            }
            "sum" | "norm" => {
                arity(1)?;
                let a = self.expr(&args[0].value)?;
                let ty = match (&a.ty, name) {
                    (Ty::Vector(_, d), "sum") if *d <= Dom::NonNeg => Ty::Scalar(Dom::NonNeg),
                    (Ty::Vector(..) | Ty::Matrix(..), "sum") => Ty::Scalar(Dom::Real),
                    (Ty::Vector(..), "norm") => Ty::Scalar(Dom::NonNeg),
                    (t, _) => return err(span, format!("`{name}` needs a vector, got {t}")),
                };
                let a = Box::new(a);
                mk(if name == "sum" { TK::Sum(a) } else { TK::Norm(a) }, ty)
            }
            "cumsum" => {
                if args.is_empty() || args.len() > 2 {
                    return err_help(span, "cumsum takes a vector or matrix, and optionally the dimension to sum along", "cumsum(v) or cumsum(M, T)");
                }
                let a = self.expr(&args[0].value)?;
                let (last, dom) = match &a.ty {
                    Ty::Vector(n, d) => (n.clone(), *d),
                    Ty::Matrix(_, c, _) => (c.clone(), Dom::Real),
                    t => return err(span, format!("cumsum needs a vector or matrix, got {t}")),
                };
                if args.len() == 2 {
                    let d = self.dim_arg(args, 1, span)?;
                    if d != last {
                        return err_help(
                            args[1].value.span,
                            format!("this cumsum runs along `{last}` (the last dimension), not `{d}`"),
                            "sums run along the last dimension; transpose the matrix to sum along the other one",
                        );
                    }
                }
                let dom = if dom <= Dom::NonNeg { Dom::NonNeg } else { Dom::Real };
                let ty = match &a.ty {
                    Ty::Vector(n, _) => Ty::Vector(n.clone(), dom),
                    t => match t {
                        Ty::Matrix(r, c, _) => Ty::Matrix(r.clone(), c.clone(), Struct::General),
                        _ => unreachable!(),
                    },
                };
                mk(TK::Cumsum(Box::new(a)), ty)
            }
            "dot" => {
                arity(2)?;
                let a = self.expr(&args[0].value)?;
                let b = self.expr(&args[1].value)?;
                match (&a.ty, &b.ty) {
                    (Ty::Vector(n1, _), Ty::Vector(n2, _)) if n1 == n2 => {}
                    (x, y) => return err(span, format!("dot needs two vectors of the same length, got {x} and {y}")),
                }
                mk(TK::Dot(Box::new(a), Box::new(b)), Ty::Scalar(Dom::Real))
            }
            "clock" => {
                arity(0)?;
                mk(TK::Clock, Ty::Scalar(Dom::Real))
            }
            "read" => err_help(span, "read can only be used as `let X: Matrix[n, p] = read(\"file\")`", "the annotation tells the compiler the shape"),
            "print" => err(span, "print is a statement and has no value"),
            "sample" => self.sample(args, span),
            _ => {
                if let Some(m) = self.models.get(name).cloned() {
                    return self.model_inst(&m, args, span);
                }
                let sig = match self.fns.get(name) {
                    Some(s) => s.clone(),
                    None => return err(span, format!("unknown function `{name}`")),
                };
                arity(sig.params.len())?;
                let mut targs = Vec::new();
                let mut map: HashMap<String, Dim> = HashMap::new();
                for ((pname, pty), a) in sig.params.iter().zip(args) {
                    let t = self.expr(&a.value)?;
                    self.unify(pty, &t.ty, &mut map).map_err(|why| Diag {
                        span: a.value.span,
                        msg: format!("argument `{pname}` of `{name}` expects {pty}, got {}: {why}", t.ty),
                        help: None,
                    })?;
                    targs.push(t);
                }
                let dims: Vec<Dim> = sig.dim_params.iter().map(|d| map[d].clone()).collect();
                let ret = match &sig.ret {
                    Some(r) => subst(r, &map),
                    None => Ty::Void,
                };
                mk(TK::Call { name: name.to_string(), args: targs, dims }, ret)
            }
        }
    }

    /// Matches a callee's declared type against an argument type, mapping the
    /// callee's dimension names onto the caller's dimensions.
    fn unify(&self, declared: &Ty, actual: &Ty, map: &mut HashMap<String, Dim>) -> Result<(), String> {
        let mut dim = |d: &Dim, a: &Dim| -> Result<(), String> {
            match d {
                Dim::Const(c) => {
                    if a != &Dim::Const(*c) {
                        return Err(format!("dimension {a} is not {c}"));
                    }
                }
                Dim::Sym(s) => match map.get(s) {
                    Some(prev) if prev != a => return Err(format!("`{s}` would have to be both {prev} and {a}")),
                    Some(_) => {}
                    None => {
                        map.insert(s.clone(), a.clone());
                    }
                },
            }
            Ok(())
        };
        match (declared, actual) {
            (Ty::Vector(n, d), Ty::Vector(an, ad)) => {
                dim(n, an)?;
                if ad > d {
                    return Err(format!("elements are {ad}, but {d} is required"));
                }
                Ok(())
            }
            (Ty::Matrix(r, c, s), Ty::Matrix(ar, ac, as_)) => {
                dim(r, ar)?;
                dim(c, ac)?;
                if as_ > s {
                    return Err(format!("the compiler can only prove the matrix is {as_}"));
                }
                Ok(())
            }
            (d, a) => match misfit(a, d) {
                None => Ok(()),
                Some(w) => Err(w),
            },
        }
    }

    fn model_inst(&mut self, m: &TModel, args: &[Arg], span: Span) -> Res<TExpr> {
        if args.len() != m.data.len() {
            let names: Vec<&str> = m.data.iter().map(|(n, _)| n.as_str()).collect();
            return err(span, format!("model `{}` takes its data ({}), got {} argument(s)", m.name, names.join(", "), args.len()));
        }
        let mut map = HashMap::new();
        let mut data = Vec::new();
        for ((dname, dty), a) in m.data.iter().zip(args) {
            let t = self.expr(&a.value)?;
            self.unify(dty, &t.ty, &mut map).map_err(|why| Diag {
                span: a.value.span,
                msg: format!("data `{dname}` of model `{}` expects {dty}, got {}: {why}", m.name, t.ty),
                help: None,
            })?;
            data.push(t);
        }
        let dims = m.dims.iter().map(|d| map[d].clone()).collect();
        Ok(TExpr { kind: TK::ModelInst { model: m.name.clone(), data, dims }, ty: Ty::Model(m.name.clone()), span })
    }

    fn sample(&mut self, args: &[Arg], span: Span) -> Res<TExpr> {
        let first = match args.first() {
            Some(a) if a.name.is_none() => a,
            _ => return err_help(span, "sample needs a model applied to data", "sample(MyModel(X, y), draws = 1000)"),
        };
        let inst = self.expr(&first.value)?;
        let model = match &inst.ty {
            Ty::Model(m) => m.clone(),
            t => return err(first.value.span, format!("sample needs a model applied to data, got {t}")),
        };
        let (mut draws, mut warmup, mut chains, mut seed) = (1000, 1000, 4, 1);
        for a in &args[1..] {
            let v = match &a.value.kind {
                ExprKind::Num(v, true) if *v >= 0.0 => *v as i64,
                _ => return err(a.value.span, "sample options take whole numbers"),
            };
            match a.name.as_deref() {
                Some("draws") => draws = v,
                Some("warmup") => warmup = v,
                Some("chains") => chains = v,
                Some("seed") => seed = v,
                _ => return err_help(a.value.span, "unknown sample option", "options are draws, warmup, chains and seed"),
            }
        }
        Ok(TExpr { kind: TK::Sample { inst: Box::new(inst), draws, warmup, chains, seed }, ty: Ty::Posterior(model), span })
    }

    // ------------------------------------------------------------ models

    fn model(&mut self, m: &ModelDecl) -> Res<TModel> {
        self.in_model = true;
        self.scopes = vec![HashMap::new()];
        self.dims = HashSet::new();
        let mut dims = Vec::new();
        let mut data = Vec::new();
        let mut seen = HashSet::new();
        for d in &m.data {
            if !seen.insert(d.name.clone()) {
                return err(d.span, format!("`{}` is declared twice", d.name));
            }
            let t = ann_ty(&d.ann);
            if matches!(t, Ty::Int) {
                return err(d.span, "data must be Real, Positive, Prob, a vector or a matrix");
            }
            sym_dims(&t, &mut dims);
            for s in &dims {
                self.dims.insert(s.clone());
            }
            self.bind(&d.name, t.clone(), false);
            data.push((d.name.clone(), t));
        }
        let mut params = Vec::new();
        for p in &m.params {
            if !seen.insert(p.name.clone()) {
                return err(p.span, format!("`{}` is declared twice", p.name));
            }
            let t = ann_ty(&p.ann);
            match &t {
                Ty::Scalar(Dom::Real) | Ty::Scalar(Dom::Positive) | Ty::Vector(_, Dom::Real) | Ty::Vector(_, Dom::Positive) | Ty::Matrix(_, _, Struct::General) => {}
                _ => return err_help(p.span, format!("parameters of type {t} are not supported yet"), "this prototype supports Real, Positive, Vector[n], Positive[n] and Matrix[m, n] parameters"),
            }
            self.check_dims_bound(&t, p.span).map_err(|mut d| {
                d.help = Some("parameter sizes must use dimensions defined by the data".into());
                d
            })?;
            self.bind(&p.name, t.clone(), false);
            params.push((p.name.clone(), t));
        }
        let mut body = Vec::new();
        for s in &m.body {
            match s {
                ModelStmt::Let { name, value, span } => {
                    if !seen.insert(name.clone()) {
                        return err(*span, format!("`{name}` is already defined"));
                    }
                    let v = self.expr(value)?;
                    let u = self.bind(name, v.ty.clone(), false);
                    body.push(TModelStmt::Let { name: u, value: v });
                }
                ModelStmt::Tilde { lhs, dist, args, span } => {
                    body.push(self.tilde(lhs, dist, args, *span)?);
                }
            }
        }
        self.in_model = false;
        Ok(TModel { name: m.name.clone(), dims, data, params, body })
    }

    fn tilde(&mut self, lhs: &Expr, dist: &str, args: &[Expr], span: Span) -> Res<TModelStmt> {
        let (d, names): (Dist, &[&str]) = match dist {
            "Normal" => (Dist::Normal, &["mean", "scale"]),
            "BernoulliLogit" => (Dist::BernoulliLogit, &["log-odds"]),
            "PoissonLog" => (Dist::PoissonLog, &["log-rate"]),
            "Exponential" => (Dist::Exponential, &["rate"]),
            _ => return err_help(span, format!("unknown distribution `{dist}`"), "available: Normal(mean, scale), BernoulliLogit(log_odds), PoissonLog(log_rate), Exponential(rate)"),
        };
        if args.len() != names.len() {
            return err(span, format!("{dist} takes {} argument(s) ({}), got {}", names.len(), names.join(", "), args.len()));
        }
        let l = self.expr(lhs)?;
        let mut targs = Vec::new();
        for a in args {
            targs.push(self.expr(a)?);
        }
        let operands: Vec<&TExpr> = std::iter::once(&l).chain(targs.iter()).collect();
        let mut mat: Option<(Dim, Dim)> = None;
        for t in &operands {
            if let Ty::Matrix(r, c, _) = &t.ty {
                match &mat {
                    Some((pr, pc)) if pr != r || pc != c => {
                        return err(t.span, format!("all matrices in a `~` statement must have the same shape; found [{pr}, {pc}] and [{r}, {c}]"));
                    }
                    _ => mat = Some((r.clone(), c.clone())),
                }
            }
        }
        let mut len: Option<Dim> = None;
        for t in &operands {
            match &t.ty {
                ty if ty.is_scalar() => {}
                Ty::Matrix(..) => {}
                Ty::Vector(n, _) => match &mat {
                    Some((r, c)) => {
                        if let Err(why) = bcast_axis(n, r, c) {
                            return err(t.span, why);
                        }
                    }
                    None => match &len {
                        Some(prev) if prev != n => {
                            return err(t.span, format!("all vectors in a `~` statement must have the same length; found {prev} and {n}"));
                        }
                        _ => len = Some(n.clone()),
                    },
                },
                ty => return err(t.span, format!("{ty} cannot appear in a `~` statement")),
            }
        }
        let shape = match (mat, len) {
            (Some((r, c)), _) => SShape::Mat(r, c),
            (None, Some(n)) => SShape::Vec(n),
            (None, None) => SShape::Scalar,
        };
        let positive_arg = |k: usize, what: &str| -> Res<()> {
            let a = &targs[k];
            if a.ty.dom() > Dom::Positive {
                let who = match &a.kind {
                    TK::Var(v) => format!("`{v}`"),
                    _ => "this expression".into(),
                };
                return err_help(
                    a.span,
                    format!("the {what} of {dist} must be Positive, but {who} is {}", a.ty.dom()),
                    "declare the parameter as `param sigma: Positive`; Mint then samples log(sigma) and adds the Jacobian itself",
                );
            }
            Ok(())
        };
        match d {
            Dist::Normal => positive_arg(1, "scale")?,
            Dist::Exponential => {
                positive_arg(0, "rate")?;
                if l.ty.dom() > Dom::NonNeg {
                    return err_help(l.span, format!("Exponential is a distribution on positive numbers, but this is {}", l.ty.dom()), "declare the parameter as Positive");
                }
            }
            Dist::BernoulliLogit | Dist::PoissonLog => {}
        }
        Ok(TModelStmt::Tilde { lhs: l, dist: d, args: targs, shape, span })
    }
}

fn widen(t: &Ty) -> Ty {
    match t {
        Ty::Scalar(_) | Ty::Int => Ty::Scalar(Dom::Real),
        Ty::Vector(n, _) => Ty::Vector(n.clone(), Dom::Real),
        Ty::Matrix(r, c, _) => Ty::Matrix(r.clone(), c.clone(), Struct::General),
        t => t.clone(),
    }
}

fn subst(t: &Ty, map: &HashMap<String, Dim>) -> Ty {
    let d = |x: &Dim| match x {
        Dim::Sym(s) => map.get(s).cloned().unwrap_or(x.clone()),
        c => c.clone(),
    };
    match t {
        Ty::Vector(n, dm) => Ty::Vector(d(n), *dm),
        Ty::Matrix(r, c, s) => Ty::Matrix(d(r), d(c), *s),
        t => t.clone(),
    }
}
