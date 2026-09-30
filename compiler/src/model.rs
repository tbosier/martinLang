//! Compiles a `model` block to a log-density-and-gradient function.
//!
//! Each `x ~ D(args)` statement becomes one fused loop over its observations.
//! For observation i the loop evaluates every argument in registers, computes
//! the log density term and its partial derivatives, and runs reverse-mode
//! differentiation through that observation's expression tree immediately.
//! A matrix-vector product like `X * beta` is evaluated as one row dot product
//! and back-propagated as one row axpy, so X is streamed once per gradient
//! and no n-length temporaries are allocated.
//!
//! Optional rewrite (`--no-suffstats` disables it): a Normal likelihood whose
//! outcome is data, whose scale is a single scalar, and whose mean is affine in
//! the parameters with data coefficients depends on the data only through
//! Z'Z, Z'y and y'y. These are computed once before sampling, and each
//! gradient then costs O(q^2) instead of O(n q).

use std::collections::HashMap;

use crate::ast::BinOp;
use crate::check::{bcast_axis, Axis, Dist, Func, SShape, TExpr, TModel, TModelStmt, TK};
use crate::codegen::{scalar_func, Opts};
use crate::ir::{fconst, for_range, rows_axpy_blocked, rows_dot_blocked, Fb, HasFb, Module};
use crate::types::{Dim, Dom, Ty};

pub fn data_global(model: &str, name: &str) -> String {
    format!("@mint_model_{model}_data_{name}")
}

pub fn dim_global(model: &str, sym: &str) -> String {
    format!("@mint_model_{model}_dim_{sym}")
}

/// How a leaf is indexed inside the loop of its statement: by the flat
/// element index, or (in a matrix-shaped statement) by row or by column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Ax {
    Flat,
    Row,
    Col,
}

/// Loop indices of one element of a statement's index space.
#[derive(Clone)]
struct Ix {
    flat: String,
    row: String,
    col: String,
}

impl Ix {
    fn vec(i: &str) -> Ix {
        Ix { flat: i.into(), row: i.into(), col: i.into() }
    }
    fn cell(g: &mut Mg, r: &str, c: &str, _rows: &str, cols: &str) -> Ix {
        let a = g.f.imul(r, cols);
        let flat = g.f.iadd(&a, c);
        Ix { flat, row: r.into(), col: c.into() }
    }
    fn at(&self, ax: Ax) -> &str {
        match ax {
            Ax::Flat => &self.flat,
            Ax::Row => &self.row,
            Ax::Col => &self.col,
        }
    }
}

#[derive(Clone, Debug)]
enum M {
    Const(f64),
    DimV(String),
    DataS(String),
    DataV(String, Ax),
    DataM(String),
    ParamS(String),
    ParamV(String, Ax),
    ParamM(String),
    MatVec { mat: String, rows: Dim, cols: Dim, vec: Box<M> },
    /// Running sum along the last dimension of `shape`; materialised before
    /// the statement's loop, its adjoint is a reverse running sum after it.
    Cumsum { inner: Box<M>, shape: SShape, ax: Ax },
    Bin(BinOp, Box<M>, Box<M>),
    Neg(Box<M>),
    Func(Func, Box<M>),
}

impl M {
    fn active(&self) -> bool {
        match self {
            M::ParamS(_) | M::ParamV(..) | M::ParamM(_) => true,
            M::MatVec { vec, .. } => vec.active(),
            M::Cumsum { inner, .. } => inner.active(),
            M::Bin(_, a, b) => a.active() || b.active(),
            M::Neg(a) | M::Func(_, a) => a.active(),
            _ => false,
        }
    }
    fn indexed(&self) -> bool {
        match self {
            M::DataV(..) | M::ParamV(..) | M::DataM(_) | M::ParamM(_) | M::MatVec { .. } | M::Cumsum { .. } => true,
            M::Bin(_, a, b) => a.indexed() || b.indexed(),
            M::Neg(a) | M::Func(_, a) => a.indexed(),
            _ => false,
        }
    }
    fn has_cumsum(&self) -> bool {
        match self {
            M::Cumsum { .. } => true,
            M::MatVec { vec, .. } => vec.has_cumsum(),
            M::Bin(_, a, b) => a.has_cumsum() || b.has_cumsum(),
            M::Neg(a) | M::Func(_, a) => a.has_cumsum(),
            _ => false,
        }
    }
}

/// Index of a scan's buffer element as seen from a statement's loop.
fn cumsum_index(ix: &Ix, ax: Ax, _shape: &SShape) -> String {
    ix.at(ax).to_string()
}

/// Active vector parameters indexed along `ax` (Row or Col).
fn axis_params(e: &M, ax: Ax, out: &mut Vec<String>) {
    match e {
        M::ParamV(n, a) if *a == ax => {
            if !out.contains(n) {
                out.push(n.clone())
            }
        }
        M::Bin(_, a, b) => {
            axis_params(a, ax, out);
            axis_params(b, ax, out);
        }
        M::Neg(a) | M::Func(_, a) => axis_params(a, ax, out),
        _ => {}
    }
}

fn shape_size(g: &mut Mg, sh: &SShape) -> String {
    match sh {
        SShape::Scalar => "1".into(),
        SShape::Vec(n) => g.dim(n),
        SShape::Mat(r, c) => {
            let (r, c) = (g.dim(r), g.dim(c));
            g.f.imul(&r, &c)
        }
    }
}

/// Emits `body` once per element of a statement's index space.
fn for_shape(g: &mut Mg, sh: &SShape, body: impl FnOnce(&mut Mg, &Ix)) {
    match sh {
        SShape::Scalar => body(g, &Ix::vec("0")),
        SShape::Vec(n) => {
            let n = g.dim(n);
            for_range(g, "0", &n, |g, i| body(g, &Ix::vec(i)));
        }
        SShape::Mat(r, c) => {
            let (r, c) = (g.dim(r), g.dim(c));
            for_range(g, "0", &r, |g, row| {
                let base = g.f.imul(row, &c);
                for_range(g, "0", &c, |g, col| {
                    let flat = g.f.iadd(&base, col);
                    body(g, &Ix { flat, row: row.to_string(), col: col.to_string() });
                });
            });
        }
    }
}

#[derive(Clone, Debug)]
enum Term {
    /// coef * param (a column of ones)
    Ones { param: String, coef: f64 },
    /// coef * data .* param
    Col { param: String, data: M, coef: f64 },
    /// coef * mat * param
    Block { param: String, mat: String, cols: Dim, coef: f64 },
}

struct SsPlan {
    terms: Vec<Term>,
    offset: Vec<(f64, M)>,
}

enum Stmt {
    Tilde { dist: Dist, lhs: M, args: Vec<M>, shape: SShape, ss: Option<SsPlan>, fission: bool },
}

struct Lowering<'a> {
    tm: &'a TModel,
    lets: HashMap<String, TExpr>,
}

impl Lowering<'_> {
    fn kind_of(&self, name: &str) -> Option<(&'static str, &Ty)> {
        if let Some((_, t)) = self.tm.data.iter().find(|(n, _)| n == name) {
            return Some(("data", t));
        }
        if let Some((_, t)) = self.tm.params.iter().find(|(n, _)| n == name) {
            return Some(("param", t));
        }
        None
    }

    /// How a value of type `t` is indexed inside a loop over `shape`.
    fn axis(t: &Ty, shape: &SShape) -> Result<Ax, String> {
        match (t, shape) {
            (Ty::Vector(n, _), SShape::Mat(r, c)) => match bcast_axis(n, r, c)? {
                Axis::Row => Ok(Ax::Row),
                Axis::Col => Ok(Ax::Col),
            },
            (Ty::Matrix(..), SShape::Vec(_)) => Err("a matrix cannot be used inside a vector-shaped statement".into()),
            _ => Ok(Ax::Flat),
        }
    }

    fn lower(&self, e: &TExpr, shape: &SShape) -> Result<M, String> {
        Ok(match &e.kind {
            TK::Num(v) => M::Const(*v),
            TK::DimVal(Dim::Sym(s)) => M::DimV(s.clone()),
            TK::DimVal(Dim::Const(c)) => M::Const(*c as f64),
            TK::Var(n) => {
                if let Some(t) = self.lets.get(n) {
                    return self.lower(t, shape);
                }
                let (kind, ty) = self.kind_of(n).ok_or_else(|| format!("unknown name `{n}`"))?;
                let ax = Self::axis(ty, shape)?;
                match (kind, ty) {
                    ("data", Ty::Vector(..)) => M::DataV(n.clone(), ax),
                    ("data", Ty::Matrix(..)) => M::DataM(n.clone()),
                    ("data", _) => M::DataS(n.clone()),
                    ("param", Ty::Vector(..)) => M::ParamV(n.clone(), ax),
                    ("param", Ty::Matrix(..)) => M::ParamM(n.clone()),
                    _ => M::ParamS(n.clone()),
                }
            }
            TK::Bin(op, a, b) => {
                let op = match op {
                    BinOp::EMul => BinOp::Mul,
                    BinOp::EDiv => BinOp::Div,
                    BinOp::Pow => {
                        if !matches!(b.kind, TK::Num(_)) {
                            return Err("powers inside a model need a literal exponent".into());
                        }
                        BinOp::Pow
                    }
                    o => *o,
                };
                M::Bin(op, Box::new(self.lower(a, shape)?), Box::new(self.lower(b, shape)?))
            }
            TK::Neg(a) => M::Neg(Box::new(self.lower(a, shape)?)),
            TK::Func(f, a) => M::Func(*f, Box::new(self.lower(a, shape)?)),
            TK::Cumsum(a) => {
                let own = match &a.ty {
                    Ty::Vector(n, _) => SShape::Vec(n.clone()),
                    Ty::Matrix(r, c, _) => SShape::Mat(r.clone(), c.clone()),
                    _ => unreachable!(),
                };
                let ax = if &own == shape { Ax::Flat } else { Self::axis(&e.ty, shape)? };
                M::Cumsum { inner: Box::new(self.lower(a, &own)?), shape: own, ax }
            }
            TK::MatVec { m, trans, v } => {
                if !matches!(shape, SShape::Vec(_)) {
                    return Err("inside a model, X * beta can only appear in a vector-shaped statement (prototype limitation)".into());
                }
                let mat = match (&m.kind, trans) {
                    (TK::Var(n), false) if matches!(self.kind_of(n), Some(("data", _))) => n.clone(),
                    _ => return Err("inside a model, matrix products must have the form X * beta with X a data matrix".into()),
                };
                let (rows, cols) = match &m.ty {
                    Ty::Matrix(r, c, _) => (r.clone(), c.clone()),
                    _ => unreachable!(),
                };
                let vec = match &v.kind {
                    TK::Var(n) if !self.lets.contains_key(n) => self.lower(v, &SShape::Vec(cols.clone()))?,
                    _ => return Err("inside a model, the vector in X * v must be a parameter or data name (prototype limitation)".into()),
                };
                M::MatVec { mat, rows, cols, vec: Box::new(vec) }
            }
            _ => return Err("this operation is not supported inside a model yet".into()),
        })
    }
}

/// Splits `mu` into terms that are linear in parameters with data
/// coefficients, plus a data-only offset.
fn affine(e: &M, coef: f64, terms: &mut Vec<Term>, offset: &mut Vec<(f64, M)>) -> bool {
    if !e.active() {
        offset.push((coef, e.clone()));
        return true;
    }
    match e {
        M::Bin(BinOp::Add, a, b) => affine(a, coef, terms, offset) && affine(b, coef, terms, offset),
        M::Bin(BinOp::Sub, a, b) => affine(a, coef, terms, offset) && affine(b, -coef, terms, offset),
        M::Neg(a) => affine(a, -coef, terms, offset),
        M::Bin(BinOp::Mul, a, b) => match (&**a, &**b) {
            (M::Const(c), x) | (x, M::Const(c)) => affine(x, coef * c, terms, offset),
            (d, M::ParamS(p)) | (M::ParamS(p), d) if !d.active() && d.indexed() => {
                terms.push(Term::Col { param: p.clone(), data: d.clone(), coef });
                true
            }
            _ => false,
        },
        M::ParamS(p) => {
            terms.push(Term::Ones { param: p.clone(), coef });
            true
        }
        M::MatVec { mat, cols, vec, .. } => match &**vec {
            M::ParamV(p, _) => {
                terms.push(Term::Block { param: p.clone(), mat: mat.clone(), cols: cols.clone(), coef });
                true
            }
            _ => false,
        },
        _ => false,
    }
}

fn matvecs<'a>(e: &'a M, out: &mut Vec<&'a M>) {
    match e {
        M::MatVec { .. } => out.push(e),
        M::Bin(_, a, b) => {
            matvecs(a, out);
            matvecs(b, out);
        }
        M::Neg(a) | M::Func(_, a) => matvecs(a, out),
        M::Cumsum { inner, .. } => matvecs(inner, out),
        _ => {}
    }
}

/// Nodes materialised around a statement's loop, children before parents:
/// every cumsum, and matrix-vector products when the loop is split.
fn globals<'a>(e: &'a M, fission: bool, out: &mut Vec<&'a M>) {
    match e {
        M::MatVec { .. } => {
            if fission {
                out.push(e)
            }
        }
        M::Cumsum { inner, .. } => {
            globals(inner, fission, out);
            out.push(e);
        }
        M::Bin(_, a, b) => {
            globals(a, fission, out);
            globals(b, fission, out);
        }
        M::Neg(a) | M::Func(_, a) => globals(a, fission, out),
        _ => {}
    }
}

fn stmt_globals<'a>(lhs: &'a M, args: &'a [M], fission: bool) -> Vec<&'a M> {
    let mut out = Vec::new();
    globals(lhs, fission, &mut out);
    for a in args {
        globals(a, fission, &mut out);
    }
    out
}

fn has_func(e: &M) -> bool {
    match e {
        M::Func(..) => true,
        M::Bin(_, a, b) => has_func(a) || has_func(b),
        M::Neg(a) => has_func(a),
        M::Cumsum { inner, .. } => has_func(inner),
        _ => false,
    }
}

/// Split the observation loop when it contains matrix-vector products and
/// transcendental functions: the row dot products and row updates then run
/// in their own passes, and the elementwise pass has no inner loops, so it
/// vectorises (including exp and log, through the vector math library).
fn wants_fission(dist: Dist, lhs: &M, args: &[M]) -> bool {
    let mut mv = Vec::new();
    matvecs(lhs, &mut mv);
    for a in args {
        matvecs(a, &mut mv);
    }
    if mv.is_empty() {
        return false;
    }
    let scale_indexed = dist == Dist::Normal && args[1].indexed();
    dist == Dist::BernoulliLogit || dist == Dist::PoissonLog || dist == Dist::Exponential || scale_indexed || has_func(lhs) || args.iter().any(has_func)
}

fn plan_suffstats(dist: Dist, lhs: &M, args: &[M], shape: &SShape) -> Option<SsPlan> {
    if dist != Dist::Normal || !matches!(shape, SShape::Vec(_)) || lhs.active() || !lhs.indexed() {
        return None;
    }
    if lhs.has_cumsum() || args.iter().any(|a| a.has_cumsum()) {
        return None;
    }
    let (mu, sigma) = (&args[0], &args[1]);
    if sigma.indexed() {
        return None;
    }
    let mut terms = Vec::new();
    let mut offset = Vec::new();
    if !affine(mu, 1.0, &mut terms, &mut offset) || terms.is_empty() {
        return None;
    }
    Some(SsPlan { terms, offset })
}

/// Per-function view of the model's data, dimensions and parameters.
struct Mg<'a> {
    f: Fb,
    m: &'a mut Module,
    dims: HashMap<String, String>,
    data_s: HashMap<String, String>,
    data_p: HashMap<String, String>,
    pval: HashMap<String, String>,
    pptr: HashMap<String, String>,
    padj: HashMap<String, String>,
    gptr: HashMap<String, String>,
    /// Materialised nodes (cumsums, split matrix-vector products):
    /// node -> (forward values, adjoints) buffers.
    split: HashMap<usize, (String, Option<String>)>,
    /// Inside a matrix-shaped loop: register accumulators for the gradient of
    /// row-indexed vector parameters, flushed once per row, so the column
    /// loop is a clean reduction that LLVM can vectorise.
    inv_acc: HashMap<(String, Ax), String>,
}

impl HasFb for Mg<'_> {
    fn fb(&mut self) -> &mut Fb {
        &mut self.f
    }
}

impl<'a> Mg<'a> {
    fn new(m: &'a mut Module, tm: &TModel, strict: bool) -> Self {
        let mut g = Mg {
            f: Fb::new(strict),
            m,
            dims: HashMap::new(),
            data_s: HashMap::new(),
            data_p: HashMap::new(),
            pval: HashMap::new(),
            pptr: HashMap::new(),
            padj: HashMap::new(),
            gptr: HashMap::new(),
            split: HashMap::new(),
            inv_acc: HashMap::new(),
        };
        for d in &tm.dims {
            let v = g.f.load_i64(&dim_global(&tm.name, d));
            g.dims.insert(d.clone(), v);
        }
        for (n, t) in &tm.data {
            let gl = data_global(&tm.name, n);
            if t.is_buffer() {
                let p = g.f.load_ptr(&gl);
                g.data_p.insert(n.clone(), p);
            } else {
                let v = g.f.load_f64(&gl);
                g.data_s.insert(n.clone(), v);
            }
        }
        g
    }

    /// (matrix pointer, vector pointer, column count) of a MatVec node.
    fn matvec_parts(&self, e: &M) -> (String, String, String) {
        match e {
            M::MatVec { mat, cols, vec, .. } => {
                let vp = match &**vec {
                    M::ParamV(n, _) => self.pptr[n].clone(),
                    M::DataV(n, _) => self.data_p[n].clone(),
                    _ => unreachable!(),
                };
                (self.data_p[mat].clone(), vp, self.dim(cols))
            }
            _ => unreachable!(),
        }
    }

    fn dim(&self, d: &Dim) -> String {
        match d {
            Dim::Const(c) => c.to_string(),
            Dim::Sym(s) => self.dims[s].clone(),
        }
    }

    /// Offsets and sizes of each parameter in the unconstrained vector.
    fn layout(&mut self, tm: &TModel) -> (Vec<(String, String, Option<String>)>, String) {
        let mut off = "0".to_string();
        let mut out = Vec::new();
        for (n, t) in &tm.params {
            let size = match t {
                Ty::Vector(d, _) => Some(self.dim(d)),
                Ty::Matrix(r, c, _) => {
                    let (r, c) = (self.dim(r), self.dim(c));
                    Some(self.f.imul(&r, &c))
                }
                _ => None,
            };
            out.push((n.clone(), off.clone(), size.clone()));
            off = self.f.iadd(&off, size.as_deref().unwrap_or("1"));
        }
        (out, off)
    }

    fn fwd(&mut self, e: &M, ix: &Ix, vals: &mut HashMap<usize, String>) -> String {
        let key = e as *const M as usize;
        let v = match e {
            M::Const(c) => fconst(*c),
            M::DimV(s) => {
                let d = self.dims[s].clone();
                self.f.sitofp(&d)
            }
            M::DataS(n) => self.data_s[n].clone(),
            M::DataV(n, ax) => {
                let p = self.data_p[n].clone();
                self.f.load(&p, ix.at(*ax))
            }
            M::DataM(n) => {
                let p = self.data_p[n].clone();
                self.f.load(&p, &ix.flat)
            }
            M::ParamS(n) => self.pval[n].clone(),
            M::ParamV(n, ax) => {
                let p = self.pptr[n].clone();
                self.f.load(&p, ix.at(*ax))
            }
            M::ParamM(n) => {
                let p = self.pptr[n].clone();
                self.f.load(&p, &ix.flat)
            }
            M::Cumsum { ax, shape, .. } => {
                let fw = self.split.get(&key).expect("cumsum materialised before use").0.clone();
                let at = cumsum_index(ix, *ax, shape);
                self.f.load(&fw, &at)
            }
            M::MatVec { .. } if self.split.contains_key(&key) => {
                let fw = self.split[&key].0.clone();
                self.f.load(&fw, &ix.flat)
            }
            M::MatVec { mat, cols, vec, .. } => {
                let mp = self.data_p[mat].clone();
                let vp = match &**vec {
                    M::ParamV(n, _) => self.pptr[n].clone(),
                    M::DataV(n, _) => self.data_p[n].clone(),
                    _ => unreachable!(),
                };
                let c = self.dim(cols);
                let row = self.f.imul(&ix.flat, &c);
                let acc = self.f.acc_new(&fconst(0.0));
                for_range(self, "0", &c, |g, k| {
                    let idx = g.f.iadd(&row, k);
                    let a = g.f.load(&mp, &idx);
                    let b = g.f.load(&vp, k);
                    let t = g.f.fmul(&a, &b);
                    g.f.acc_add(&acc, &t);
                });
                self.f.acc_get(&acc)
            }
            M::Bin(op, a, b) => {
                let x = self.fwd(a, ix, vals);
                let y = self.fwd(b, ix, vals);
                match op {
                    BinOp::Add => self.f.fadd(&x, &y),
                    BinOp::Sub => self.f.fsub(&x, &y),
                    BinOp::Mul => self.f.fmul(&x, &y),
                    BinOp::Div => self.f.fdiv(&x, &y),
                    BinOp::Pow => match &**b {
                        M::Const(k) if *k == 2.0 => self.f.fmul(&x, &x),
                        _ => self.f.intrinsic2(self.m, "llvm.pow.f64", &x, &y),
                    },
                    _ => unreachable!(),
                }
            }
            M::Neg(a) => {
                let x = self.fwd(a, ix, vals);
                self.f.fneg(&x)
            }
            M::Func(func, a) => {
                let x = self.fwd(a, ix, vals);
                scalar_func(&mut self.f, self.m, *func, &x)
            }
        };
        vals.insert(key, v.clone());
        v
    }

    fn val(vals: &HashMap<usize, String>, e: &M) -> String {
        vals[&(e as *const M as usize)].clone()
    }

    /// Reverse sweep for one observation: adds `adj * d(e)/d(params)` to the
    /// gradient accumulators.
    fn bwd(&mut self, e: &M, adj: &str, ix: &Ix, vals: &HashMap<usize, String>) {
        if !e.active() {
            return;
        }
        let key = e as *const M as usize;
        if let Some((_, Some(ad))) = self.split.get(&key) {
            let ad = ad.clone();
            let at = match e {
                M::Cumsum { ax, shape, .. } => cumsum_index(ix, *ax, shape),
                _ => ix.flat.clone(),
            };
            self.f.add_to(&ad, &at, adj);
            return;
        }
        let i = ix;
        match e {
            M::ParamS(n) => {
                let acc = self.padj[n].clone();
                self.f.acc_add(&acc, adj);
            }
            M::ParamV(n, ax) => {
                if let Some(acc) = self.inv_acc.get(&(n.clone(), *ax)).cloned() {
                    self.f.acc_add(&acc, adj);
                    return;
                }

                let g = self.gptr[n].clone();
                self.f.add_to(&g, ix.at(*ax), adj);
            }
            M::ParamM(n) => {
                let g = self.gptr[n].clone();
                self.f.add_to(&g, &ix.flat, adj);
            }
            M::Cumsum { .. } => unreachable!("cumsum adjoints go through its buffer"),
            M::MatVec { mat, cols, vec, .. } => {
                let n = match &**vec {
                    M::ParamV(n, _) => n,
                    _ => unreachable!(),
                };
                let mp = self.data_p[mat].clone();
                let gp = self.gptr[n].clone();
                let c = self.dim(cols);
                let row = self.f.imul(&ix.flat, &c);
                let adj = adj.to_string();
                for_range(self, "0", &c, |g, k| {
                    let idx = g.f.iadd(&row, k);
                    let a = g.f.load(&mp, &idx);
                    let t = g.f.fmul(&adj, &a);
                    g.f.add_to(&gp, k, &t);
                });
            }
            M::Bin(op, a, b) => {
                let (va, vb) = (Self::val(vals, a), Self::val(vals, b));
                match op {
                    BinOp::Add => {
                        self.bwd(a, adj, i, vals);
                        self.bwd(b, adj, i, vals);
                    }
                    BinOp::Sub => {
                        self.bwd(a, adj, i, vals);
                        if b.active() {
                            let n = self.f.fneg(adj);
                            self.bwd(b, &n, i, vals);
                        }
                    }
                    BinOp::Mul => {
                        if a.active() {
                            let t = self.f.fmul(adj, &vb);
                            self.bwd(a, &t, i, vals);
                        }
                        if b.active() {
                            let t = self.f.fmul(adj, &va);
                            self.bwd(b, &t, i, vals);
                        }
                    }
                    BinOp::Div => {
                        let q = self.f.fdiv(adj, &vb);
                        if a.active() {
                            self.bwd(a, &q, i, vals);
                        }
                        if b.active() {
                            // d(a/b)/db = -(a/b)/b
                            let out = Self::val(vals, e);
                            let t = self.f.fmul(&q, &out);
                            let t = self.f.fneg(&t);
                            self.bwd(b, &t, i, vals);
                        }
                    }
                    BinOp::Pow => {
                        let k = match &**b {
                            M::Const(k) => *k,
                            _ => unreachable!(),
                        };
                        // d(x^k)/dx = k x^(k-1)
                        if k == 0.0 {
                            return;
                        }
                        let d = if k == 1.0 {
                            fconst(1.0)
                        } else if k == 2.0 {
                            self.f.fmul(&fconst(2.0), &va)
                        } else {
                            let p = self.f.intrinsic2(self.m, "llvm.pow.f64", &va, &fconst(k - 1.0));
                            self.f.fmul(&fconst(k), &p)
                        };
                        let t = self.f.fmul(adj, &d);
                        self.bwd(a, &t, i, vals);
                    }
                    _ => unreachable!(),
                }
            }
            M::Neg(a) => {
                let n = self.f.fneg(adj);
                self.bwd(a, &n, i, vals);
            }
            M::Func(func, a) => {
                let (x, y) = (Self::val(vals, a), Self::val(vals, e));
                let d = match func {
                    Func::Exp => y,
                    Func::Log => self.f.fdiv(&fconst(1.0), &x),
                    Func::Log1p => {
                        let t = self.f.fadd(&fconst(1.0), &x);
                        self.f.fdiv(&fconst(1.0), &t)
                    }
                    Func::Sqrt => {
                        let t = self.f.fmul(&fconst(2.0), &y);
                        self.f.fdiv(&fconst(1.0), &t)
                    }
                    Func::Sigmoid => {
                        let t = self.f.fsub(&fconst(1.0), &y);
                        self.f.fmul(&y, &t)
                    }
                    Func::Abs => {
                        let c = self.f.reg();
                        self.f.emit(format!("{c} = fcmp olt double {x}, {}", fconst(0.0)));
                        let r = self.f.reg();
                        self.f.emit(format!("{r} = select i1 {c}, double {}, double {}", fconst(-1.0), fconst(1.0)));
                        r
                    }
                };
                let t = self.f.fmul(adj, &d);
                self.bwd(a, &t, i, vals);
            }
            _ => {}
        }
    }

    /// Log density of one observation and its partial derivatives with
    /// respect to the outcome and each argument (normalising constants that do
    /// not depend on parameters are dropped).
    fn lpdf(&mut self, dist: Dist, x: &str, a: &[String]) -> (String, Vec<String>) {
        let one = fconst(1.0);
        match dist {
            Dist::Normal => {
                let (mu, s) = (&a[0], &a[1]);
                let diff = self.f.fsub(x, mu);
                let inv_s = self.f.fdiv(&one, s);
                let z = self.f.fmul(&diff, &inv_s);
                let z2 = self.f.fmul(&z, &z);
                let h = self.f.fmul(&fconst(-0.5), &z2);
                let ls = self.f.intrinsic1(self.m, "llvm.log.f64", s);
                let lp = self.f.fsub(&h, &ls);
                let dmu = self.f.fmul(&z, &inv_s);
                let dx = self.f.fneg(&dmu);
                let zm1 = self.f.fsub(&z2, &one);
                let ds = self.f.fmul(&zm1, &inv_s);
                (lp, vec![dx, dmu, ds])
            }
            Dist::BernoulliLogit => {
                // log p(y | eta) = y*eta - log(1 + exp(eta)), evaluated stably
                let eta = &a[0];
                let abs = self.f.intrinsic1(self.m, "llvm.fabs.f64", eta);
                let na = self.f.fneg(&abs);
                let e = self.f.intrinsic1(self.m, "llvm.exp.f64", &na);
                // log(1 + e) rather than log1p(e): e = exp(-|eta|) is in (0, 1], so the
                // absolute error is at most ~1e-16, and llvm.log has a vector form.
                // --strict-fp keeps libm's log1p.
                let l1p = if self.f.strict {
                    self.f.intrinsic1(self.m, "log1p", &e)
                } else {
                    let onep = self.f.fadd(&one, &e);
                    self.f.intrinsic1(self.m, "llvm.log.f64", &onep)
                };
                let pos = self.f.intrinsic2(self.m, "llvm.maxnum.f64", eta, &fconst(0.0));
                let softplus = self.f.fadd(&pos, &l1p);
                let ye = self.f.fmul(x, eta);
                let lp = self.f.fsub(&ye, &softplus);
                let den = self.f.fadd(&one, &e);
                let c = self.f.reg();
                self.f.emit(format!("{c} = fcmp oge double {eta}, {}", fconst(0.0)));
                let num = self.f.reg();
                self.f.emit(format!("{num} = select i1 {c}, double {one}, double {e}"));
                let sig = self.f.fdiv(&num, &den);
                let deta = self.f.fsub(x, &sig);
                (lp, vec![fconst(0.0), deta])
            }
            Dist::PoissonLog => {
                // log p(y | eta) = y*eta - exp(eta) - log(y!)  (the last term is data only)
                let eta = &a[0];
                let e = self.f.intrinsic1(self.m, "llvm.exp.f64", eta);
                let ye = self.f.fmul(x, eta);
                let lp = self.f.fsub(&ye, &e);
                let deta = self.f.fsub(x, &e);
                (lp, vec![fconst(0.0), deta])
            }
            Dist::Exponential => {
                let rate = &a[0];
                let lr = self.f.intrinsic1(self.m, "llvm.log.f64", rate);
                let rx = self.f.fmul(rate, x);
                let lp = self.f.fsub(&lr, &rx);
                let dx = self.f.fneg(rate);
                let inv = self.f.fdiv(&one, rate);
                let drate = self.f.fsub(&inv, x);
                (lp, vec![dx, drate])
            }
        }
    }

    fn finish(self, header: &str, epi: &[String]) {
        let Mg { f, m, .. } = self;
        m.funcs.push(f.finish(header, epi));
    }
}

pub fn gen_model(m: &mut Module, tm: &TModel, opts: &Opts) {
    m.declare("declare noalias ptr @mint_ws_slot(i64, i64)");
    let name = &tm.name;
    for d in &tm.dims {
        m.globals.push(format!("{} = internal global i64 0", dim_global(name, d)));
    }
    for (n, t) in &tm.data {
        if t.is_buffer() {
            m.globals.push(format!("{} = internal global ptr null", data_global(name, n)));
        } else {
            m.globals.push(format!("{} = internal global double 0.0", data_global(name, n)));
        }
    }

    // Lower the body, inlining `let`s.
    let mut low = Lowering { tm, lets: HashMap::new() };
    let mut stmts = Vec::new();
    for s in &tm.body {
        match s {
            TModelStmt::Let { name, value } => {
                low.lets.insert(name.clone(), value.clone());
            }
            TModelStmt::Tilde { lhs, dist, args, shape, .. } => {
                let l = low.lower(lhs, shape).unwrap_or_else(|e| panic_model(tm, &e));
                let a: Vec<M> = args.iter().map(|x| low.lower(x, shape).unwrap_or_else(|e| panic_model(tm, &e))).collect();
                if matches!(dist, Dist::BernoulliLogit | Dist::PoissonLog) && l.active() {
                    panic_model(tm, "the outcome of BernoulliLogit and PoissonLog must be data");
                }
                if matches!(dist, Dist::BernoulliLogit | Dist::PoissonLog) && l.has_cumsum() {
                    // its support could not be checked before sampling
                    panic_model(tm, "the outcome of BernoulliLogit and PoissonLog cannot contain cumsum");
                }
                let ss = if opts.suffstats { plan_suffstats(*dist, &l, &a, shape) } else { None };
                let fission = opts.fission && ss.is_none() && matches!(shape, SShape::Vec(_)) && wants_fission(*dist, &l, &a);
                stmts.push(Stmt::Tilde { dist: *dist, lhs: l, args: a, shape: shape.clone(), ss, fission });
            }
        }
    }
    let n_ss = stmts.iter().filter(|s| matches!(s, Stmt::Tilde { ss: Some(_), .. })).count();
    if n_ss > 0 {
        eprintln!("mintc: model {name}: {n_ss} likelihood term(s) rewritten to sufficient statistics");
    }
    for (k, s) in stmts.iter().enumerate() {
        if let Stmt::Tilde { ss: Some(_), .. } = s {
            for g in ["G", "b"] {
                m.globals.push(format!("@mint_model_{name}_ss{k}_{g} = internal global ptr null"));
            }
            m.globals.push(format!("@mint_model_{name}_ss{k}_c = internal global double 0.0"));
            m.globals.push(format!("@mint_model_{name}_ss{k}_q = internal global i64 0"));
        }
    }

    gen_init(m, tm, &stmts, opts);
    gen_logp(m, tm, &stmts, opts);
    gen_constrain(m, tm, opts);
    gen_sample_fn(m, tm, opts);
}

fn panic_model(tm: &TModel, msg: &str) -> ! {
    eprintln!("error: in model `{}`: {msg}", tm.name);
    std::process::exit(1);
}

fn ss_q(g: &mut Mg, plan: &SsPlan) -> String {
    let mut q = "0".to_string();
    for t in &plan.terms {
        let s = match t {
            Term::Block { cols, .. } => g.dim(cols),
            _ => "1".into(),
        };
        q = g.f.iadd(&q, &s);
    }
    q
}

fn gen_init(m: &mut Module, tm: &TModel, stmts: &[Stmt], opts: &Opts) {
    let mut g = Mg::new(m, tm, opts.strict_fp);
    // BernoulliLogit and PoissonLog outcomes are data; check their support once.
    for s in stmts {
        let Stmt::Tilde { dist: d @ (Dist::BernoulliLogit | Dist::PoissonLog), lhs, shape, .. } = s else { continue };
        let msg = g.m.string(&format!("model {}", tm.name));
        let checker = if *d == Dist::BernoulliLogit { "mint_check_binary" } else { "mint_check_count" };
        for_shape(&mut g, shape, |g, ix| {
            let v = g.fwd(lhs, ix, &mut HashMap::new());
            g.f.emit(format!("call void @{checker}(double {v}, i64 {}, ptr {msg})", ix.flat));
        });
    }
    for (k, s) in stmts.iter().enumerate() {
        let Stmt::Tilde { lhs, shape: SShape::Vec(len), ss: Some(plan), .. } = s else { continue };
        let name = tm.name.clone();
        let q = ss_q(&mut g, plan);
        let qq = g.f.imul(&q, &q);
        let gm = g.f.reg();
        g.f.emit(format!("{gm} = call ptr @mint_alloc(i64 {qq})"));
        g.f.memzero(g.m, &gm, &qq);
        let b = g.f.reg();
        g.f.emit(format!("{b} = call ptr @mint_alloc(i64 {q})"));
        g.f.memzero(g.m, &b, &q);
        let z = g.f.reg();
        g.f.emit(format!("{z} = call ptr @mint_alloc(i64 {q})"));
        let c = g.f.acc_new(&fconst(0.0));
        let n = g.dim(len);
        for_range(&mut g, "0", &n, |g, i| {
            let ix = Ix::vec(i);
            let mut vals = HashMap::new();
            let mut off = "0".to_string();
            for t in &plan.terms {
                match t {
                    Term::Block { mat, cols, coef, .. } => {
                        let mp = g.data_p[mat].clone();
                        let cn = g.dim(cols);
                        let row = g.f.imul(i, &cn);
                        let o = off.clone();
                        for_range(g, "0", &cn, |g, kk| {
                            let idx = g.f.iadd(&row, kk);
                            let x = g.f.load(&mp, &idx);
                            let x = g.f.fmul(&fconst(*coef), &x);
                            let zi = g.f.iadd(&o, kk);
                            g.f.store(&x, &z, &zi);
                        });
                        off = g.f.iadd(&off, &cn);
                    }
                    Term::Ones { coef, .. } => {
                        g.f.store(&fconst(*coef), &z, &off);
                        off = g.f.iadd(&off, "1");
                    }
                    Term::Col { data, coef, .. } => {
                        let x = g.fwd(data, &ix, &mut vals);
                        let x = g.f.fmul(&fconst(*coef), &x);
                        g.f.store(&x, &z, &off);
                        off = g.f.iadd(&off, "1");
                    }
                }
            }
            let mut y = g.fwd(lhs, &ix, &mut vals);
            for (coef, e) in &plan.offset {
                let x = g.fwd(e, &ix, &mut vals);
                let x = g.f.fmul(&fconst(*coef), &x);
                y = g.f.fsub(&y, &x);
            }
            let yy = g.f.fmul(&y, &y);
            g.f.acc_add(&c, &yy);
            for_range(g, "0", &q, |g, j| {
                let zj = g.f.load(&z, j);
                let t = g.f.fmul(&zj, &y);
                g.f.add_to(&b, j, &t);
                let jrow = g.f.imul(j, &q);
                for_range(g, j, &q, |g, kk| {
                    let zk = g.f.load(&z, kk);
                    let t = g.f.fmul(&zj, &zk);
                    let idx = g.f.iadd(&jrow, kk);
                    g.f.add_to(&gm, &idx, &t);
                });
            });
        });
        for_range(&mut g, "0", &q, |g, j| {
            for_range(g, "0", j, |g, kk| {
                let a = g.f.imul(kk, &q);
                let a = g.f.iadd(&a, j);
                let x = g.f.load(&gm, &a);
                let bi = g.f.imul(j, &q);
                let bi = g.f.iadd(&bi, kk);
                g.f.store(&x, &gm, &bi);
            });
        });
        let cv = g.f.acc_get(&c);
        g.f.emit(format!("call void @mint_free(ptr {z})"));
        g.f.emit(format!("store ptr {gm}, ptr @mint_model_{name}_ss{k}_G"));
        g.f.emit(format!("store ptr {b}, ptr @mint_model_{name}_ss{k}_b"));
        g.f.emit(format!("store double {cv}, ptr @mint_model_{name}_ss{k}_c"));
        g.f.emit(format!("store i64 {q}, ptr @mint_model_{name}_ss{k}_q"));
    }
    let header = format!("define void @mint_model_{}_init()", tm.name);
    g.finish(&header, &["ret void".into()]);
}

fn gen_logp(m: &mut Module, tm: &TModel, stmts: &[Stmt], opts: &Opts) {
    let mut g = Mg::new(m, tm, opts.strict_fp);
    let (layout, total) = g.layout(tm);
    g.f.memzero(g.m, "%grad", &total);
    let lp = g.f.acc_new(&fconst(0.0));

    // Scratch buffers: constrained values and adjoints of Positive vector
    // parameters, and the forward/adjoint buffers of materialised nodes.
    let mut pos_slots: Vec<(String, String)> = Vec::new(); // (param, len)
    for (n, t) in &tm.params {
        if let Ty::Vector(d, Dom::Positive) = t {
            let len = g.dim(d);
            pos_slots.push((n.clone(), len));
        }
    }
    let mut ws_slots: Vec<(usize, String, bool)> = Vec::new(); // (node, len, active)
    for s in stmts {
        let Stmt::Tilde { lhs, args, fission, ss: None, .. } = s else { continue };
        for node in stmt_globals(lhs, args, *fission) {
            let n = match node {
                M::MatVec { rows, .. } => g.dim(rows),
                M::Cumsum { shape, .. } => shape_size(&mut g, shape),
                _ => unreachable!(),
            };
            ws_slots.push((node as *const M as usize, n, node.active()));
        }
    }
    let mut pos_bufs: HashMap<String, (String, String, String)> = HashMap::new(); // param -> (value, adjoint, len)
    {
        // Each scratch buffer is a separate per-thread allocation returned
        // `noalias`, so LLVM knows they never overlap.
        let mut slot = 0usize;
        let mut buf = |g: &mut Mg, n: &str| -> String {
            let r = g.f.reg();
            g.f.emit(format!("{r} = call noalias ptr @mint_ws_slot(i64 {slot}, i64 {n})"));
            slot += 1;
            r
        };
        for (node, n, active) in ws_slots {
            let fw = buf(&mut g, &n);
            let ad = if active { Some(buf(&mut g, &n)) } else { None };
            g.split.insert(node, (fw, ad));
        }
        for (name, len) in pos_slots {
            let v = buf(&mut g, &len);
            let a = buf(&mut g, &len);
            pos_bufs.insert(name, (v, a, len));
        }
    }

    for ((n, t), (_, off, _)) in tm.params.iter().zip(&layout) {
        match t {
            Ty::Vector(_, Dom::Positive) => {
                // sigma_k = exp(u_k); log-Jacobian sum_k u_k
                let (v, a, len) = pos_bufs[n].clone();
                let u = g.f.gep("%theta", off);
                g.f.memzero(g.m, &a, &len);
                for_range(&mut g, "0", &len, |g, k| {
                    let x = g.f.load(&u, k);
                    g.f.acc_add(&lp, &x);
                    let e = g.f.intrinsic1(g.m, "llvm.exp.f64", &x);
                    g.f.store(&e, &v, k);
                });
                g.pptr.insert(n.clone(), v);
                g.gptr.insert(n.clone(), a);
            }
            Ty::Vector(..) | Ty::Matrix(..) => {
                let p = g.f.gep("%theta", off);
                let gp = g.f.gep("%grad", off);
                g.pptr.insert(n.clone(), p);
                g.gptr.insert(n.clone(), gp);
            }
            Ty::Scalar(d) => {
                let u = g.f.load("%theta", off);
                let v = if *d == Dom::Positive {
                    // sigma = exp(u); log |d sigma / d u| = u
                    g.f.acc_add(&lp, &u);
                    g.f.intrinsic1(g.m, "llvm.exp.f64", &u)
                } else {
                    u
                };
                g.pval.insert(n.clone(), v);
                let a = g.f.acc_new(&fconst(0.0));
                g.padj.insert(n.clone(), a);
            }
            _ => unreachable!(),
        }
    }

    for (k, s) in stmts.iter().enumerate() {
        let Stmt::Tilde { dist, lhs, args, shape, ss, fission } = s;
        if let Some(plan) = ss {
            let n = match shape {
                SShape::Vec(n) => n,
                _ => unreachable!(),
            };
            gen_ss_logp(&mut g, tm, k, plan, &args[1], n, &lp);
            continue;
        }
        let nodes = stmt_globals(lhs, args, *fission);
        // before the loop: materialise each node (children first)
        for node in &nodes {
            let key = *node as *const M as usize;
            let (fw, ad) = g.split.remove(&key).unwrap();
            match node {
                M::MatVec { rows, .. } => {
                    let (mp, vp, c) = g.matvec_parts(node);
                    let n = g.dim(rows);
                    let fw2 = fw.clone();
                    rows_dot_blocked(&mut g, &mp, &vp, &c, &n, &move |g: &mut Mg, i: &str, s: &str| g.f.store(s, &fw2, i));
                }
                M::Cumsum { inner, shape: own, .. } => {
                    let (rows, cols) = match own {
                        SShape::Vec(n) => ("1".to_string(), g.dim(n)),
                        SShape::Mat(r, c) => (g.dim(r), g.dim(c)),
                        SShape::Scalar => unreachable!(),
                    };
                    let is_vec = matches!(own, SShape::Vec(_));
                    let fw2 = fw.clone();
                    // Running sums are sequential along a row, so four rows are
                    // interleaved to give the CPU four independent chains.
                    let scan = |g: &mut Mg, r0: &str, width: usize| {
                        let mut rs = Vec::new();
                        let mut bases = Vec::new();
                        let mut accs = Vec::new();
                        for l in 0..width {
                            let r = g.f.iadd(r0, &l.to_string());
                            bases.push(g.f.imul(&r, &cols));
                            rs.push(r);
                            accs.push(g.f.acc_new(&fconst(0.0)));
                        }
                        for_range(g, "0", &cols, |g, c| {
                            for l in 0..width {
                                let flat = g.f.iadd(&bases[l], c);
                                let ix = if is_vec { Ix::vec(c) } else { Ix { flat: flat.clone(), row: rs[l].clone(), col: c.to_string() } };
                                let v = g.fwd(inner, &ix, &mut HashMap::new());
                                let old = g.f.acc_get(&accs[l]);
                                let sum = g.f.fadd(&old, &v); // sequential: a running sum is not a reduction
                                g.f.emit(format!("store double {sum}, ptr {}", accs[l]));
                                g.f.store(&sum, &fw2, &flat);
                            }
                        });
                    };
                    let nb = g.f.iop("sdiv", &rows, "4");
                    for_range(&mut g, "0", &nb, |g, b| {
                        let r0 = g.f.imul(b, "4");
                        scan(g, &r0, 4);
                    });
                    let done = g.f.imul(&nb, "4");
                    for_range(&mut g, &done, &rows, |g, r| scan(g, r, 1));
                }
                _ => unreachable!(),
            }
            if let Some(ad) = &ad {
                let n = match node {
                    M::MatVec { rows, .. } => g.dim(rows),
                    M::Cumsum { shape, .. } => shape_size(&mut g, shape),
                    _ => unreachable!(),
                };
                g.f.memzero(g.m, ad, &n);
            }
            g.split.insert(key, (fw, ad));
        }
        let body = |g: &mut Mg, ix: &Ix| {
            let mut vals = HashMap::new();
            let x = g.fwd(lhs, ix, &mut vals);
            let a: Vec<String> = args.iter().map(|e| g.fwd(e, ix, &mut vals)).collect();
            let (term, partials) = g.lpdf(*dist, &x, &a);
            g.f.acc_add(&lp, &term);
            g.bwd(lhs, &partials[0], ix, &vals);
            for (e, d) in args.iter().zip(&partials[1..]) {
                g.bwd(e, d, ix, &vals);
            }
        };
        match shape {
            SShape::Mat(r, c) => {
                // Outer loop over one dimension, inner over the other. Gradient
                // contributions to parameters indexed by the outer dimension are
                // summed in registers and written once per outer step.
                // Row-major loop. Gradients of row-indexed parameters are summed
                // in registers along each row and written once per row, so the
                // column loop is a clean reduction that LLVM vectorises.
                let mut rp = Vec::new();
                axis_params(lhs, Ax::Row, &mut rp);
                for a in args.iter() {
                    axis_params(a, Ax::Row, &mut rp);
                }
                let (r, c) = (g.dim(r), g.dim(c));
                for_range(&mut g, "0", &r, |g, row| {
                    for p in &rp {
                        let acc = g.f.acc_new(&fconst(0.0));
                        g.inv_acc.insert((p.clone(), Ax::Row), acc);
                    }
                    for_range(g, "0", &c, |g, col| {
                        let ix = Ix::cell(g, row, col, &r, &c);
                        body(g, &ix);
                    });
                    for p in &rp {
                        let acc = g.inv_acc.remove(&(p.clone(), Ax::Row)).unwrap();
                        let v = g.f.acc_get(&acc);
                        let gp = g.gptr[p].clone();
                        g.f.add_to(&gp, row, &v);
                    }
                });
            }
            _ => for_shape(&mut g, shape, body),
        }
        // after the loop: propagate each node's adjoints (parents first)
        for node in nodes.iter().rev() {
            let key = *node as *const M as usize;
            let Some((fw, Some(ad))) = g.split.remove(&key) else { continue };
            match node {
                M::MatVec { rows, vec, .. } => {
                    let (mp, _, c) = g.matvec_parts(node);
                    let n = g.dim(rows);
                    let gp = match &**vec {
                        M::ParamV(p, _) => g.gptr[p].clone(),
                        _ => unreachable!(),
                    };
                    let ad2 = ad.clone();
                    rows_axpy_blocked(&mut g, &mp, &c, &n, &move |g: &mut Mg, i: &str| g.f.load(&ad2, i), &gp);
                }
                M::Cumsum { inner, shape: own, .. } => {
                    // adjoint of a running sum: a reverse running sum
                    let (rows, cols) = match own {
                        SShape::Vec(n) => ("1".to_string(), g.dim(n)),
                        SShape::Mat(r, c) => (g.dim(r), g.dim(c)),
                        SShape::Scalar => unreachable!(),
                    };
                    let is_vec = matches!(own, SShape::Vec(_));
                    let last = g.f.iop("sub nsw", &cols, "1");
                    let ad2 = ad.clone();
                    let rscan = |g: &mut Mg, r0: &str, width: usize| {
                        let mut rs = Vec::new();
                        let mut bases = Vec::new();
                        let mut accs = Vec::new();
                        for l in 0..width {
                            let r = g.f.iadd(r0, &l.to_string());
                            bases.push(g.f.imul(&r, &cols));
                            rs.push(r);
                            accs.push(g.f.acc_new(&fconst(0.0)));
                        }
                        for_range(g, "0", &cols, |g, k| {
                            let c = g.f.iop("sub nsw", &last, k);
                            for l in 0..width {
                                let flat = g.f.iadd(&bases[l], &c);
                                let ix = if is_vec { Ix::vec(&c) } else { Ix { flat: flat.clone(), row: rs[l].clone(), col: c.clone() } };
                                let a = g.f.load(&ad2, &flat);
                                let old = g.f.acc_get(&accs[l]);
                                let sum = g.f.fadd(&old, &a);
                                g.f.emit(format!("store double {sum}, ptr {}", accs[l]));
                                let mut vals = HashMap::new();
                                g.fwd(inner, &ix, &mut vals);
                                g.bwd(inner, &sum, &ix, &vals);
                            }
                        });
                    };
                    let nb = g.f.iop("sdiv", &rows, "4");
                    for_range(&mut g, "0", &nb, |g, b| {
                        let r0 = g.f.imul(b, "4");
                        rscan(g, &r0, 4);
                    });
                    let done = g.f.imul(&nb, "4");
                    for_range(&mut g, &done, &rows, |g, r| rscan(g, r, 1));
                }
                _ => unreachable!(),
            }
            g.split.insert(key, (fw, Some(ad)));
        }
    }

    for ((n, t), (_, off, _)) in tm.params.iter().zip(&layout) {
        if let Ty::Vector(_, Dom::Positive) = t {
            // d/du_k = adj_k * sigma_k + 1
            let (v, a, len) = pos_bufs[n].clone();
            let gp = g.f.gep("%grad", off);
            for_range(&mut g, "0", &len, |g, k| {
                let ad = g.f.load(&a, k);
                let x = g.f.load(&v, k);
                let t = g.f.fmul(&ad, &x);
                let t = g.f.fadd(&t, &fconst(1.0));
                g.f.store(&t, &gp, k);
            });
        }
        if let Ty::Scalar(d) = t {
            let a = g.padj[n].clone();
            let adj = g.f.acc_get(&a);
            let gv = if *d == Dom::Positive {
                let v = g.pval[n].clone();
                let t = g.f.fmul(&adj, &v);
                g.f.fadd(&t, &fconst(1.0))
            } else {
                adj
            };
            g.f.store(&gv, "%grad", off);
        }
    }
    let r = g.f.acc_get(&lp);
    let header = format!("define double @mint_model_{}_logp(ptr noalias %theta, ptr noalias %grad)", tm.name);
    g.finish(&header, &[format!("ret double {r}")]);
}

fn gen_ss_logp(g: &mut Mg, tm: &TModel, k: usize, plan: &SsPlan, sigma: &M, len: &Dim, lp: &str) {
    let ix0 = Ix::vec("0");
    let name = &tm.name;
    let gm = g.f.load_ptr(&format!("@mint_model_{name}_ss{k}_G"));
    let b = g.f.load_ptr(&format!("@mint_model_{name}_ss{k}_b"));
    let c = g.f.load_f64(&format!("@mint_model_{name}_ss{k}_c"));
    let q = g.f.load_i64(&format!("@mint_model_{name}_ss{k}_q"));
    let th = g.f.reg();
    g.f.emit(format!("{th} = alloca double, i64 {q}"));
    let gt = g.f.reg();
    g.f.emit(format!("{gt} = alloca double, i64 {q}"));
    // gather parameters into theta
    let mut off = "0".to_string();
    let mut offs = Vec::new();
    for t in &plan.terms {
        offs.push(off.clone());
        match t {
            Term::Block { param, cols, .. } => {
                let p = g.pptr[param].clone();
                let cn = g.dim(cols);
                let o = off.clone();
                for_range(g, "0", &cn, |g, kk| {
                    let x = g.f.load(&p, kk);
                    let idx = g.f.iadd(&o, kk);
                    g.f.store(&x, &th, &idx);
                });
                off = g.f.iadd(&off, &cn);
            }
            Term::Ones { param, .. } | Term::Col { param, .. } => {
                let v = g.pval[param].clone();
                g.f.store(&v, &th, &off);
                off = g.f.iadd(&off, "1");
            }
        }
    }
    // Gt = G theta, t1 = theta'b, t2 = theta'G theta
    let t1 = g.f.acc_new(&fconst(0.0));
    let t2 = g.f.acc_new(&fconst(0.0));
    for_range(g, "0", &q, |g, j| {
        let row = g.f.imul(j, &q);
        let acc = g.f.acc_new(&fconst(0.0));
        for_range(g, "0", &q, |g, kk| {
            let idx = g.f.iadd(&row, kk);
            let a = g.f.load(&gm, &idx);
            let x = g.f.load(&th, kk);
            let t = g.f.fmul(&a, &x);
            g.f.acc_add(&acc, &t);
        });
        let s = g.f.acc_get(&acc);
        g.f.store(&s, &gt, j);
        let tj = g.f.load(&th, j);
        let bj = g.f.load(&b, j);
        let x = g.f.fmul(&tj, &bj);
        g.f.acc_add(&t1, &x);
        let y = g.f.fmul(&tj, &s);
        g.f.acc_add(&t2, &y);
    });
    let t1v = g.f.acc_get(&t1);
    let t2v = g.f.acc_get(&t2);
    let two_t1 = g.f.fmul(&fconst(2.0), &t1v);
    let qv = g.f.fsub(&c, &two_t1);
    let qv = g.f.fadd(&qv, &t2v); // residual sum of squares
    let mut vals = HashMap::new();
    let s = g.fwd(sigma, &ix0, &mut vals);
    let s2 = g.f.fmul(&s, &s);
    let inv_s2 = g.f.fdiv(&fconst(1.0), &s2);
    let nd = g.dim(len);
    let nf = g.f.sitofp(&nd);
    let ls = g.f.intrinsic1(g.m, "llvm.log.f64", &s);
    let a = g.f.fmul(&nf, &ls);
    let bq = g.f.fmul(&qv, &inv_s2);
    let bq = g.f.fmul(&fconst(0.5), &bq);
    let term = g.f.fadd(&a, &bq);
    let term = g.f.fneg(&term);
    g.f.acc_add(lp, &term);
    // d/dtheta = (b - G theta) / s^2, scattered back to the parameters
    for (t, o) in plan.terms.iter().zip(&offs) {
        match t {
            Term::Block { param, cols, .. } => {
                let gp = g.gptr[param].clone();
                let cn = g.dim(cols);
                let o = o.clone();
                for_range(g, "0", &cn, |g, kk| {
                    let idx = g.f.iadd(&o, kk);
                    let bj = g.f.load(&b, &idx);
                    let gj = g.f.load(&gt, &idx);
                    let d = g.f.fsub(&bj, &gj);
                    let d = g.f.fmul(&d, &inv_s2);
                    g.f.add_to(&gp, kk, &d);
                });
            }
            Term::Ones { param, .. } | Term::Col { param, .. } => {
                let bj = g.f.load(&b, o);
                let gj = g.f.load(&gt, o);
                let d = g.f.fsub(&bj, &gj);
                let d = g.f.fmul(&d, &inv_s2);
                let acc = g.padj[param].clone();
                g.f.acc_add(&acc, &d);
            }
        }
    }
    if sigma.active() {
        // d/ds [-n log s - Q/(2 s^2)] = -n/s + Q/s^3
        let a = g.f.fdiv(&nf, &s);
        let bq = g.f.fmul(&qv, &inv_s2);
        let bq = g.f.fdiv(&bq, &s);
        let ds = g.f.fsub(&bq, &a);
        g.bwd(sigma, &ds, &ix0, &vals);
    }
}

fn gen_constrain(m: &mut Module, tm: &TModel, opts: &Opts) {
    let mut g = Mg::new(m, tm, opts.strict_fp);
    let (layout, total) = g.layout(tm);
    g.f.memcpy(g.m, "%out", "%unc", &total);
    for ((_, t), (_, off, _)) in tm.params.iter().zip(&layout) {
        match t {
            Ty::Scalar(Dom::Positive) => {
                let u = g.f.load("%unc", off);
                let v = g.f.intrinsic1(g.m, "llvm.exp.f64", &u);
                g.f.store(&v, "%out", off);
            }
            Ty::Vector(d, Dom::Positive) => {
                let len = g.dim(d);
                let off = off.clone();
                for_range(&mut g, "0", &len, |g, k| {
                    let i = g.f.iadd(&off, k);
                    let u = g.f.load("%unc", &i);
                    let v = g.f.intrinsic1(g.m, "llvm.exp.f64", &u);
                    g.f.store(&v, "%out", &i);
                });
            }
            _ => {}
        }
    }
    let header = format!("define void @mint_model_{}_constrain(ptr %unc, ptr %out)", tm.name);
    g.finish(&header, &["ret void".into()]);
}

fn gen_sample_fn(m: &mut Module, tm: &TModel, opts: &Opts) {
    let mut g = Mg::new(m, tm, opts.strict_fp);
    let (layout, total) = g.layout(tm);
    let k = tm.params.len();
    let names = g.f.alloca(&format!("[{k} x ptr]"));
    let sizes = g.f.alloca(&format!("[{k} x i64]"));
    for (j, (n, _, size)) in layout.iter().enumerate() {
        let s = g.m.string(n);
        let p = g.f.reg();
        g.f.emit(format!("{p} = getelementptr inbounds [{k} x ptr], ptr {names}, i64 0, i64 {j}"));
        g.f.emit(format!("store ptr {s}, ptr {p}"));
        let q = g.f.reg();
        g.f.emit(format!("{q} = getelementptr inbounds [{k} x i64], ptr {sizes}, i64 0, i64 {j}"));
        let sz = size.clone().unwrap_or("-1".into());
        g.f.emit(format!("store i64 {sz}, ptr {q}"));
    }
    let r = g.f.reg();
    let name = &tm.name;
    g.f.emit(format!(
        "{r} = call ptr @mint_sample(ptr @mint_model_{name}_logp, ptr @mint_model_{name}_constrain, i64 {total}, i64 %draws, i64 %warmup, i64 %chains, i64 %seed, i64 {k}, ptr {names}, ptr {sizes})"
    ));
    let header = format!("define ptr @mint_model_{name}_sample(i64 %draws, i64 %warmup, i64 %chains, i64 %seed)");
    g.finish(&header, &[format!("ret ptr {r}")]);
}
