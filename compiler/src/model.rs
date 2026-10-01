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

use std::collections::{HashMap, HashSet};

use crate::ast::BinOp;
use crate::check::{bcast_axis, Axis, Dist, Func, SShape, TExpr, TModel, TModelStmt, TK};
use crate::codegen::{scalar_func, Opts};
use crate::explain::{self, ModelRep, ParamRep, Poly, StmtRep};
use crate::ir::{fconst, for_range, for_range_md, if_then, rows_axpy_blocked, rows_dot_blocked, Fb, HasFb, Module, Narrow, ROW_BLOCK};
use crate::types::{Dim, Dom, Ty};

pub fn data_global(model: &str, name: &str) -> String {
    format!("@mint_model_{model}_data_{name}")
}

/// The narrow copy of a data buffer (null when its values need doubles).
fn narrow_global(model: &str, name: &str) -> String {
    format!("@mint_model_{model}_nd_{name}")
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

/// Whether any leaf is indexed by row or column (rather than by element).
fn uses_axes(e: &M) -> bool {
    match e {
        M::DataV(_, ax) | M::ParamV(_, ax) | M::Cumsum { ax, .. } => *ax != Ax::Flat,
        M::MatVec { .. } => true,
        M::Bin(_, a, b) => uses_axes(a) || uses_axes(b),
        M::Neg(a) | M::Func(_, a) => uses_axes(a),
        _ => false,
    }
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
            let cm = g.is_cm(r, c);
            let (r, c) = (g.dim(r), g.dim(c));
            if cm {
                // the scan layout (see Mg::cm_row)
                for_range(g, "0", &c, |g, col| {
                    for_range(g, "0", &r, |g, row| {
                        let flat = g.cm_flat(row, col, &r, &c);
                        body(g, &Ix { flat, row: row.to_string(), col: col.to_string() });
                    });
                });
            } else {
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
}

/// Matrix shapes (rows, cols) that some cumsum runs along. The compiler stores
/// every model quantity of such a shape column-major (parameters inside the
/// sampler's vector, data copied once at initialisation, scratch buffers), so
/// the running sum along a row becomes an elementwise add of whole columns,
/// and every loop over the shape walks contiguous memory. Users see the
/// original layout: draws and printed gradients are converted back.
fn scan_shapes(e: &M, out: &mut Vec<(Dim, Dim)>) {
    match e {
        M::Cumsum { inner, shape, .. } => {
            if let SShape::Mat(r, c) = shape {
                let k = (r.clone(), c.clone());
                if !out.contains(&k) {
                    out.push(k)
                }
            }
            scan_shapes(inner, out);
        }
        M::MatVec { vec, .. } => scan_shapes(vec, out),
        M::Bin(_, a, b) => {
            scan_shapes(a, out);
            scan_shapes(b, out);
        }
        M::Neg(a) | M::Func(_, a) => scan_shapes(a, out),
        _ => {}
    }
}

/// Shapes of the data matrices in matrix-vector products; those stay row-major.
fn matvec_shapes(e: &M, out: &mut Vec<(Dim, Dim)>) {
    let mut mv = Vec::new();
    matvecs(e, &mut mv);
    for m in mv {
        if let M::MatVec { rows, cols, .. } = m {
            out.push((rows.clone(), cols.clone()));
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
    fission_why(dist, lhs, args).is_ok()
}

/// Why a statement wants loop fission (see `wants_fission`), or why not.
fn fission_why(dist: Dist, lhs: &M, args: &[M]) -> Result<String, String> {
    let mut mv = Vec::new();
    matvecs(lhs, &mut mv);
    for a in args {
        matvecs(a, &mut mv);
    }
    if mv.is_empty() {
        return Err("no matrix-vector product".into());
    }
    let scale_indexed = dist == Dist::Normal && args[1].indexed();
    if dist == Dist::BernoulliLogit || dist == Dist::PoissonLog || dist == Dist::Exponential {
        Ok(format!("{dist:?} needs exp or log"))
    } else if scale_indexed {
        Ok("the scale varies by observation, so the density needs a log per observation".into())
    } else if has_func(lhs) || args.iter().any(has_func) {
        Ok("the expression calls a function (exp, log, ...)".into())
    } else {
        Err("no transcendental function next to the product".into())
    }
}

fn plan_suffstats(dist: Dist, lhs: &M, args: &[M], shape: &SShape) -> Result<SsPlan, &'static str> {
    if dist != Dist::Normal {
        return Err("not Normal");
    }
    if !matches!(shape, SShape::Vec(_)) {
        return Err("not vector-shaped");
    }
    if lhs.active() || !lhs.indexed() {
        return Err("the outcome is not a data vector");
    }
    if lhs.has_cumsum() || args.iter().any(|a| a.has_cumsum()) {
        return Err("it contains a running sum");
    }
    let (mu, sigma) = (&args[0], &args[1]);
    if sigma.indexed() {
        return Err("the scale is not one scalar");
    }
    let mut terms = Vec::new();
    let mut offset = Vec::new();
    if !affine(mu, 1.0, &mut terms, &mut offset) || terms.is_empty() {
        return Err("the mean is not affine in the parameters with data coefficients");
    }
    Ok(SsPlan { terms, offset })
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
    /// Matrix shapes stored column-major (see `scan_shapes`).
    cm: Vec<(Dim, Dim)>,
    /// In a fused scan statement: the adjoint of each scan node for the
    /// current element, summed in a register and stored once.
    node_acc: HashMap<usize, String>,
    /// In a fused scan statement: per-lane partial-sum buffers (cols x 4)
    /// for the gradients of column-indexed vector parameters.
    col_part: HashMap<String, String>,
    /// The buffers behind `col_part`, for the whole function; `col_part`
    /// holds only the current kernel's, so other statements' uses of the same
    /// parameter are unaffected.
    part_bufs: HashMap<String, String>,
    /// In a vectorised kernel: vector accumulators for scalar parameters.
    vpadj: HashMap<String, String>,
    /// In a fused scan kernel: where the current element's running-sum
    /// adjoints live in the group scratch.
    ad_at: Option<String>,
    /// In a fused scan kernel: per-element register accumulators for the
    /// gradients of matrix parameters the kernel owns (stored, not added).
    elem_acc: HashMap<String, String>,
    /// Fused scan kernels: per-group scratch (keyed by the first running sum).
    kscratch: HashMap<usize, String>,
    /// A precomputed exp(eta) for the next PoissonLog density, or
    /// exp(-|eta|) for the next BernoulliLogit density.
    exp_override: Option<String>,
    /// A precomputed log(1 + exp(-|eta|)) for the next BernoulliLogit density.
    log_override: Option<String>,
    /// A precomputed 1/(1 + exp(-|eta|)) for the next BernoulliLogit density.
    q_override: Option<String>,
    /// Fused scan kernel: the adjoint C computed for copy k, for R.
    pending_ad: HashMap<usize, String>,
    /// Scratch buffer register -> its `mint_ws_slot` index, so an outlined
    /// kernel can ask for the same per-thread slot.
    ws_slot_of: HashMap<String, usize>,
    /// When fused scan kernels may run on several threads: the register
    /// holding the requested thread count (`mint_par_threads`).
    par_nt: Option<String>,
    /// The leap entry point (`gen_logp` with `leap`): the matrix parameters
    /// whose leapfrog work the fused scan kernels hand to the runtime's hook,
    /// with their offsets in theta.
    leap_cov: Vec<(String, String)>,
    /// The model's parameters (for `leap_blocks_of`).
    leap_tm_params: Vec<(String, Ty)>,
    /// Write to the explain report: only the first copy of `logp` (not the
    /// narrow-data variants or the leap entry point, which repeat it).
    rec: bool,
    /// The statement being emitted, for `Mg::note`.
    cur_k: usize,
    /// The current fused scan kernel's pass structure has been reported.
    scan_noted: bool,
    /// The switch that keeps fused scan kernels on one thread, if any.
    par_off: Option<&'static str>,
    /// Report lines waiting for the current statement's kernel description.
    pending: Vec<String>,
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
            cm: Vec::new(),
            node_acc: HashMap::new(),
            col_part: HashMap::new(),
            part_bufs: HashMap::new(),
            vpadj: HashMap::new(),
            ad_at: None,
            elem_acc: HashMap::new(),
            kscratch: HashMap::new(),
            exp_override: None,
            log_override: None,
            q_override: None,
            pending_ad: HashMap::new(),
            ws_slot_of: HashMap::new(),
            par_nt: None,
            leap_cov: Vec::new(),
            leap_tm_params: tm.params.clone(),
            rec: false,
            cur_k: 0,
            scan_noted: false,
            par_off: None,
            pending: Vec::new(),
        };
        for d in &tm.dims {
            let v = g.f.load_i64(&dim_global(&tm.name, d));
            g.dims.insert(d.clone(), v);
        }
        for (n, t) in &tm.data {
            let gl = data_global(&tm.name, n);
            if t.is_buffer() {
                let p = g.f.load_ptr(&gl);
                if let Some(k) = g.m.narrow_data.get(n).copied() {
                    let np = g.f.load_ptr(&narrow_global(&tm.name, n));
                    g.f.narrow.insert(p.clone(), (np, k));
                }
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

    fn is_cm(&self, r: &Dim, c: &Dim) -> bool {
        self.cm.iter().any(|(a, b)| a == r && b == c)
    }

    /// Where row `row` of a matrix in the scan layout lives: element (row,
    /// col) is at base + col * stride. The rows are stored in blocks of
    /// CM_BLOCK (a fused scan kernel's group of rows), each block column by
    /// column, CM_BLOCK elements per column; the last rows mod CM_BLOCK rows
    /// form a last block of their own, as many elements per column. So a
    /// group's elements are one contiguous range, and so are a thread's
    /// groups.
    fn cm_row(&mut self, row: &str, rows: &str, cols: &str) -> (String, String) {
        let b = CM_BLOCK.to_string();
        let nfull = self.f.iop("sdiv", rows, &b);
        let full = self.f.imul(&nfull, &b);
        let infull = self.f.reg();
        self.f.emit(format!("{infull} = icmp slt i64 {row}, {full}"));
        let rb = self.f.iop("srem", row, &b);
        let r0 = self.f.iop("sub nsw", row, &rb);
        let fb = self.f.imul(&r0, cols);
        let fb = self.f.iadd(&fb, &rb);
        let tb = self.f.imul(&full, cols);
        let to = self.f.iop("sub nsw", row, &full);
        let tb = self.f.iadd(&tb, &to);
        let ts = self.f.iop("sub nsw", rows, &full);
        let base = self.f.reg();
        self.f.emit(format!("{base} = select i1 {infull}, i64 {fb}, i64 {tb}"));
        let stride = self.f.reg();
        self.f.emit(format!("{stride} = select i1 {infull}, i64 {b}, i64 {ts}"));
        (base, stride)
    }

    /// Index of element (row, col) of a matrix in the scan layout.
    fn cm_flat(&mut self, row: &str, col: &str, rows: &str, cols: &str) -> String {
        let (b, st) = self.cm_row(row, rows, cols);
        let o = self.f.imul(col, &st);
        self.f.iadd(&b, &o)
    }

    /// Emits `store(p[col], acc)` style flushes of column accumulators.
    fn col_accs_begin(&mut self, params: &[String]) {
        for p in params {
            let acc = self.f.acc_new(&fconst(0.0));
            self.inv_acc.insert((p.clone(), Ax::Col), acc);
        }
    }

    fn col_accs_flush(&mut self, params: &[String], col: &str) {
        for p in params {
            let acc = self.inv_acc.remove(&(p.clone(), Ax::Col)).unwrap();
            let v = self.f.acc_get(&acc);
            let gp = self.gptr[p].clone();
            self.f.add_to(&gp, col, &v);
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
        if let Some(v) = vals.get(&key) {
            return v.clone();
        }
        let v = match e {
            M::Const(c) => fconst(*c),
            M::DimV(s) => {
                let d = self.dims[s].clone();
                let v = self.f.sitofp(&d);
                self.f.splat(&v)
            }
            M::DataS(n) => {
                let v = self.data_s[n].clone();
                self.f.splat(&v)
            }
            M::DataV(n, Ax::Col) if self.f.lanes > 1 => {
                // lanes run along rows: one value per column, broadcast
                let p = self.data_p[n].clone();
                let v = self.f.load_scalar(&p, &ix.col);
                self.f.splat(&v)
            }
            M::DataV(n, ax) => {
                let p = self.data_p[n].clone();
                self.f.load(&p, ix.at(*ax))
            }
            M::DataM(n) => {
                let p = self.data_p[n].clone();
                self.f.load(&p, &ix.flat)
            }
            M::ParamS(n) => {
                let v = self.pval[n].clone();
                self.f.splat(&v)
            }
            M::ParamV(n, Ax::Col) if self.f.lanes > 1 => {
                let p = self.pptr[n].clone();
                let v = self.f.load_scalar(&p, &ix.col);
                self.f.splat(&v)
            }
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
        if let Some(acc) = self.node_acc.get(&key).cloned() {
            self.f.acc_add(&acc, adj);
            return;
        }
        if let Some((_, Some(ad))) = self.split.get(&key) {
            let ad = ad.clone();
            let at = match (e, &self.ad_at) {
                (M::Cumsum { .. }, Some(at)) => at.clone(),
                (M::Cumsum { ax, shape, .. }, None) => cumsum_index(ix, *ax, shape),
                _ => ix.flat.clone(),
            };
            self.f.add_to(&ad, &at, adj);
            return;
        }
        let i = ix;
        match e {
            M::ParamS(n) => {
                let acc = self.vpadj.get(n).unwrap_or(&self.padj[n]).clone();
                self.f.acc_add(&acc, adj);
            }
            M::ParamV(n, Ax::Col) if self.col_part.contains_key(n) && !self.inv_acc.contains_key(&(n.clone(), Ax::Col)) => {
                // per-lane partial sums at [col * 4 + lane], reduced later
                let buf = self.col_part[n].clone();
                let at = self.f.imul(&ix.col, "4");
                self.f.add_to(&buf, &at, adj);
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
                if let Some(acc) = self.elem_acc.get(n).cloned() {
                    self.f.acc_add(&acc, adj);
                    return;
                }
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
                // values are needed by *, / and ^ only (a sum may be swept
                // without them)
                let va = || Self::val(vals, a);
                let vb = || Self::val(vals, b);
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
                            let t = self.f.fmul(adj, &vb());
                            self.bwd(a, &t, i, vals);
                        }
                        if b.active() {
                            let t = self.f.fmul(adj, &va());
                            self.bwd(b, &t, i, vals);
                        }
                    }
                    BinOp::Div => {
                        let q = self.f.fdiv(adj, &vb());
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
                            self.f.fmul(&fconst(2.0), &va())
                        } else {
                            let p = self.f.intrinsic2(self.m, "llvm.pow.f64", &va(), &fconst(k - 1.0));
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
                        let c = self.f.fcmp("olt", &x, &fconst(0.0));
                        self.f.select(&c, &fconst(-1.0), &fconst(1.0))
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
                let e = match self.exp_override.take() {
                    Some(e) => e,
                    None => self.bl_exp(eta),
                };
                // In vector code with Mint's log: log1p(e) from Mint's log1p on
                // [0, 1], with q = 1/(1 + e) shared with the sigmoid
                let mint_l1p = self.f.lanes > 1 && self.f.inline_log && self.m.inline_log;
                let l1p = match self.log_override.take() {
                    Some(l) => l,
                    None if mint_l1p => {
                        let u = self.f.fadd(&one, &e);
                        let q = self.f.fdiv(&one, &u);
                        self.q_override = Some(q.clone());
                        self.f.log1p01(self.m, &e, &q)
                    }
                    None => self.bl_log(&e),
                };
                let pos = self.f.intrinsic2(self.m, "llvm.maxnum.f64", eta, &fconst(0.0));
                let softplus = self.f.fadd(&pos, &l1p);
                let ye = self.f.fmul(x, eta);
                let lp = self.f.fsub(&ye, &softplus);
                let c = self.f.fcmp("oge", eta, &fconst(0.0));
                let sig = match self.q_override.take() {
                    // sigmoid(eta) = q for eta >= 0, e q otherwise
                    Some(q) => {
                        let eq = self.f.fmul(&e, &q);
                        self.f.select(&c, &q, &eq)
                    }
                    None => {
                        let den = self.f.fadd(&one, &e);
                        let num = self.f.select(&c, &one, &e);
                        self.f.fdiv(&num, &den)
                    }
                };
                let deta = self.f.fsub(x, &sig);
                (lp, vec![fconst(0.0), deta])
            }
            Dist::PoissonLog => {
                // log p(y | eta) = y*eta - exp(eta) - log(y!)  (the last term is data only)
                let eta = &a[0];
                let e = match self.exp_override.take() {
                    Some(e) => e,
                    None => self.f.intrinsic1(self.m, "llvm.exp.f64", eta),
                };
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

    /// exp(-|eta|), BernoulliLogit's exponential.
    fn bl_exp(&mut self, eta: &str) -> String {
        let abs = self.f.intrinsic1(self.m, "llvm.fabs.f64", eta);
        let na = self.f.fneg(&abs);
        self.f.intrinsic1(self.m, "llvm.exp.f64", &na)
    }

    /// log(1 + e) for e = exp(-|eta|) in (0, 1]: log rather than log1p, since
    /// the absolute error is then at most ~1e-16 and log has a vector form.
    /// --strict-fp keeps libm's log1p.
    fn bl_log(&mut self, e: &str) -> String {
        if self.f.strict {
            self.f.intrinsic1(self.m, "log1p", e)
        } else {
            let onep = self.f.fadd(&fconst(1.0), e);
            self.f.intrinsic1(self.m, "llvm.log.f64", &onep)
        }
    }

    fn finish(self, header: &str, epi: &[String]) {
        let Mg { f, m, .. } = self;
        m.funcs.push(f.finish(header, epi));
    }

    /// Adds a line to the explain report of the current statement.
    fn note(&mut self, s: impl Into<String>) {
        if self.rec {
            note(self.m, self.cur_k, s);
        }
    }
}

/// The transcendental functions in `names` (Module::math), for the report.
pub fn math_summary(names: &[String]) -> Option<String> {
    let mut out: Vec<String> = Vec::new();
    for n in names {
        let lanes = |v: &str| if v == "f64" || v.is_empty() { "scalar".to_string() } else { format!("{} lanes", v.trim_start_matches('v').trim_end_matches("f64")) };
        let l = if let Some(rest) = n.strip_prefix("mint_") {
            let (base, v) = rest.split_once("_v").map(|(b, v)| (b, format!("v{v}"))).unwrap_or((rest, String::new()));
            let base = if base == "log1p01" { "log1p" } else { base };
            format!("Mint's {base} ({})", lanes(&v))
        } else if let Some(rest) = n.strip_prefix("llvm.") {
            let (base, v) = rest.split_once('.').unwrap_or((rest, "f64"));
            format!("llvm.{base} ({})", lanes(v))
        } else {
            format!("libm {n}")
        };
        if !out.contains(&l) {
            out.push(l);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(format!("math: {}", out.join(", ")))
    }
}

/// The explain report's entry for the model being compiled (None outside
/// `mintc explain`).
fn rep(m: &mut Module) -> Option<&mut ModelRep> {
    m.log.as_deref_mut().map(|r| r.model())
}

/// Adds a line to the report of statement k (the k-th `~`).
fn note(m: &mut Module, k: usize, s: impl Into<String>) {
    if let Some(r) = rep(m) {
        r.stmts[k].lines.push(s.into());
    }
}

/// Source lines of the `~` statements, in order.
fn stmt_lines(tm: &TModel) -> Vec<u32> {
    tm.body.iter().filter_map(|s| if let TModelStmt::Tilde { span, .. } = s { Some(span.line) } else { None }).collect()
}

/// A lowered expression, for the report.
fn show_m(e: &M) -> String {
    match e {
        M::Const(c) => explain::num(*c),
        M::DimV(n) | M::DataS(n) | M::DataV(n, _) | M::DataM(n) | M::ParamS(n) | M::ParamV(n, _) | M::ParamM(n) => n.clone(),
        M::MatVec { mat, vec, .. } => format!("{mat} * {}", show_m(vec)),
        M::Cumsum { inner, .. } => format!("cumsum({})", show_m(inner)),
        M::Bin(op, a, b) => format!("({} {} {})", show_m(a), op.symbol(), show_m(b)),
        M::Neg(a) => format!("-{}", show_m(a)),
        M::Func(f, a) => format!("{}({})", f.name(), show_m(a)),
    }
}

fn narrow_name(k: Narrow) -> &'static str {
    match k {
        Narrow::I8 => "int8",
        Narrow::I16 => "int16",
        Narrow::F32 => "float",
    }
}

/// The columns of Z and their number q for a sufficient-statistics plan.
fn ss_cols(plan: &SsPlan) -> (Vec<String>, Poly) {
    let mut q = Poly::default();
    let mut cols = Vec::new();
    for t in &plan.terms {
        let c = |coef: f64| if coef == 1.0 { String::new() } else { format!("{} * ", explain::num(coef)) };
        match t {
            Term::Ones { param, coef } => {
                cols.push(format!("{}1 for {param}", c(*coef)));
                q.add(&Poly::constant(1));
            }
            Term::Col { param, data, coef } => {
                cols.push(format!("{}{} for {param}", c(*coef), show_m(data)));
                q.add(&Poly::constant(1));
            }
            Term::Block { param, mat, cols: d, coef } => {
                cols.push(format!("{}{mat} ({d} columns) for {param}", c(*coef)));
                q.add(&Poly::dims(&[d]));
            }
        }
    }
    (cols, q)
}

pub fn gen_model(m: &mut Module, tm: &TModel, opts: &Opts) {
    m.declare("declare noalias ptr @mint_ws_slot(i64, i64)");
    let name = &tm.name;
    if let Some(r) = m.log.as_deref_mut() {
        let lets: HashMap<String, TExpr> = tm.body.iter().filter_map(|s| if let TModelStmt::Let { name, value } = s { Some((name.clone(), value.clone())) } else { None }).collect();
        let mut decl = HashMap::new();
        for (n, t) in &tm.data {
            decl.insert(n.clone(), (t.clone(), format!("data declared {}", explain::src_ty(t))));
        }
        for (n, t) in &tm.params {
            decl.insert(n.clone(), (t.clone(), format!("param declared {}", explain::src_ty(t))));
        }
        let stmts = tm
            .body
            .iter()
            .filter_map(|s| {
                let TModelStmt::Tilde { lhs, dist, args, span, .. } = s else { return None };
                let text = format!("{} ~ {dist:?}({})", explain::show(lhs), args.iter().map(explain::show).collect::<Vec<_>>().join(", "));
                // the model lets it uses, and the lets those use
                let mut used = Vec::new();
                for e in std::iter::once(lhs).chain(args) {
                    explain::vars_of(e, &mut used);
                }
                let mut k = 0;
                while k < used.len() {
                    if let Some(v) = lets.get(&used[k]) {
                        explain::vars_of(v, &mut used);
                    }
                    k += 1;
                }
                let mut lines: Vec<String> = used.iter().filter_map(|n| lets.get(n).map(|v| format!("with {n} = {}", explain::show(v)))).collect();
                lines.reverse();
                // the checker's proof that a scale or rate is Positive (when
                // it is not a literal)
                let pos = match dist {
                    Dist::Normal => Some(("scale", &args[1])),
                    Dist::Exponential => Some(("rate", &args[0])),
                    _ => None,
                };
                if let Some((what, e)) = pos.filter(|(_, e)| !matches!(e.kind, TK::Num(_))) {
                    let sc = explain::Scope { lets: &lets, decl: &decl };
                    let mut chain = Vec::new();
                    explain::fact_chain(e, &sc, 0, &mut HashSet::new(), &mut chain);
                    lines.extend(chain.into_iter().enumerate().map(|(i, (d, l))| format!("{}{}{l}", "  ".repeat(d), if i == 0 { format!("{what} ") } else { String::new() })));
                }
                Some(StmtRep { line: span.line, text, lines })
            })
            .collect();
        r.models.push(ModelRep {
            name: name.clone(),
            data: tm.data.iter().map(|(n, t)| format!("{n}: {}", explain::src_ty(t))).collect(),
            stmts,
            ..Default::default()
        });
    }
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
                let k = stmts.len();
                let l = low.lower(lhs, shape).unwrap_or_else(|e| panic_model(tm, &e));
                let a: Vec<M> = args.iter().map(|x| low.lower(x, shape).unwrap_or_else(|e| panic_model(tm, &e))).collect();
                if matches!(dist, Dist::BernoulliLogit | Dist::PoissonLog) && l.active() {
                    panic_model(tm, "the outcome of BernoulliLogit and PoissonLog must be data");
                }
                if matches!(dist, Dist::BernoulliLogit | Dist::PoissonLog) && l.has_cumsum() {
                    // its support could not be checked before sampling
                    panic_model(tm, "the outcome of BernoulliLogit and PoissonLog cannot contain cumsum");
                }
                let try_ss = plan_suffstats(*dist, &l, &a, shape);
                match (&try_ss, opts.suffstats) {
                    (Ok(plan), true) => {
                        let (cols, q) = ss_cols(plan);
                        note(m, k, format!("sufficient statistics: Normal with a data outcome, one scalar scale and a mean affine in the parameters, so the likelihood depends on the data only through Z'Z, Z'y and y'y; Z = [{}], q = {q}", cols.join(", ")));
                        if !plan.offset.is_empty() {
                            let off: Vec<String> = plan.offset.iter().map(|(c, e)| if *c == 1.0 { show_m(e) } else { format!("{} * {}", explain::num(*c), show_m(e)) }).collect();
                            note(m, k, format!("data-only terms of the mean subtracted from the outcome first: {}", off.join(", ")));
                        }
                    }
                    (Ok(_), false) => note(m, k, "sufficient statistics: off (--no-suffstats)"),
                    (Err(why), _) if *dist == Dist::Normal && matches!(shape, SShape::Vec(_)) && !l.active() && l.indexed() => note(m, k, format!("sufficient statistics: no, {why}")),
                    _ => {}
                }
                let ss = if opts.suffstats { try_ss.ok() } else { None };
                let fission = opts.fission && ss.is_none() && matches!(shape, SShape::Vec(_)) && wants_fission(*dist, &l, &a);
                match fission_why(*dist, &l, &a) {
                    Ok(why) if fission => note(m, k, format!("loop fission: {why}")),
                    Ok(_) if !opts.fission && ss.is_none() => note(m, k, format!("loop fission: off (--no-fission); {} inside the loop, one row dot product per element", mv_list(&l, &a))),
                    Err(why) if ss.is_none() && why != "no matrix-vector product" => note(m, k, format!("loop fission: no, {why}; {} inside the loop, one row dot product per element", mv_list(&l, &a))),
                    _ => {}
                }
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

    let mut cm = Vec::new();
    let mut scanned = Vec::new();
    let mut mv = Vec::new();
    for Stmt::Tilde { lhs, args, .. } in &stmts {
        for e in std::iter::once(lhs).chain(args) {
            scan_shapes(e, &mut scanned);
            matvec_shapes(e, &mut mv);
        }
    }
    if opts.scan_layout {
        cm = scanned.clone();
        cm.retain(|s| !mv.contains(s));
    }
    if let Some(r) = rep(m) {
        for (rd, cd) in &scanned {
            r.layout.push(if cm.contains(&(rd.clone(), cd.clone())) {
                format!("Matrix[{rd}, {cd}]: scan layout, because a running sum runs along {cd}: rows in blocks of {CM_BLOCK}, each block stored column by column (the {CM_BLOCK} rows at column 0, then column 1, ...); the last {rd} mod {CM_BLOCK} rows form a block of their own")
            } else if !opts.scan_layout {
                format!("Matrix[{rd}, {cd}]: row-major (--no-scan-layout), although a running sum runs along {cd}")
            } else {
                format!("Matrix[{rd}, {cd}]: row-major: a matrix-vector product uses the same shape")
            });
        }
        if scanned.is_empty() {
            r.layout.push("row-major throughout (no running sum over a matrix)".into());
        }
    }

    // Narrow data: `init` picks, per sample() call, a variant of `logp`
    // whose vector kernels read a narrow copy of each data buffer whose
    // values it holds exactly (variant 0 reads only doubles).
    let mut nlog = Vec::new();
    let cands = narrow_candidates(tm, &stmts, opts, &cm, m.avx2, &mut nlog);
    for (n, _) in &cands {
        m.globals.push(format!("{} = internal global ptr null", narrow_global(name, n)));
    }
    let nv: usize = cands.iter().map(|(_, k)| k.len() + 1).product();
    if nv > 1 {
        m.globals.push(format!("@mint_model_{name}_variant = internal global i64 0"));
        nlog.push(format!("{nv} variants of logp are compiled (variant 0 reads only doubles); init records which one the data allows, and sample() passes that one to the sampler"));
    }
    if let Some(r) = rep(m) {
        if nlog.is_empty() {
            r.other.push("narrow data: none, no data is read by Mint's own vector kernels".into());
        }
        r.narrow = nlog;
    }
    gen_init(m, tm, &stmts, opts, &cm, &cands);
    gen_logp(m, tm, &stmts, opts, &cm, false);
    for v in 1..nv {
        m.narrow_data = narrow_variant(&cands, v).into_iter().collect();
        m.variant = format!("_n{v}");
        gen_logp(m, tm, &stmts, opts, &cm, false);
    }
    m.narrow_data.clear();
    m.variant.clear();
    if nv > 1 {
        let fns: Vec<String> = (0..nv).map(|v| if v == 0 { format!("ptr @mint_model_{name}_logp") } else { format!("ptr @mint_model_{name}_logp_n{v}") }).collect();
        m.globals.push(format!("@mint_model_{name}_logp_table = internal constant [{nv} x ptr] [{}]", fns.join(", ")));
    }
    // the fused leapfrog's copy reads the wide data (variant 0)
    let leap = opts.fused_leapfrog && gen_logp(m, tm, &stmts, opts, &cm, true);
    gen_constrain(m, tm, opts, &cm);
    let permute = gen_permute(m, tm, opts, &cm);
    gen_sample_fn(m, tm, opts, permute, nv, leap);
}

/// Most variants of `logp` generated for narrow data. Each is a full copy
/// for clang to compile: the logistic model took 0.14 s to build with one,
/// 0.73 s with 8.
const MAX_NARROW_VARIANTS: usize = 4;

/// The data buffers that Mint's own vector kernels load (the vectorised
/// fused scan kernel, with the statements it absorbs, and the fission
/// kernel), each with the narrow types to try for it, narrowest first.
/// Which one is used, if any, is decided at run time from the values.
fn narrow_candidates(tm: &TModel, stmts: &[Stmt], opts: &Opts, cm: &[(Dim, Dim)], avx2: bool, log: &mut Vec<String>) -> Vec<(String, Vec<Narrow>)> {
    fn leaves(e: &M, out: &mut Vec<String>) {
        match e {
            // a column-indexed vector in a kernel whose lanes run along rows
            // is one scalar load per column: left wide
            M::DataV(n, ax) if *ax != Ax::Col => out.push(n.clone()),
            M::DataM(n) => out.push(n.clone()),
            M::MatVec { mat, vec, .. } => {
                out.push(mat.clone());
                leaves(vec, out);
            }
            M::Cumsum { inner, .. } => leaves(inner, out),
            M::Bin(_, a, b) => {
                leaves(a, out);
                leaves(b, out);
            }
            M::Neg(a) | M::Func(_, a) => leaves(a, out),
            _ => {}
        }
    }
    let mut names: Vec<String> = Vec::new();
    for s in stmts {
        let Stmt::Tilde { dist, lhs, args, shape, ss: None, fission } = s else { continue };
        let nodes = stmt_globals(lhs, args, *fission);
        let scan = opts.scan_fusion && fused_scan_rows_cm(cm, lhs, args, shape, *fission).is_some() && kernel_lanes(*dist, lhs, args, false) > 1;
        let fk = opts.fission_kernel && !opts.strict_fp && *fission && fission_kernel_ok(lhs, args, &nodes);
        if !(scan || fk) {
            continue;
        }
        leaves(lhs, &mut names);
        for a in args {
            leaves(a, &mut names);
        }
        if scan {
            // element-wise statements over the same shape run in its reverse loop
            for s2 in stmts {
                let Stmt::Tilde { dist: d2, lhs: l2, args: a2, shape: sh2, ss: None, fission: f2 } = s2 else { continue };
                if sh2 == shape && stmt_globals(l2, a2, *f2).is_empty() && !uses_axes(l2) && !a2.iter().any(uses_axes) && kernel_lanes(*d2, l2, a2, *f2) > 1 {
                    leaves(l2, &mut names);
                    for a in a2 {
                        leaves(a, &mut names);
                    }
                }
            }
        }
    }
    // Off under --strict-fp, and without AVX2 (the barrier in Fb::opaque
    // needs a vector register for <4 x double>; programs are built for the host).
    if !opts.narrow_data || opts.strict_fp || !avx2 {
        let read: Vec<&String> = tm.data.iter().map(|(n, _)| n).filter(|n| names.contains(n)).collect();
        if !read.is_empty() {
            let why = if !opts.narrow_data {
                "--no-narrow-data"
            } else if opts.strict_fp {
                "--strict-fp"
            } else {
                "the host has no AVX2"
            };
            log.push(format!("off ({why}); the vector kernels read {} as doubles", read.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
        }
        return Vec::new();
    }
    // (name, types, whether the model makes it integer-valued)
    // MINTC_NARROW_VARIANTS overrides the limit (for tests that need more
    // types per buffer than the default allows)
    let max_variants = std::env::var("MINTC_NARROW_VARIANTS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|v| *v >= 1).unwrap_or(MAX_NARROW_VARIANTS);
    let mut cands: Vec<(String, Vec<Narrow>, bool)> = Vec::new();
    // (for the report: why a candidate's types are what they are)
    let mut why: HashMap<String, &'static str> = HashMap::new();
    for (n, _) in &tm.data {
        if !names.contains(n) {
            continue;
        }
        // A BernoulliLogit outcome is checked to be 0 or 1 before sampling,
        // so int8 holds it (unless it contains -0.0, which the check accepts
        // and int8 cannot hold: it then stays double), and a PoissonLog
        // outcome is checked to be counts. Anything else may be small
        // integers (indicators, codes) or values exact in float.
        let outcome = |d: Dist| {
            stmts.iter().any(|s| {
                let Stmt::Tilde { dist, lhs, .. } = s;
                *dist == d && matches!(lhs, M::DataV(x, _) | M::DataM(x) if x == n)
            })
        };
        let (binary, counts) = (outcome(Dist::BernoulliLogit), outcome(Dist::PoissonLog));
        let kinds = if binary { vec![Narrow::I8] } else { vec![Narrow::I8, Narrow::I16, Narrow::F32] };
        cands.push((n.clone(), kinds, binary || counts));
        why.insert(
            n.clone(),
            if binary {
                " (a BernoulliLogit outcome, checked to be 0 or 1, so int8 only)"
            } else if counts {
                " (a PoissonLog outcome, checked to be counts)"
            } else {
                ""
            },
        );
    }
    // Too many variants: give up the less likely types first (small
    // integers in real-valued data, then float and int16 for counts),
    // then whole candidates, real-valued ones first.
    let count = |c: &[(String, Vec<Narrow>, bool)]| c.iter().fold(1usize, |a, (_, k, _)| a.saturating_mul(k.len() + 1));
    let mut dropped: Vec<(String, Narrow)> = Vec::new();
    let mut removed: Vec<String> = Vec::new();
    for (ints, drop) in [(false, Narrow::I16), (false, Narrow::I8), (true, Narrow::F32), (true, Narrow::I16)] {
        if count(&cands) <= max_variants {
            break;
        }
        for (n, ks, i) in cands.iter_mut() {
            if *i == ints && ks.len() > 1 && ks.contains(&drop) {
                ks.retain(|k| *k != drop);
                dropped.push((n.clone(), drop));
            }
        }
    }
    while count(&cands) > max_variants {
        let at = cands.iter().rposition(|c| !c.2).unwrap_or(cands.len() - 1);
        removed.push(format!("{}: stays double (at most {max_variants} variants of logp)", cands[at].0));
        cands.remove(at);
    }
    for (n, ks, _) in &cands {
        let tries: Vec<&str> = ks.iter().map(|k| narrow_name(*k)).collect();
        let what = why.get(n).copied();
        let skipped: Vec<&str> = dropped.iter().filter(|(d, _)| d == n).map(|(_, k)| narrow_name(*k)).collect();
        let skipped = if skipped.is_empty() { String::new() } else { format!("; {} not tried (at most {max_variants} variants of logp)", skipped.join(" and ")) };
        log.push(format!("{n}: tries {}; the vector kernels read a copy in the first type that holds every value exactly, else the doubles{}{skipped}", tries.join(", then "), what.unwrap_or("")));
    }
    log.extend(removed);
    cands.into_iter().map(|(n, k, _)| (n, k)).collect()
}

/// The narrow data of variant v: candidate j takes digit j of v in mixed
/// radix (radix 1 + its number of types; 0 is the wide data).
fn narrow_variant(cands: &[(String, Vec<Narrow>)], v: usize) -> Vec<(String, Narrow)> {
    let mut rem = v;
    let mut out = Vec::new();
    for (n, kinds) in cands {
        let base = kinds.len() + 1;
        let d = rem % base;
        rem /= base;
        if d > 0 {
            out.push((n.clone(), kinds[d - 1]));
        }
    }
    out
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

fn gen_init(m: &mut Module, tm: &TModel, stmts: &[Stmt], opts: &Opts, cm: &[(Dim, Dim)], cands: &[(String, Vec<Narrow>)]) {
    let mut g = Mg::new(m, tm, opts.strict_fp);
    g.cm = cm.to_vec();
    // Column-major data matrices: a transposed copy, made once per sample().
    for (n, t) in &tm.data {
        let Ty::Matrix(r, c, _) = t else { continue };
        if !g.is_cm(r, c) {
            continue;
        }
        let (rd, cd) = (g.dim(r), g.dim(c));
        let src = g.data_p[n].clone();
        let size = g.f.imul(&rd, &cd);
        // the previous sample()'s copy is released (free(NULL) is a no-op)
        let keep = format!("@mint_model_{}_cm_{n}", tm.name);
        g.m.globals.push(format!("{keep} = internal global ptr null"));
        let old = g.f.load_ptr(&keep);
        g.f.emit(format!("call void @mint_free(ptr {old})"));
        let dst = g.f.reg();
        g.f.emit(format!("{dst} = call ptr @mint_alloc(i64 {size})"));
        g.f.emit(format!("store ptr {dst}, ptr {keep}"));
        transpose(&mut g, &src, &dst, &rd, &cd);
        g.f.emit(format!("store ptr {dst}, ptr {}", data_global(&tm.name, n)));
        g.data_p.insert(n.clone(), dst);
        if let Some(r) = rep(g.m) {
            r.layout.push(format!("data {n}: copied into the scan layout when sample() starts"));
        }
    }
    // BernoulliLogit and PoissonLog outcomes are data; check their support once.
    for (k, s) in stmts.iter().enumerate() {
        let Stmt::Tilde { dist: d @ (Dist::BernoulliLogit | Dist::PoissonLog), lhs, shape, .. } = s else { continue };
        note(g.m, k, format!("the outcome is checked to be {} when sample() starts", if *d == Dist::BernoulliLogit { "0 or 1" } else { "whole numbers >= 0" }));
        let msg = g.m.string(&format!("model {}", tm.name));
        let checker = if *d == Dist::BernoulliLogit { "mint_check_binary" } else { "mint_check_count" };
        for_shape(&mut g, shape, |g, ix| {
            let v = g.fwd(lhs, ix, &mut HashMap::new());
            // report the row-major index whatever the storage order
            let at = match shape {
                SShape::Mat(_, c) => {
                    let c = g.dim(c);
                    let a = g.f.imul(&ix.row, &c);
                    g.f.iadd(&a, &ix.col)
                }
                _ => ix.flat.clone(),
            };
            g.f.emit(format!("call void @{checker}(double {v}, i64 {at}, ptr {msg})"));
        });
    }
    for (k, s) in stmts.iter().enumerate() {
        let Stmt::Tilde { lhs, shape: SShape::Vec(len), ss: Some(plan), .. } = s else { continue };
        let (_, qs) = ss_cols(plan);
        note(g.m, k, format!("init computes Z'Z ({} x {}, one triangle, mirrored), Z'y ({qs} values) and y'y once per sample() call, in one pass over the {len} observations", explain::factor(&qs), explain::factor(&qs)));
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
    // Narrow copies of the data the vector kernels read (after the layout
    // copy, so they share its order), and the variant of logp that reads
    // them: candidate j's choice is digit j of the variant (narrow_variant).
    if !cands.is_empty() {
        g.m.declare("declare ptr @mint_narrow(ptr, i64, i64, ptr, ptr)");
        let kind_at = g.f.alloca("i64");
        let mut idx = "0".to_string();
        let mut stride = 1usize;
        for (n, kinds) in cands {
            let gl = narrow_global(&tm.name, n);
            // the previous sample()'s copy is released (free(NULL) is a no-op)
            let old = g.f.load_ptr(&gl);
            g.f.emit(format!("call void @mint_free(ptr {old})"));
            let size = match tm.data.iter().find(|(d, _)| d == n).map(|(_, t)| t) {
                Some(Ty::Vector(d, _)) => g.dim(d),
                Some(Ty::Matrix(r, c, _)) => {
                    let (r, c) = (g.dim(r), g.dim(c));
                    g.f.imul(&r, &c)
                }
                _ => unreachable!("narrow candidates are data buffers"),
            };
            let mask: u32 = kinds.iter().map(|k| 1 << (k.code() - 1)).sum();
            let msg = g.m.string(&format!("model {} data {n}", tm.name));
            let src = g.data_p[n].clone();
            let np = g.f.reg();
            g.f.emit(format!("{np} = call ptr @mint_narrow(ptr {src}, i64 {size}, i64 {mask}, ptr {kind_at}, ptr {msg})"));
            g.f.emit(format!("store ptr {np}, ptr {gl}"));
            let kind = g.f.load_i64(&kind_at);
            let mut digit = "0".to_string();
            for (j, k) in kinds.iter().enumerate() {
                let c = g.f.reg();
                g.f.emit(format!("{c} = icmp eq i64 {kind}, {}", k.code()));
                let d = g.f.reg();
                g.f.emit(format!("{d} = select i1 {c}, i64 {}, i64 {digit}", j + 1));
                digit = d;
            }
            let t = g.f.imul(&digit, &stride.to_string());
            idx = g.f.iadd(&idx, &t);
            stride *= kinds.len() + 1;
        }
        g.f.emit(format!("store i64 {idx}, ptr @mint_model_{}_variant", tm.name));
    }
    let header = format!("define void @mint_model_{}_init()", tm.name);
    g.finish(&header, &["ret void".into()]);
}

/// Row-major src into the scan layout (see Mg::cm_row) in dst.
fn transpose(g: &mut Mg, src: &str, dst: &str, rows: &str, cols: &str) {
    for_range(g, "0", rows, |g, row| {
        let base = g.f.imul(row, cols);
        let (b, st) = g.cm_row(row, rows, cols);
        for_range(g, "0", cols, |g, col| {
            let i = g.f.iadd(&base, col);
            let x = g.f.load(src, &i);
            let j = g.f.imul(col, &st);
            let j = g.f.iadd(&j, &b);
            g.f.store(&x, dst, &j);
        });
    });
}

/// The scan layout in src back to row-major in dst.
fn untranspose(g: &mut Mg, src: &str, dst: &str, rows: &str, cols: &str) {
    for_range(g, "0", rows, |g, row| {
        let base = g.f.imul(row, cols);
        let (b, st) = g.cm_row(row, rows, cols);
        for_range(g, "0", cols, |g, col| {
            let j = g.f.imul(col, &st);
            let j = g.f.iadd(&j, &b);
            let x = g.f.load(src, &j);
            let i = g.f.iadd(&base, col);
            g.f.store(&x, dst, &i);
        });
    });
}

/// min(1, n): the column range of a peeled first iteration.
fn first_of(g: &mut Mg, n: &str) -> String {
    let c = g.f.reg();
    g.f.emit(format!("{c} = icmp sgt i64 {n}, 0"));
    let r = g.f.reg();
    g.f.emit(format!("{r} = select i1 {c}, i64 1, i64 0"));
    r
}

/// Emits the model's log density and gradient, `logp(theta, grad)`.
///
/// With `leap`, emits instead `leap(theta, grad, hook, hctx)`, the same
/// function with a hook for the sampler (the fused leapfrog): the matrix
/// parameters owned by exactly one fused scan kernel (all their gradient is
/// summed in its reverse loop) are covered (at most LEAP_MAX_BLOCKS). Each
/// thread of a kernel, once its groups of rows are done, so that their
/// gradient is final, calls
///
///   hook(hctx, slot, lo, len)
///
/// for each covered parameter: in the scan layout (Mg::cm_row) the
/// thread's rows are the one contiguous range theta[lo .. lo + len). The
/// runtime then does its leaf work on those elements on that thread
/// (leaf_block in runtime/mint_rt.c). `slot` is the kernel's thread index;
/// thread 0 (the calling thread) also runs the single vectors and leftover
/// rows and hands them over as one more range, and when the kernel runs
/// serially the calling thread hands over the whole parameter as slot 0
/// (leap_hook_calls). Also emits
/// `leap_blocks(out)`, which writes (offset, length) of each covered
/// parameter and returns their number, so the runtime can do the rest of
/// theta itself. Returns false, emitting nothing, when nothing would be
/// covered.
fn gen_logp(m: &mut Module, tm: &TModel, stmts: &[Stmt], opts: &Opts, cm: &[(Dim, Dim)], leap: bool) -> bool {
    let mut g = Mg::new(m, tm, opts.strict_fp);
    g.cm = cm.to_vec();
    g.rec = !leap && g.m.variant.is_empty() && g.m.log.is_some();
    let (layout, _total) = g.layout(tm);
    let lines = stmt_lines(tm);
    // explain: why a statement over a host's shape stays out of it, and why
    // a matrix parameter of the host's shape is not owned
    let mut absorb_why: HashMap<usize, String> = HashMap::new();
    let mut own_why: HashMap<usize, Vec<String>> = HashMap::new();

    // Statement fusion. An element-wise statement over the same column-major
    // shape as a fused scan (a prior on the scanned matrix, say) is absorbed
    // into the scan's reverse loop instead of making its own pass. A matrix
    // parameter all of whose gradient contributions happen in that loop, one
    // visit per element, is owned by it: its gradient is summed in a register
    // and stored once, and needs no zeroing.
    let mut absorbed: Vec<Option<usize>> = vec![None; stmts.len()];
    let mut owned: HashMap<usize, Vec<String>> = HashMap::new();
    if opts.scan_fusion {
        for (h, s) in stmts.iter().enumerate() {
            let Stmt::Tilde { dist, lhs, args, shape, ss: None, fission } = s else { continue };
            if absorbed[h].is_some() || fused_scan_rows(&g, lhs, args, shape, *fission).is_none() {
                continue;
            }
            let vec_host = kernel_lanes(*dist, lhs, args, *fission) > 1;
            for (t, s2) in stmts.iter().enumerate() {
                let Stmt::Tilde { dist: d2, lhs: l2, args: a2, shape: sh2, ss: None, fission: f2 } = s2 else { continue };
                if t == h || absorbed[t].is_some() || sh2 != shape || !stmt_globals(l2, a2, *f2).is_empty() {
                    continue;
                }
                if uses_axes(l2) || a2.iter().any(uses_axes) {
                    absorb_why.entry(t).or_insert(format!("not absorbed into the fused scan kernel of line {}: it reads row- or column-indexed vectors", lines[h]));
                    continue;
                }
                if vec_host && kernel_lanes(*d2, l2, a2, *f2) == 1 {
                    absorb_why.entry(t).or_insert(format!("not absorbed into the fused scan kernel of line {}: it has no vector form", lines[h]));
                    continue;
                }
                absorbed[t] = Some(h);
            }
            for (n, ty) in &tm.params {
                let Ty::Matrix(r, c, _) = ty else { continue };
                if &SShape::Mat(r.clone(), c.clone()) != shape {
                    continue;
                }
                let only_here = stmts.iter().enumerate().all(|(t, s2)| {
                    let Stmt::Tilde { lhs: l2, args: a2, .. } = s2;
                    let m = mentions(l2, n) || a2.iter().any(|a| mentions(a, n));
                    !m || t == h || absorbed[t] == Some(h)
                });
                let in_scans_only = !mentions_outside_scans(lhs, n) && !args.iter().any(|a| mentions_outside_scans(a, n));
                if only_here && in_scans_only {
                    owned.entry(h).or_default().push(n.clone());
                } else if mentions(lhs, n) || args.iter().any(|a| mentions(a, n)) {
                    let why = if !only_here {
                        let other: Vec<String> = stmts
                            .iter()
                            .enumerate()
                            .filter(|(t, s2)| {
                                let Stmt::Tilde { lhs: l2, args: a2, .. } = s2;
                                *t != h && absorbed[*t] != Some(h) && (mentions(l2, n) || a2.iter().any(|a| mentions(a, n)))
                            })
                            .map(|(t, _)| lines[t].to_string())
                            .collect();
                        if other.len() == 1 {
                            format!("line {} also uses it", other[0])
                        } else {
                            format!("lines {} also use it", other.join(", "))
                        }
                    } else {
                        "it is used outside the running sums".to_string()
                    };
                    own_why.entry(h).or_default().push(format!("gradient of {n} accumulated in memory, not owned: {why}"));
                }
            }
        }
    }
    if leap {
        for ((n, t), (_, off, _)) in tm.params.iter().zip(&layout) {
            // (at most LEAP_MAX_BLOCKS; any further parameter is left to the runtime)
            if matches!(t, Ty::Matrix(..)) && owned.values().filter(|v| v.contains(n)).count() == 1 && g.leap_cov.len() < LEAP_MAX_BLOCKS {
                g.leap_cov.push((n.clone(), off.clone()));
            }
        }
        if let Some(r) = rep(g.m) {
            r.other.push(if g.leap_cov.is_empty() {
                "fused leapfrog (--fused-leapfrog): not emitted, no matrix parameter is owned by exactly one fused scan kernel".to_string()
            } else {
                format!(
                    "fused leapfrog (--fused-leapfrog): the leap entry point runs the sampler's leaf work on {} from inside the fused scan kernel, on each range of rows once its gradient is final (on the kernel's threads when it runs in parallel, else on the calling thread)",
                    g.leap_cov.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")
                )
            });
        }
        if g.leap_cov.is_empty() {
            return false;
        }
        gen_leap_blocks(g.m, tm, opts, cm, &g.leap_cov.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>());
    }
    // Zero the gradient of every parameter whose gradient is accumulated
    // (scalars and Positive vectors are stored at the end; owned matrices
    // are stored by their kernel).
    let owned_all: Vec<String> = owned.values().flatten().cloned().collect();
    for ((n, t), (_, off, size)) in tm.params.iter().zip(&layout) {
        let zero = match t {
            Ty::Vector(_, d) => *d != Dom::Positive,
            Ty::Matrix(..) => !owned_all.contains(n),
            _ => false,
        };
        if zero {
            let p = g.f.gep("%grad", off);
            g.f.memzero(g.m, &p, size.as_ref().unwrap());
        }
    }
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
    // Fused scan kernels whose groups of rows can be split across the
    // chain's threads: the requested thread count is read once per call.
    g.par_off = if !opts.parallel_kernel {
        Some("--no-parallel-kernel")
    } else if opts.strict_fp {
        Some("--strict-fp")
    } else {
        None
    };
    if opts.parallel_kernel && opts.scan_fusion && !opts.strict_fp {
        let any = stmts.iter().enumerate().any(|(k, s)| {
            let Stmt::Tilde { dist, lhs, args, shape, ss: None, fission } = s else { return false };
            absorbed[k].is_none() && fused_scan_rows(&g, lhs, args, shape, *fission).is_some() && kernel_lanes(*dist, lhs, args, false) > 1
        });
        if any {
            g.m.declare("declare i64 @mint_par_threads()");
            let r = g.f.reg();
            g.f.emit(format!("{r} = call i64 @mint_par_threads()"));
            g.par_nt = Some(r);
        }
    }
    let mut ws_slots: Vec<(usize, String, bool)> = Vec::new(); // (node, len, active)
    let mut part_slots: Vec<(String, String)> = Vec::new(); // (param, 4 * cols)
    let mut kscr_slots: Vec<(usize, String)> = Vec::new(); // (first node, doubles)
    for s in stmts {
        let Stmt::Tilde { lhs, args, fission, ss: None, shape, .. } = s else { continue };
        if opts.scan_fusion && fused_scan_rows(&g, lhs, args, shape, *fission).is_some() {
            let SShape::Mat(_, c) = shape else { unreachable!() };
            let nodes = stmt_globals(lhs, args, *fission);
            let per = g.dim(c);
            let per = g.f.imul(&per, &(KERNEL_WIDTH * (1 + nodes.len() as u32)).to_string());
            kscr_slots.push((nodes[0] as *const M as usize, per));
            let n4 = g.dim(c);
            let mut n4 = g.f.imul(&n4, "4");
            if let Some(nt) = g.par_nt.clone() {
                // one slice per thread of a parallel kernel
                n4 = g.f.imul(&n4, &nt);
            }
            for p in fused_col_params(lhs, args, *fission) {
                if !part_slots.iter().any(|(q, _)| q == &p) {
                    part_slots.push((p, n4.clone()));
                }
            }
        }
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
            g.ws_slot_of.insert(r.clone(), slot);
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
        for (p, n4) in part_slots {
            let b = buf(&mut g, &n4);
            g.part_bufs.insert(p, b);
        }
        for (k, n) in kscr_slots {
            let b = buf(&mut g, &n);
            g.kscratch.insert(k, b);
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
        g.cur_k = k;
        let math0 = g.m.math.len();
        if let Some(h) = absorbed[k] {
            g.note(format!("absorbed into the fused scan kernel of line {}: it runs in that kernel's reverse loop, with no pass of its own", lines[h]));
            continue; // emitted inside its host's kernel
        }
        if let Some(w) = absorb_why.get(&k) {
            g.note(w.clone());
        }
        if let Some(plan) = ss {
            let n = match shape {
                SShape::Vec(n) => n,
                _ => unreachable!(),
            };
            let (_, q) = ss_cols(plan);
            g.note(format!("each gradient: O(q^2) = O({}^2) arithmetic on the precomputed statistics, no loop over the {n} observations", explain::factor(&q)));
            gen_ss_logp(&mut g, tm, k, plan, &args[1], n, &lp);
            let used = g.m.math[math0..].to_vec();
            if let Some(l) = math_summary(&used) {
                g.note(l);
            }
            continue;
        }
        let nodes = stmt_globals(lhs, args, *fission);
        let scan = fused_scan_why(&g.cm, lhs, args, shape, *fission);
        if scan.is_ok() && opts.scan_fusion {
            let guests: Vec<&Stmt> = stmts.iter().enumerate().filter(|(t, _)| absorbed[*t] == Some(k)).map(|(_, s)| s).collect();
            let own = owned.get(&k).cloned().unwrap_or_default();
            if g.rec {
                // reported by gen_fused_scan after its own description
                for (t, _) in stmts.iter().enumerate().filter(|(t, _)| absorbed[*t] == Some(k)) {
                    let text = rep(g.m).map(|r| r.stmts[t].text.clone()).unwrap_or_default();
                    g.pending.push(format!("absorbed: line {} ({text}), in the reverse loop", lines[t]));
                }
                for n in &own {
                    g.pending.push(format!("owned gradient: {n}; every contribution happens in this kernel's reverse loop, so it is summed in a register and stored once, never zeroed"));
                }
                g.pending.extend(own_why.get(&k).into_iter().flatten().cloned());
            }
            gen_fused_scan(&mut g, tm, k, *dist, lhs, args, shape, &nodes, &guests, &own, &lp);
            let used = g.m.math[math0..].to_vec();
            if let Some(l) = math_summary(&used) {
                g.note(l);
            }
            continue;
        }
        if matches!(shape, SShape::Mat(..)) && nodes.iter().any(|n| matches!(n, M::Cumsum { .. })) {
            match &scan {
                Ok(_) => g.note("fused scan kernel: off (--no-scan-fusion)"),
                Err(why) => g.note(format!("not a fused scan kernel: {why}")),
            }
        }
        if opts.fission_kernel && !opts.strict_fp && *fission && fission_kernel_ok(lhs, args, &nodes) {
            let SShape::Vec(nd) = shape else { unreachable!() };
            let n = g.dim(nd);
            gen_fission_kernel(&mut g, *dist, lhs, args, &nodes, &n, nd, &lp);
            if g.rec {
                let used = g.m.math[math0..].to_vec();
                if let Some(l) = math_summary(&used) {
                    g.note(l);
                }
            }
            continue;
        }
        if *fission {
            g.note(match fission_kernel_why_not(lhs, args, &nodes) {
                _ if !opts.fission_kernel => "fission kernel: off (--no-fission-kernel)".to_string(),
                _ if opts.strict_fp => "fission kernel: off (--strict-fp)".to_string(),
                Some(why) => format!("not a fission kernel: {why}"),
                None => unreachable!(),
            });
        }
        if g.rec {
            generic_notes(&mut g, lhs, args, shape, &nodes);
        }
        // before the loop: materialise each node (children first)
        for node in &nodes {
            let key = *node as *const M as usize;
            let (fw, ad) = g.split.remove(&key).unwrap();
            match node {
                M::MatVec { rows, .. } => {
                    let (mp, vp, c) = g.matvec_parts(node);
                    let n = g.dim(rows);
                    let fw2 = fw.clone();
                    rows_dot_blocked(&mut g, &mp, &vp, &c, "0", &n, &move |g: &mut Mg, i: &str, s: &str| g.f.store(s, &fw2, i));
                }
                M::Cumsum { inner, shape: SShape::Mat(rd, cd), .. } if g.is_cm(rd, cd) => {
                    // Scan layout: column c of the running sum is column c-1
                    // plus column c of the operand.
                    let (rows, cols) = (g.dim(rd), g.dim(cd));
                    let fw2 = fw.clone();
                    let pass = |g: &mut Mg, lo: &str, hi: &str, first: bool| {
                        for_range(g, lo, hi, |g, col| {
                            for_range(g, "0", &rows, |g, row| {
                                let (b, st) = g.cm_row(row, &rows, &cols);
                                let o = g.f.imul(col, &st);
                                let flat = g.f.iadd(&b, &o);
                                let ix = Ix { flat: flat.clone(), row: row.to_string(), col: col.to_string() };
                                let v = g.fwd(inner, &ix, &mut HashMap::new());
                                let s = if first {
                                    v
                                } else {
                                    let p = g.f.iop("sub nsw", &flat, &st);
                                    let prev = g.f.load(&fw2, &p);
                                    g.f.fadd(&prev, &v)
                                };
                                g.f.store(&s, &fw2, &flat);
                            });
                        });
                    };
                    let one = first_of(&mut g, &cols);
                    pass(&mut g, "0", &one, true);
                    pass(&mut g, &one, &cols, false);
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
            SShape::Mat(r, c) if nodes.is_empty() && !uses_axes(lhs) && !args.iter().any(uses_axes) => {
                // every leaf is indexed by the element itself: one flat loop,
                // whatever the storage order
                let (r, c) = (g.dim(r), g.dim(c));
                let n = g.f.imul(&r, &c);
                for_range(&mut g, "0", &n, |g, i| body(g, &Ix::vec(i)));
            }
            SShape::Mat(r, c) if g.is_cm(r, c) => {
                // Scan layout: column outer, row inner. Gradients of
                // column-indexed parameters are summed in registers per column.
                let mut cp = Vec::new();
                axis_params(lhs, Ax::Col, &mut cp);
                for a in args.iter() {
                    axis_params(a, Ax::Col, &mut cp);
                }
                let (r, c) = (g.dim(r), g.dim(c));
                for_range(&mut g, "0", &c, |g, col| {
                    g.col_accs_begin(&cp);
                    for_range(g, "0", &r, |g, row| {
                        let flat = g.cm_flat(row, col, &r, &c);
                        body(g, &Ix { flat, row: row.to_string(), col: col.to_string() });
                    });
                    g.col_accs_flush(&cp, col);
                });
            }
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
                    rows_axpy_blocked(&mut g, &mp, &c, "0", &n, &move |g: &mut Mg, i: &str| g.f.load(&ad2, i), &gp);
                }
                M::Cumsum { inner, shape: SShape::Mat(rd, cd), .. } if g.is_cm(rd, cd) => {
                    // Reverse running sum, column by column from the last, in
                    // place in the adjoint buffer.
                    let (rows, cols) = (g.dim(rd), g.dim(cd));
                    let mut cp = Vec::new();
                    axis_params(inner, Ax::Col, &mut cp);
                    let last = g.f.iop("sub nsw", &cols, "1");
                    let ad2 = ad.clone();
                    let pass = |g: &mut Mg, lo: &str, hi: &str, first: bool| {
                        for_range(g, lo, hi, |g, k| {
                            let col = g.f.iop("sub nsw", &last, k);
                            g.col_accs_begin(&cp);
                            for_range(g, "0", &rows, |g, row| {
                                let (b, st) = g.cm_row(row, &rows, &cols);
                                let o = g.f.imul(&col, &st);
                                let flat = g.f.iadd(&b, &o);
                                let a = g.f.load(&ad2, &flat);
                                let sum = if first {
                                    a
                                } else {
                                    let nx = g.f.iadd(&flat, &st);
                                    let b = g.f.load(&ad2, &nx);
                                    let s = g.f.fadd(&a, &b);
                                    g.f.store(&s, &ad2, &flat);
                                    s
                                };
                                let ix = Ix { flat: flat.clone(), row: row.to_string(), col: col.clone() };
                                let mut vals = HashMap::new();
                                g.fwd(inner, &ix, &mut vals);
                                g.bwd(inner, &sum, &ix, &vals);
                            });
                            g.col_accs_flush(&cp, &col);
                        });
                    };
                    let one = first_of(&mut g, &cols);
                    pass(&mut g, "0", &one, true);
                    pass(&mut g, &one, &cols, false);
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
        let used = g.m.math[math0..].to_vec();
        if let Some(l) = math_summary(&used) {
            g.note(l);
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
    let header = if leap {
        // grad is not noalias here: the hook reads it through the sampler's
        // own pointers
        format!("define double @mint_model_{}_leap(ptr noalias %theta, ptr %grad, ptr %hook, ptr %hctx)", tm.name)
    } else {
        format!("define double @mint_model_{}_logp{}(ptr noalias %theta, ptr noalias %grad)", tm.name, g.m.variant)
    };
    let rec = g.rec;
    g.finish(&header, &[format!("ret double {r}")]);
    if rec {
        let slots = m.funcs.last().map_or(0, |f| f.matches("@mint_ws_slot(").count());
        if let Some(r) = rep(m) {
            if slots > 0 {
                r.other.push(format!("scratch: {} in logp, each a per-thread allocation from mint_ws_slot, declared noalias (so LLVM needs no overlap checks between them)", plural(slots, "buffer")));
            }
        }
    }
    true
}

/// The explain report for a statement that takes the general path of
/// `gen_logp` (one loop over its index space, with materialised nodes
/// around it). Mirrors the branches there; a gradient pass exists only for
/// a node with an adjoint (`node.active()`, as in `ws_slots`).
fn generic_notes(g: &mut Mg, lhs: &M, args: &[M], shape: &SShape, nodes: &[&M]) {
    let prods: Vec<String> = nodes.iter().filter(|n| matches!(n, M::MatVec { .. })).map(|n| show_m(n)).collect();
    let grads: Vec<String> = nodes.iter().filter(|n| matches!(n, M::MatVec { .. }) && n.active()).map(|n| show_m(n)).collect();
    let text = match shape {
        SShape::Scalar => "one scalar term".to_string(),
        SShape::Vec(n) if !prods.is_empty() => format!(
            "separate passes over the {n} elements: the row dot products of {} ({ROW_BLOCK} rows at a time), then an elementwise loop for the density and its derivatives (left to LLVM){}",
            products(&prods),
            if grads.is_empty() { String::new() } else { format!(", then the gradient row updates of {} ({ROW_BLOCK} rows at a time)", products(&grads)) }
        ),
        SShape::Vec(n) => format!("one loop over the {n} elements: value, density and reverse sweep per element"),
        SShape::Mat(r, c) if nodes.is_empty() && !uses_axes(lhs) && !args.iter().any(uses_axes) => format!("one flat loop over the {r} x {c} elements"),
        SShape::Mat(r, c) if g.is_cm(r, c) => {
            let mut cp = Vec::new();
            axis_params(lhs, Ax::Col, &mut cp);
            for a in args {
                axis_params(a, Ax::Col, &mut cp);
            }
            let acc = if cp.is_empty() { String::new() } else { format!("; gradients of {} (indexed by {c}) summed per column in registers", cp.join(", ")) };
            format!("one loop over the {r} x {c} elements in the scan layout, columns outer and rows inner{acc}")
        }
        SShape::Mat(r, c) => {
            let mut rp = Vec::new();
            axis_params(lhs, Ax::Row, &mut rp);
            for a in args {
                axis_params(a, Ax::Row, &mut rp);
            }
            let acc = if rp.is_empty() { String::new() } else { format!("; gradients of {} (indexed by {r}) summed per row in registers", rp.join(", ")) };
            format!("one row-major loop over the {r} x {c} elements{acc}")
        }
    };
    g.note(text);
    for n in nodes {
        let M::Cumsum { inner, shape: own, .. } = n else { continue };
        let how = match own {
            SShape::Mat(r, c) if g.is_cm(r, c) => "column by column in the scan layout",
            SShape::Mat(..) => "four rows interleaved",
            _ => "one sequential pass",
        };
        g.note(format!("running sum of {} materialised before the loop ({how}); its adjoint, a reverse running sum, after it", show_m(inner)));
    }
}

/// Most parameters the leap entry point covers: the size of the runtime's
/// buffer for `leap_blocks` (MAX_LEAP_BLOCKS in runtime/mint_rt.c).
const LEAP_MAX_BLOCKS: usize = 64;

/// `leap_blocks(out)`: (offset, length) in theta of each parameter the leap
/// entry point covers (see `gen_logp`); returns their number.
fn gen_leap_blocks(m: &mut Module, tm: &TModel, opts: &Opts, cm: &[(Dim, Dim)], covered: &[String]) {
    let mut g = Mg::new(m, tm, opts.strict_fp);
    g.cm = cm.to_vec();
    let (layout, _) = g.layout(tm);
    let mut k = 0usize;
    for (n, off, size) in &layout {
        if !covered.contains(n) {
            continue;
        }
        for (j, v) in [off.clone(), size.clone().unwrap()].iter().enumerate() {
            let a = g.f.reg();
            g.f.emit(format!("{a} = getelementptr inbounds i64, ptr %out, i64 {}", 2 * k + j));
            g.f.emit(format!("store i64 {v}, ptr {a}"));
        }
        k += 1;
    }
    let header = format!("define i64 @mint_model_{}_leap_blocks(ptr %out)", tm.name);
    g.finish(&header, &[format!("ret i64 {k}")]);
}

/// The leap entry point: rows r0..r1 of the covered parameters among
/// `owned` are final (r0 a multiple of CM_BLOCK, and r1 too unless it is
/// the last row). In the scan layout (Mg::cm_row) those rows are one
/// contiguous range of each parameter, which the hook takes, as `slot`.
fn leap_hook_calls(g: &mut Mg, owned: &[String], slot: &str, r0: &str, r1: &str, hook: &str, hctx: &str) {
    let cov: Vec<(String, String)> = g.leap_cov.iter().filter(|(n, _)| owned.contains(n)).cloned().collect();
    for (n, off) in cov {
        let Some((_, Ty::Matrix(_, c, _))) = g.leap_tm_params.iter().find(|(m, _)| *m == n).cloned() else { unreachable!() };
        let cols = g.dim(&c);
        let a = g.f.imul(r0, &cols);
        let a = g.f.iadd(&off, &a);
        let n = g.f.iop("sub nsw", r1, r0);
        let len = g.f.imul(&n, &cols);
        g.f.emit(format!("call void {hook}(ptr {hctx}, i64 {slot}, i64 {a}, i64 {len})"));
    }
}

/// The start of a register sum of adjoints inside a vector kernel (one
/// element's running-sum or product adjoint, one column's gradient): -0.0,
/// since -0.0 + x is x for every x, so LLVM drops the first add, which it
/// cannot do for 0.0 + x (that is +0.0 when x is -0.0). That is not the
/// same arithmetic after contraction: with the add gone, the product that
/// was its operand can be fused into the next add, unrounded, which can
/// change the result in the last bits (it does on
/// `x ~ Normal(c * y, 1); y ~ Normal(cumsum(a * x, T), exp(b))`). Mint's
/// floating-point rules allow that (every add and multiply carries
/// `contract`). `--no-negzero-sums` and `--strict-fp` keep 0.0.
fn adj_zero(negzero: bool) -> String {
    fconst(if negzero { -0.0 } else { 0.0 })
}

/// Rows per chunk of a fission kernel: the chunk's rows of X (CHUNK * p
/// doubles) are read by the dot products and are still in L1 when the
/// gradient updates read them again. 32 was faster than 16, 64 and 128 on
/// the logistic benchmark (p = 20).
const CHUNK: u32 = 32;

/// Whether a split (fissioned) statement can run as a fission kernel: its
/// materialised nodes are all matrix-vector products, and every other
/// operation has a vector form.
fn fission_kernel_ok(lhs: &M, args: &[M], nodes: &[&M]) -> bool {
    fission_kernel_why_not(lhs, args, nodes).is_none()
}

/// Why a split statement cannot run as a fission kernel (None: it can).
fn fission_kernel_why_not(lhs: &M, args: &[M], nodes: &[&M]) -> Option<&'static str> {
    fn ok(e: &M) -> bool {
        match e {
            M::Func(Func::Log1p, _) | M::Cumsum { .. } => false,
            M::Func(_, a) | M::Neg(a) => ok(a),
            M::Bin(_, a, b) => ok(a) && ok(b),
            _ => true,
        }
    }
    if nodes.is_empty() || !nodes.iter().all(|n| matches!(n, M::MatVec { .. })) {
        return Some("it materialises something other than matrix-vector products");
    }
    if !(ok(lhs) && args.iter().all(ok)) {
        return Some("log1p or a running sum has no vector form in it");
    }
    None
}

/// A split likelihood over n observations as one loop over chunks of CHUNK
/// rows. For each chunk:
///
///   - the row dot products, four rows at a time, in Mint's vector form;
///   - the density, its derivatives and the elementwise part of the backward
///     sweep as Mint's own <4 x double> code with Mint's exp and log inline
///     (no calls for them, so nothing is spilled around them; a power other
///     than ^2 still calls the vector math library's pow). For BernoulliLogit
///     and PoissonLog the density's exp, and BernoulliLogit's log1p, run in
///     loops of their own first (see run_chunk);
///   - the row updates of the gradient, four rows at a time, reading the
///     chunk's rows of X again from L1.
///
/// X is read from memory once per gradient instead of twice. The rows left
/// over (n mod CHUNK) take the same steps in groups of four, and the last
/// n mod 4 rows in scalar code.
#[allow(clippy::too_many_arguments)]
fn gen_fission_kernel(g: &mut Mg, dist: Dist, lhs: &M, args: &[M], nodes: &[&M], n: &str, nd: &Dim, lp: &str) {
    const L: u32 = 4;
    let keys: Vec<usize> = nodes.iter().map(|n| *n as *const M as usize).collect();
    let bufs: Vec<(String, Option<String>)> = keys.iter().map(|k| g.split[k].clone()).collect();
    // row dot products and gradient updates for rows lo..hi: Mint's vector
    // form when hi - lo is a multiple of 4, LLVM's otherwise
    let dots = |g: &mut Mg, lo: &str, hi: &str, vec: bool| {
        for (node, (fw, _)) in nodes.iter().zip(&bufs) {
            let (mp, vp, c) = g.matvec_parts(node);
            if vec {
                dot4_vec(g, &mp, &vp, &c, lo, hi, fw);
            } else {
                let fw2 = fw.clone();
                rows_dot_blocked(g, &mp, &vp, &c, lo, hi, &move |g: &mut Mg, i: &str, s: &str| g.f.store(s, &fw2, i));
            }
        }
    };
    let axpys = |g: &mut Mg, lo: &str, hi: &str, vec: bool| {
        for (node, (_, ad)) in nodes.iter().zip(&bufs).rev() {
            let Some(ad) = ad else { continue };
            let M::MatVec { vec: v, .. } = node else { unreachable!() };
            let M::ParamV(p, _) = &**v else { unreachable!() };
            let (mp, _, c) = g.matvec_parts(node);
            let gp = g.gptr[p].clone();
            if vec {
                axpy4_vec(g, &mp, &c, lo, hi, ad, &gp);
            } else {
                let ad2 = ad.clone();
                rows_axpy_blocked(g, &mp, &c, lo, hi, &move |g: &mut Mg, i: &str| g.f.load(&ad2, i), &gp);
            }
        }
    };
    // one observation (or L of them): each product's adjoint is summed in a
    // register and stored once, so its buffer needs no zeroing
    let body = |g: &mut Mg, i: &str, lpa: &str| {
        let ix = Ix::vec(i);
        for (k, (_, ad)) in keys.iter().zip(&bufs) {
            if ad.is_some() {
                let acc = g.f.acc_new(&adj_zero(g.m.negzero_sums));
                g.node_acc.insert(*k, acc);
            }
        }
        let mut vals = HashMap::new();
        let x = g.fwd(lhs, &ix, &mut vals);
        let a: Vec<String> = args.iter().map(|e| g.fwd(e, &ix, &mut vals)).collect();
        let (term, partials) = g.lpdf(dist, &x, &a);
        g.f.acc_add(lpa, &term);
        g.bwd(lhs, &partials[0], &ix, &vals);
        for (e, d) in args.iter().zip(&partials[1..]) {
            g.bwd(e, d, &ix, &vals);
        }
        for (k, (_, ad)) in keys.iter().zip(&bufs) {
            if let Some(ad) = ad {
                let acc = g.node_acc.remove(k).unwrap();
                let v = g.f.acc_get(&acc);
                g.f.store(&v, ad, i);
            }
        }
    };
    // vector accumulators for the log density and the scalar parameters
    g.f.lanes = L;
    let lpv = g.f.acc_new(&fconst(0.0));
    let scalars: Vec<String> = g.padj.keys().cloned().collect();
    let vps: HashMap<String, String> = scalars.iter().map(|s| (s.clone(), g.f.acc_new(&fconst(0.0)))).collect();
    g.f.lanes = 1;
    let chunk = CHUNK.to_string();
    // L1 scratch for the density's own exp (PoissonLog's exp(eta),
    // BernoulliLogit's exp(-|eta|)) when the linear predictor is cheap to
    // evaluate twice, and for BernoulliLogit's log1p(e) and 1/(1 + e)
    let cheap = !has_func(&args[0]);
    let scratch = |g: &mut Mg, on: bool| if on { Some(g.f.alloca(&format!("[{CHUNK} x double]"))) } else { None };
    let scr_e = scratch(g, cheap && matches!(dist, Dist::BernoulliLogit | Dist::PoissonLog));
    let scr_l = scratch(g, scr_e.is_some() && dist == Dist::BernoulliLogit);
    let scr_q = scratch(g, scr_l.is_some() && g.m.inline_log);
    if g.rec {
        let prods: Vec<String> = nodes.iter().map(|n| show_m(n)).collect();
        // (the gradient updates run for the products with an adjoint buffer, as in `axpys`)
        let grads: Vec<String> = nodes.iter().zip(&bufs).filter(|(_, (_, ad))| ad.is_some()).map(|(n, _)| show_m(n)).collect();
        let upd = if grads.is_empty() { String::new() } else { format!(", then the gradient updates of {} ({L} rows at a time, reading the chunk's rows of the matrix again, now from cache)", products(&grads)) };
        g.note(format!("fission kernel: one loop over chunks of {CHUNK} rows; per chunk, the row dot products of {} ({L} rows at a time), then the density and its derivatives on {L} rows per vector{upd}; all in Mint's own <{L} x double> code", products(&prods)));
        if scr_e.is_some() {
            g.note(format!("the density's {} runs first, in a loop of its own over the chunk, into a {CHUNK}-value scratch", if dist == Dist::PoissonLog { "exp(eta)" } else { "exp(-|eta|)" }));
        } else if matches!(dist, Dist::BernoulliLogit | Dist::PoissonLog) {
            g.note("the density's exp stays in the main loop: the linear predictor calls a function, so it is not evaluated twice");
        }
        if scr_l.is_some() {
            g.note(format!("log(1 + exp(-|eta|)) likewise, in a second loop{}", if scr_q.is_some() { ", with 1/(1 + e) computed once for it and the sigmoid" } else { "" }));
        }
        g.note(format!("rows left over: the last {nd} mod {CHUNK} in groups of {L} in the same vector code, then the last {nd} mod {L} in scalar code"));
    }
    // The elementwise part and the gradient updates of `rows` rows from lo,
    // whose dot products are in the buffers.
    let run_chunk = |g: &mut Mg, lo: &str, rows: u32| {
        let hi = g.f.iadd(lo, &rows.to_string());
        let nv = (rows / L).to_string();
        g.f.lanes = L;
        g.vpadj = vps.clone();
        g.f.inline_log = true;
        // The density's exp, then its log, each in a loop of its own over the
        // chunk, into L1 scratch: each iteration is then a short dependency
        // chain, and several overlap (measured on the logistic gradient: exp
        // and log1p in one loop, 16% slower; everything in one loop, 12%).
        if let Some(se) = &scr_e {
            for_range(g, "0", &nv, |g, j| {
                let o = g.f.imul(j, &L.to_string());
                let i = g.f.iadd(lo, &o);
                let eta = g.fwd(&args[0], &Ix::vec(&i), &mut HashMap::new());
                let e = if dist == Dist::PoissonLog { g.f.intrinsic1(g.m, "llvm.exp.f64", &eta) } else { g.bl_exp(&eta) };
                g.f.store(&e, se, &o);
            });
        }
        if let (Some(se), Some(sl)) = (&scr_e, &scr_l) {
            for_range(g, "0", &nv, |g, j| {
                let o = g.f.imul(j, &L.to_string());
                let e = g.f.load(se, &o);
                let l = match &scr_q {
                    // 1/(1 + e) once, for the sigmoid and for Mint's log1p
                    Some(sq) => {
                        let u = g.f.fadd(&fconst(1.0), &e);
                        let q = g.f.fdiv(&fconst(1.0), &u);
                        g.f.store(&q, sq, &o);
                        g.f.log1p01(g.m, &e, &q)
                    }
                    None => g.bl_log(&e),
                };
                g.f.store(&l, sl, &o);
            });
        }
        for_range(g, "0", &nv, |g, j| {
            let o = g.f.imul(j, &L.to_string());
            let i = g.f.iadd(lo, &o);
            if let Some(se) = &scr_e {
                let e = g.f.load(se, &o);
                g.exp_override = Some(e);
            }
            if let Some(sl) = &scr_l {
                let l = g.f.load(sl, &o);
                g.log_override = Some(l);
            }
            if let Some(sq) = &scr_q {
                let q = g.f.load(sq, &o);
                g.q_override = Some(q);
            }
            body(g, &i, &lpv);
        });
        g.vpadj.clear();
        g.f.inline_log = false;
        g.f.lanes = 1;
        axpys(g, lo, &hi, true);
    };
    let nb = g.f.iop("sdiv", n, &chunk);
    for_range(g, "0", &nb, |g, b| {
        let lo = g.f.imul(b, &chunk);
        let hi = g.f.iadd(&lo, &chunk);
        dots(g, &lo, &hi, true);
        run_chunk(g, &lo, CHUNK);
    });
    // the rows left over: groups of four in the same vector code, then
    // single rows in scalar code
    let full = g.f.imul(&nb, &chunk);
    let rest = g.f.iop("sub nsw", n, &full);
    let quads = g.f.iop("sdiv", &rest, &L.to_string());
    for_range(g, "0", &quads, |g, q| {
        let o = g.f.imul(q, &L.to_string());
        let lo = g.f.iadd(&full, &o);
        let hi = g.f.iadd(&lo, &L.to_string());
        dots(g, &lo, &hi, true);
        run_chunk(g, &lo, L);
    });
    let q4 = g.f.imul(&quads, &L.to_string());
    let done = g.f.iadd(&full, &q4);
    dots(g, &done, n, false);
    for_range(g, &done, n, |g, i| body(g, i, lp));
    axpys(g, &done, n, false);
    g.f.lanes = L;
    let v = g.f.acc_get(&lpv);
    let s = g.f.hsum(g.m, &v);
    let mut sums = Vec::new();
    for p in &scalars {
        let v = g.f.acc_get(&vps[p]);
        sums.push((p.clone(), g.f.hsum(g.m, &v)));
    }
    g.f.lanes = 1;
    g.f.acc_add(lp, &s);
    for (p, v) in sums {
        let acc = g.padj[&p].clone();
        g.f.acc_add(&acc, &v);
    }
}

/// The lanes k0 + l < c (l = 0..3) as a <4 x i1> mask.
fn tail_mask(g: &mut Mg, k0: &str, c: &str) -> String {
    let splat = |g: &mut Mg, x: &str| {
        let a = g.f.reg();
        g.f.emit(format!("{a} = insertelement <4 x i64> poison, i64 {x}, i64 0"));
        let r = g.f.reg();
        g.f.emit(format!("{r} = shufflevector <4 x i64> {a}, <4 x i64> poison, <4 x i32> zeroinitializer"));
        r
    };
    let ks = splat(g, k0);
    let cs = splat(g, c);
    let idx = g.f.reg();
    g.f.emit(format!("{idx} = add <4 x i64> {ks}, <i64 0, i64 1, i64 2, i64 3>"));
    let m = g.f.reg();
    g.f.emit(format!("{m} = icmp slt <4 x i64> {idx}, {cs}"));
    m
}

/// A <4 x double> load of p[idx..idx+4] with the lanes outside `mask` zero.
fn masked_load(g: &mut Mg, p: &str, idx: &str, mask: &str) -> String {
    if let Some((np, k)) = g.f.narrow.get(p).cloned() {
        // a narrow copy of data: the same lanes, converted (masked-off lanes
        // are zero either way)
        let (et, s, b) = (k.llty(), k.mname(), k.bytes());
        g.m.declare(&format!("declare <4 x {et}> @llvm.masked.load.v4{s}.p0(ptr, i32, <4 x i1>, <4 x {et}>)"));
        let a = g.f.reg();
        g.f.emit(format!("{a} = getelementptr inbounds {et}, ptr {np}, i64 {idx}"));
        let x = g.f.reg();
        g.f.emit(format!("{x} = call <4 x {et}> @llvm.masked.load.v4{s}.p0(ptr {a}, i32 {b}, <4 x i1> {mask}, <4 x {et}> zeroinitializer)"));
        let r = g.f.reg();
        g.f.emit(format!("{r} = {} <4 x {et}> {x} to <4 x double>", k.conv()));
        // opaque to the optimiser, as in Fb::load_narrow
        let l = g.f.lanes;
        g.f.lanes = 4;
        let o = g.f.opaque(&r);
        g.f.lanes = l;
        return o;
    }
    g.m.declare("declare <4 x double> @llvm.masked.load.v4f64.p0(ptr, i32, <4 x i1>, <4 x double>)");
    let a = g.f.gep(p, idx);
    let r = g.f.reg();
    g.f.emit(format!("{r} = call <4 x double> @llvm.masked.load.v4f64.p0(ptr {a}, i32 8, <4 x i1> {mask}, <4 x double> zeroinitializer)"));
    r
}

/// Runs `body(k)` for k = 0, 4, .. below c rounded down to a multiple of 4,
/// then `tail(k, mask)` once if c is not a multiple of 4.
fn vec_cols(g: &mut Mg, c: &str, body: &mut dyn FnMut(&mut Mg, &str), tail: &mut dyn FnMut(&mut Mg, &str, &str)) {
    let c4 = g.f.iop("sdiv", c, "4");
    // no runtime unrolling: for short rows (p = 20 is 5 iterations) an
    // unrolled copy plus its remainder loop cost more than they save
    let md = g.m.loop_as_written();
    for_range_md(g, "0", &c4, Some(&md), |g, kb| {
        let k = g.f.imul(kb, "4");
        body(g, &k);
    });
    let k0 = g.f.imul(&c4, "4");
    let has = g.f.reg();
    g.f.emit(format!("{has} = icmp slt i64 {k0}, {c}"));
    let t = g.f.label("tail");
    let e = g.f.label("tail_done");
    g.f.emit(format!("br i1 {has}, label %{t}, label %{e}"));
    g.f.start_block(&t);
    let mask = tail_mask(g, &k0, c);
    tail(g, &k0, &mask);
    g.f.br(&e);
    g.f.start_block(&e);
}

/// fw[i] = M[i, :] . v for rows lo..hi (hi - lo a multiple of 4), four rows
/// at a time in Mint's vector form: one accumulator per row, vectorised
/// along the columns (a masked tail when c is not a multiple of 4), so each
/// load of v feeds four FMAs; then a 4 x 4 transpose-and-add leaves the four
/// dot products in one vector, stored with one instruction. (Eight rows per
/// group measured the same.)
fn dot4_vec(g: &mut Mg, mp: &str, vp: &str, c: &str, lo: &str, hi: &str, fw: &str) {
    let r = 4usize;
    let len = g.f.iop("sub nsw", hi, lo);
    let groups = g.f.iop("sdiv", &len, &r.to_string());
    for_range(g, "0", &groups, |g, gi| {
        g.f.lanes = 4;
        let o = g.f.imul(gi, &r.to_string());
        let i0 = g.f.iadd(lo, &o);
        let rows: Vec<String> = (0..r)
            .map(|l| {
                let i = g.f.iadd(&i0, &l.to_string());
                g.f.imul(&i, c)
            })
            .collect();
        let accs: Vec<String> = (0..r).map(|_| g.f.acc_new(&fconst(0.0))).collect();
        let (rows2, accs2) = (rows.clone(), accs.clone());
        vec_cols(
            g,
            c,
            &mut |g: &mut Mg, k: &str| {
                let b = g.f.load(vp, k);
                for l in 0..r {
                    let idx = g.f.iadd(&rows[l], k);
                    let x = g.f.load(mp, &idx);
                    let t = g.f.fmul(&x, &b);
                    g.f.acc_add(&accs[l], &t);
                }
            },
            &mut |g: &mut Mg, k: &str, mask: &str| {
                let b = masked_load(g, vp, k, mask);
                for l in 0..r {
                    let idx = g.f.iadd(&rows2[l], k);
                    let x = masked_load(g, mp, &idx, mask);
                    let t = g.f.fmul(&x, &b);
                    g.f.acc_add(&accs2[l], &t);
                }
            },
        );
        let a: Vec<String> = accs.iter().map(|acc| g.f.acc_get(acc)).collect();
        let fl = g.f.rflags;
        let shuf = |g: &mut Mg, x: &str, y: &str, m: &str| {
            let r = g.f.reg();
            g.f.emit(format!("{r} = shufflevector <4 x double> {x}, <4 x double> {y}, <4 x i32> <{m}>"));
            r
        };
        let add = |g: &mut Mg, x: &str, y: &str| {
            let r = g.f.reg();
            g.f.emit(format!("{r} = fadd {fl}<4 x double> {x}, {y}"));
            r
        };
        for q in 0..r / 4 {
            let a = &a[4 * q..4 * q + 4];
            // h01 = [a0_01, a1_01, a0_23, a1_23], h23 likewise for a2, a3
            let e01 = shuf(g, &a[0], &a[1], "i32 0, i32 4, i32 2, i32 6");
            let o01 = shuf(g, &a[0], &a[1], "i32 1, i32 5, i32 3, i32 7");
            let h01 = add(g, &e01, &o01);
            let e23 = shuf(g, &a[2], &a[3], "i32 0, i32 4, i32 2, i32 6");
            let o23 = shuf(g, &a[2], &a[3], "i32 1, i32 5, i32 3, i32 7");
            let h23 = add(g, &e23, &o23);
            let lo2 = shuf(g, &h01, &h23, "i32 0, i32 1, i32 4, i32 5");
            let hi2 = shuf(g, &h01, &h23, "i32 2, i32 3, i32 6, i32 7");
            let s = add(g, &lo2, &hi2);
            let at = g.f.iadd(&i0, &(4 * q).to_string());
            g.f.store(&s, fw, &at);
        }
        g.f.lanes = 1;
    });
}

/// gp[:] += sum_i ad[i] M[i, :] for rows lo..hi (hi - lo a multiple of 4),
/// four rows per pass over gp, in Mint's vector form (a masked tail when c
/// is not a multiple of 4).
fn axpy4_vec(g: &mut Mg, mp: &str, c: &str, lo: &str, hi: &str, ad: &str, gp: &str) {
    let r = 4usize;
    g.m.declare("declare void @llvm.masked.store.v4f64.p0(<4 x double>, ptr, i32, <4 x i1>)");
    let len = g.f.iop("sub nsw", hi, lo);
    let groups = g.f.iop("sdiv", &len, &r.to_string());
    for_range(g, "0", &groups, |g, gi| {
        g.f.lanes = 4;
        let o = g.f.imul(gi, &r.to_string());
        let i0 = g.f.iadd(lo, &o);
        let rows: Vec<String> = (0..r)
            .map(|l| {
                let i = g.f.iadd(&i0, &l.to_string());
                g.f.imul(&i, c)
            })
            .collect();
        let mut cs: Vec<String> = Vec::new();
        for q in 0..r / 4 {
            let at = g.f.iadd(&i0, &(4 * q).to_string());
            let cv = g.f.load(ad, &at);
            for l in 0..4 {
                let x = g.f.reg();
                g.f.emit(format!("{x} = shufflevector <4 x double> {cv}, <4 x double> poison, <4 x i32> splat (i32 {l})"));
                cs.push(x);
            }
        }
        // a balanced sum of the R products
        let sum = |g: &mut Mg, xs: Vec<String>| {
            let mut t: Vec<String> = xs.iter().zip(&cs).map(|(x, c)| g.f.fmul(c, x)).collect();
            while t.len() > 1 {
                t = t.chunks(2).map(|p| g.f.fadd(&p[0], &p[1])).collect();
            }
            t.pop().unwrap()
        };
        let rows2 = rows.clone();
        vec_cols(
            g,
            c,
            &mut |g: &mut Mg, k: &str| {
                let xs: Vec<String> = rows
                    .iter()
                    .map(|r| {
                        let idx = g.f.iadd(r, k);
                        g.f.load(mp, &idx)
                    })
                    .collect();
                let s = sum(g, xs);
                g.f.add_to(gp, k, &s);
            },
            &mut |g: &mut Mg, k: &str, mask: &str| {
                let xs: Vec<String> = rows2
                    .iter()
                    .map(|r| {
                        let idx = g.f.iadd(r, k);
                        masked_load(g, mp, &idx, mask)
                    })
                    .collect();
                let s = sum(g, xs);
                let old = masked_load(g, gp, k, mask);
                let nw = g.f.fadd(&old, &s);
                let a = g.f.gep(gp, k);
                g.f.emit(format!("call void @llvm.masked.store.v4f64.p0(<4 x double> {nw}, ptr {a}, i32 8, <4 x i1> {mask})"));
            },
        );
        g.f.lanes = 1;
    });
}

/// A matrix-shaped statement whose only materialised nodes are running sums
/// over its own (column-major) shape can run as one fused, blocked loop
/// nest. Returns its row dimension when that applies.
fn fused_scan_rows(g: &Mg, lhs: &M, args: &[M], shape: &SShape, fission: bool) -> Option<Dim> {
    fused_scan_rows_cm(&g.cm, lhs, args, shape, fission)
}

fn fused_scan_rows_cm(cm: &[(Dim, Dim)], lhs: &M, args: &[M], shape: &SShape, fission: bool) -> Option<Dim> {
    fused_scan_why(cm, lhs, args, shape, fission).ok()
}

/// `fused_scan_rows_cm`, with the reason when the statement is not a fused
/// scan.
fn fused_scan_why(cm: &[(Dim, Dim)], lhs: &M, args: &[M], shape: &SShape, fission: bool) -> Result<Dim, &'static str> {
    let SShape::Mat(r, c) = shape else { return Err("not matrix-shaped") };
    if !cm.iter().any(|(a, b)| a == r && b == c) {
        return Err("its shape is not in the scan layout");
    }
    let nodes = stmt_globals(lhs, args, fission);
    if nodes.is_empty() {
        return Err("no running sum");
    }
    for n in &nodes {
        match n {
            M::Cumsum { shape: own, ax: Ax::Flat, .. } if own == shape && n.active() => {}
            M::Cumsum { .. } if !n.active() => return Err("a running sum of data only"),
            _ => return Err("a running sum over another shape"),
        }
    }
    Ok(r.clone())
}

/// Whether `e` refers to the name `n` anywhere (inside running sums too).
fn mentions(e: &M, n: &str) -> bool {
    match e {
        M::DataV(x, _) | M::ParamV(x, _) | M::DataM(x) | M::ParamM(x) | M::ParamS(x) | M::DataS(x) => x == n,
        M::MatVec { mat, vec, .. } => mat == n || mentions(vec, n),
        M::Cumsum { inner, .. } => mentions(inner, n),
        M::Bin(_, a, b) => mentions(a, n) || mentions(b, n),
        M::Neg(a) | M::Func(_, a) => mentions(a, n),
        _ => false,
    }
}

/// Whether `e` refers to `n` outside every running sum.
fn mentions_outside_scans(e: &M, n: &str) -> bool {
    match e {
        M::Cumsum { .. } => false,
        M::MatVec { vec, .. } => mentions_outside_scans(vec, n),
        M::Bin(_, a, b) => mentions_outside_scans(a, n) || mentions_outside_scans(b, n),
        M::Neg(a) | M::Func(_, a) => mentions_outside_scans(a, n),
        _ => mentions(e, n),
    }
}

/// Vector width of the fused kernel of a scan statement: 4, or 1 when some
/// operation has no vector form yet.
fn kernel_lanes(dist: Dist, lhs: &M, args: &[M], fission: bool) -> u32 {
    lanes_why(dist, lhs, args, fission).0
}

/// `kernel_lanes`, with the reason for one lane.
fn lanes_why(dist: Dist, lhs: &M, args: &[M], fission: bool) -> (u32, String) {
    let inner_ok = stmt_globals(lhs, args, fission).iter().all(|n| vec_ok(n));
    if dist == Dist::BernoulliLogit {
        (1, "BernoulliLogit has no vector form in this kernel".into())
    } else if !(vec_ok(lhs) && args.iter().all(vec_ok) && inner_ok) {
        // which operations `vec_ok` refused
        fn no_vec(e: &M, out: &mut Vec<&'static str>) {
            let mut put = |s: &'static str| {
                if !out.contains(&s) {
                    out.push(s)
                }
            };
            match e {
                M::Func(Func::Abs, _) => put("abs"),
                M::Func(Func::Log1p, _) => put("log1p"),
                M::MatVec { .. } => put("a matrix-vector product"),
                _ => {}
            }
            match e {
                M::Func(_, a) | M::Neg(a) => no_vec(a, out),
                M::Bin(_, a, b) => {
                    no_vec(a, out);
                    no_vec(b, out);
                }
                M::Cumsum { inner, .. } => no_vec(inner, out),
                _ => {}
            }
        }
        let mut what = Vec::new();
        for e in std::iter::once(lhs).chain(args) {
            no_vec(e, &mut what);
        }
        (1, format!("{} has no vector form in this kernel", what.join(" and ")))
    } else {
        (4, String::new())
    }
}

/// Rows per group in a fused scan kernel: UNROLL vectors of 4 lanes.
const KERNEL_UNROLL: u32 = 2;
const KERNEL_WIDTH: u32 = 4 * KERNEL_UNROLL;
/// Rows per block of the scan layout (see `Mg::cm_row`): a kernel's group.
const CM_BLOCK: u32 = KERNEL_WIDTH;

/// A sum of terms: its backward sweep needs no values of its parts.
fn additive(e: &M) -> bool {
    match e {
        M::Bin(BinOp::Add | BinOp::Sub, a, b) => additive(a) && additive(b),
        M::Neg(a) => additive(a),
        M::Const(_) | M::DimV(_) | M::DataS(_) | M::DataV(..) | M::DataM(_) | M::ParamS(_) | M::ParamV(..) | M::ParamM(_) | M::Cumsum { .. } => true,
        _ => false,
    }
}

/// Column-indexed vector parameters of a fused scan statement.
fn fused_col_params(lhs: &M, args: &[M], fission: bool) -> Vec<String> {
    let mut cp = Vec::new();
    axis_params(lhs, Ax::Col, &mut cp);
    for a in args {
        axis_params(a, Ax::Col, &mut cp);
    }
    for n in stmt_globals(lhs, args, fission) {
        if let M::Cumsum { inner, .. } = n {
            axis_params(inner, Ax::Col, &mut cp);
        }
    }
    cp
}

/// Whether the vectorised kernel can emit this expression (everything but
/// the few operations still written as scalar instructions).
fn vec_ok(e: &M) -> bool {
    match e {
        M::Func(Func::Abs | Func::Log1p, _) => false,
        M::Func(_, a) | M::Neg(a) => vec_ok(a),
        M::Bin(_, a, b) => vec_ok(a) && vec_ok(b),
        M::Cumsum { inner, .. } => vec_ok(inner),
        M::MatVec { .. } => false,
        _ => true,
    }
}

/// Fused, vectorised form of a statement over a column-major R x C shape
/// whose running sums run along C (the time axis of a panel of series).
///
/// The rows are taken four at a time, one row per vector lane; because the
/// storage is column-major, four adjacent rows at one column are one
/// contiguous vector load. For each group of rows:
///
///   forward, column by column: each running sum is a vector register (its
///   carry) plus the operand; the likelihood term and its derivatives follow
///   immediately, and each running sum's adjoint is stored once;
///   reverse, from the last column: a second register carries each adjoint's
///   reverse running sum, which is pushed through the operand.
///
/// Gradients of row-indexed parameters stay in registers for the whole
/// group; those of column-indexed parameters go to per-lane partial sums
/// (C x 4, reduced once at the end); scalar parameters and the log density
/// use vector accumulators. The rows left over (R mod 4) run through the
/// same generator with one lane. Nothing but the adjoints (C x 4 doubles per
/// group, in L1) is written besides the gradient.
/// Everything one group of rows needs, as registers of the function being
/// emitted: the logp function itself, or an outlined parallel kernel.
struct Scan<'s> {
    dist: Dist,
    lhs: &'s M,
    args: &'s [M],
    guests: &'s [&'s Stmt],
    owned: &'s [String],
    keys: Vec<usize>,
    inner: Vec<&'s M>,
    /// row- and column-indexed vector parameters
    rp: Vec<String>,
    cp: Vec<String>,
    rows: String,
    cols: String,
    last: String,
    /// the running sums' adjoint buffers
    ad: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn scan_new<'s>(g: &mut Mg, dist: Dist, lhs: &'s M, args: &'s [M], shape: &SShape, nodes: &[&'s M], guests: &'s [&'s Stmt], owned: &'s [String]) -> Scan<'s> {
    let SShape::Mat(rd, cd) = shape else { unreachable!() };
    let (rows, cols) = (g.dim(rd), g.dim(cd));
    let keys: Vec<usize> = nodes.iter().map(|n| *n as *const M as usize).collect();
    let ad: Vec<String> = keys.iter().map(|k| g.split[k].1.clone().expect("a running sum over parameters")).collect();
    let inner: Vec<&M> = nodes
        .iter()
        .map(|n| match n {
            M::Cumsum { inner, .. } => &**inner,
            _ => unreachable!(),
        })
        .collect();
    let mut rp = Vec::new();
    axis_params(lhs, Ax::Row, &mut rp);
    for a in args {
        axis_params(a, Ax::Row, &mut rp);
    }
    for e in &inner {
        axis_params(e, Ax::Row, &mut rp);
    }
    let cp: Vec<String> = fused_col_params(lhs, args, false);
    let last = g.f.iop("sub nsw", &cols, "1");
    Scan { dist, lhs, args, guests, owned, keys, inner, rp, cp, rows, cols, last, ad }
}

/// Whether `e` contains a matrix-vector product (whose backward sweep
/// writes a whole gradient vector, so it cannot be split across threads).
fn has_matvec(e: &M) -> bool {
    match e {
        M::MatVec { .. } => true,
        M::Cumsum { inner, .. } => has_matvec(inner),
        M::Bin(_, a, b) => has_matvec(a) || has_matvec(b),
        M::Neg(a) | M::Func(_, a) => has_matvec(a),
        _ => false,
    }
}

/// Fused, vectorised form of a statement over a column-major R x C shape
/// whose running sums run along C (the time axis of a panel of series).
///
/// The rows are taken four at a time, one row per vector lane; because the
/// storage is column-major, four adjacent rows at one column are one
/// contiguous vector load. For each group of rows:
///
///   forward, column by column: each running sum is a vector register (its
///   carry) plus the operand; the likelihood term and its derivatives follow
///   immediately, and each running sum's adjoint is stored once;
///   reverse, from the last column: a second register carries each adjoint's
///   reverse running sum, which is pushed through the operand.
///
/// Gradients of row-indexed parameters stay in registers for the whole
/// group; those of column-indexed parameters go to per-lane partial sums
/// (C x 4, reduced once at the end); scalar parameters and the log density
/// use vector accumulators. The rows left over (R mod 4) run through the
/// same generator with one lane. Nothing but the adjoints (C x 4 doubles per
/// group, in L1) is written besides the gradient.
///
/// Groups touch only their own rows, so with `par_nt` set the groups of
/// eight rows run in an outlined function (`gen_par_kernel`) that the
/// runtime splits across the chain's threads (`mint_par_groups`). Single
/// vectors and leftover rows stay on the calling thread.
#[allow(clippy::too_many_arguments)]
fn gen_fused_scan<'s>(
    g: &mut Mg,
    tm: &TModel,
    k: usize,
    dist: Dist,
    lhs: &'s M,
    args: &'s [M],
    shape: &SShape,
    nodes: &[&'s M],
    guests: &'s [&'s Stmt],
    owned: &'s [String],
    lp: &str,
) {
    let sc = scan_new(g, dist, lhs, args, shape, nodes, guests, owned);
    let lanes = kernel_lanes(dist, lhs, args, false);
    let guest_matvec = guests.iter().any(|s| {
        let Stmt::Tilde { lhs: l2, args: a2, .. } = s;
        has_matvec(l2) || a2.iter().any(has_matvec)
    });
    let par = match &g.par_nt {
        Some(nt) if lanes > 1 && !guest_matvec => Some(nt.clone()),
        _ => None,
    };
    g.scan_noted = false;
    if g.rec {
        let SShape::Mat(rd, cd) = shape else { unreachable!() };
        let (_, why1) = lanes_why(dist, lhs, args, false);
        g.note(format!("fused scan kernel over Matrix[{rd}, {cd}]: {} along {cd}, each carried in a register along its row", plural(nodes.len(), "running sum")));
        let wide = lanes * KERNEL_UNROLL;
        g.note(if lanes > 1 {
            format!("vector code: {lanes} lanes (<{lanes} x double>), one row per lane, in groups of {wide} rows ({KERNEL_UNROLL} vectors); the last {rd} mod {wide} rows as vectors of {lanes}, then single rows")
        } else {
            format!("scalar code, one row at a time: {why1}")
        });
    }
    let (rows, cols) = (sc.rows.clone(), sc.cols.clone());
    let cp = sc.cp.clone();
    g.col_part = cp.iter().map(|p| (p.clone(), g.part_bufs[p].clone())).collect();
    let n4 = g.f.imul(&cols, "4");
    // (a parallel kernel zeroes each thread's slice itself)
    let zero_parts = |g: &mut Mg| {
        for p in &cp {
            let b = g.col_part[p].clone();
            g.f.memzero(g.m, &b, &n4);
        }
    };
    if par.is_none() {
        zero_parts(g);
    }

    // groups of UNROLL vectors, then single vectors, then single rows (all of
    // them when the statement is not vectorised)
    const UNROLL: u32 = KERNEL_UNROLL;
    let wide = (lanes * UNROLL).to_string();
    let groups = g.f.iop("sdiv", &rows, &wide);
    let done_wide = g.f.imul(&groups, &wide);
    let rest = g.f.iop("sub nsw", &rows, &done_wide);
    let singles = g.f.iop("sdiv", &rest, &lanes.to_string());
    let s_end = g.f.imul(&singles, &lanes.to_string());
    let done = if lanes > 1 { g.f.iadd(&done_wide, &s_end) } else { "0".to_string() };
    // column partial sums written: one slice per thread that ran
    let used_at = g.f.alloca("i64");
    g.f.emit(format!("store i64 1, ptr {used_at}"));
    if lanes > 1 {
        // one vector accumulator per copy for the log density and each scalar
        // parameter, so the copies do not wait on each other
        g.f.lanes = lanes;
        let lpv: Vec<String> = (0..UNROLL).map(|_| g.f.acc_new(&fconst(0.0))).collect();
        let mut scalars: Vec<String> = g.padj.keys().cloned().collect();
        scalars.sort();
        let vps: Vec<HashMap<String, String>> =
            (0..UNROLL).map(|_| scalars.iter().map(|n| (n.clone(), g.f.acc_new(&fconst(0.0)))).collect()).collect();
        g.f.lanes = 1;
        // the vector accumulators into the log density and the scalar
        // adjoints (after the groups and single vectors of the serial code)
        let reduce_vec = |g: &mut Mg| {
            g.f.lanes = lanes;
            let mut tot = g.f.acc_get(&lpv[0]);
            for a in &lpv[1..] {
                let v = g.f.acc_get(a);
                tot = g.f.fadd(&tot, &v);
            }
            let s = g.f.hsum(g.m, &tot);
            let mut sums = Vec::new();
            for n in &scalars {
                let mut tot = g.f.acc_get(&vps[0][n]);
                for m in &vps[1..] {
                    let v = g.f.acc_get(&m[n]);
                    tot = g.f.fadd(&tot, &v);
                }
                sums.push((n.clone(), g.f.hsum(g.m, &tot)));
            }
            g.f.lanes = 1;
            g.f.acc_add(lp, &s);
            for (n, v) in sums {
                let acc = g.padj[&n].clone();
                g.f.acc_add(&acc, &v);
            }
        };
        // groups, single vectors, then leftover rows, on the calling thread
        let serial_all = |g: &mut Mg| {
            for_range(g, "0", &groups, |g, b| {
                let r0 = g.f.imul(b, &wide);
                scan_group(g, &sc, lanes, UNROLL, &r0, &lpv, &vps);
            });
            let d2 = done_wide.clone();
            for_range(g, "0", &singles, |g, b| {
                let off = g.f.imul(b, &lanes.to_string());
                let r0 = g.f.iadd(&d2, &off);
                scan_group(g, &sc, lanes, 1, &r0, &lpv, &vps);
            });
            reduce_vec(g);
            let lp1 = [lp.to_string()];
            for_range(g, &done, &rows, |g, r| scan_group(g, &sc, 1, 1, r, &lp1, &[]));
            leap_hook_calls(g, owned, "0", "0", &rows, "%hook", "%hctx");
        };
        if let Some(nt) = &par {
            // With more than one thread requested, the outlined kernel (its
            // thread 0 also runs the single vectors and leftover rows);
            // otherwise exactly the serial code (and its arithmetic).
            let kn = gen_par_kernel(g, tm, k, &sc, nodes, &scalars);
            let (l_par, l_ser, l_join) = (g.f.label("par"), g.f.label("ser"), g.f.label("pjoin"));
            let c = g.f.reg();
            g.f.emit(format!("{c} = icmp sgt i64 {nt}, 1"));
            g.f.emit(format!("br i1 {c}, label %{l_par}, label %{l_ser}"));
            g.f.start_block(&l_par);
            let u = par_kernel_call(g, tm, &kn, &cp, &scalars, &groups, nt, lp);
            g.f.emit(format!("store i64 {u}, ptr {used_at}"));
            g.f.br(&l_join);
            g.f.start_block(&l_ser);
            zero_parts(g);
            serial_all(g);
            g.f.br(&l_join);
            g.f.start_block(&l_join);
        } else {
            serial_all(g);
        }
    } else {
        let lp1 = [lp.to_string()];
        for_range(g, &done, &rows, |g, r| scan_group(g, &sc, 1, 1, r, &lp1, &[]));
        leap_hook_calls(g, owned, "0", "0", &rows, "%hook", "%hctx");
    }
    let used = if par.is_some() {
        let r = g.f.reg();
        g.f.emit(format!("{r} = load i64, ptr {used_at}"));
        r
    } else {
        "1".to_string()
    };
    // column-indexed gradients: reduce the four per-lane partial sums (of
    // every thread's slice, in thread order)
    for p in &cp {
        let b = g.col_part[p].clone();
        let gp = g.gptr[p].clone();
        let one = used == "1";
        let used2 = used.clone();
        let n42 = n4.clone();
        for_range(g, "0", &cols, |g, c| {
            let at = g.f.imul(c, "4");
            let lanes4 = |g: &mut Mg, at: &str| {
                let mut tot = g.f.load(&b, at);
                for l in 1..4 {
                    let i = g.f.iadd(at, &l.to_string());
                    let x = g.f.load(&b, &i);
                    tot = g.f.fadd(&tot, &x);
                }
                tot
            };
            let tot = lanes4(g, &at);
            if one {
                g.f.add_to(&gp, c, &tot);
            } else {
                let acc = g.f.acc_new(&tot);
                for_range(g, "1", &used2, |g, t| {
                    let o = g.f.imul(t, &n42);
                    let a2 = g.f.iadd(&o, &at);
                    let v = lanes4(g, &a2);
                    add_in_order(g, &acc, &v);
                });
                let tot = g.f.acc_get(&acc);
                g.f.add_to(&gp, c, &tot);
            }
        });
    }
    g.col_part.clear();
    if g.rec {
        let SShape::Mat(rd, cd) = shape else { unreachable!() };
        for l in std::mem::take(&mut g.pending) {
            g.note(l);
        }
        if !sc.rp.is_empty() {
            g.note(format!("gradients of {} (indexed by {rd}): summed in registers per group of rows", sc.rp.join(", ")));
        }
        if !sc.cp.is_empty() {
            g.note(if lanes > 1 {
                format!("gradients of {} (indexed by {cd}): per-lane partial sums, {cd} x 4 per thread, reduced once at the end", sc.cp.join(", "))
            } else {
                format!("gradients of {} (indexed by {cd}): partial sums per column, reduced once at the end", sc.cp.join(", "))
            });
        }
        let wide = lanes * KERNEL_UNROLL;
        g.note(match (&par, g.par_off) {
            (Some(_), _) => format!("threads: the groups of {wide} rows are split across the chain's threads when more than one is requested at run time (mint_par_groups; thread 0 also runs the rows left over); with one thread it runs the serial loop"),
            (None, _) if lanes == 1 => "threads: one (only vector kernels are split)".to_string(),
            (None, Some(flag)) => format!("threads: one (parallel kernel off, {flag})"),
            (None, _) if guest_matvec => "threads: one (an absorbed statement has a matrix-vector product)".to_string(),
            (None, _) => unreachable!("par is set whenever par_nt is, lanes > 1 and no guest has a matrix-vector product"),
        });
    }
}

/// Matrix-vector products for the report; a product that occurs k times
/// is computed k times, and says so.
fn products(ps: &[String]) -> String {
    let mut seen: Vec<(String, usize)> = Vec::new();
    for p in ps {
        match seen.iter_mut().find(|(q, _)| q == p) {
            Some((_, n)) => *n += 1,
            None => seen.push((p.clone(), 1)),
        }
    }
    seen.iter().map(|(p, n)| if *n == 1 { p.clone() } else { format!("{p} ({n} times, once per occurrence)") }).collect::<Vec<_>>().join(" and ")
}

/// The matrix-vector products of a statement, for the report.
fn mv_list(lhs: &M, args: &[M]) -> String {
    let mut mv = Vec::new();
    for e in std::iter::once(lhs).chain(args) {
        matvecs(e, &mut mv);
    }
    let names: Vec<String> = mv.iter().map(|n| show_m(n)).collect();
    products(&names)
}

/// "1 running sum", "2 running sums".
fn plural(n: usize, what: &str) -> String {
    if n == 1 {
        format!("1 {what}")
    } else {
        format!("{n} {what}s")
    }
}

/// Slots of the context a fused scan statement passes to its parallel
/// kernel (8 bytes each): theta, grad, the per-thread log density and scalar
/// adjoint outputs, then the column partial-sum buffers, the constrained
/// values and adjoints of each Positive vector parameter, and the values of
/// the scalar parameters (sorted by name).
const CTX_THETA: usize = 0;
const CTX_GRAD: usize = 1;
const CTX_OUT_LP: usize = 2;
const CTX_OUT_SC: usize = 3;
const CTX_FIXED: usize = 4;
/// Most threads a parallel kernel may use (MINT_PAR_MAX in the runtime).
const PAR_MAX_THREADS: usize = 64;

fn pos_vectors(tm: &TModel) -> Vec<String> {
    tm.params.iter().filter(|(_, t)| matches!(t, Ty::Vector(_, Dom::Positive))).map(|(n, _)| n.clone()).collect()
}

/// Emits the outlined kernel of a fused scan statement: groups g0..g1 of
/// eight rows on thread `tid`, with its own scratch (per-thread
/// `mint_ws_slot`s), its own slice of the column partial sums, and its log
/// density and scalar adjoints written to entry `tid` of the output arrays.
/// Returns the name of the entry point `(ctx, g0, g1, tid)`.
fn gen_par_kernel(g: &mut Mg, tm: &TModel, k: usize, sc: &Scan, nodes: &[&M], scalars: &[String]) -> String {
    let leap = !g.leap_cov.is_empty();
    let kn = format!("mint_model_{}_{}scan{k}{}", tm.name, if leap { "leap_" } else { "" }, g.m.variant);
    let pos = pos_vectors(tm);
    let ncp = sc.cp.len();
    let leap_base = CTX_FIXED + ncp + 2 * pos.len() + scalars.len();
    let covered: Vec<String> = g.leap_cov.iter().map(|(n, _)| n.clone()).collect();
    // The same per-thread slots as the caller's buffers, requested at the
    // same sizes: on the calling thread they are the caller's own buffers,
    // and are never reallocated under it.
    let ad_slots: Vec<usize> = sc.ad.iter().map(|a| g.ws_slot_of[a]).collect();
    let k_slot = g.ws_slot_of[&g.kscratch[&sc.keys[0]]];
    let strict = g.f.strict;
    let cm = g.cm.clone();
    let lanes = kernel_lanes(sc.dist, sc.lhs, sc.args, false);
    let shape = match nodes[0] {
        M::Cumsum { shape, .. } => shape.clone(),
        _ => unreachable!(),
    };
    let SShape::Mat(_, cd) = &shape else { unreachable!() };
    {
        let mut km = Mg::new(&mut *g.m, tm, strict);
        km.cm = cm;
        let (layout, _) = km.layout(tm);
        km.leap_cov = tm.params.iter().zip(&layout).filter(|((n, _), _)| covered.contains(n)).map(|((n, _), (_, off, _))| (n.clone(), off.clone())).collect();
        let sc_base = CTX_FIXED + ncp + 2 * pos.len();
        for ((n, t), (_, off, _)) in tm.params.iter().zip(&layout) {
            match t {
                Ty::Vector(_, Dom::Positive) => {
                    let i = pos.iter().position(|q| q == n).unwrap();
                    km.pptr.insert(n.clone(), format!("%kpv{i}"));
                    km.gptr.insert(n.clone(), format!("%kpa{i}"));
                }
                Ty::Vector(..) | Ty::Matrix(..) => {
                    let p = km.f.gep("%theta", off);
                    let gp = km.f.gep("%grad", off);
                    km.pptr.insert(n.clone(), p);
                    km.gptr.insert(n.clone(), gp);
                }
                Ty::Scalar(_) => {
                    let j = scalars.iter().position(|q| q == n).unwrap();
                    let a = km.f.reg();
                    km.f.emit(format!("{a} = getelementptr inbounds i64, ptr %ctx, i64 {}", sc_base + j));
                    let v = km.f.reg();
                    km.f.emit(format!("{v} = load double, ptr {a}"));
                    km.pval.insert(n.clone(), v);
                    // the leftover rows' (one lane): the groups' and single
                    // vectors' go to the vector accumulators
                    let acc = km.f.acc_new(&fconst(0.0));
                    km.padj.insert(n.clone(), acc);
                }
                _ => unreachable!(),
            }
        }
        for (j, (node, slot)) in nodes.iter().zip(&ad_slots).enumerate() {
            let M::Cumsum { shape, .. } = node else { unreachable!() };
            let n = shape_size(&mut km, shape);
            let r = km.f.reg();
            km.f.emit(format!("{r} = call noalias ptr @mint_ws_slot(i64 {slot}, i64 {n})"));
            // forward values come from registers and the group scratch; a
            // load from the caller's forward buffer would be a bug
            km.split.insert(sc.keys[j], ("null".to_string(), Some(r)));
        }
        let cols = km.dim(cd);
        let per = km.f.imul(&cols, &(KERNEL_WIDTH * (1 + nodes.len() as u32)).to_string());
        let r = km.f.reg();
        km.f.emit(format!("{r} = call noalias ptr @mint_ws_slot(i64 {k_slot}, i64 {per})"));
        km.kscratch.insert(sc.keys[0], r);
        let n4 = km.f.imul(&cols, "4");
        let off = km.f.imul("%tid", &n4);
        for (i, p) in sc.cp.iter().enumerate() {
            let b = km.f.gep(&format!("%kp{i}"), &off);
            km.f.memzero(km.m, &b, &n4);
            km.col_part.insert(p.clone(), b);
        }
        let sck = scan_new(&mut km, sc.dist, sc.lhs, sc.args, &shape, nodes, sc.guests, sc.owned);
        const UNROLL: u32 = KERNEL_UNROLL;
        let wide = (lanes * UNROLL).to_string();
        km.f.lanes = lanes;
        let lpv: Vec<String> = (0..UNROLL).map(|_| km.f.acc_new(&fconst(0.0))).collect();
        let vps: Vec<HashMap<String, String>> =
            (0..UNROLL).map(|_| scalars.iter().map(|n| (n.clone(), km.f.acc_new(&fconst(0.0)))).collect()).collect();
        km.f.lanes = 1;
        for_range(&mut km, "%g0", "%g1", |km, b| {
            let r0 = km.f.imul(b, &wide);
            scan_group(km, &sck, lanes, UNROLL, &r0, &lpv, &vps);
        });
        // the leap entry point: this thread's rows are final, one contiguous
        // range of each covered parameter (the scan layout)
        let mut hp = Vec::new();
        if leap {
            for j in 0..2 {
                let a = km.f.reg();
                km.f.emit(format!("{a} = getelementptr inbounds i64, ptr %ctx, i64 {}", leap_base + j));
                let v = km.f.reg();
                km.f.emit(format!("{v} = load ptr, ptr {a}"));
                hp.push(v);
            }
            let r0 = km.f.imul("%g0", &wide);
            let r1 = km.f.imul("%g1", &wide);
            leap_hook_calls(&mut km, sc.owned, "%tid", &r0, &r1, &hp[0], &hp[1]);
        }
        // thread 0 (the calling thread) also runs the single vectors and the
        // leftover rows (and their leap hook)
        let lp1 = km.f.acc_new(&fconst(0.0));
        {
            let rows = sck.rows.clone();
            let groups = km.f.iop("sdiv", &rows, &wide);
            let done_wide = km.f.imul(&groups, &wide);
            let rest = km.f.iop("sub nsw", &rows, &done_wide);
            let singles = km.f.iop("sdiv", &rest, &lanes.to_string());
            let s_end = km.f.imul(&singles, &lanes.to_string());
            let done = km.f.iadd(&done_wide, &s_end);
            let t0 = km.f.reg();
            km.f.emit(format!("{t0} = icmp eq i64 %tid, 0"));
            if_then(&mut km, &t0, |km| {
                for_range(km, "0", &singles, |km, b| {
                    let off = km.f.imul(b, &lanes.to_string());
                    let r0 = km.f.iadd(&done_wide, &off);
                    scan_group(km, &sck, lanes, 1, &r0, &lpv, &vps);
                });
                let lpa = [lp1.clone()];
                for_range(km, &done, &rows, |km, r| scan_group(km, &sck, 1, 1, r, &lpa, &[]));
                if leap {
                    leap_hook_calls(km, sc.owned, "%tid", &done_wide, &rows, &hp[0], &hp[1]);
                }
            });
        }
        km.f.lanes = lanes;
        let mut tot = km.f.acc_get(&lpv[0]);
        for a in &lpv[1..] {
            let v = km.f.acc_get(a);
            tot = km.f.fadd(&tot, &v);
        }
        let s = km.f.hsum(km.m, &tot);
        let mut sums = Vec::new();
        for n in scalars {
            let mut tot = km.f.acc_get(&vps[0][n]);
            for m in &vps[1..] {
                let v = km.f.acc_get(&m[n]);
                tot = km.f.fadd(&tot, &v);
            }
            sums.push(km.f.hsum(km.m, &tot));
        }
        km.f.lanes = 1;
        // the leftover rows' scalar accumulators
        let l1 = km.f.acc_get(&lp1);
        let s = km.f.fadd(&s, &l1);
        let sums: Vec<String> = sums
            .iter()
            .zip(scalars)
            .map(|(v, n)| {
                let a = km.padj[n].clone();
                let x = km.f.acc_get(&a);
                km.f.fadd(v, &x)
            })
            .collect();
        let out_lp = km.f.reg();
        km.f.emit(format!("{out_lp} = getelementptr inbounds i64, ptr %ctx, i64 {CTX_OUT_LP}"));
        let out_lp2 = km.f.reg();
        km.f.emit(format!("{out_lp2} = load ptr, ptr {out_lp}"));
        km.f.store(&s, &out_lp2, "%tid");
        let out_sc = km.f.reg();
        km.f.emit(format!("{out_sc} = getelementptr inbounds i64, ptr %ctx, i64 {CTX_OUT_SC}"));
        let out_sc2 = km.f.reg();
        km.f.emit(format!("{out_sc2} = load ptr, ptr {out_sc}"));
        let base = km.f.imul("%tid", &scalars.len().to_string());
        for (j, v) in sums.iter().enumerate() {
            let at = km.f.iadd(&base, &j.to_string());
            km.f.store(v, &out_sc2, &at);
        }
        // (grad is read by the leap entry point's hook: not noalias there)
        let gq = if leap { "ptr %grad" } else { "ptr noalias %grad" };
        let mut params = vec!["ptr noalias %theta".to_string(), gq.to_string()];
        params.extend((0..ncp).map(|i| format!("ptr noalias %kp{i}")));
        for i in 0..pos.len() {
            params.push(format!("ptr noalias %kpv{i}"));
            params.push(format!("ptr noalias %kpa{i}"));
        }
        params.extend(["ptr %ctx".to_string(), "i64 %g0".to_string(), "i64 %g1".to_string(), "i64 %tid".to_string()]);
        let header = format!("define internal void @{kn}_body({})", params.join(", "));
        km.finish(&header, &["ret void".to_string()]);
    }
    // the entry point the runtime calls: unpack the context
    let mut w = format!("define internal void @{kn}(ptr %ctx, i64 %g0, i64 %g1, i64 %tid) {{\nentry:\n");
    let mut argv = Vec::new();
    let mut slots = vec![CTX_THETA, CTX_GRAD];
    slots.extend((0..ncp + 2 * pos.len()).map(|i| CTX_FIXED + i));
    for (i, s) in slots.iter().enumerate() {
        w += &format!("  %a{i} = getelementptr inbounds i64, ptr %ctx, i64 {s}\n  %p{i} = load ptr, ptr %a{i}\n");
        argv.push(format!("ptr %p{i}"));
    }
    argv.extend(["ptr %ctx".to_string(), "i64 %g0".to_string(), "i64 %g1".to_string(), "i64 %tid".to_string()]);
    w += &format!("  call void @{kn}_body({})\n  ret void\n}}\n", argv.join(", "));
    g.m.funcs.push(w);
    kn
}

/// Fills the context of a parallel scan kernel, runs it over `groups`
/// groups with up to `nt` threads, and adds the threads' log densities and
/// scalar adjoints to the caller's accumulators in thread order. Returns
/// the register holding the number of threads that ran (the number of
/// column partial-sum slices to reduce).
#[allow(clippy::too_many_arguments)]
fn par_kernel_call(g: &mut Mg, tm: &TModel, kn: &str, cp: &[String], scalars: &[String], groups: &str, nt: &str, lp: &str) -> String {
    let pos = pos_vectors(tm);
    let leap = !g.leap_cov.is_empty();
    let leap_base = CTX_FIXED + cp.len() + 2 * pos.len() + scalars.len();
    let nslots = leap_base + if leap { 2 } else { 0 };
    let ctx = g.f.alloca(&format!("[{nslots} x i64]"));
    let out_lp = g.f.alloca(&format!("[{PAR_MAX_THREADS} x double]"));
    let out_sc = g.f.alloca(&format!("[{} x double]", PAR_MAX_THREADS * scalars.len().max(1)));
    let put = |g: &mut Mg, i: usize, ty: &str, v: &str| {
        let a = g.f.reg();
        g.f.emit(format!("{a} = getelementptr inbounds i64, ptr {ctx}, i64 {i}"));
        g.f.emit(format!("store {ty} {v}, ptr {a}"));
    };
    put(g, CTX_THETA, "ptr", "%theta");
    put(g, CTX_GRAD, "ptr", "%grad");
    put(g, CTX_OUT_LP, "ptr", &out_lp);
    put(g, CTX_OUT_SC, "ptr", &out_sc);
    for (i, p) in cp.iter().enumerate() {
        let b = g.part_bufs[p].clone();
        put(g, CTX_FIXED + i, "ptr", &b);
    }
    for (i, n) in pos.iter().enumerate() {
        let (v, a) = (g.pptr[n].clone(), g.gptr[n].clone());
        put(g, CTX_FIXED + cp.len() + 2 * i, "ptr", &v);
        put(g, CTX_FIXED + cp.len() + 2 * i + 1, "ptr", &a);
    }
    for (j, n) in scalars.iter().enumerate() {
        let v = g.pval[n].clone();
        let v = g.f.opnd(&v);
        put(g, CTX_FIXED + cp.len() + 2 * pos.len() + j, "double", &v);
    }
    if leap {
        // the leap entry point's hook and its context
        put(g, leap_base, "ptr", "%hook");
        put(g, leap_base + 1, "ptr", "%hctx");
    }
    g.m.declare("declare i64 @mint_par_groups(ptr, ptr, i64, i64)");
    let used = g.f.reg();
    g.f.emit(format!("{used} = call i64 @mint_par_groups(ptr @{kn}, ptr {ctx}, i64 {groups}, i64 {nt})"));
    let ns = scalars.len().to_string();
    for_range(g, "0", &used, |g, t| {
        let v = g.f.load(&out_lp, t);
        add_in_order(g, lp, &v);
        let base = g.f.imul(t, &ns);
        for (j, n) in scalars.iter().enumerate() {
            let at = g.f.iadd(&base, &j.to_string());
            let v = g.f.load(&out_sc, &at);
            let acc = g.padj[n].clone();
            add_in_order(g, &acc, &v);
        }
    });
    used
}

/// acc += v without `reassoc`, so a loop of these sums in loop order (the
/// threads' partial results are added in thread order).
fn add_in_order(g: &mut Mg, acc: &str, v: &str) {
    let old = g.f.acc_get(acc);
    let s = g.f.fadd(&old, v);
    let t = g.f.ty();
    g.f.emit(format!("store {t} {s}, ptr {acc}"));
}

/// One group of `u` vectors of `l` rows starting at r0 (u * l rows in all),
/// accumulating the log density in `lpa`. The u copies are independent
/// (their own carries and accumulators), which gives the CPU u chains and
/// uses whole cache lines.
fn scan_group(g: &mut Mg, sc: &Scan, l: u32, u: u32, r0: &str, lpa: &[String], vps: &[HashMap<String, String>]) {
    let Scan { dist, lhs, args, guests, owned, keys, inner, rp, cp, rows, cols, last, ad } = sc;
    let (dist, lhs, args, guests, owned) = (*dist, *lhs, *args, *guests, *owned);
    let (rows, cols, last) = (rows.clone(), cols.clone(), last.clone());
    g.f.lanes = l;
    let use_copy_accs = |g: &mut Mg, k: usize| {
        if !vps.is_empty() {
            g.vpadj = vps[k].clone();
        }
    };
    // one register per column-indexed parameter per column, shared by the
    // copies, then one update of the per-lane partial sums
    let col_begin = |g: &mut Mg| {
        for p in cp.iter() {
            let acc = g.f.acc_new(&adj_zero(g.m.negzero_sums));
            g.inv_acc.insert((p.clone(), Ax::Col), acc);
        }
    };
    let col_flush = |g: &mut Mg, col: &str| {
        for p in cp.iter() {
            let acc = g.inv_acc.remove(&(p.clone(), Ax::Col)).unwrap();
            let v = g.f.acc_get(&acc);
            let b = g.col_part[p].clone();
            let at = g.f.imul(col, "4");
            g.f.add_to(&b, &at, &v);
        }
    };
    let r0s: Vec<String> = (0..u).map(|k| g.f.iadd(r0, &(k * l).to_string())).collect();
    // where each copy's rows live (the scan layout, Mg::cm_row): element
    // (r0s[k] + lane, col) is at base + col * stride + lane
    let rbs: Vec<(String, String)> = r0s.iter().map(|r| g.cm_row(r, &rows, &cols)).collect();
    let flat_at = |g: &mut Mg, col: &str, k: usize| {
        let o = g.f.imul(col, &rbs[k].1);
        g.f.iadd(&rbs[k].0, &o)
    };
    // The adjoints of this group's running sums live in a small contiguous
    // scratch, [col * (u * l) + k * l + lane], reused by every group: it
    // stays in L1, where the full-matrix index would scatter it.
    let width = (u * l).to_string();
    let ad_at = |g: &mut Mg, col: &str, k: usize| {
        let a = g.f.imul(col, &width);
        g.f.iadd(&a, &(k as u32 * l).to_string())
    };
    let row_accs: Vec<Vec<String>> = (0..u).map(|_| rp.iter().map(|_| g.f.acc_new(&fconst(0.0))).collect()).collect();
    let carry: Vec<Vec<String>> = (0..u).map(|_| keys.iter().map(|_| g.f.acc_new(&fconst(0.0))).collect()).collect();
    let use_row_accs = |g: &mut Mg, k: usize| {
        for (p, acc) in rp.iter().zip(&row_accs[k]) {
            g.inv_acc.insert((p.clone(), Ax::Row), acc.clone());
        }
    };
    // The density's own exp runs in a pass of its own when there is one
    // (PoissonLog): A computes the running sums and eta into an L1
    // scratch, B takes exp over the scratch with nothing else live (so
    // its constants stay in registers), C finishes the density and its
    // derivatives. Otherwise the three are one loop.
    //
    // One scratch slot holds eta, then exp(eta) (B works in place), then
    // the first running sum's adjoint (C overwrites each exp value after
    // reading it); the running sums have one slot each. With 8 rows and
    // 150 columns that is 19 KB, which stays in L1.
    let split = dist == Dist::PoissonLog;
    let first_report = g.rec && !g.scan_noted;
    let scr = g.kscratch[&keys[0]].clone();
    let cw = g.f.imul(&cols, &width);
    let eta_p = scr.clone();
    let ex_p = scr.clone();
    let s_p: Vec<String> = (0..keys.len())
        .map(|j| {
            let o = g.f.imul(&cw, &(1 + j).to_string());
            g.f.gep(&scr, &o)
        })
        .collect();
    // adjoints: the first running sum's in the shared slot when split
    // (only with a single running sum: a nested one's parent adds into
    // its child's adjoint through the general buffer)
    let alias = split && keys.len() == 1;
    let ad: Vec<String> = ad.iter().enumerate().map(|(j, a)| if alias && j == 0 { scr.clone() } else { a.clone() }).collect();
    // When the density's argument is a sum of terms (beta + state), its
    // backward sweep needs no values, so C can reload eta instead of the
    // running sum: A then stores eta alone and B writes exp(eta) into the
    // running sum's (unused) slot. With a single running sum, C hands its
    // adjoint to R in a register.
    let lean = alias && additive(&args[0]) && !lhs.has_cumsum();
    let ex_p = if lean { s_p[0].clone() } else { ex_p };
    if first_report {
        g.scan_noted = true;
        g.note(if split {
            format!(
                "passes per group: A, forward in time, the running sums and the density's argument into the group's scratch; B, exp over the scratch with nothing else live (the exp split); C (density and derivatives) and R (reverse running sums of the adjoints) together, from the last column{}",
                if lean { "; C reloads the argument (a sum of terms) instead of the running sum, and hands its adjoint to R in a register" } else { "" }
            )
        } else {
            "passes per group: forward in time, the running sums, the density and its derivatives; then backwards from the last column, the reverse running sums of the adjoints".to_string()
        });
    }
    if split {
        for_range(g, "0", &cols, |g, col| {
            for k in 0..u as usize {
                let flat = flat_at(g, col, k);
                let ix = Ix { flat: flat.clone(), row: r0s[k].clone(), col: col.to_string() };
                let at = ad_at(g, col, k);
                let mut vals = HashMap::new();
                for (j, e) in inner.iter().enumerate() {
                    let v = g.fwd(e, &ix, &mut vals);
                    let prev = g.f.acc_get(&carry[k][j]);
                    let s = g.f.fadd(&prev, &v); // sequential in the column index: not a reduction
                    let t = g.f.ty();
                    g.f.emit(format!("store {t} {s}, ptr {}", carry[k][j]));
                    if !lean {
                        g.f.store(&s, &s_p[j], &at);
                    }
                    vals.insert(keys[j], s);
                }
                let eta = g.fwd(&args[0], &ix, &mut vals);
                g.f.store(&eta, &eta_p, &at);
            }
        });
        let n = g.f.imul(&cols, &u.to_string());
        for_range(g, "0", &n, |g, i| {
            let at = g.f.imul(i, &l.to_string());
            let eta = g.f.load(&eta_p, &at);
            let e = g.f.intrinsic1(g.m, "llvm.exp.f64", &eta);
            g.f.store(&e, &ex_p, &at);
        });
    }
    // C: the density and its derivatives at column col (each running
    // sum's adjoint stored); R: the reverse running sums at column col
    let c_col = |g: &mut Mg, col: &str| {
    for k in 0..u as usize {
        use_row_accs(g, k);
        use_copy_accs(g, k);
        let flat = flat_at(g, col, k);
        let ix = Ix { flat: flat.clone(), row: r0s[k].clone(), col: col.to_string() };
        let at = ad_at(g, col, k);
        let mut vals = HashMap::new();
        if lean {
            let eta = g.f.load(&eta_p, &at);
            vals.insert(&args[0] as *const M as usize, eta);
        }
        for (j, e) in inner.iter().enumerate() {
            if lean {
                break;
            }
            let s = if split {
                g.f.load(&s_p[j], &at)
            } else {
                let v = g.fwd(e, &ix, &mut vals);
                let prev = g.f.acc_get(&carry[k][j]);
                let s = g.f.fadd(&prev, &v); // sequential in the column index: not a reduction
                let t = g.f.ty();
                g.f.emit(format!("store {t} {s}, ptr {}", carry[k][j]));
                // kept for the reverse pass, where a nested running
                // sum's operand may need its child's value
                g.f.store(&s, &s_p[j], &at);
                s
            };
            vals.insert(keys[j], s);
        }
        for key in keys.iter() {
            let acc = g.f.acc_new(&adj_zero(g.m.negzero_sums));
            g.node_acc.insert(*key, acc);
        }
        let x = g.fwd(lhs, &ix, &mut vals);
        let a: Vec<String> = args.iter().map(|e| g.fwd(e, &ix, &mut vals)).collect();
        if split {
            let e = g.f.load(&ex_p, &at);
            g.exp_override = Some(e);
        }
        let (term, partials) = g.lpdf(dist, &x, &a);
        g.f.acc_add(&lpa[k % lpa.len()], &term);
        g.bwd(lhs, &partials[0], &ix, &vals);
        for (e, d) in args.iter().zip(&partials[1..]) {
            g.bwd(e, d, &ix, &vals);
        }
        for (j, key) in keys.iter().enumerate() {
            let acc = g.node_acc.remove(key).unwrap();
            let v = g.f.acc_get(&acc);
            if alias {
                g.pending_ad.insert(k, v);
            } else {
                g.f.store(&v, &ad[j], &at);
            }
        }
    }
    };
    let r_col = |g: &mut Mg, col: &str| {
    for k in 0..u as usize {
        use_row_accs(g, k);
        use_copy_accs(g, k);
        let flat = flat_at(g, col, k);
        let ix = Ix { flat: flat.clone(), row: r0s[k].clone(), col: col.to_string() };
        let at = ad_at(g, col, k);
        g.ad_at = Some(at.clone());
        for n in owned {
            let acc = g.f.acc_new(&adj_zero(g.m.negzero_sums));
            g.elem_acc.insert(n.clone(), acc);
        }
        for j in (0..keys.len()).rev() {
            let a = match g.pending_ad.remove(&k) {
                Some(v) if alias => v,
                _ => g.f.load(&ad[j], &at),
            };
            let prev = g.f.acc_get(&carry[k][j]);
            let sum = g.f.fadd(&prev, &a);
            let t = g.f.ty();
            g.f.emit(format!("store {t} {sum}, ptr {}", carry[k][j]));
            // the values of the running sums nested inside this one
            let mut vals = HashMap::new();
            for i in 0..j {
                let v = g.f.load(&s_p[i], &at);
                vals.insert(keys[i], v);
            }
            g.fwd(inner[j], &ix, &mut vals);
            g.bwd(inner[j], &sum, &ix, &vals);
        }
        g.ad_at = None;
        // absorbed element-wise statements over the same elements
        for Stmt::Tilde { dist: d2, lhs: l2, args: a2, .. } in guests {
            let mut vals = HashMap::new();
            let x = g.fwd(l2, &ix, &mut vals);
            let av: Vec<String> = a2.iter().map(|e| g.fwd(e, &ix, &mut vals)).collect();
            let (term, partials) = g.lpdf(*d2, &x, &av);
            g.f.acc_add(&lpa[k % lpa.len()], &term);
            g.bwd(l2, &partials[0], &ix, &vals);
            for (e, d) in a2.iter().zip(&partials[1..]) {
                g.bwd(e, d, &ix, &vals);
            }
        }
        for n in owned {
            let acc = g.elem_acc.remove(n).unwrap();
            let v = g.f.acc_get(&acc);
            let gp = g.gptr[n].clone();
            g.f.store(&v, &gp, &flat);
        }
    }
    };
    let t = g.f.ty();
    let zero = g.f.opnd(&fconst(0.0));
    if split {
        // With the running sums already in the scratch, C does not depend
        // on the column order, so it runs backwards together with R: each
        // adjoint goes straight into the reverse running sum.
        for c in carry.iter().flatten() {
            g.f.emit(format!("store {t} {zero}, ptr {c}"));
        }
        for_range(g, "0", &cols, |g, kk| {
            let col = g.f.iop("sub nsw", &last, kk);
            col_begin(g);
            c_col(g, &col);
            r_col(g, &col);
            col_flush(g, &col);
        });
    } else {
        for_range(g, "0", &cols, |g, col| {
            col_begin(g);
            c_col(g, col);
            col_flush(g, col);
        });
        // reverse: parents before children, so a parent's contribution to
        // a child's adjoint at this element is in place before the child
        // reads it
        for c in carry.iter().flatten() {
            g.f.emit(format!("store {t} {zero}, ptr {c}"));
        }
        for_range(g, "0", &cols, |g, kk| {
            let col = g.f.iop("sub nsw", &last, kk);
            col_begin(g);
            r_col(g, &col);
            col_flush(g, &col);
        });
    }
    g.vpadj.clear();
    for k in 0..u as usize {
        for (p, acc) in rp.iter().zip(&row_accs[k]) {
            g.inv_acc.remove(&(p.clone(), Ax::Row));
            let v = g.f.acc_get(acc);
            let gp = g.gptr[p].clone();
            g.f.add_to(&gp, &r0s[k], &v);
        }
    }
    g.f.lanes = 1;
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

fn gen_constrain(m: &mut Module, tm: &TModel, opts: &Opts, cm: &[(Dim, Dim)]) {
    let mut g = Mg::new(m, tm, opts.strict_fp);
    g.cm = cm.to_vec();
    let (layout, total) = g.layout(tm);
    g.f.memcpy(g.m, "%out", "%unc", &total);
    for ((n, t), (_, off, _)) in tm.params.iter().zip(&layout) {
        // (the log-Jacobian of the exp is added in logp, gen_logp)
        let transform = match t {
            Ty::Matrix(r, c, _) if g.is_cm(r, c) => {
                // back to row-major for the draws
                let (rd, cd) = (g.dim(r), g.dim(c));
                let src = g.f.gep("%unc", off);
                let dst = g.f.gep("%out", off);
                untranspose(&mut g, &src, &dst, &rd, &cd);
                "unconstrained; scan layout inside the sampler, draws written row-major"
            }
            Ty::Scalar(Dom::Positive) => {
                let u = g.f.load("%unc", off);
                let v = g.f.intrinsic1(g.m, "llvm.exp.f64", &u);
                g.f.store(&v, "%out", off);
                "sampled as log; exp maps it back, log-Jacobian added"
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
                "each entry sampled as log; exp maps it back, log-Jacobian added"
            }
            _ => "unconstrained",
        };
        if let Some(r) = rep(g.m) {
            r.params.push(ParamRep { name: n.clone(), ty: explain::src_ty(t), size: Poly::size_of(t).to_string(), transform: transform.into() });
        }
    }
    let header = format!("define void @mint_model_{}_constrain(ptr %unc, ptr %out)", tm.name);
    g.finish(&header, &["ret void".into()]);
}

/// Converts an unconstrained vector between the user's row-major layout and
/// the internal one (`_to_internal`, `_to_user`), for the runtime's gradient
/// benchmark and printout. Returns false when the layouts are the same.
fn gen_permute(m: &mut Module, tm: &TModel, opts: &Opts, cm: &[(Dim, Dim)]) -> bool {
    let any = tm.params.iter().any(|(_, t)| matches!(t, Ty::Matrix(r, c, _) if cm.contains(&(r.clone(), c.clone()))));
    if !any {
        return false;
    }
    for inward in [true, false] {
        let mut g = Mg::new(m, tm, opts.strict_fp);
        g.cm = cm.to_vec();
        let (layout, total) = g.layout(tm);
        g.f.memcpy(g.m, "%dst", "%src", &total);
        for ((n, t), (_, off, _)) in tm.params.iter().zip(&layout) {
            let Ty::Matrix(r, c, _) = t else { continue };
            if !g.is_cm(r, c) {
                continue;
            }
            if inward {
                if let Some(rp) = rep(g.m) {
                    rp.layout.push(format!("param {n}: in the scan layout inside the sampler's vector; the runtime converts its benchmark point and printed gradients (to_internal, to_user)"));
                }
            }
            let (rd, cd) = (g.dim(r), g.dim(c));
            let src = g.f.gep("%src", off);
            let dst = g.f.gep("%dst", off);
            if inward {
                transpose(&mut g, &src, &dst, &rd, &cd);
            } else {
                untranspose(&mut g, &src, &dst, &rd, &cd);
            }
        }
        let which = if inward { "to_internal" } else { "to_user" };
        let header = format!("define void @mint_model_{}_{which}(ptr %src, ptr %dst)", tm.name);
        g.finish(&header, &["ret void".into()]);
    }
    true
}

fn gen_sample_fn(m: &mut Module, tm: &TModel, opts: &Opts, permute: bool, variants: usize, leap: bool) {
    let mut g = Mg::new(m, tm, opts.strict_fp);
    g.m.declare("declare void @mint_set_layout(ptr, ptr)");
    g.m.declare("declare void @mint_set_leap(ptr, ptr)");
    if leap {
        let n = &tm.name;
        g.f.emit(format!("call void @mint_set_leap(ptr @mint_model_{n}_leap, ptr @mint_model_{n}_leap_blocks)"));
    } else {
        g.f.emit("call void @mint_set_leap(ptr null, ptr null)");
    }
    if permute {
        let n = &tm.name;
        g.f.emit(format!("call void @mint_set_layout(ptr @mint_model_{n}_to_internal, ptr @mint_model_{n}_to_user)"));
    } else {
        g.f.emit("call void @mint_set_layout(ptr null, ptr null)");
    }
    let (layout, total) = g.layout(tm);
    if let Some(r) = rep(g.m) {
        let mut d = Poly::default();
        for (_, t) in &tm.params {
            d.add(&Poly::size_of(t));
        }
        r.total = d.to_string();
        r.dim = d;
    }
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
    let name = &tm.name;
    // the variant of logp chosen by init for this call's data
    let logp = if variants > 1 {
        let v = g.f.load_i64(&format!("@mint_model_{name}_variant"));
        let a = g.f.reg();
        g.f.emit(format!("{a} = getelementptr inbounds [{variants} x ptr], ptr @mint_model_{name}_logp_table, i64 0, i64 {v}"));
        g.f.load_ptr(&a)
    } else {
        format!("@mint_model_{name}_logp")
    };
    let r = g.f.reg();
    g.f.emit(format!(
        "{r} = call ptr @mint_sample(ptr {logp}, ptr @mint_model_{name}_constrain, i64 {total}, i64 %draws, i64 %warmup, i64 %chains, i64 %seed, i64 {k}, ptr {names}, ptr {sizes})"
    ));
    let header = format!("define ptr @mint_model_{name}_sample(i64 %draws, i64 %warmup, i64 %chains, i64 %seed)");
    g.finish(&header, &[format!("ret ptr {r}")]);
}
