//! `mintc explain`: what the compiler found in a program and what it did.
//!
//! The report is a decision log. codegen.rs and model.rs write to it
//! (`Module::log`) at the points where they make each decision, during the
//! same compilation that `build` runs, so it describes the code that is
//! emitted and cannot drift from it. This file holds the log itself, the
//! helpers that describe sizes, expressions and type facts, and the renderer.
//!
//! Sizes that depend on data dimensions are symbolic (`Poly`): dimensions
//! such as n or G are known only when the program reads its data.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ast::BinOp;
use crate::check::{TExpr, TK};
use crate::types::{Dim, Dom, Struct, Ty};

#[derive(Default)]
pub struct Report {
    /// Compiler switches that differ from the defaults.
    pub switches: Vec<String>,
    /// Facts about the host that change code generation.
    pub host: Vec<String>,
    pub models: Vec<ModelRep>,
    pub fns: Vec<FnRep>,
}

#[derive(Default)]
pub struct ModelRep {
    pub name: String,
    pub data: Vec<String>,
    pub params: Vec<ParamRep>,
    /// Number of unconstrained values NUTS samples.
    pub total: String,
    /// sample() calls of this model and the dimensions they bind.
    pub calls: Vec<String>,
    pub layout: Vec<String>,
    pub stmts: Vec<StmtRep>,
    pub narrow: Vec<String>,
    pub other: Vec<String>,
}

pub struct ParamRep {
    pub name: String,
    pub ty: String,
    pub size: String,
    pub transform: String,
}

pub struct StmtRep {
    pub line: u32,
    pub text: String,
    pub lines: Vec<String>,
}

#[derive(Default)]
pub struct FnRep {
    pub sig: String,
    /// (indent, text) in emission order.
    pub lines: Vec<(usize, String)>,
}

impl Report {
    pub fn model(&mut self) -> &mut ModelRep {
        self.models.last_mut().expect("a model is being compiled")
    }
    pub fn func(&mut self) -> &mut FnRep {
        self.fns.last_mut().expect("a function is being compiled")
    }
}

// ------------------------------------------------------------ sizes

/// A polynomial in dimension names with integer coefficients, e.g.
/// G x T + G + T + 1.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Poly(BTreeMap<Vec<String>, i64>);

impl Poly {
    pub fn constant(c: i64) -> Poly {
        let mut p = Poly::default();
        if c != 0 {
            p.0.insert(Vec::new(), c);
        }
        p
    }
    /// The product of some dimensions.
    pub fn dims(ds: &[&Dim]) -> Poly {
        let mut c = 1i64;
        let mut names = Vec::new();
        for d in ds {
            match d {
                Dim::Const(k) => c *= k,
                Dim::Sym(s) => names.push(s.clone()),
            }
        }
        names.sort();
        let mut p = Poly::default();
        if c != 0 {
            p.0.insert(names, c);
        }
        p
    }
    /// Number of values in a scalar, vector or matrix of type `t`.
    pub fn size_of(t: &Ty) -> Poly {
        match t {
            Ty::Vector(n, _) => Poly::dims(&[n]),
            Ty::Matrix(r, c, _) => Poly::dims(&[r, c]),
            _ => Poly::constant(1),
        }
    }
    pub fn add(&mut self, o: &Poly) {
        for (k, v) in &o.0 {
            let e = self.0.entry(k.clone()).or_insert(0);
            *e += v;
            if *e == 0 {
                self.0.remove(k);
            }
        }
    }

    /// The value when every dimension is known.
    pub fn eval(&self, dims: &HashMap<String, i64>) -> Option<i64> {
        let mut s = 0i64;
        for (k, v) in &self.0 {
            let mut t = *v;
            for d in k {
                t *= dims.get(d)?;
            }
            s += t;
        }
        Some(s)
    }
    /// More than one term: needs parentheses as a factor.
    pub fn is_sum(&self) -> bool {
        self.0.len() > 1
    }
}

impl std::fmt::Display for Poly {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.0.is_empty() {
            return f.write_str("0");
        }
        let mut terms: Vec<(&Vec<String>, &i64)> = self.0.iter().collect();
        terms.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(b.0)));
        let parts: Vec<String> = terms
            .into_iter()
            .map(|(k, v)| match (k.is_empty(), *v) {
                (true, v) => v.to_string(),
                (false, 1) => k.join(" x "),
                (false, v) => format!("{v} x {}", k.join(" x ")),
            })
            .collect();
        f.write_str(&parts.join(" + "))
    }
}

/// `p` as a factor of a product: parenthesised when it is a sum.
pub fn factor(p: &Poly) -> String {
    if p.is_sum() {
        format!("({p})")
    } else {
        p.to_string()
    }
}

/// A type as it is written in source.
pub fn src_ty(t: &Ty) -> String {
    match t {
        Ty::Scalar(d) => d.to_string(),
        Ty::Vector(n, Dom::Real) => format!("Vector[{n}]"),
        Ty::Vector(n, d) => format!("{d}[{n}]"),
        Ty::Matrix(r, c, Struct::General) => format!("Matrix[{r}, {c}]"),
        Ty::Matrix(r, _, Struct::Spd) => format!("SPD[{r}]"),
        Ty::Matrix(r, _, Struct::Psd) => format!("PSD[{r}]"),
        t => t.to_string(),
    }
}

// ------------------------------------------------------------ expressions

/// A name as the user wrote it (the checker renames function locals to
/// `name.N`).
pub fn plain(name: &str) -> &str {
    match name.rsplit_once('.') {
        Some((a, b)) if !b.is_empty() && b.bytes().all(|c| c.is_ascii_digit()) => a,
        _ => name,
    }
}

pub fn num(v: f64) -> String {
    format!("{v}")
}

/// An expression of the typed tree, printed close to its source form.
pub fn show(e: &TExpr) -> String {
    show_p(e, 0)
}

fn prec(e: &TExpr) -> u8 {
    match &e.kind {
        TK::Bin(BinOp::Add | BinOp::Sub, ..) => 1,
        TK::Bin(BinOp::Pow, ..) => 3,
        TK::Bin(..) | TK::MatVec { .. } | TK::Gram { .. } | TK::MatMul { .. } | TK::Neg(_) => 2,
        _ => 4,
    }
}

fn show_p(e: &TExpr, ctx: u8) -> String {
    let s = match &e.kind {
        TK::Num(v) => num(*v),
        TK::Str(s) => format!("\"{s}\""),
        TK::Var(n) => plain(n).to_string(),
        TK::VecLit(xs) => format!("[{}]", xs.iter().map(show).collect::<Vec<_>>().join(", ")),
        TK::DimVal(d) => d.to_string(),
        TK::Bin(op, a, b) => {
            let (l, r) = match op {
                BinOp::Add => (1, 1),
                BinOp::Sub => (1, 2),
                BinOp::Pow => (4, 4),
                _ => (2, 3),
            };
            format!("{} {} {}", show_p(a, l), op.symbol(), show_p(b, r))
        }
        TK::Neg(a) => format!("-{}", show_p(a, 3)),
        TK::Func(f, a) => format!("{}({})", f.name(), show(a)),
        TK::MatVec { m, trans, v } => format!("{}{} * {}", show_p(m, 4), if *trans { "'" } else { "" }, show_p(v, 3)),
        TK::Gram { a, w } => {
            let a = show_p(a, 4);
            match w {
                Some(w) => format!("{a}' * diag({}) * {a}", show(w)),
                None => format!("{a}' * {a}"),
            }
        }
        TK::MatMul { a, ta, b, tb } => {
            format!("{}{} * {}{}", show_p(a, 4), if *ta { "'" } else { "" }, show_p(b, 4), if *tb { "'" } else { "" })
        }
        TK::Transpose(a) => format!("{}'", show_p(a, 4)),
        TK::Identity(d) => format!("I({d})"),
        TK::Fill(v) => {
            let f = if *v == 0.0 { "zeros" } else { "ones" };
            match &e.ty {
                Ty::Vector(n, _) => format!("{f}({n})"),
                Ty::Matrix(r, c, _) => format!("{f}({r}, {c})"),
                _ => num(*v),
            }
        }
        TK::Solve { h, g } => format!("solve({}, {})", show(h), show(g)),
        TK::AssumeSpd(a) => format!("assume_spd({})", show(a)),
        TK::Sum(a) => format!("sum({})", show(a)),
        TK::Cumsum(a) => match &a.ty {
            Ty::Matrix(_, c, _) => format!("cumsum({}, {c})", show(a)),
            _ => format!("cumsum({})", show(a)),
        },
        TK::Dot(a, b) => format!("dot({}, {})", show(a), show(b)),
        TK::Norm(a) => format!("norm({})", show(a)),
        TK::Call { name, args, .. } => format!("{name}({})", args.iter().map(show).collect::<Vec<_>>().join(", ")),
        TK::Read(p) => format!("read(\"{p}\")"),
        TK::ModelInst { model, data, .. } => format!("{model}({})", data.iter().map(show).collect::<Vec<_>>().join(", ")),
        TK::Sample { inst, .. } => format!("sample({}, ...)", show(inst)),
        TK::Clock => "clock()".into(),
    };
    if prec(e) < ctx {
        format!("({s})")
    } else {
        s
    }
}

/// Names of variables an expression reads, in order of first use.
pub fn vars_of(e: &TExpr, out: &mut Vec<String>) {
    if let TK::Var(n) = &e.kind {
        if !out.contains(n) {
            out.push(n.clone());
        }
    }
    for c in children(e) {
        vars_of(c, out);
    }
}

fn children(e: &TExpr) -> Vec<&TExpr> {
    match &e.kind {
        TK::Bin(_, a, b) | TK::Dot(a, b) => vec![a, b],
        TK::Neg(a) | TK::Func(_, a) | TK::Transpose(a) | TK::AssumeSpd(a) | TK::Sum(a) | TK::Cumsum(a) | TK::Norm(a) => vec![a],
        TK::MatVec { m, v, .. } => vec![m, v],
        TK::Gram { a, w } => std::iter::once(&**a).chain(w.as_deref()).collect(),
        TK::MatMul { a, b, .. } => vec![a, b],
        TK::Solve { h, g } => vec![h, g],
        TK::VecLit(xs) => xs.iter().collect(),
        TK::Call { args, .. } | TK::ModelInst { data: args, .. } => args.iter().collect(),
        TK::Sample { inst, .. } => vec![inst],
        _ => vec![],
    }
}

// ------------------------------------------------------------ type facts

/// What the checker proved about a value: its domain, or its matrix
/// structure. None when there is nothing to say (Real, general).
pub fn fact(t: &Ty) -> Option<String> {
    match t {
        Ty::Scalar(d) | Ty::Vector(_, d) if *d != Dom::Real => Some(d.to_string()),
        Ty::Int => Some("NonNeg".into()),
        Ty::Matrix(_, _, s) if *s != Struct::General => Some(s.to_string()),
        _ => None,
    }
}

fn short(t: &Ty) -> String {
    match t {
        Ty::Scalar(d) | Ty::Vector(_, d) => d.to_string(),
        Ty::Int => "NonNeg".into(),
        Ty::Matrix(_, _, s) => s.to_string(),
        t => t.to_string(),
    }
}

/// Where the names of a function or model come from, for `fact_chain`.
pub struct Scope<'a> {
    /// let name -> value (names assigned later are left out: their type is
    /// the declared, widened one)
    pub lets: &'a HashMap<String, TExpr>,
    /// parameter name -> (type, how it is declared)
    pub decl: &'a HashMap<String, (Ty, String)>,
}

/// The chain of type facts behind `e`'s fact: one line per step, each
/// "expression: fact, rule", children indented below their parent. Only
/// operands whose facts the rule used are followed. The facts are the
/// checker's (the types in the typed tree); the rule names say which of the
/// checker's rules (types.rs) combined them.
pub fn fact_chain(e: &TExpr, sc: &Scope, depth: usize, seen: &mut HashSet<String>, out: &mut Vec<(usize, String)>) {
    fact_chain_as(e, None, sc, depth, seen, out)
}

fn fact_chain_as(e: &TExpr, label: Option<String>, sc: &Scope, depth: usize, seen: &mut HashSet<String>, out: &mut Vec<(usize, String)>) {
    if depth > 12 {
        return;
    }
    let Some(f) = fact(&e.ty) else { return };
    let text = label.unwrap_or_else(|| show(e));
    let mut follow: Vec<&TExpr> = Vec::new();
    let why: String = match &e.kind {
        TK::Var(n) => {
            if let Some(v) = sc.lets.get(n) {
                if seen.insert(n.clone()) {
                    return fact_chain_as(v, Some(format!("{} = {}", plain(n), show(v))), sc, depth, seen, out);
                }
                out.push((depth, format!("{} is {f} (shown above)", plain(n))));
                return;
            }
            match sc.decl.get(n) {
                Some((_, how)) => how.clone(),
                None => "declared type".into(),
            }
        }
        TK::Num(_) => "a literal".into(),
        TK::VecLit(_) => format!("every entry is {}", short(&e.ty)),
        TK::Fill(v) => if *v == 0.0 { "zeros".into() } else { "ones".into() },
        TK::DimVal(_) => "a dimension".into(),
        TK::Identity(_) => "the identity".into(),
        TK::AssumeSpd(_) => "assume_spd, checked at run time (symmetry and a Cholesky factorisation)".into(),
        TK::Read(_) => "a read with a domain annotation, checked when it is read".into(),
        TK::Func(func, a) => {
            let u = func.name();
            match func {
                crate::check::Func::Exp => format!("{u} is always Positive"),
                crate::check::Func::Sigmoid => format!("{u} is always in (0, 1)"),
                crate::check::Func::Abs => format!("{u} is always >= 0"),
                _ => {
                    follow.push(a);
                    format!("{u} of {}", short(&a.ty))
                }
            }
        }
        TK::Gram { w: Some(w), .. } => {
            follow.push(w);
            match &e.ty {
                Ty::Matrix(_, _, Struct::Psd) => format!("A' * diag(w) * A with weights w >= 0 (here {})", short(&w.ty)),
                _ => "A' * diag(w) * A with weights of unknown sign".into(),
            }
        }
        TK::Gram { w: None, .. } => "A' * A".into(),
        TK::Bin(op, a, b) => {
            // (a literal operand is visible in the rule itself)
            follow.extend([&**a, &**b].into_iter().filter(|c| fact(&c.ty).is_some() && !matches!(c.kind, TK::Num(_))));
            let lit = |x: &TExpr, s: String| if matches!(x.kind, TK::Num(_)) { show(x) } else { s };
            let (sa, sb) = (lit(a, short(&a.ty)), lit(b, short(&b.ty)));
            match (op, &a.ty, &b.ty) {
                (BinOp::EMul, Ty::Matrix(..), Ty::Matrix(..)) => format!("{sa} .* {sb} (Schur product theorem)"),
                _ => format!("{sa} {} {sb}", op.symbol()),
            }
        }
        TK::Neg(_) => "negation".into(),
        TK::Transpose(a) => {
            follow.push(a);
            "transpose".into()
        }
        TK::Sum(a) | TK::Norm(a) => {
            follow.push(a);
            "a sum of non-negative values".into()
        }
        TK::Call { name, .. } => format!("the declared return type of {name}"),
        _ => "derived by the checker".into(),
    };
    out.push((depth, format!("{text} is {f}: {why}")));
    for c in follow {
        fact_chain_as(c, None, sc, depth + 1, seen, out);
    }
}

// ------------------------------------------------------------ rendering

pub fn render(r: &Report) -> String {
    let mut o = String::new();
    let mut put = |indent: usize, s: &str| {
        o += &"  ".repeat(indent);
        o += s;
        o.push('\n');
    };
    put(0, &format!("switches: {}", if r.switches.is_empty() { "defaults".to_string() } else { r.switches.join(", ") }));
    for h in &r.host {
        put(0, h);
    }
    for m in &r.models {
        put(0, "");
        put(0, &format!("model {}", m.name));
        if !m.data.is_empty() {
            put(1, &format!("data: {}", m.data.join(", ")));
        }
        put(1, &format!("parameters: NUTS samples {} unconstrained values", m.total));
        let w0 = m.params.iter().map(|p| p.name.len()).max().unwrap_or(0);
        let w1 = m.params.iter().map(|p| p.ty.len()).max().unwrap_or(0);
        let w2 = m.params.iter().map(|p| p.size.len()).max().unwrap_or(0);
        for p in &m.params {
            put(2, &format!("{:w0$}  {:w1$}  {:w2$}  {}", p.name, p.ty, p.size, p.transform));
        }
        for c in &m.calls {
            put(1, c);
        }
        if !m.layout.is_empty() {
            put(1, "layout:");
            for l in &m.layout {
                put(2, l);
            }
        }
        put(1, "statements:");
        for s in &m.stmts {
            put(2, &format!("line {}: {}", s.line, s.text));
            for l in &s.lines {
                put(3, l);
            }
        }
        if !m.narrow.is_empty() {
            put(1, "narrow data (checked when sample() starts):");
            for l in &m.narrow {
                put(2, l);
            }
        }
        for l in &m.other {
            put(1, l);
        }
    }
    for f in &r.fns {
        put(0, "");
        put(0, &f.sig);
        for (d, l) in &f.lines {
            put(1 + d, l);
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_sizes() {
        let (g, t, one) = (Dim::Sym("G".into()), Dim::Sym("T".into()), Poly::constant(1));
        let mut p = Poly::dims(&[&g, &t]);
        p.add(&Poly::dims(&[&g]));
        p.add(&Poly::dims(&[&t]));
        p.add(&one);
        assert_eq!(p.to_string(), "G x T + G + T + 1");
        let mut q = Poly::dims(&[&Dim::Sym("p".into())]);
        q.add(&one);
        q.add(&one);
        assert_eq!(q.to_string(), "p + 2");

        assert_eq!(q.eval(&HashMap::from([("p".to_string(), 3)])), Some(5));
        assert_eq!(Poly::dims(&[&Dim::Const(8), &Dim::Const(2)]).to_string(), "16");
        assert_eq!(plain("H.12"), "H");
        assert_eq!(plain("x"), "x");
    }
}
