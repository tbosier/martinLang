//! Types: shapes with symbolic dimensions, value domains and matrix structure.
//!
//! Domains and structures are both chains ordered by inclusion, so "a fits b"
//! is simply `a <= b`:
//!
//!   Prob (0,1)  ⊂  Positive (0,∞)  ⊂  NonNeg [0,∞)  ⊂  Real
//!   SPD  ⊂  PSD  ⊂  Sym  ⊂  General
//!
//! These are facts about real numbers. Floating point can still round a
//! Positive value to 0 or break a Cholesky factorisation of a matrix that is
//! SPD in exact arithmetic; the runtime reports that case instead of
//! producing garbage.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Sym(String),
    Const(i64),
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Dim::Sym(s) => write!(f, "{s}"),
            Dim::Const(c) => write!(f, "{c}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dom {
    Prob,
    Positive,
    NonNeg,
    Real,
}

impl fmt::Display for Dom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Dom::Prob => "Prob",
            Dom::Positive => "Positive",
            Dom::NonNeg => "NonNeg",
            Dom::Real => "Real",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Struct {
    Spd,
    Psd,
    Sym,
    General,
}

impl fmt::Display for Struct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Struct::Spd => "SPD",
            Struct::Psd => "PSD",
            Struct::Sym => "symmetric",
            Struct::General => "general",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Scalar(Dom),
    /// A dimension used as a number.
    Int,
    Str,
    Void,
    Vector(Dim, Dom),
    Matrix(Dim, Dim, Struct),
    /// A model applied to its data, e.g. `Regression(X, y)`.
    Model(String),
    Posterior(String),
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ty::Scalar(d) => write!(f, "{d}"),
            Ty::Int => write!(f, "Int"),
            Ty::Str => write!(f, "String"),
            Ty::Void => write!(f, "nothing"),
            Ty::Vector(n, Dom::Real) => write!(f, "Vector[{n}]"),
            Ty::Vector(n, d) => write!(f, "Vector[{n}] of {d}"),
            Ty::Matrix(r, _, Struct::Spd) => write!(f, "SPD[{r}]"),
            Ty::Matrix(r, _, Struct::Psd) => write!(f, "PSD[{r}]"),
            Ty::Matrix(r, c, Struct::Sym) => write!(f, "Matrix[{r}, {c}] (symmetric)"),
            Ty::Matrix(r, c, Struct::General) => write!(f, "Matrix[{r}, {c}]"),
            Ty::Model(m) => write!(f, "model {m}"),
            Ty::Posterior(m) => write!(f, "posterior of {m}"),
        }
    }
}

impl Ty {
    pub fn is_scalar(&self) -> bool {
        matches!(self, Ty::Scalar(_) | Ty::Int)
    }
    pub fn is_buffer(&self) -> bool {
        matches!(self, Ty::Vector(..) | Ty::Matrix(..))
    }
    /// Element domain of a scalar or vector.
    pub fn dom(&self) -> Dom {
        match self {
            Ty::Scalar(d) | Ty::Vector(_, d) => *d,
            Ty::Int => Dom::NonNeg,
            _ => Dom::Real,
        }
    }
}

pub fn const_dom(v: f64) -> Dom {
    if v > 0.0 && v < 1.0 {
        Dom::Prob
    } else if v >= 1.0 {
        Dom::Positive
    } else if v == 0.0 {
        Dom::NonNeg
    } else {
        Dom::Real
    }
}

pub fn dom_add(a: Dom, b: Dom) -> Dom {
    if a <= Dom::NonNeg && b <= Dom::NonNeg {
        if a <= Dom::Positive || b <= Dom::Positive {
            Dom::Positive
        } else {
            Dom::NonNeg
        }
    } else {
        Dom::Real
    }
}

/// `a - b`; `a_const` is the value of `a` when it is a literal.
pub fn dom_sub(a_const: Option<f64>, b: Dom) -> Dom {
    match a_const {
        Some(c) if c == 1.0 && b == Dom::Prob => Dom::Prob,
        Some(c) if c >= 1.0 && b == Dom::Prob => Dom::Positive,
        _ => Dom::Real,
    }
}

pub fn dom_mul(a: Dom, b: Dom) -> Dom {
    if a == Dom::Prob && b == Dom::Prob {
        Dom::Prob
    } else if a <= Dom::Positive && b <= Dom::Positive {
        Dom::Positive
    } else if a <= Dom::NonNeg && b <= Dom::NonNeg {
        Dom::NonNeg
    } else {
        Dom::Real
    }
}

pub fn dom_div(a: Dom, b: Dom) -> Dom {
    if b <= Dom::Positive {
        if a <= Dom::Positive {
            Dom::Positive
        } else if a <= Dom::NonNeg {
            Dom::NonNeg
        } else {
            Dom::Real
        }
    } else {
        Dom::Real
    }
}

/// `a ^ b`; `b_const` is the exponent when it is a literal.
pub fn dom_pow(a: Dom, b_const: Option<f64>) -> Dom {
    match b_const {
        Some(e) if a == Dom::Prob && e > 0.0 => Dom::Prob,
        _ if a <= Dom::Positive => Dom::Positive,
        Some(e) if e.fract() == 0.0 && (e as i64) % 2 == 0 => Dom::NonNeg,
        Some(e) if a <= Dom::NonNeg && e > 0.0 => Dom::NonNeg,
        _ => Dom::Real,
    }
}

pub fn struct_add(a: Struct, b: Struct) -> Struct {
    if (a == Struct::Spd && b <= Struct::Psd) || (b == Struct::Spd && a <= Struct::Psd) {
        Struct::Spd
    } else if a <= Struct::Psd && b <= Struct::Psd {
        Struct::Psd
    } else if a <= Struct::Sym && b <= Struct::Sym {
        Struct::Sym
    } else {
        Struct::General
    }
}

pub fn struct_scale(c: Dom, s: Struct) -> Struct {
    if c <= Dom::Positive {
        s
    } else if c <= Dom::NonNeg {
        if s <= Struct::Psd {
            Struct::Psd
        } else {
            s
        }
    } else if s <= Struct::Sym {
        Struct::Sym
    } else {
        Struct::General
    }
}

/// Elementwise (Hadamard) product: by the Schur product theorem the product
/// of two PSD matrices is PSD, and of two SPD matrices is SPD.
pub fn struct_hadamard(a: Struct, b: Struct) -> Struct {
    if a == Struct::Spd && b == Struct::Spd {
        Struct::Spd
    } else if a <= Struct::Psd && b <= Struct::Psd {
        Struct::Psd
    } else if a <= Struct::Sym && b <= Struct::Sym {
        Struct::Sym
    } else {
        Struct::General
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logistic_hessian_weights_are_prob() {
        // mu = sigmoid(..) : Prob;  1 - mu : Prob;  mu .* (1 - mu) : Prob
        let one_minus = dom_sub(Some(1.0), Dom::Prob);
        assert_eq!(one_minus, Dom::Prob);
        assert_eq!(dom_mul(Dom::Prob, one_minus), Dom::Prob);
    }

    #[test]
    fn gram_plus_ridge_is_spd() {
        let ridge = struct_scale(Dom::Positive, Struct::Spd); // lambda * I
        assert_eq!(struct_add(Struct::Psd, ridge), Struct::Spd);
        // X'X alone is only PSD, and a Real multiple of I is not PSD
        assert_eq!(struct_add(Struct::Psd, Struct::Psd), Struct::Psd);
        assert_eq!(struct_add(Struct::Psd, struct_scale(Dom::Real, Struct::Spd)), Struct::Sym);
    }

    #[test]
    fn domain_arithmetic() {
        assert_eq!(dom_add(Dom::Positive, Dom::NonNeg), Dom::Positive);
        assert_eq!(dom_add(Dom::Positive, Dom::Real), Dom::Real);
        assert_eq!(dom_div(Dom::NonNeg, Dom::Positive), Dom::NonNeg);
        assert_eq!(dom_div(Dom::Positive, Dom::Real), Dom::Real);
        assert_eq!(dom_pow(Dom::Real, Some(2.0)), Dom::NonNeg);
        assert_eq!(dom_sub(Some(2.0), Dom::Prob), Dom::Positive);
        assert_eq!(dom_sub(None, Dom::Prob), Dom::Real);
        assert_eq!(const_dom(0.5), Dom::Prob);
        assert_eq!(const_dom(0.0), Dom::NonNeg);
    }
}
