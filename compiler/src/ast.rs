//! Untyped syntax tree produced by the parser.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DimAnn {
    Sym(String),
    Const(i64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeAnn {
    Real,
    Positive,
    Prob,
    Int,
    Vector(DimAnn),
    /// `Positive[n]`, `Prob[n]`, `Real[n]`: a vector whose entries lie in a domain.
    VecOf(DimAnn, String),
    Matrix(DimAnn, DimAnn),
    Psd(DimAnn),
    Spd(DimAnn),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    EMul,
    EDiv,
    Pow,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::EMul => ".*",
            BinOp::EDiv => "./",
            BinOp::Pow => "^",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    /// Numeric literal; the flag records whether it was written as an integer.
    Num(f64, bool),
    Str(String),
    Var(String),
    VecLit(Vec<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Transpose(Box<Expr>),
    Call(String, Vec<Arg>),
}

#[derive(Clone, Debug)]
pub struct Arg {
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let {
        name: String,
        mutable: bool,
        ann: Option<TypeAnn>,
        value: Expr,
        span: Span,
    },
    Assign {
        name: String,
        value: Expr,
        span: Span,
    },
    Repeat {
        count: Expr,
        body: Block,
        span: Span,
    },
    Expr(Expr),
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ann: TypeAnn,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Option<TypeAnn>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ModelStmt {
    Let { name: String, value: Expr, span: Span },
    Tilde { lhs: Expr, dist: String, args: Vec<Expr>, span: Span },
}

#[derive(Clone, Debug)]
pub struct ModelDecl {
    pub name: String,
    pub data: Vec<Param>,
    pub params: Vec<Param>,
    pub body: Vec<ModelStmt>,
}

#[derive(Clone, Debug, Default)]
pub struct Program {
    pub fns: Vec<FnDecl>,
    pub models: Vec<ModelDecl>,
}
