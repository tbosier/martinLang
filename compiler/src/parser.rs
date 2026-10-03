//! Recursive-descent parser.
//!
//! Precedence, loosest first: `+ -`, `* / .* ./`, unary `-`, `^` (right
//! associative), postfix `'` (transpose) and calls.

use crate::ast::*;
use crate::diag::{err, err_help, Res};
use crate::lexer::{Tok, Token};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Parser { toks, pos: 0 }
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }
    fn peek_at(&self, k: usize) -> &Tok {
        &self.toks[(self.pos + k).min(self.toks.len() - 1)].tok
    }
    fn span(&self) -> Span {
        self.toks[self.pos].span
    }
    fn next(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.next();
            true
        } else {
            false
        }
    }
    fn expect(&mut self, t: &Tok, what: &str) -> Res<Span> {
        let sp = self.span();
        if self.eat(t) {
            Ok(sp)
        } else {
            err(sp, format!("expected {what}, found {}", describe(self.peek())))
        }
    }
    fn skip_newlines(&mut self) {
        while self.peek() == &Tok::Newline {
            self.next();
        }
    }
    fn ident(&mut self, what: &str) -> Res<(String, Span)> {
        let sp = self.span();
        match self.peek().clone() {
            Tok::Ident(s) => {
                self.next();
                Ok((s, sp))
            }
            t => err(sp, format!("expected {what}, found {}", describe(&t))),
        }
    }
    fn end_stmt(&mut self) -> Res<()> {
        match self.peek() {
            Tok::Newline => {
                self.skip_newlines();
                Ok(())
            }
            Tok::RBrace | Tok::Eof => Ok(()),
            t => err(self.span(), format!("expected end of line, found {}", describe(t))),
        }
    }

    pub fn program(&mut self) -> Res<Program> {
        let mut prog = Program::default();
        self.skip_newlines();
        while self.peek() != &Tok::Eof {
            match self.peek().clone() {
                Tok::Ident(k) if k == "fn" => prog.fns.push(self.fn_decl()?),
                Tok::Ident(k) if k == "model" => prog.models.push(self.model_decl()?),
                t => {
                    return err_help(
                        self.span(),
                        format!("expected `fn` or `model`, found {}", describe(&t)),
                        "a Martin file is a list of `fn` and `model` declarations",
                    )
                }
            }
            self.skip_newlines();
        }
        Ok(prog)
    }

    fn type_ann(&mut self) -> Res<TypeAnn> {
        let (name, sp) = self.ident("a type")?;
        let dims = |p: &mut Parser, n: usize| -> Res<Vec<DimAnn>> {
            p.expect(&Tok::LBracket, "`[`")?;
            let mut out = Vec::new();
            loop {
                let sp = p.span();
                match p.next().tok {
                    Tok::Ident(s) => out.push(DimAnn::Sym(s)),
                    Tok::Num(v, true) if v >= 0.0 => out.push(DimAnn::Const(v as i64)),
                    t => return err(sp, format!("expected a dimension name or size, found {}", describe(&t))),
                }
                if !p.eat(&Tok::Comma) {
                    break;
                }
            }
            p.expect(&Tok::RBracket, "`]`")?;
            if out.len() != n {
                return err(sp, format!("this type takes {n} dimension(s), got {}", out.len()));
            }
            Ok(out)
        };
        if matches!(name.as_str(), "Real" | "Positive" | "Prob") && self.peek() == &Tok::LBracket {
            return Ok(TypeAnn::VecOf(dims(self, 1)?.remove(0), name));
        }
        Ok(match name.as_str() {
            "Real" => TypeAnn::Real,
            "Positive" => TypeAnn::Positive,
            "Prob" => TypeAnn::Prob,
            "Int" => TypeAnn::Int,
            "Vector" => TypeAnn::Vector(dims(self, 1)?.remove(0)),
            "Matrix" => {
                let mut d = dims(self, 2)?;
                let b = d.pop().unwrap();
                TypeAnn::Matrix(d.pop().unwrap(), b)
            }
            "SPD" => TypeAnn::Spd(dims(self, 1)?.remove(0)),
            "PSD" => TypeAnn::Psd(dims(self, 1)?.remove(0)),
            _ => {
                return err_help(
                    sp,
                    format!("unknown type `{name}`"),
                    "types are Real, Positive, Prob, Int, Vector[n], Positive[n], Prob[n], Matrix[m, n], PSD[n], SPD[n]",
                )
            }
        })
    }

    fn param(&mut self) -> Res<Param> {
        let (name, span) = self.ident("a name")?;
        self.expect(&Tok::Colon, "`:` and a type")?;
        let ann = self.type_ann()?;
        Ok(Param { name, ann, span })
    }

    fn fn_decl(&mut self) -> Res<FnDecl> {
        let span = self.span();
        self.next(); // fn
        let (name, _) = self.ident("a function name")?;
        self.expect(&Tok::LParen, "`(`")?;
        let mut params = Vec::new();
        if !self.eat(&Tok::RParen) {
            loop {
                params.push(self.param()?);
                if self.eat(&Tok::RParen) {
                    break;
                }
                self.expect(&Tok::Comma, "`,` or `)`")?;
            }
        }
        let ret = if self.eat(&Tok::Arrow) { Some(self.type_ann()?) } else { None };
        let body = self.block()?;
        Ok(FnDecl { name, params, ret, body, span })
    }

    fn block(&mut self) -> Res<Block> {
        self.expect(&Tok::LBrace, "`{`")?;
        self.skip_newlines();
        let mut b = Block::default();
        while self.peek() != &Tok::RBrace {
            if self.peek() == &Tok::Eof {
                return err(self.span(), "unexpected end of file: missing `}`");
            }
            let st = self.stmt()?;
            if let Stmt::Expr(e) = &st {
                if self.peek() == &Tok::RBrace
                    || (self.peek() == &Tok::Newline && {
                        let mut k = 0;
                        while self.peek_at(k) == &Tok::Newline {
                            k += 1;
                        }
                        self.peek_at(k) == &Tok::RBrace
                    })
                {
                    self.skip_newlines();
                    b.tail = Some(e.clone());
                    break;
                }
            }
            b.stmts.push(st);
            self.end_stmt()?;
        }
        self.expect(&Tok::RBrace, "`}`")?;
        Ok(b)
    }

    fn stmt(&mut self) -> Res<Stmt> {
        let span = self.span();
        match self.peek().clone() {
            Tok::Ident(k) if k == "let" => {
                self.next();
                let mutable = matches!(self.peek(), Tok::Ident(m) if m == "mut");
                if mutable {
                    self.next();
                }
                let (name, _) = self.ident("a variable name")?;
                let ann = if self.eat(&Tok::Colon) { Some(self.type_ann()?) } else { None };
                self.expect(&Tok::Assign, "`=`")?;
                self.skip_newlines();
                let value = self.expr()?;
                Ok(Stmt::Let { name, mutable, ann, value, span })
            }
            Tok::Ident(k) if k == "repeat" => {
                self.next();
                let count = self.expr()?;
                let body = self.block()?;
                Ok(Stmt::Repeat { count, body, span })
            }
            Tok::Ident(name) if self.peek_at(1) == &Tok::Assign => {
                self.next();
                self.next();
                self.skip_newlines();
                let value = self.expr()?;
                Ok(Stmt::Assign { name, value, span })
            }
            _ => Ok(Stmt::Expr(self.expr()?)),
        }
    }

    fn model_decl(&mut self) -> Res<ModelDecl> {
        self.next(); // model
        let (name, _) = self.ident("a model name")?;
        self.expect(&Tok::LBrace, "`{`")?;
        self.skip_newlines();
        let mut m = ModelDecl { name, data: vec![], params: vec![], body: vec![] };
        while !self.eat(&Tok::RBrace) {
            let sp = self.span();
            match self.peek().clone() {
                Tok::Eof => return err(sp, "unexpected end of file: missing `}`"),
                Tok::Ident(k) if k == "data" => {
                    self.next();
                    m.data.push(self.param()?);
                }
                Tok::Ident(k) if k == "param" => {
                    self.next();
                    m.params.push(self.param()?);
                }
                Tok::Ident(k) if k == "let" => {
                    self.next();
                    let (name, _) = self.ident("a name")?;
                    self.expect(&Tok::Assign, "`=`")?;
                    self.skip_newlines();
                    let value = self.expr()?;
                    m.body.push(ModelStmt::Let { name, value, span: sp });
                }
                _ => {
                    let lhs = self.expr()?;
                    if !self.eat(&Tok::Tilde) {
                        return err_help(
                            self.span(),
                            "expected `~` in a model statement",
                            "model bodies contain `data`, `param`, `let` and `x ~ Distribution(...)` lines",
                        );
                    }
                    let (dist, _) = self.ident("a distribution name")?;
                    self.expect(&Tok::LParen, "`(`")?;
                    let mut args = Vec::new();
                    if !self.eat(&Tok::RParen) {
                        loop {
                            args.push(self.expr()?);
                            if self.eat(&Tok::RParen) {
                                break;
                            }
                            self.expect(&Tok::Comma, "`,` or `)`")?;
                        }
                    }
                    m.body.push(ModelStmt::Tilde { lhs, dist, args, span: sp });
                }
            }
            self.end_stmt()?;
        }
        Ok(m)
    }

    pub fn expr(&mut self) -> Res<Expr> {
        let mut lhs = self.term()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => return Ok(lhs),
            };
            let span = self.span();
            self.next();
            self.skip_newlines();
            let rhs = self.term()?;
            lhs = Expr { kind: ExprKind::Bin(op, Box::new(lhs), Box::new(rhs)), span };
        }
    }

    fn term(&mut self) -> Res<Expr> {
        let mut lhs = self.unary()?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::DotStar => BinOp::EMul,
                Tok::DotSlash => BinOp::EDiv,
                _ => return Ok(lhs),
            };
            let span = self.span();
            self.next();
            self.skip_newlines();
            let rhs = self.unary()?;
            lhs = Expr { kind: ExprKind::Bin(op, Box::new(lhs), Box::new(rhs)), span };
        }
    }

    fn unary(&mut self) -> Res<Expr> {
        if self.peek() == &Tok::Minus {
            let span = self.span();
            self.next();
            let e = self.unary()?;
            return Ok(Expr { kind: ExprKind::Neg(Box::new(e)), span });
        }
        self.power()
    }

    fn power(&mut self) -> Res<Expr> {
        let base = self.postfix()?;
        if self.peek() == &Tok::Caret {
            let span = self.span();
            self.next();
            let exp = self.unary()?;
            return Ok(Expr { kind: ExprKind::Bin(BinOp::Pow, Box::new(base), Box::new(exp)), span });
        }
        Ok(base)
    }

    fn postfix(&mut self) -> Res<Expr> {
        let mut e = self.primary()?;
        while self.peek() == &Tok::Quote {
            let span = self.span();
            self.next();
            e = Expr { kind: ExprKind::Transpose(Box::new(e)), span };
        }
        Ok(e)
    }

    fn primary(&mut self) -> Res<Expr> {
        let span = self.span();
        match self.next().tok {
            Tok::Num(v, is_int) => Ok(Expr { kind: ExprKind::Num(v, is_int), span }),
            Tok::Str(s) => Ok(Expr { kind: ExprKind::Str(s), span }),
            Tok::LParen => {
                let e = self.expr()?;
                self.expect(&Tok::RParen, "`)`")?;
                Ok(e)
            }
            Tok::LBracket => {
                let mut items = Vec::new();
                if !self.eat(&Tok::RBracket) {
                    loop {
                        items.push(self.expr()?);
                        if self.eat(&Tok::RBracket) {
                            break;
                        }
                        self.expect(&Tok::Comma, "`,` or `]`")?;
                    }
                }
                Ok(Expr { kind: ExprKind::VecLit(items), span })
            }
            Tok::Ident(name) => {
                if self.peek() != &Tok::LParen {
                    return Ok(Expr { kind: ExprKind::Var(name), span });
                }
                self.next();
                let mut args = Vec::new();
                if !self.eat(&Tok::RParen) {
                    loop {
                        let arg_name = match (self.peek().clone(), self.peek_at(1)) {
                            (Tok::Ident(n), Tok::Assign) => {
                                self.next();
                                self.next();
                                Some(n)
                            }
                            _ => None,
                        };
                        let value = self.expr()?;
                        args.push(Arg { name: arg_name, value });
                        if self.eat(&Tok::RParen) {
                            break;
                        }
                        self.expect(&Tok::Comma, "`,` or `)`")?;
                    }
                }
                Ok(Expr { kind: ExprKind::Call(name, args), span })
            }
            t => err(span, format!("expected an expression, found {}", describe(&t))),
        }
    }
}

pub fn describe(t: &Tok) -> String {
    match t {
        Tok::Ident(s) => format!("`{s}`"),
        Tok::Num(v, _) => format!("number {v}"),
        Tok::Str(s) => format!("string \"{s}\""),
        Tok::Newline => "end of line".into(),
        Tok::Eof => "end of file".into(),
        other => format!("`{}`", match other {
            Tok::LParen => "(",
            Tok::RParen => ")",
            Tok::LBracket => "[",
            Tok::RBracket => "]",
            Tok::LBrace => "{",
            Tok::RBrace => "}",
            Tok::Comma => ",",
            Tok::Colon => ":",
            Tok::Assign => "=",
            Tok::Tilde => "~",
            Tok::Plus => "+",
            Tok::Minus => "-",
            Tok::Star => "*",
            Tok::Slash => "/",
            Tok::DotStar => ".*",
            Tok::DotSlash => "./",
            Tok::Caret => "^",
            Tok::Quote => "'",
            Tok::Arrow => "->",
            _ => "?",
        }),
    }
}
