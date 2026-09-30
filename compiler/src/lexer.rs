//! Tokeniser. Newlines end statements, except inside ( ) and [ ].

use crate::ast::Span;
use crate::diag::{err, Res};

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String),
    Num(f64, bool),
    Str(String),
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Assign,
    Tilde,
    Plus,
    Minus,
    Star,
    Slash,
    DotStar,
    DotSlash,
    Caret,
    Quote,
    Arrow,
    Newline,
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

pub fn lex(src: &str) -> Res<Vec<Token>> {
    let chars: Vec<char> = src.chars().collect();
    let mut toks: Vec<Token> = Vec::new();
    let (mut i, mut line, mut col) = (0usize, 1u32, 1u32);
    let mut depth = 0i32; // nesting of ( and [
    while i < chars.len() {
        let c = chars[i];
        let span = Span { line, col };
        let bump = |i: &mut usize, col: &mut u32, n: usize| {
            *i += n;
            *col += n as u32;
        };
        if c == '\n' {
            if depth == 0 {
                toks.push(Token { tok: Tok::Newline, span });
            }
            i += 1;
            line += 1;
            col = 1;
            continue;
        }
        if c.is_whitespace() {
            bump(&mut i, &mut col, 1);
            continue;
        }
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let start = i;
            let mut is_int = true;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i < chars.len() && chars[i] == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()) {
                is_int = false;
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            } else if i < chars.len() && chars[i] == '.' && !chars.get(i + 1).is_some_and(|d| *d == '*' || *d == '/') {
                // "1." is a real literal
                is_int = false;
                i += 1;
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let save = i;
                i += 1;
                if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                    i += 1;
                }
                if i < chars.len() && chars[i].is_ascii_digit() {
                    is_int = false;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                } else {
                    i = save;
                }
            }
            let text: String = chars[start..i].iter().collect();
            let v: f64 = match text.parse() {
                Ok(v) => v,
                Err(_) => return err(span, format!("bad number '{text}'")),
            };
            col += (i - start) as u32;
            toks.push(Token { tok: Tok::Num(v, is_int), span });
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            col += (i - start) as u32;
            toks.push(Token { tok: Tok::Ident(chars[start..i].iter().collect()), span });
            continue;
        }
        if c == '"' {
            let start = i + 1;
            i += 1;
            while i < chars.len() && chars[i] != '"' && chars[i] != '\n' {
                i += 1;
            }
            if i >= chars.len() || chars[i] != '"' {
                return err(span, "unterminated string");
            }
            let s: String = chars[start..i].iter().collect();
            col += (i + 1 - (start - 1)) as u32;
            i += 1;
            toks.push(Token { tok: Tok::Str(s), span });
            continue;
        }
        let two = |a: char| chars.get(i + 1) == Some(&a);
        let (tok, n) = match c {
            '(' => {
                depth += 1;
                (Tok::LParen, 1)
            }
            ')' => {
                depth -= 1;
                (Tok::RParen, 1)
            }
            '[' => {
                depth += 1;
                (Tok::LBracket, 1)
            }
            ']' => {
                depth -= 1;
                (Tok::RBracket, 1)
            }
            '{' => (Tok::LBrace, 1),
            '}' => (Tok::RBrace, 1),
            ',' => (Tok::Comma, 1),
            ':' => (Tok::Colon, 1),
            ';' => (Tok::Newline, 1),
            '=' => (Tok::Assign, 1),
            '~' => (Tok::Tilde, 1),
            '+' => (Tok::Plus, 1),
            '-' if two('>') => (Tok::Arrow, 2),
            '-' => (Tok::Minus, 1),
            '*' => (Tok::Star, 1),
            '/' => (Tok::Slash, 1),
            '^' => (Tok::Caret, 1),
            '\'' => (Tok::Quote, 1),
            '.' if two('*') => (Tok::DotStar, 2),
            '.' if two('/') => (Tok::DotSlash, 2),
            _ => return err(span, format!("unexpected character '{c}'")),
        };
        if depth < 0 {
            return err(span, "unbalanced closing bracket");
        }
        toks.push(Token { tok, span });
        bump(&mut i, &mut col, n);
    }
    toks.push(Token { tok: Tok::Newline, span: Span { line, col } });
    toks.push(Token { tok: Tok::Eof, span: Span { line, col } });
    Ok(toks)
}
