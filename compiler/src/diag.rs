//! Compile errors with a source excerpt and an optional hint.

use crate::ast::Span;

#[derive(Debug, Clone)]
pub struct Diag {
    pub span: Span,
    pub msg: String,
    pub help: Option<String>,
}

pub type Res<T> = Result<T, Diag>;

pub fn err<T>(span: Span, msg: impl Into<String>) -> Res<T> {
    Err(Diag { span, msg: msg.into(), help: None })
}

pub fn err_help<T>(span: Span, msg: impl Into<String>, help: impl Into<String>) -> Res<T> {
    Err(Diag { span, msg: msg.into(), help: Some(help.into()) })
}

impl Diag {
    pub fn render(&self, path: &str, src: &str) -> String {
        let mut out = format!("error: {}\n  --> {}:{}:{}\n", self.msg, path, self.span.line, self.span.col);
        if let Some(line) = src.lines().nth(self.span.line.saturating_sub(1) as usize) {
            let n = self.span.line.to_string();
            let pad = " ".repeat(n.len());
            out += &format!("{pad} |\n{n} | {line}\n{pad} | {}^\n", " ".repeat(self.span.col.saturating_sub(1) as usize));
        }
        if let Some(h) = &self.help {
            out += &format!("help: {h}\n");
        }
        out
    }
}
