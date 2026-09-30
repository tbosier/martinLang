//! A small textual LLVM IR builder.
//!
//! Local variables live in `alloca` slots and are promoted to registers by
//! LLVM's mem2reg pass, which keeps loop emission simple. Reductions carry the
//! `reassoc` flag so LLVM may vectorise them; that is the one place where Mint
//! deliberately departs from strict IEEE evaluation order (see
//! docs/architecture.md). `--strict-fp` turns every fast-math flag off.

use std::collections::{BTreeSet, HashMap};

pub fn fconst(v: f64) -> String {
    format!("0x{:016X}", v.to_bits())
}

#[derive(Default)]
pub struct Module {
    pub decls: BTreeSet<String>,
    pub globals: Vec<String>,
    pub funcs: Vec<String>,
    strings: HashMap<String, String>,
    /// Use Mint's own `exp` (mint_exp_defs) instead of llvm.exp.f64.
    pub inline_exp: bool,
    /// The host has AVX2 (programs are built for the host): table lookups use
    /// its gather instruction directly, since LLVM's generic gather is split
    /// into scalar loads on some CPUs where the instruction is fast.
    pub avx2: bool,
}

/// 2^(j/256) for j = 0..255, each correctly rounded to double (computed with
/// 60-digit decimal arithmetic).
const EXP_TAB: [u64; 256] = [
    0x3FF0000000000000, 0x3FF00B1AFA5ABCBF, 0x3FF0163DA9FB3335, 0x3FF02168143B0281,
    0x3FF02C9A3E778061, 0x3FF037D42E11BBCC, 0x3FF04315E86E7F85, 0x3FF04E5F72F654B1,
    0x3FF059B0D3158574, 0x3FF0650A0E3C1F89, 0x3FF0706B29DDF6DE, 0x3FF07BD42B72A836,
    0x3FF0874518759BC8, 0x3FF092BDF66607E0, 0x3FF09E3ECAC6F383, 0x3FF0A9C79B1F3919,
    0x3FF0B5586CF9890F, 0x3FF0C0F145E46C85, 0x3FF0CC922B7247F7, 0x3FF0D83B23395DEC,
    0x3FF0E3EC32D3D1A2, 0x3FF0EFA55FDFA9C5, 0x3FF0FB66AFFED31B, 0x3FF1073028D7233E,
    0x3FF11301D0125B51, 0x3FF11EDBAB5E2AB6, 0x3FF12ABDC06C31CC, 0x3FF136A814F204AB,
    0x3FF1429AAEA92DE0, 0x3FF14E95934F312E, 0x3FF15A98C8A58E51, 0x3FF166A45471C3C2,
    0x3FF172B83C7D517B, 0x3FF17ED48695BBC0, 0x3FF18AF9388C8DEA, 0x3FF1972658375D2F,
    0x3FF1A35BEB6FCB75, 0x3FF1AF99F8138A1C, 0x3FF1BBE084045CD4, 0x3FF1C82F95281C6B,
    0x3FF1D4873168B9AA, 0x3FF1E0E75EB44027, 0x3FF1ED5022FCD91D, 0x3FF1F9C18438CE4D,
    0x3FF2063B88628CD6, 0x3FF212BE3578A819, 0x3FF21F49917DDC96, 0x3FF22BDDA27912D1,
    0x3FF2387A6E756238, 0x3FF2451FFB82140A, 0x3FF251CE4FB2A63F, 0x3FF25E85711ECE75,
    0x3FF26B4565E27CDD, 0x3FF2780E341DDF29, 0x3FF284DFE1F56381, 0x3FF291BA7591BB70,
    0x3FF29E9DF51FDEE1, 0x3FF2AB8A66D10F13, 0x3FF2B87FD0DAD990, 0x3FF2C57E39771B2F,
    0x3FF2D285A6E4030B, 0x3FF2DF961F641589, 0x3FF2ECAFA93E2F56, 0x3FF2F9D24ABD886B,
    0x3FF306FE0A31B715, 0x3FF31432EDEEB2FD, 0x3FF32170FC4CD831, 0x3FF32EB83BA8EA32,
    0x3FF33C08B26416FF, 0x3FF3496266E3FA2D, 0x3FF356C55F929FF1, 0x3FF36431A2DE883B,
    0x3FF371A7373AA9CB, 0x3FF37F26231E754A, 0x3FF38CAE6D05D866, 0x3FF39A401B7140EF,
    0x3FF3A7DB34E59FF7, 0x3FF3B57FBFEC6CF4, 0x3FF3C32DC313A8E5, 0x3FF3D0E544EDE173,
    0x3FF3DEA64C123422, 0x3FF3EC70DF1C5175, 0x3FF3FA4504AC801C, 0x3FF40822C367A024,
    0x3FF4160A21F72E2A, 0x3FF423FB2709468A, 0x3FF431F5D950A897, 0x3FF43FFA3F84B9D4,
    0x3FF44E086061892D, 0x3FF45C2042A7D232, 0x3FF46A41ED1D0057, 0x3FF4786D668B3237,
    0x3FF486A2B5C13CD0, 0x3FF494E1E192AED2, 0x3FF4A32AF0D7D3DE, 0x3FF4B17DEA6DB7D7,
    0x3FF4BFDAD5362A27, 0x3FF4CE41B817C114, 0x3FF4DCB299FDDD0D, 0x3FF4EB2D81D8ABFF,
    0x3FF4F9B2769D2CA7, 0x3FF508417F4531EE, 0x3FF516DAA2CF6642, 0x3FF5257DE83F4EEF,
    0x3FF5342B569D4F82, 0x3FF542E2F4F6AD27, 0x3FF551A4CA5D920F, 0x3FF56070DDE910D2,
    0x3FF56F4736B527DA, 0x3FF57E27DBE2C4CF, 0x3FF58D12D497C7FD, 0x3FF59C0827FF07CC,
    0x3FF5AB07DD485429, 0x3FF5BA11FBA87A03, 0x3FF5C9268A5946B7, 0x3FF5D84590998B93,
    0x3FF5E76F15AD2148, 0x3FF5F6A320DCEB71, 0x3FF605E1B976DC09, 0x3FF6152AE6CDF6F4,
    0x3FF6247EB03A5585, 0x3FF633DD1D1929FD, 0x3FF6434634CCC320, 0x3FF652B9FEBC8FB7,
    0x3FF6623882552225, 0x3FF671C1C70833F6, 0x3FF68155D44CA973, 0x3FF690F4B19E9538,
    0x3FF6A09E667F3BCD, 0x3FF6B052FA75173E, 0x3FF6C012750BDABF, 0x3FF6CFDCDDD47645,
    0x3FF6DFB23C651A2F, 0x3FF6EF9298593AE5, 0x3FF6FF7DF9519484, 0x3FF70F7466F42E87,
    0x3FF71F75E8EC5F74, 0x3FF72F8286EAD08A, 0x3FF73F9A48A58174, 0x3FF74FBD35D7CBFD,
    0x3FF75FEB564267C9, 0x3FF77024B1AB6E09, 0x3FF780694FDE5D3F, 0x3FF790B938AC1CF6,
    0x3FF7A11473EB0187, 0x3FF7B17B0976CFDB, 0x3FF7C1ED0130C132, 0x3FF7D26A62FF86F0,
    0x3FF7E2F336CF4E62, 0x3FF7F3878491C491, 0x3FF80427543E1A12, 0x3FF814D2ADD106D9,
    0x3FF82589994CCE13, 0x3FF8364C1EB941F7, 0x3FF8471A4623C7AD, 0x3FF857F4179F5B21,
    0x3FF868D99B4492ED, 0x3FF879CAD931A436, 0x3FF88AC7D98A6699, 0x3FF89BD0A478580F,
    0x3FF8ACE5422AA0DB, 0x3FF8BE05BAD61778, 0x3FF8CF3216B5448C, 0x3FF8E06A5E0866D9,
    0x3FF8F1AE99157736, 0x3FF902FED0282C8A, 0x3FF9145B0B91FFC6, 0x3FF925C353AA2FE2,
    0x3FF93737B0CDC5E5, 0x3FF948B82B5F98E5, 0x3FF95A44CBC8520F, 0x3FF96BDD9A7670B3,
    0x3FF97D829FDE4E50, 0x3FF98F33E47A22A2, 0x3FF9A0F170CA07BA, 0x3FF9B2BB4D53FE0D,
    0x3FF9C49182A3F090, 0x3FF9D674194BB8D5, 0x3FF9E86319E32323, 0x3FF9FA5E8D07F29E,
    0x3FFA0C667B5DE565, 0x3FFA1E7AED8EB8BB, 0x3FFA309BEC4A2D33, 0x3FFA42C980460AD8,
    0x3FFA5503B23E255D, 0x3FFA674A8AF46052, 0x3FFA799E1330B358, 0x3FFA8BFE53C12E59,
    0x3FFA9E6B5579FDBF, 0x3FFAB0E521356EBA, 0x3FFAC36BBFD3F37A, 0x3FFAD5FF3A3C2774,
    0x3FFAE89F995AD3AD, 0x3FFAFB4CE622F2FF, 0x3FFB0E07298DB666, 0x3FFB20CE6C9A8952,
    0x3FFB33A2B84F15FB, 0x3FFB468415B749B1, 0x3FFB59728DE5593A, 0x3FFB6C6E29F1C52A,
    0x3FFB7F76F2FB5E47, 0x3FFB928CF22749E4, 0x3FFBA5B030A1064A, 0x3FFBB8E0B79A6F1F,
    0x3FFBCC1E904BC1D2, 0x3FFBDF69C3F3A207, 0x3FFBF2C25BD71E09, 0x3FFC06286141B33D,
    0x3FFC199BDD85529C, 0x3FFC2D1CD9FA652C, 0x3FFC40AB5FFFD07A, 0x3FFC544778FAFB22,
    0x3FFC67F12E57D14B, 0x3FFC7BA88988C933, 0x3FFC8F6D9406E7B5, 0x3FFCA3405751C4DB,
    0x3FFCB720DCEF9069, 0x3FFCCB0F2E6D1675, 0x3FFCDF0B555DC3FA, 0x3FFCF3155B5BAB74,
    0x3FFD072D4A07897C, 0x3FFD1B532B08C968, 0x3FFD2F87080D89F2, 0x3FFD43C8EACAA1D6,
    0x3FFD5818DCFBA487, 0x3FFD6C76E862E6D3, 0x3FFD80E316C98398, 0x3FFD955D71FF6075,
    0x3FFDA9E603DB3285, 0x3FFDBE7CD63A8315, 0x3FFDD321F301B460, 0x3FFDE7D5641C0658,
    0x3FFDFC97337B9B5F, 0x3FFE11676B197D17, 0x3FFE264614F5A129, 0x3FFE3B333B16EE12,
    0x3FFE502EE78B3FF6, 0x3FFE653924676D76, 0x3FFE7A51FBC74C83, 0x3FFE8F7977CDB740,
    0x3FFEA4AFA2A490DA, 0x3FFEB9F4867CCA6E, 0x3FFECF482D8E67F1, 0x3FFEE4AAA2188510,
    0x3FFEFA1BEE615A27, 0x3FFF0F9C1CB6412A, 0x3FFF252B376BBA97, 0x3FFF3AC948DD7274,
    0x3FFF50765B6E4540, 0x3FFF6632798844F8, 0x3FFF7BFDAD9CBE14, 0x3FFF91D802243C89,
    0x3FFFA7C1819E90D8, 0x3FFFBDBA3692D514, 0x3FFFD3C22B8F71F1, 0x3FFFE9D96B2A23D9,
];

/// Mint's exp, emitted into every module that uses it (scalar and vector
/// forms). It is plain arithmetic, so LLVM inlines and vectorises it with no
/// call: a call to a vector math library forces every vector register to be
/// spilled around it.
///
/// x = k ln2 + r with |r| <= ln2/2 (k from the round-to-nearest shift trick,
/// ln2 split in two for an exact reduction); e^r is its degree-13 Taylor
/// polynomial evaluated by Estrin's scheme (depth 4 instead of 13); 2^k is
/// applied in two exact steps so that k = 1024 does not overflow and
/// subnormal results round once. Worst error found against a long double
/// reference over 2e7 inputs: 2 ulp. NaN, +-inf, overflow and underflow
/// match libm.
/// The IR types and helpers for one width: (double, i64, i1 types, fma name,
/// suffix), double and i64 constant formatters.
fn exp_types(lanes: u32) -> (String, String, String, String, String) {
    if lanes == 1 {
        ("double".into(), "i64".into(), "i1".into(), "llvm.fma.f64".into(), String::new())
    } else {
        let l = lanes;
        (format!("<{l} x double>"), format!("<{l} x i64>"), format!("<{l} x i1>"), format!("llvm.fma.v{l}f64"), format!("_v{l}"))
    }
}

/// Every definition and declaration `mint_exp{_vL}` needs.
fn mint_exp_defs(lanes: u32, avx2: bool) -> Vec<String> {
    let (d, it, b, fma, sfx) = exp_types(lanes);
    let mut out = vec![
        format!("declare {d} @{fma}({d}, {d}, {d})"),
        mint_exp_table(),
        mint_exp_full(lanes),
        mint_exp_fast(lanes, avx2),
    ];
    let l = lanes;
    if lanes == 1 {
        out.push("declare double @llvm.fabs.f64(double)".into());
    } else {
        out.push(format!("declare {d} @llvm.fabs.v{l}f64({d})"));
        out.push(format!("declare {d} @llvm.masked.gather.v{l}f64.v{l}p0(<{l} x ptr>, i32, {b}, {d})"));
        if avx2 && lanes == 4 {
            out.push("declare <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double>, ptr, <4 x i64>, <4 x double>, i8)".into());
        }
        out.push(format!("declare i1 @llvm.vector.reduce.or.v{l}i1({b})"));
    }
    out.push("declare i1 @llvm.expect.i1(i1, i1)".into());
    let _ = (it, sfx);
    out
}

fn mint_exp_table() -> String {
    let vals: Vec<String> = EXP_TAB.iter().map(|v| format!("double 0x{v:016X}")).collect();
    format!("@mint_exp_tab = internal unnamed_addr constant [256 x double] [{}], align 64", vals.join(", "))
}

/// The fast exp: x = (256 k' + j) ln2/256 + r with |r| <= ln2/512; e^x =
/// 2^k' * T[j] * e^r with T[j] = 2^(j/256) from a table (one gather), and
/// e^r - 1 = q a degree-4 polynomial (3 FMAs), formed as T + T*q for accuracy;
/// 2^k' is added to the exponent bits. Inputs with |x| > 708 (and NaN) take
/// the full-range version below, behind a branch marked unlikely.
fn mint_exp_fast(lanes: u32, avx2: bool) -> String {
    let (d, it, b, fma, sfx) = exp_types(lanes);
    let l = lanes;
    let c = |x: f64| if lanes == 1 { fconst(x) } else { format!("splat (double {})", fconst(x)) };
    let i = |n: i64| if lanes == 1 { n.to_string() } else { format!("splat (i64 {n})") };
    let fm = |r: &str, a: &str, x: &str, y: &str| format!("  %{r} = call {d} @{fma}({d} {a}, {d} {x}, {d} {y})\n");
    let shift = 6755399441055744.0; // 0x1.8p52
    let mut s = format!("define internal {d} @mint_exp{sfx}({d} %x) alwaysinline {{\nentry:\n");
    s += &fm("kd0", "%x", &c(std::f64::consts::LOG2_E * 256.0), &c(shift));
    s += &format!("  %kb = bitcast {d} %kd0 to {it}\n  %kd = fsub {d} %kd0, {}\n", c(shift));
    s += &fm("r0", "%kd", &c(-f64::from_bits(0x3FE62E42FEFA39EF) / 256.0), "%x");
    s += &fm("r", "%kd", &c(-f64::from_bits(0x3C7ABC9E3B39803F) / 256.0), "%r0");
    s += &format!("  %r2 = fmul {d} %r, %r\n");
    s += &fm("qa", "%r", &c(1.0 / 6.0), &c(0.5));
    s += &fm("qb", "%r2", &c(1.0 / 24.0), "%qa");
    s += &fm("q", "%r2", "%qb", "%r");
    s += &format!("  %j = and {it} %kb, {}\n", i(255));
    if lanes == 1 {
        s += "  %tp = getelementptr inbounds double, ptr @mint_exp_tab, i64 %j\n  %t = load double, ptr %tp, align 8\n";
    } else if avx2 && lanes == 4 {
        // all-ones mask (sign bits set): load every lane
        s += "  %t = call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> poison, ptr @mint_exp_tab, <4 x i64> %j, <4 x double> splat (double 0xFFFFFFFFFFFFFFFF), i8 8)\n";
    } else {
        s += &format!("  %tp = getelementptr inbounds double, ptr @mint_exp_tab, {it} %j\n");
        s += &format!("  %t = call {d} @llvm.masked.gather.v{l}f64.v{l}p0(<{l} x ptr> %tp, i32 8, {b} splat (i1 true), {d} poison)\n");
    }
    s += &fm("m", "%t", "%q", "%t");
    s += &format!("  %hi = and {it} %kb, {}\n  %sc = shl {it} %hi, {}\n", i(-256), i(44));
    s += &format!("  %mb = bitcast {d} %m to {it}\n  %fb = add {it} %mb, %sc\n  %fast = bitcast {it} %fb to {d}\n");
    if lanes == 1 {
        s += "  %ax = call double @llvm.fabs.f64(double %x)\n";
    } else {
        s += &format!("  %ax = call {d} @llvm.fabs.v{l}f64({d} %x)\n");
    }
    s += &format!("  %out = fcmp ugt {d} %ax, {}\n", c(708.0));
    if lanes == 1 {
        s += "  %any = call i1 @llvm.expect.i1(i1 %out, i1 false)\n";
    } else {
        s += &format!("  %any0 = call i1 @llvm.vector.reduce.or.v{l}i1({b} %out)\n  %any = call i1 @llvm.expect.i1(i1 %any0, i1 false)\n");
    }
    s += "  br i1 %any, label %slow, label %done\nslow:\n";
    s += &format!("  %full = call {d} @mint_exp_full{sfx}({d} %x)\n  %sel = select {b} %out, {d} %full, {d} %fast\n  br label %done\n");
    s += &format!("done:\n  %res = phi {d} [ %fast, %entry ], [ %sel, %slow ]\n  ret {d} %res\n}}");
    s
}

/// The full-range exp: correct for every input (see the doc comment above).
fn mint_exp_full(lanes: u32) -> String {
    let (d, it, b, fma, sfx) = exp_types(lanes);
    let name = format!("mint_exp_full{sfx}");
    let c = |x: f64| if lanes == 1 { fconst(x) } else { format!("splat (double {})", fconst(x)) };
    let i = |n: i64| if lanes == 1 { n.to_string() } else { format!("splat (i64 {n})") };
    let mut coef = vec![1.0f64];
    let mut fact = 1.0f64;
    for k in 1..=13 {
        fact *= k as f64;
        coef.push(1.0 / fact);
    }
    let fm = |r: &str, a: &str, x: &str, y: &str| format!("  %{r} = call {d} @{fma}({d} {a}, {d} {x}, {d} {y})\n");
    let mut s = format!("define internal {d} @{name}({d} %x) noinline {{\n");
    s += &fm("kd0", "%x", &c(std::f64::consts::LOG2_E), &c(6755399441055744.0)); // 0x1.8p52
    s += &format!("  %kb = bitcast {d} %kd0 to {it}\n");
    s += &format!("  %kd = fsub {d} %kd0, {}\n", c(6755399441055744.0));
    s += &fm("r0", "%kd", &c(-f64::from_bits(0x3FE62E42FEFA39EF)), "%x");
    s += &fm("r", "%kd", &c(-f64::from_bits(0x3C7ABC9E3B39803F)), "%r0");
    s += &format!("  %r2 = fmul {d} %r, %r\n  %r4 = fmul {d} %r2, %r2\n  %r8 = fmul {d} %r4, %r4\n");
    for q in 0..7 {
        s += &fm(&format!("q{q}"), "%r", &c(coef[2 * q + 1]), &c(coef[2 * q]));
    }
    s += &fm("s0", "%q1", "%r2", "%q0");
    s += &fm("s1", "%q3", "%r2", "%q2");
    s += &fm("s2", "%q5", "%r2", "%q4");
    s += &fm("t0", "%s1", "%r4", "%s0");
    s += &fm("t1", "%q6", "%r4", "%s2");
    s += &fm("p", "%t1", "%r8", "%t0");
    s += &format!("  %neg = fcmp olt {d} %x, {}\n", c(0.0));
    s += &format!("  %bias = select {b} %neg, {it} {}, {it} {}\n", i(1087), i(959));
    s += &format!("  %e0 = add {it} %kb, %bias\n  %e1 = shl {it} %e0, {}\n", i(52));
    s += &format!("  %sc = bitcast {it} %e1 to {d}\n  %m0 = fmul {d} %p, %sc\n");
    s += &format!("  %s = select {b} %neg, {d} {}, {d} {}\n", c(f64::from_bits(0x3BF0000000000000)), c(f64::from_bits(0x43F0000000000000)));
    s += &format!("  %m1 = fmul {d} %m0, %s\n");
    s += &format!("  %big = fcmp ogt {d} %x, {}\n", c(f64::from_bits(0x40862E42FEFA39EF)));
    s += &format!("  %m2 = select {b} %big, {d} {}, {d} %m1\n", c(f64::INFINITY));
    s += &format!("  %tiny = fcmp olt {d} %x, {}\n", c(f64::from_bits(0xC0874910D52D3051)));
    s += &format!("  %m3 = select {b} %tiny, {d} {}, {d} %m2\n  ret {d} %m3\n}}", c(0.0));
    s
}

impl Module {
    pub fn string(&mut self, s: &str) -> String {
        if let Some(g) = self.strings.get(s) {
            return g.clone();
        }
        let name = format!("@.str.{}", self.strings.len());
        let mut bytes = String::new();
        for b in s.bytes().chain(std::iter::once(0u8)) {
            if b.is_ascii_alphanumeric() || b == b' ' || (b.is_ascii_punctuation() && b != b'"' && b != b'\\') {
                bytes.push(b as char);
            } else {
                bytes += &format!("\\{:02X}", b);
            }
        }
        self.globals.push(format!("{name} = private unnamed_addr constant [{} x i8] c\"{bytes}\"", s.len() + 1));
        self.strings.insert(s.to_string(), name.clone());
        name
    }

    pub fn declare(&mut self, line: &str) {
        self.decls.insert(line.to_string());
    }

    pub fn finish(self) -> String {
        let mut out = String::from("; generated by mintc\n\n");
        for d in &self.decls {
            out += d;
            out.push('\n');
        }
        out.push('\n');
        for g in &self.globals {
            out += g;
            out.push('\n');
        }
        out.push('\n');
        for f in &self.funcs {
            out += f;
            out.push('\n');
        }
        out
    }
}

pub struct Fb {
    lines: Vec<String>,
    allocas: Vec<String>,
    n: usize,
    hoist: Option<(usize, Vec<String>)>,
    pub frees: Vec<String>,
    /// fast-math flags for ordinary arithmetic and for reductions
    pub flags: &'static str,
    pub rflags: &'static str,
    pub strict: bool,
    /// 1 for scalar code; otherwise every floating-point value this builder
    /// makes is a vector of `lanes` doubles (constants are splatted).
    pub lanes: u32,
    /// Use Mint's exp in scalar code too (in loops LLVM will not vectorise).
    pub scalar_inline_exp: bool,
}

impl Fb {
    pub fn new(strict_fp: bool) -> Self {
        Fb {
            lines: vec![],
            allocas: vec![],
            n: 0,
            hoist: None,
            frees: vec![],
            flags: if strict_fp { "" } else { "contract " },
            rflags: if strict_fp { "" } else { "reassoc contract " },
            strict: strict_fp,
            lanes: 1,
            scalar_inline_exp: false,
        }
    }

    /// The floating-point value type: `double` or `<L x double>`.
    pub fn ty(&self) -> String {
        if self.lanes == 1 {
            "double".into()
        } else {
            format!("<{} x double>", self.lanes)
        }
    }

    /// An operand in the current type: literal constants are splatted in
    /// vector mode (registers are assumed to be of the current type already).
    pub fn opnd(&self, x: &str) -> String {
        let lit = x.starts_with("0x") || x.parse::<f64>().is_ok();
        if self.lanes > 1 && lit {
            format!("splat (double {x})")
        } else {
            x.to_string()
        }
    }

    /// Broadcasts a scalar double register or constant to the current type.
    pub fn splat(&mut self, x: &str) -> String {
        if self.lanes == 1 {
            return x.to_string();
        }
        if x.starts_with("0x") || x.parse::<f64>().is_ok() {
            return x.to_string(); // opnd() splats literals where they are used
        }
        let l = self.lanes;
        let a = self.reg();
        self.emit(format!("{a} = insertelement <{l} x double> poison, double {x}, i64 0"));
        let r = self.reg();
        self.emit(format!("{r} = shufflevector <{l} x double> {a}, <{l} x double> poison, <{l} x i32> zeroinitializer"));
        r
    }

    /// A scalar load, whatever the current type.
    pub fn load_scalar(&mut self, p: &str, idx: &str) -> String {
        let a = self.gep(p, idx);
        let r = self.reg();
        self.emit(format!("{r} = load double, ptr {a}"));
        r
    }

    /// Horizontal sum of a value of the current type, as a scalar double.
    pub fn hsum(&mut self, m: &mut Module, v: &str) -> String {
        if self.lanes == 1 {
            return v.to_string();
        }
        let l = self.lanes;
        m.declare(&format!("declare double @llvm.vector.reduce.fadd.v{l}f64(double, <{l} x double>)"));
        let r = self.reg();
        let fl = self.rflags;
        self.emit(format!("{r} = call {fl}double @llvm.vector.reduce.fadd.v{l}f64(double -0.0, <{l} x double> {v})"));
        r
    }

    pub fn reg(&mut self) -> String {
        self.n += 1;
        format!("%r{}", self.n)
    }

    pub fn label(&mut self, base: &str) -> String {
        self.n += 1;
        format!("{base}{}", self.n)
    }

    pub fn emit(&mut self, s: impl AsRef<str>) {
        self.lines.push(format!("  {}", s.as_ref()));
    }

    pub fn start_block(&mut self, l: &str) {
        self.lines.push(format!("{l}:"));
    }

    pub fn br(&mut self, l: &str) {
        self.emit(format!("br label %{l}"));
    }

    pub fn alloca(&mut self, ty: &str) -> String {
        let r = self.reg();
        self.allocas.push(format!("  {r} = alloca {ty}"));
        r
    }

    /// Emits instructions at the start of the current top-level statement
    /// instead of here, so buffers used inside loops are allocated once.
    pub fn begin_hoist(&mut self) {
        self.hoist = Some((self.lines.len(), vec![]));
    }

    pub fn end_hoist(&mut self) {
        if let Some((at, h)) = self.hoist.take() {
            self.lines.splice(at..at, h);
        }
    }

    pub fn hoisted<T>(&mut self, f: impl FnOnce(&mut Fb) -> T) -> T {
        if self.hoist.is_none() {
            return f(self);
        }
        let saved = std::mem::take(&mut self.lines);
        let r = f(self);
        let produced = std::mem::replace(&mut self.lines, saved);
        self.hoist.as_mut().unwrap().1.extend(produced);
        r
    }

    // ---- arithmetic helpers

    pub fn fop(&mut self, op: &str, a: &str, b: &str) -> String {
        let r = self.reg();
        let flags = self.flags;
        let (t, a, b) = (self.ty(), self.opnd(a), self.opnd(b));
        self.emit(format!("{r} = {op} {flags}{t} {a}, {b}"));
        r
    }
    pub fn fadd(&mut self, a: &str, b: &str) -> String {
        self.fop("fadd", a, b)
    }
    pub fn fsub(&mut self, a: &str, b: &str) -> String {
        self.fop("fsub", a, b)
    }
    pub fn fmul(&mut self, a: &str, b: &str) -> String {
        self.fop("fmul", a, b)
    }
    pub fn fdiv(&mut self, a: &str, b: &str) -> String {
        self.fop("fdiv", a, b)
    }
    pub fn fneg(&mut self, a: &str) -> String {
        let r = self.reg();
        let (t, a) = (self.ty(), self.opnd(a));
        self.emit(format!("{r} = fneg {t} {a}"));
        r
    }
    pub fn iop(&mut self, op: &str, a: &str, b: &str) -> String {
        let r = self.reg();
        self.emit(format!("{r} = {op} i64 {a}, {b}"));
        r
    }
    pub fn imul(&mut self, a: &str, b: &str) -> String {
        self.iop("mul nsw", a, b)
    }
    pub fn iadd(&mut self, a: &str, b: &str) -> String {
        self.iop("add nsw", a, b)
    }
    pub fn sitofp(&mut self, a: &str) -> String {
        let r = self.reg();
        self.emit(format!("{r} = sitofp i64 {a} to double"));
        r
    }
    /// The name of `name` (an llvm.*.f64 intrinsic) for the current type.
    fn vname(&self, name: &str) -> String {
        if self.lanes == 1 {
            return name.to_string();
        }
        match name.strip_suffix(".f64") {
            Some(base) if name.starts_with("llvm.") => format!("{base}.v{}f64", self.lanes),
            _ => panic!("no vector form of {name}"),
        }
    }
    pub fn intrinsic1(&mut self, m: &mut Module, name: &str, a: &str) -> String {
        let (t, a) = (self.ty(), self.opnd(a));
        // Mint's own exp only where Mint emits the vector code itself: in
        // scalar code LLVM's loop vectoriser maps llvm.exp to the vector math
        // library, and the scalar form's out-of-range branch would stop it.
        if name == "llvm.exp.f64" && m.inline_exp && (self.lanes > 1 || self.scalar_inline_exp) {
            for d in mint_exp_defs(self.lanes, m.avx2) {
                m.declare(&d);
            }
            let f = if self.lanes == 1 { "mint_exp".to_string() } else { format!("mint_exp_v{}", self.lanes) };
            let r = self.reg();
            self.emit(format!("{r} = call {t} @{f}({t} {a})"));
            return r;
        }
        let name = self.vname(name);
        m.declare(&format!("declare {t} @{name}({t})"));
        let r = self.reg();
        self.emit(format!("{r} = call {t} @{name}({t} {a})"));
        r
    }
    pub fn intrinsic2(&mut self, m: &mut Module, name: &str, a: &str, b: &str) -> String {
        let (t, a, b) = (self.ty(), self.opnd(a), self.opnd(b));
        let name = self.vname(name);
        m.declare(&format!("declare {t} @{name}({t}, {t})"));
        let r = self.reg();
        self.emit(format!("{r} = call {t} @{name}({t} {a}, {t} {b})"));
        r
    }

    // ---- memory helpers

    pub fn gep(&mut self, p: &str, idx: &str) -> String {
        let r = self.reg();
        self.emit(format!("{r} = getelementptr inbounds double, ptr {p}, i64 {idx}"));
        r
    }
    pub fn load(&mut self, p: &str, idx: &str) -> String {
        let a = self.gep(p, idx);
        let r = self.reg();
        let t = self.ty();
        self.emit(format!("{r} = load {t}, ptr {a}, align 8"));
        r
    }
    pub fn store(&mut self, v: &str, p: &str, idx: &str) {
        let a = self.gep(p, idx);
        let (t, v) = (self.ty(), self.opnd(v));
        self.emit(format!("store {t} {v}, ptr {a}, align 8"));
    }
    /// p[idx] += v
    pub fn add_to(&mut self, p: &str, idx: &str, v: &str) {
        let a = self.gep(p, idx);
        let old = self.reg();
        let t = self.ty();
        self.emit(format!("{old} = load {t}, ptr {a}, align 8"));
        let s = self.fadd(&old, v);
        self.emit(format!("store {t} {s}, ptr {a}, align 8"));
    }
    pub fn load_ptr(&mut self, g: &str) -> String {
        let r = self.reg();
        self.emit(format!("{r} = load ptr, ptr {g}"));
        r
    }
    pub fn load_i64(&mut self, g: &str) -> String {
        let r = self.reg();
        self.emit(format!("{r} = load i64, ptr {g}"));
        r
    }
    pub fn load_f64(&mut self, g: &str) -> String {
        let r = self.reg();
        self.emit(format!("{r} = load double, ptr {g}"));
        r
    }
    pub fn memcpy(&mut self, m: &mut Module, dst: &str, src: &str, n_doubles: &str) {
        m.declare("declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)");
        let bytes = self.imul(n_doubles, "8");
        self.emit(format!("call void @llvm.memcpy.p0.p0.i64(ptr {dst}, ptr {src}, i64 {bytes}, i1 false)"));
    }
    pub fn memzero(&mut self, m: &mut Module, dst: &str, n_doubles: &str) {
        m.declare("declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)");
        let bytes = self.imul(n_doubles, "8");
        self.emit(format!("call void @llvm.memset.p0.i64(ptr {dst}, i8 0, i64 {bytes}, i1 false)"));
    }

    // ---- accumulators (alloca-backed, promoted to phis by mem2reg)

    pub fn acc_new(&mut self, init: &str) -> String {
        let t = self.ty();
        let a = self.alloca(&t);
        let init = self.opnd(init);
        self.emit(format!("store {t} {init}, ptr {a}"));
        a
    }
    /// Reduction update: carries `reassoc` so the loop can be vectorised.
    pub fn acc_add(&mut self, acc: &str, v: &str) {
        let t = self.ty();
        let old = self.reg();
        self.emit(format!("{old} = load {t}, ptr {acc}"));
        let r = self.reg();
        let fl = self.rflags;
        let v = self.opnd(v);
        self.emit(format!("{r} = fadd {fl}{t} {old}, {v}"));
        self.emit(format!("store {t} {r}, ptr {acc}"));
    }
    pub fn acc_get(&mut self, acc: &str) -> String {
        let t = self.ty();
        let r = self.reg();
        self.emit(format!("{r} = load {t}, ptr {acc}"));
        r
    }

    // ---- control flow

    pub fn finish(self, header: &str, epilogue: &[String]) -> String {
        let mut out = format!("{header} {{\nentry:\n");
        for a in &self.allocas {
            out += a;
            out.push('\n');
        }
        for l in &self.lines {
            out += l;
            out.push('\n');
        }
        for l in epilogue {
            out += "  ";
            out += l;
            out.push('\n');
        }
        out += "}\n";
        out
    }
}

pub trait HasFb {
    fn fb(&mut self) -> &mut Fb;
}

/// for i in lo..hi { body(i) }
pub fn for_range<C: HasFb>(cx: &mut C, lo: &str, hi: &str, body: impl FnOnce(&mut C, &str)) {
    let f = cx.fb();
    let iv = f.alloca("i64");
    f.emit(format!("store i64 {lo}, ptr {iv}"));
    let hdr = f.label("loop");
    let bb = f.label("body");
    let exit = f.label("done");
    f.br(&hdr);
    f.start_block(&hdr);
    let i = f.reg();
    f.emit(format!("{i} = load i64, ptr {iv}"));
    let c = f.reg();
    f.emit(format!("{c} = icmp slt i64 {i}, {hi}"));
    f.emit(format!("br i1 {c}, label %{bb}, label %{exit}"));
    f.start_block(&bb);
    body(cx, &i);
    let f = cx.fb();
    let i2 = f.reg();
    f.emit(format!("{i2} = add nsw i64 {i}, 1"));
    f.emit(format!("store i64 {i2}, ptr {iv}"));
    f.br(&hdr);
    f.start_block(&exit);
}

/// For i in 0..n: store(i, sum_k M[i*c + k] * v[k]).
/// Rows are processed four at a time so each load of v feeds four FMAs;
/// LLVM vectorises the four independent reductions.
pub fn rows_dot_blocked<C: HasFb>(cx: &mut C, m: &str, v: &str, c: &str, n: &str, store: &dyn Fn(&mut C, &str, &str)) {
    let nb = cx.fb().iop("sdiv", n, "4");
    for_range(cx, "0", &nb, |cx, b| {
        let f = cx.fb();
        let i0 = f.imul(b, "4");
        let mut ids = Vec::new();
        let mut rows = Vec::new();
        for l in 0..4 {
            let i = f.iadd(&i0, &l.to_string());
            rows.push(f.imul(&i, c));
            ids.push(i);
        }
        let accs: Vec<String> = (0..4).map(|_| f.acc_new(&fconst(0.0))).collect();
        for_range(cx, "0", c, |cx, k| {
            let f = cx.fb();
            let x = f.load(v, k);
            for l in 0..4 {
                let idx = f.iadd(&rows[l], k);
                let a = f.load(m, &idx);
                let t = f.fmul(&a, &x);
                f.acc_add(&accs[l], &t);
            }
        });
        for l in 0..4 {
            let s = cx.fb().acc_get(&accs[l]);
            store(cx, &ids[l], &s);
        }
    });
    let done = cx.fb().imul(&nb, "4");
    for_range(cx, &done, n, |cx, i| {
        let f = cx.fb();
        let row = f.imul(i, c);
        let acc = f.acc_new(&fconst(0.0));
        for_range(cx, "0", c, |cx, k| {
            let f = cx.fb();
            let idx = f.iadd(&row, k);
            let a = f.load(m, &idx);
            let x = f.load(v, k);
            let t = f.fmul(&a, &x);
            f.acc_add(&acc, &t);
        });
        let s = cx.fb().acc_get(&acc);
        store(cx, i, &s);
    });
}

/// For k in 0..c: g[k] += sum_i coef(i) * M[i*c + k], four rows per pass over g.
pub fn rows_axpy_blocked<C: HasFb>(cx: &mut C, m: &str, c: &str, n: &str, coef: &dyn Fn(&mut C, &str) -> String, g: &str) {
    let nb = cx.fb().iop("sdiv", n, "4");
    for_range(cx, "0", &nb, |cx, b| {
        let i0 = cx.fb().imul(b, "4");
        let mut rows = Vec::new();
        let mut cs = Vec::new();
        for l in 0..4 {
            let i = cx.fb().iadd(&i0, &l.to_string());
            cs.push(coef(cx, &i));
            rows.push(cx.fb().imul(&i, c));
        }
        for_range(cx, "0", c, |cx, k| {
            let f = cx.fb();
            let mut sum: Option<String> = None;
            for l in 0..4 {
                let idx = f.iadd(&rows[l], k);
                let a = f.load(m, &idx);
                let t = f.fmul(&cs[l], &a);
                sum = Some(match sum {
                    None => t,
                    Some(s) => f.fadd(&s, &t),
                });
            }
            f.add_to(g, k, &sum.unwrap());
        });
    });
    let done = cx.fb().imul(&nb, "4");
    for_range(cx, &done, n, |cx, i| {
        let a = coef(cx, i);
        let row = cx.fb().imul(i, c);
        for_range(cx, "0", c, |cx, k| {
            let f = cx.fb();
            let idx = f.iadd(&row, k);
            let x = f.load(m, &idx);
            let t = f.fmul(&a, &x);
            f.add_to(g, k, &t);
        });
    });
}
