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

/// A narrow storage type for a copy of model data whose every value it holds
/// exactly (chosen at run time, see `mint_narrow` in the runtime). Vector
/// code loads it and converts in registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Narrow {
    I8,
    I16,
    F32,
}

impl Narrow {
    /// The runtime's code for this type (`mint_narrow`).
    pub fn code(self) -> u32 {
        match self {
            Narrow::I8 => 1,
            Narrow::I16 => 2,
            Narrow::F32 => 3,
        }
    }
    pub fn llty(self) -> &'static str {
        match self {
            Narrow::I8 => "i8",
            Narrow::I16 => "i16",
            Narrow::F32 => "float",
        }
    }
    /// The element suffix of a masked-load intrinsic for this type.
    pub fn mname(self) -> &'static str {
        match self {
            Narrow::I8 => "i8",
            Narrow::I16 => "i16",
            Narrow::F32 => "f32",
        }
    }
    pub fn bytes(self) -> u32 {
        match self {
            Narrow::I8 => 1,
            Narrow::I16 => 2,
            Narrow::F32 => 4,
        }
    }
    /// The conversion to double, exact for every value of the type.
    pub fn conv(self) -> &'static str {
        match self {
            Narrow::F32 => "fpext",
            _ => "sitofp",
        }
    }
}

#[derive(Default)]
pub struct Module {
    pub decls: BTreeSet<String>,
    pub globals: Vec<String>,
    pub funcs: Vec<String>,
    strings: HashMap<String, String>,
    /// Metadata nodes (loop hints), printed after the functions.
    meta: Vec<String>,
    /// Use Mint's own `exp` (mint_exp_defs) instead of llvm.exp.f64.
    pub inline_exp: bool,
    /// Use Mint's own `log` (mint_log_defs) where a builder asks for it
    /// (`Fb::inline_log`).
    pub inline_log: bool,
    /// The host has AVX2 (programs are built for the host): table lookups use
    /// its gather instruction directly, since LLVM's generic gather is split
    /// into scalar loads on some CPUs where the instruction is fast.
    pub avx2: bool,
    /// While a model's narrow-data variant of `logp` is generated: the data
    /// that variant reads through a narrow copy, and the suffix of the
    /// variant's function names (empty for the wide one).
    pub narrow_data: HashMap<String, Narrow>,
    pub variant: String,
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

/// invc_j and logc_j = -log(invc_j) for Mint's log, j = 0..127 (generated by
/// tests/log/make_table.py with 50-digit decimal arithmetic; worst rounding
/// error of logc 8e-4 ulp, so logc is nearly exact).
const LOG_TAB: [(u64, u64); 128] = [
    (0x3FF745D0A5422C0B, 0xBFD7FAF8024E0454),
    (0x3FF7242924AB9EB3, 0xBFD79E2831E797C2),
    (0x3FF702E0D464AB05, 0xBFD741D9C57E73EC),
    (0x3FF6E1F793413BEE, 0xBFD6E60F56898CAE),
    (0x3FF6C16B7BBB3215, 0xBFD68AC68A9AE122),
    (0x3FF6A13D2B3D50CC, 0xBFD63004098412D1),
    (0x3FF6816742299A86, 0xBFD5D5BB837FA37C),
    (0x3FF661EC07B777F8, 0xBFD57BF639D96A97),
    (0x3FF642C822A66AA6, 0xBFD522AD6AD715B6),
    (0x3FF623FACF5C3383, 0xBFD4C9E19D7DAF25),
    (0x3FF6058114CD6945, 0xBFD4718CE6E6BC93),
    (0x3FF5E75BA1BC2A3B, 0xBFD419B3E067B408),
    (0x3FF5C9876409FFFB, 0xBFD3C2502D1086F9),
    (0x3FF5AC05B495A7AC, 0xBFD36B68500802CE),
    (0x3FF58ED2AF7742BA, 0xBFD314F35ABD66A6),
    (0x3FF571EC9145433C, 0xBFD2BEEE7E674007),
    (0x3FF555563CA05B6B, 0xBFD26964C715DF22),
    (0x3FF53909FCD4009E, 0xBFD214478B7E01B6),
    (0x3FF51D08E5A453FB, 0xBFD1BF9C5B747884),
    (0x3FF501507AF9BB26, 0xBFD16B5E025DDEB9),
    (0x3FF4E5DFED473718, 0xBFD1178C48D1F351),
    (0x3FF4CAB8EEA06D54, 0xBFD0C42EA5E73051),
    (0x3FF4AFD6DA44DFDC, 0xBFD0713913926BF8),
    (0x3FF4953AD0C57870, 0xBFD01EB1374CB878),
    (0x3FF47AE05921F872, 0xBFCF991699C7E257),
    (0x3FF460CB52DBA849, 0xBFCEF5AB0550B0AF),
    (0x3FF446F8350B473F, 0xBFCE530DCEBE4064),
    (0x3FF42D66946EBCC5, 0xBFCDB1406E72FF3F),
    (0x3FF41413F339B8B8, 0xBFCD103720F55765),
    (0x3FF3FB01DA9E3DC6, 0xBFCC6FFFA775BC25),
    (0x3FF3E22BD0F62F0B, 0xBFCBD081496C8451),
    (0x3FF3C995E3C406C2, 0xBFCB31D9F0B7D765),
    (0x3FF3B13BDA8544D2, 0xBFCA93F248ECFFBC),
    (0x3FF3991C18D696C7, 0xBFC9F6C389421EEA),
    (0x3FF38138E4BF59BD, 0xBFC95A603A1EA1A2),
    (0x3FF3698D97BAB3DC, 0xBFC8BEAD8BB00ADD),
    (0x3FF3521CB7B71BDF, 0xBFC823BFA66EB95B),
    (0x3FF33AE38B68AC41, 0xBFC789881D34E9C9),
    (0x3FF323E3C14F0106, 0xBFC6F015A835B7E4),
    (0x3FF30D1909A212CC, 0xBFC6574EF7450178),
    (0x3FF2F685957D4844, 0xBFC5BF461C61C518),
    (0x3FF2E025EE11D235, 0xBFC527E71B0A8D8C),
    (0x3FF2C9FAC0B87F99, 0xBFC49139C41C8C9A),
    (0x3FF2B405630B06AB, 0xBFC3FB4A836C86BF),
    (0x3FF29E413392E447, 0xBFC365FCF2A7716D),
    (0x3FF288B0107E1876, 0xBFC2D160FE6D5A28),
    (0x3FF2734FC81EFC20, 0xBFC23D6AA6A08CB0),
    (0x3FF25E231F663C04, 0xBFC1AA3040F452BB),
    (0x3FF2492466A0074E, 0xBFC1178D50881631),
    (0x3FF23457538DA00A, 0xBFC0859EB919E323),
    (0x3FF21FB76557956B, 0xBFBFE88FB150F2C0),
    (0x3FF20B474B639444, 0xBFBEC73D009ED28D),
    (0x3FF1F70557A676D8, 0xBFBDA73384881A9A),
    (0x3FF1E2EF3CCEC7B5, 0xBFBC8858180BD571),
    (0x3FF1CF06A272E8E7, 0xBFBB6AC7ECE12EE2),
    (0x3FF1BB4AB03498A6, 0xBFBA4E7C90A90428),
    (0x3FF1A7B9E2CAC1EC, 0xBFB93365B5D52A6C),
    (0x3FF19452D54CB643, 0xBFB81974716DD2AD),
    (0x3FF181185989DF4E, 0xBFB700D7286B1A3E),
    (0x3FF16E060EA33624, 0xBFB5E9534494A43D),
    (0x3FF15B1E0DD7FDB0, 0xBFB4D30CA9525182),
    (0x3FF1485E5F654D1C, 0xBFB3BDEB8CDE4184),
    (0x3FF135C8269E08E4, 0xBFB2AA05E2B5DB1E),
    (0x3FF12359A2F1979C, 0xBFB19746C33EE22B),
    (0x3FF1111145CF90D3, 0xBFB0859BCCC7B19D),
    (0x3FF0FEF0D306425F, 0xBFAEEA489522C595),
    (0x3FF0ECF66A3BE0C1, 0xBFACCB91DB701FCE),
    (0x3FF0DB2194010CA5, 0xBFAAAF0EC09A2DF5),
    (0x3FF0C9719803C989, 0xBFA894B2B113238C),
    (0x3FF0B7E65A027858, 0xBFA67C8377601477),
    (0x3FF0A6808256291F, 0xBFA4669E7C07119A),
    (0x3FF0953E7A3889C5, 0xBFA252DC2D1E3185),
    (0x3FF0842091AC1360, 0xBFA0414F2DBD6957),
    (0x3FF07325E38B9D5E, 0xBF9C63C980C7A3F6),
    (0x3FF0624CFC6144C7, 0xBF9848F0C686BE52),
    (0x3FF05198A0EADD79, 0xBF9432D29728FF77),
    (0x3FF0410467063914, 0xBF90206BB31B2B34),
    (0x3FF030921FFC09BD, 0xBF88247D7192C449),
    (0x3FF0204033DB1A17, 0xBF800FEF2F3A744B),
    (0x3FF0100F847C25C2, 0xBF70077A50FFCB63),
    (0x3FF0000000000000, 0x0000000000000000),
    (0x3FEFC07E3720ECED, 0x3F7FE090A37A4AA4),
    (0x3FEF81F7A028F666, 0x3F8FC0C90695B556),
    (0x3FEF4465C5EC086B, 0x3F97B915F5E3A0A1),
    (0x3FEF07C106741142, 0x3F9F82B939826B42),
    (0x3FEECC07071F419D, 0x3FA39E92E39739B4),
    (0x3FEE91325C225BC4, 0x3FA7744D6EF37D23),
    (0x3FEE573A56906830, 0x3FAB42E4FC54A1CF),
    (0x3FEE1E1D94D6FD00, 0x3FAF0A39DDCAAA6D),
    (0x3FEDE5D6DF46EAE9, 0x3FB1653716D43F79),
    (0x3FEDAE60102C24C9, 0x3FB341DB0A9F1A83),
    (0x3FED77B58F3599A7, 0x3FB51B0DF2E487DB),
    (0x3FED41D3DAA6C3B4, 0x3FB6F0D4D1B23D7C),
    (0x3FED0CB584739304, 0x3FB8C34636F71BA0),
    (0x3FECD8567579AC17, 0x3FBA926DE7B2FE52),
    (0x3FECA4B22A0B3730, 0x3FBC5E5C3797A757),
    (0x3FEC71C6DF5F0CF1, 0x3FBE270993D380BB),
    (0x3FEC3F8F04F87AF5, 0x3FBFEC9114D01070),
    (0x3FEC0E07A35BD30B, 0x3FC0D77BA37DA395),
    (0x3FEBDD2B314B929C, 0x3FC1B72C6ABC40A6),
    (0x3FEBACF8256B8581, 0x3FC295574B0EBE61),
    (0x3FEB7D6C211CB54E, 0x3FC371FCA5F0E387),
    (0x3FEB4E80CD412CAF, 0x3FC44D2FAAAA0DA8),
    (0x3FEB203547ECE061, 0x3FC526EA783C0669),
    (0x3FEAF286FD6BEB7D, 0x3FC5FF2F3CE6F6B4),
    (0x3FEAC56F57A8EAE0, 0x3FC6D6138BFA3C65),
    (0x3FEA98EFD6FE820F, 0x3FC7AB86C7680C19),
    (0x3FEA6D025CCD361A, 0x3FC87F9CF39ED6DE),
    (0x3FEA41A4F9A38D91, 0x3FC952565BF716AA),
    (0x3FEA16D3952B66FE, 0x3FCA23BE0C05C7C7),
    (0x3FE9EC8DE8D2C890, 0x3FCAF3CCA0F00CB5),
    (0x3FE9C2D13642474A, 0x3FCBC286EE943AD9),
    (0x3FE99998D25E1540, 0x3FCC8FFBABC43F06),
    (0x3FE970E52608730E, 0x3FCD5C2083F53984),
    (0x3FE948B0998D829D, 0x3FCE270964CE9186),
    (0x3FE920FAE6E7805F, 0x3FCEF0AFC3B12732),
    (0x3FE8F9C2687986DF, 0x3FCFB91415EF7DCD),
    (0x3FE8D300A29FE772, 0x3FD04027F1709AB8),
    (0x3FE8ACB94167EC72, 0x3FD0A32460CE041F),
    (0x3FE886E5A6A3706F, 0x3FD1058CBADBD297),
    (0x3FE861855B17763B, 0x3FD1675E9C8E173C),
    (0x3FE83C978B7D6B66, 0x3FD1C89895126A71),
    (0x3FE818175AC980FA, 0x3FD22943F2A84285),
    (0x3FE7F406EFDF524F, 0x3FD289578AF5E392),
    (0x3FE7D05F71863767, 0x3FD2E8E239C868CE),
    (0x3FE7AD21F9CD5704, 0x3FD347DDC3596E2B),
    (0x3FE78A4BAE6758D4, 0x3FD3A64E93405676),
    (0x3FE767DD2D015EAC, 0x3FD4042FBF76EA3B),
];

/// Every definition and declaration `mint_log{_vL}` needs.
fn mint_log_defs(lanes: u32, avx2: bool) -> Vec<String> {
    let (d, _, b, fma, _) = exp_types(lanes);
    let l = lanes;
    let mut out = vec![format!("declare {d} @{fma}({d}, {d}, {d})"), mint_log_tables(), mint_log_full(lanes, avx2), mint_log_fast(lanes, avx2)];
    if lanes > 1 {
        out.push(format!("declare {d} @llvm.masked.gather.v{l}f64.v{l}p0(<{l} x ptr>, i32, {b}, {d})"));
        if avx2 && lanes == 4 {
            out.push("declare <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double>, ptr, <4 x i64>, <4 x double>, i8)".into());
        }
        out.push(format!("declare i1 @llvm.vector.reduce.or.v{l}i1({b})"));
    }
    out.push("declare i1 @llvm.expect.i1(i1, i1)".into());
    out
}

fn mint_log_tables() -> String {
    let inv: Vec<String> = LOG_TAB.iter().map(|(v, _)| format!("double 0x{v:016X}")).collect();
    let lc: Vec<String> = LOG_TAB.iter().map(|(_, v)| format!("double 0x{v:016X}")).collect();
    format!(
        "@mint_log_invc = internal unnamed_addr constant [128 x double] [{}], align 64\n@mint_log_logc = internal unnamed_addr constant [128 x double] [{}], align 64",
        inv.join(", "),
        lc.join(", ")
    )
}

/// The body of Mint's log for a positive normal input `%x` (a value of the
/// given width), leaving the result in `%y`; `kadj` (a register or None) is
/// added to the exponent.
///
/// x = 2^k z with z in about [0.684, 1.371) (the bit pattern of x minus OFF
/// gives k in its top bits and the table index j in the next 7); c_j is a
/// point of j's subinterval with 1/c_j = invc_j exact, so z/c_j = 1 + r with
/// r = fma(z, invc_j, -1) (one rounding) and |r| < 2^-8 + 2^-20, and
/// log x = k ln2 + logc_j + log1p(r). The subinterval that holds 1 has
/// c = 1 (logc 0), so near 1 the result is r + poly(r) with no
/// cancellation. log1p(r) - r is its degree-7 Taylor polynomial (truncation
/// below 2^-66). k ln2 + logc + r is summed as hi + lo (Fast2Sum: |k ln2 +
/// logc| >= |r| or it is 0), with ln2 split so that k ln2hi is exact.
fn mint_log_core(lanes: u32, avx2: bool, kadj: Option<&str>) -> String {
    let (d, it, b, fma, _) = exp_types(lanes);
    let l = lanes;
    let c = |x: f64| if lanes == 1 { fconst(x) } else { format!("splat (double {})", fconst(x)) };
    let i = |n: u64| if lanes == 1 { (n as i64).to_string() } else { format!("splat (i64 {})", n as i64) };
    let fm = |r: &str, a: &str, x: &str, y: &str| format!("  %{r} = call {d} @{fma}({d} {a}, {d} {x}, {d} {y})\n");
    let mut s = String::new();
    s += &format!("  %ix = bitcast {d} %x to {it}\n");
    s += &format!("  %tmp = sub {it} %ix, {}\n", i(0x3FE5F00000000000));
    s += &format!("  %j0 = lshr {it} %tmp, {}\n  %j = and {it} %j0, {}\n", i(45), i(127));
    s += &format!("  %top = and {it} %tmp, {}\n", i(0xFFF0000000000000));
    s += &format!("  %iz = sub {it} %ix, %top\n  %z = bitcast {it} %iz to {d}\n");
    // k = tmp >> 52 (arithmetic) as a double. AVX2 has neither a 64-bit
    // arithmetic shift nor a 64-bit integer conversion, so: the high 32 bits
    // shifted by 20 (one 32-bit shift), then a 32-bit conversion.
    let (i32t, i32h) = if lanes == 1 { ("i32".to_string(), "i32".to_string()) } else { (format!("<{} x i32>", 2 * l), format!("<{l} x i32>")) };
    if lanes == 1 {
        s += "  %th = lshr i64 %tmp, 32\n  %th32 = trunc i64 %th to i32\n  %k32 = ashr i32 %th32, 20\n";
    } else {
        let odd: Vec<String> = (0..l).map(|q| format!("i32 {}", 2 * q + 1)).collect();
        s += &format!("  %t32 = bitcast {it} %tmp to {i32t}\n  %s32 = ashr {i32t} %t32, splat (i32 20)\n");
        s += &format!("  %k32 = shufflevector {i32t} %s32, {i32t} poison, <{l} x i32> <{}>\n", odd.join(", "));
    }
    match kadj {
        Some(a) => s += &format!("  %kd1 = sitofp {i32h} %k32 to {d}\n  %kd = fadd {d} %kd1, {a}\n"),
        None => s += &format!("  %kd = sitofp {i32h} %k32 to {d}\n"),
    }
    for (name, tab) in [("invc", "@mint_log_invc"), ("logc", "@mint_log_logc")] {
        if lanes == 1 {
            s += &format!("  %{name}p = getelementptr inbounds double, ptr {tab}, i64 %j\n  %{name} = load double, ptr %{name}p, align 8\n");
        } else if avx2 && lanes == 4 {
            s += &format!("  %{name} = call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> poison, ptr {tab}, <4 x i64> %j, <4 x double> splat (double 0xFFFFFFFFFFFFFFFF), i8 8)\n");
        } else {
            s += &format!("  %{name}p = getelementptr inbounds double, ptr {tab}, {it} %j\n");
            s += &format!("  %{name} = call {d} @llvm.masked.gather.v{l}f64.v{l}p0(<{l} x ptr> %{name}p, i32 8, {b} splat (i1 true), {d} poison)\n");
        }
    }
    let ln2hi = f64::from_bits(0x3FE62E42FEFA3800);
    let ln2lo = f64::from_bits(0x3D2EF35793C76730);
    s += &fm("r", "%z", "%invc", &c(-1.0));
    s += &fm("w", "%kd", &c(ln2hi), "%logc");
    s += &format!("  %hi = fadd {d} %w, %r\n  %lo0 = fsub {d} %w, %hi\n  %lo1 = fadd {d} %lo0, %r\n");
    s += &fm("lo", "%kd", &c(ln2lo), "%lo1");
    s += &format!("  %r2 = fmul {d} %r, %r\n  %r3 = fmul {d} %r, %r2\n");
    s += &fm("q1", "%r", &c(-1.0 / 6.0), &c(1.0 / 5.0));
    s += &fm("q1b", "%r2", &c(1.0 / 7.0), "%q1");
    s += &fm("q2", "%r", &c(-1.0 / 4.0), &c(1.0 / 3.0));
    s += &fm("q", "%r2", "%q1b", "%q2");
    s += &fm("t", "%r2", &c(-0.5), "%lo");
    s += &fm("t2", "%r3", "%q", "%t");
    s += &format!("  %y = fadd {d} %t2, %hi\n");
    s
}

/// The fast log: the core for every lane, and a branch (marked unlikely) to
/// the full-range version when any lane is zero, subnormal, negative,
/// infinite or NaN (one unsigned compare of the bits finds them all).
fn mint_log_fast(lanes: u32, avx2: bool) -> String {
    let (d, it, b, _, sfx) = exp_types(lanes);
    let l = lanes;
    let i = |n: u64| if lanes == 1 { (n as i64).to_string() } else { format!("splat (i64 {})", n as i64) };
    let mut s = format!("define internal {d} @mint_log{sfx}({d} %x) alwaysinline {{\nentry:\n");
    s += &mint_log_core(lanes, avx2, None);
    // (ix - 0x0010...) >=u 0x7FE0... as a signed compare (AVX2 has no
    // unsigned one): flipping the sign bit of both sides
    s += &format!("  %e0 = add {it} %ix, {}\n  %out = icmp sgt {it} %e0, {}\n", i(0x7FF0000000000000), i(0xFFDFFFFFFFFFFFFF));
    if lanes == 1 {
        s += "  %any = call i1 @llvm.expect.i1(i1 %out, i1 false)\n";
    } else {
        s += &format!("  %any0 = call i1 @llvm.vector.reduce.or.v{l}i1({b} %out)\n  %any = call i1 @llvm.expect.i1(i1 %any0, i1 false)\n");
    }
    s += "  br i1 %any, label %slow, label %done\nslow:\n";
    s += &format!("  %full = call {d} @mint_log_full{sfx}({d} %x)\n  %sel = select {b} %out, {d} %full, {d} %y\n  br label %done\n");
    s += &format!("done:\n  %res = phi {d} [ %y, %entry ], [ %sel, %slow ]\n  ret {d} %res\n}}");
    s
}

/// The full-range log: subnormal inputs are scaled by 2^52 first (and 52
/// taken off k); log(+-0) = -inf, log(x < 0) = NaN, log(+inf) = +inf, and a
/// NaN input is returned (quieted), as in libm.
fn mint_log_full(lanes: u32, avx2: bool) -> String {
    let (d, _, b, _, sfx) = exp_types(lanes);
    let c = |x: f64| if lanes == 1 { fconst(x) } else { format!("splat (double {})", fconst(x)) };
    let mut s = format!("define internal {d} @mint_log_full{sfx}({d} %a) noinline {{\nentry:\n");
    s += &format!("  %sub0 = fcmp olt {d} %a, {}\n  %sub1 = fcmp ogt {d} %a, {}\n  %sub = and {b} %sub0, %sub1\n", c(f64::MIN_POSITIVE), c(0.0));
    s += &format!("  %as = fmul {d} %a, {}\n  %x = select {b} %sub, {d} %as, {d} %a\n", c(4503599627370496.0));
    s += &format!("  %kadj = select {b} %sub, {d} {}, {d} {}\n", c(-52.0), c(0.0));
    s += &mint_log_core(lanes, avx2, Some("%kadj"));
    s += &format!("  %zero = fcmp oeq {d} %a, {}\n  %y1 = select {b} %zero, {d} {}, {d} %y\n", c(0.0), c(f64::NEG_INFINITY));
    s += &format!("  %neg = fcmp olt {d} %a, {}\n  %y2 = select {b} %neg, {d} {}, {d} %y1\n", c(0.0), c(f64::NAN));
    s += &format!("  %inf = fcmp oeq {d} %a, {}\n  %y3 = select {b} %inf, {d} {}, {d} %y2\n", c(f64::INFINITY), c(f64::INFINITY));
    s += &format!("  %nan = fcmp uno {d} %a, %a\n  %qa = fadd {d} %a, %a\n  %y4 = select {b} %nan, {d} %qa, {d} %y3\n  ret {d} %y4\n}}");
    s
}

/// -log(1 - m/256) for m = 0..128, correctly rounded (then zeros), for
/// Mint's log1p on [0, 1] (generated by tests/log/make_table_log1p.py).
const LOG1P_TAB: [u64; 256] = [
    0x0000000000000000, 0x3F70080559588B35, 0x3F8010157588DE71, 0x3F882448A388A2AA,
    0x3F90205658935847, 0x3F9432A925980CC1, 0x3F98492528C8CABF, 0x3F9C63D2EC14AAF2,
    0x3FA0415D89E74444, 0x3FA252F32F8D183F, 0x3FA466AED42DE3EA, 0x3FA67C94F2D4BB58,
    0x3FA894AA149FB343, 0x3FAAAEF2D0FB10FC, 0x3FACCB73CDDDB2CC, 0x3FAEEA31C006B87C,
    0x3FB08598B59E3A07, 0x3FB1973BD1465567, 0x3FB2AA04A44717A5, 0x3FB3BDF5A7D1EE64,
    0x3FB4D3115D207EAC, 0x3FB5E95A4D9791CB, 0x3FB700D30AEAC0E1, 0x3FB8197E2F40E3F0,
    0x3FB9335E5D594989, 0x3FBA4E7640B1BC38, 0x3FBB6AC88DAD5B1C, 0x3FBC885801BC4B23,
    0x3FBDA727638446A2, 0x3FBEC739830A1120, 0x3FBFE89139DBD566, 0x3FC08598B59E3A07,
    0x3FC1178E8227E47C, 0x3FC1AA2B7E23F72A, 0x3FC23D712A49C202, 0x3FC2D1610C86813A,
    0x3FC365FCB0159016, 0x3FC3FB45A59928CC, 0x3FC4913D8333B561, 0x3FC527E5E4A1B58D,
    0x3FC5BF406B543DB2, 0x3FC6574EBE8C133A, 0x3FC6F0128B756ABC, 0x3FC7898D85444C73,
    0x3FC823C16551A3C2, 0x3FC8BEAFEB38FE8C, 0x3FC95A5ADCF7017F, 0x3FC9F6C407089664,
    0x3FCA93ED3C8AD9E3, 0x3FCB31D8575BCE3D, 0x3FCBD087383BD8AD, 0x3FCC6FFBC6F00F71,
    0x3FCD1037F2655E7B, 0x3FCDB13DB0D48940, 0x3FCE530EFFE71012, 0x3FCEF5ADE4DCFFE6,
    0x3FCF991C6CB3B379, 0x3FD01EAE5626C691, 0x3FD07138604D5862, 0x3FD0C42D676162E3,
    0x3FD1178E8227E47C, 0x3FD16B5CCBACFB73, 0x3FD1BF99635A6B95, 0x3FD214456D0EB8D4,
    0x3FD269621134DB92, 0x3FD2BEF07CDC9354, 0x3FD314F1E1D35CE4, 0x3FD36B6776BE1117,
    0x3FD3C25277333184, 0x3FD419B423D5E8C7, 0x3FD4718DC271C41B, 0x3FD4C9E09E172C3C,
    0x3FD522AE0738A3D8, 0x3FD57BF753C8D1FB, 0x3FD5D5BDDF595F30, 0x3FD630030B3AAC49,
    0x3FD68AC83E9C6A14, 0x3FD6E60EE6AF1972, 0x3FD741D876C67BB1, 0x3FD79E26687CFB3E,
    0x3FD7FAFA3BD8151C, 0x3FD85855776DCBFB, 0x3FD8B639A88B2DF5, 0x3FD914A8635BF68A,
    0x3FD973A3431356AE, 0x3FD9D32BEA15ED3B, 0x3FDA33440224FA79, 0x3FDA93ED3C8AD9E3,
    0x3FDAF5295248CDD0, 0x3FDB56FA04462909, 0x3FDBB9611B80E2FB, 0x3FDC1C60693FA39E,
    0x3FDC7FF9C74554C9, 0x3FDCE42F18064743, 0x3FDD490246DEFA6B, 0x3FDDAE75484C9616,
    0x3FDE148A1A2726CE, 0x3FDE7B42C3DDAD73, 0x3FDEE2A156B413E5, 0x3FDF4AA7EE03192D,
    0x3FDFB358AF7A4884, 0x3FE00E5AE5B207AB, 0x3FE04360BE7603AD, 0x3FE078BF0533C568,
    0x3FE0AE76E2D054FA, 0x3FE0E4898611CCE1, 0x3FE11AF823C75AA8, 0x3FE151C3F6F29612,
    0x3FE188EE40F23CA6, 0x3FE1C07849AE6007, 0x3FE1F8635FC61659, 0x3FE230B0D8BEBC98,
    0x3FE269621134DB92, 0x3FE2A2786D0EC107, 0x3FE2DBF557B0DF43, 0x3FE315DA4434068B,
    0x3FE35028AD9D8C86, 0x3FE38AE2171976E7, 0x3FE3C6080C36BFB5, 0x3FE4019C2125CA93,
    0x3FE43D9FF2F923C5, 0x3FE47A1527E8A2D3, 0x3FE4B6FD6F970C1F, 0x3FE4F45A835A4E19,
    0x3FE5322E26867857, 0x3FE5707A26BB8C66, 0x3FE5AF405C3649E0, 0x3FE5EE82AA241920,
    0x3FE62E42FEFA39EF, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
];

/// Every definition and declaration `mint_log1p01_v{L}` needs.
fn mint_log1p01_defs(lanes: u32, avx2: bool) -> Vec<String> {
    let (d, _, b, fma, _) = exp_types(lanes);
    let l = lanes;
    let vals: Vec<String> = LOG1P_TAB.iter().map(|v| format!("double 0x{v:016X}")).collect();
    let mut out = vec![
        format!("declare {d} @{fma}({d}, {d}, {d})"),
        format!("@mint_log1p_tab = internal unnamed_addr constant [256 x double] [{}], align 64", vals.join(", ")),
        mint_log1p01(lanes, avx2),
        format!("declare {d} @llvm.masked.gather.v{l}f64.v{l}p0(<{l} x ptr>, i32, {b}, {d})"),
    ];
    if avx2 && lanes == 4 {
        out.push("declare <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double>, ptr, <4 x i64>, <4 x double>, i8)".into());
    }
    out
}

/// log1p(e) for e in [0, 1] (or NaN), given q = 1/(1 + e) (which the
/// caller has anyway: BernoulliLogit's sigmoid is q or e q). With
/// m = round(256 (1 - q)) and invc = 1 - m/256 (both exact),
/// (1 + e) invc = 1 + r with r = e invc - m/256 formed by one fma from the
/// exact e (1 + e is never rounded), |r| <= 2^-8, and
/// log1p(e) = -log(invc) + log1p(r), -log(invc) from a table (one gather)
/// and log1p(r) - r the degree-7 Taylor polynomial, summed as in mint_log.
/// For m = 0 (e < 1/511) the table entry is 0 and r = e exactly, so tiny e
/// keep full relative accuracy. No special cases: e is finite and at most
/// 1, and a NaN e gives r = NaN.
fn mint_log1p01(lanes: u32, avx2: bool) -> String {
    let (d, it, b, fma, sfx) = exp_types(lanes);
    let l = lanes;
    let c = |x: f64| format!("splat (double {})", fconst(x));
    let fm = |r: &str, a: &str, x: &str, y: &str| format!("  %{r} = call {d} @{fma}({d} {a}, {d} {x}, {d} {y})\n");
    let magic = 6755399441055744.0; // 0x1.8p52: adding it rounds to an integer
    let mut s = format!("define internal {d} @mint_log1p01{sfx}({d} %e, {d} %q) alwaysinline {{\nentry:\n");
    s += &fm("t", "%q", &c(-256.0), &c(magic + 256.0));
    s += &format!("  %md = fsub {d} %t, {}\n", c(magic));
    s += &fm("invc", "%md", &c(-1.0 / 256.0), &c(1.0));
    s += &format!("  %dm = fmul {d} %md, {}\n  %ndm = fneg {d} %dm\n", c(1.0 / 256.0));
    s += &format!("  %tb = bitcast {d} %t to {it}\n  %j = and {it} %tb, splat (i64 255)\n");
    if avx2 && lanes == 4 {
        s += "  %logc = call <4 x double> @llvm.x86.avx2.gather.q.pd.256(<4 x double> poison, ptr @mint_log1p_tab, <4 x i64> %j, <4 x double> splat (double 0xFFFFFFFFFFFFFFFF), i8 8)\n";
    } else {
        s += &format!("  %lp = getelementptr inbounds double, ptr @mint_log1p_tab, {it} %j\n");
        s += &format!("  %logc = call {d} @llvm.masked.gather.v{l}f64.v{l}p0(<{l} x ptr> %lp, i32 8, {b} splat (i1 true), {d} poison)\n");
    }
    s += &fm("r", "%e", "%invc", "%ndm");
    s += &format!("  %hi = fadd {d} %logc, %r\n  %lo0 = fsub {d} %logc, %hi\n  %lo = fadd {d} %lo0, %r\n");
    s += &format!("  %r2 = fmul {d} %r, %r\n  %r3 = fmul {d} %r, %r2\n");
    s += &fm("q1", "%r", &c(-1.0 / 6.0), &c(1.0 / 5.0));
    s += &fm("q1b", "%r2", &c(1.0 / 7.0), "%q1");
    s += &fm("q2", "%r", &c(-1.0 / 4.0), &c(1.0 / 3.0));
    s += &fm("qq", "%r2", "%q1b", "%q2");
    s += &fm("u", "%r2", &c(-0.5), "%lo");
    s += &fm("u2", "%r3", "%qq", "%u");
    s += &format!("  %y = fadd {d} %u2, %hi\n  ret {d} %y\n}}");
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
        for m in &self.meta {
            out += m;
            out.push('\n');
        }
        out
    }

    /// Loop metadata asking LLVM to leave a loop as written: no unrolling,
    /// vectorisation or interleaving (for loops Mint has already vectorised
    /// and whose trip count is small, where a runtime-unrolled copy and its
    /// remainder cost more than they save).
    pub fn loop_as_written(&mut self) -> String {
        let n = self.meta.len();
        let (id, a, b, c) = (n, n + 1, n + 2, n + 3);
        self.meta.push(format!("!{id} = distinct !{{!{id}, !{a}, !{b}, !{c}}}"));
        self.meta.push(format!("!{a} = !{{!\"llvm.loop.unroll.disable\"}}"));
        self.meta.push(format!("!{b} = !{{!\"llvm.loop.vectorize.width\", i32 1}}"));
        self.meta.push(format!("!{c} = !{{!\"llvm.loop.interleave.count\", i32 1}}"));
        format!("!{id}")
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
    /// Use Mint's log in vector code (lanes > 1): set by the fission kernel.
    pub inline_log: bool,
    /// Data pointers (registers) with a narrow copy: wide pointer -> (narrow
    /// pointer, its type). Only vector code (lanes > 1) loads the narrow
    /// copy. That code is Mint's own, so its arithmetic is the same either
    /// way; a scalar loop is left to LLVM's vectoriser, whose choice of
    /// vector width and interleaving (and so the order of a reassociated
    /// sum) could change with the type it loads.
    pub narrow: HashMap<String, (String, Narrow)>,
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
            inline_log: false,
            narrow: HashMap::new(),
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
    /// A floating-point comparison in the current type (i1 or <L x i1>).
    pub fn fcmp(&mut self, pred: &str, a: &str, b: &str) -> String {
        let r = self.reg();
        let (t, a, b) = (self.ty(), self.opnd(a), self.opnd(b));
        self.emit(format!("{r} = fcmp {pred} {t} {a}, {b}"));
        r
    }
    /// select(c, a, b) in the current type; `c` comes from `fcmp`.
    pub fn select(&mut self, c: &str, a: &str, b: &str) -> String {
        let r = self.reg();
        let (t, a, b) = (self.ty(), self.opnd(a), self.opnd(b));
        let ct = if self.lanes == 1 { "i1".to_string() } else { format!("<{} x i1>", self.lanes) };
        self.emit(format!("{r} = select {ct} {c}, {t} {a}, {t} {b}"));
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
        // Mint's own log likewise, where the builder asks for it; a constant
        // argument keeps llvm.log, which LLVM folds.
        let lit = a.starts_with("0x") || a.starts_with("splat (double 0x") || a.parse::<f64>().is_ok();
        if name == "llvm.log.f64" && m.inline_log && self.inline_log && self.lanes > 1 && !lit {
            for d in mint_log_defs(self.lanes, m.avx2) {
                m.declare(&d);
            }
            let r = self.reg();
            self.emit(format!("{r} = call {t} @mint_log_v{}({t} {a})", self.lanes));
            return r;
        }
        let name = self.vname(name);
        m.declare(&format!("declare {t} @{name}({t})"));
        let r = self.reg();
        self.emit(format!("{r} = call {t} @{name}({t} {a})"));
        r
    }
    /// log1p(e) for e in [0, 1] given q = 1/(1 + e), in vector code
    /// (mint_log1p01).
    pub fn log1p01(&mut self, m: &mut Module, e: &str, q: &str) -> String {
        assert!(self.lanes > 1, "log1p01 is emitted in vector code only");
        for d in mint_log1p01_defs(self.lanes, m.avx2) {
            m.declare(&d);
        }
        let t = self.ty();
        let r = self.reg();
        self.emit(format!("{r} = call {t} @mint_log1p01_v{}({t} {e}, {t} {q})", self.lanes));
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
        if self.lanes > 1 {
            if let Some((np, k)) = self.narrow.get(p).cloned() {
                return self.load_narrow(&np, k, idx);
            }
        }
        let a = self.gep(p, idx);
        let r = self.reg();
        let t = self.ty();
        self.emit(format!("{r} = load {t}, ptr {a}, align 8"));
        r
    }
    /// Elements idx.. of a narrow copy, converted to the current type.
    pub fn load_narrow(&mut self, np: &str, k: Narrow, idx: &str) -> String {
        let (et, b) = (k.llty(), k.bytes());
        let a = self.reg();
        self.emit(format!("{a} = getelementptr inbounds {et}, ptr {np}, i64 {idx}"));
        let (nt, t) = if self.lanes == 1 { (et.to_string(), "double".to_string()) } else { (format!("<{} x {et}>", self.lanes), self.ty()) };
        let x = self.reg();
        self.emit(format!("{x} = load {nt}, ptr {a}, align {b}"));
        let r = self.reg();
        self.emit(format!("{r} = {} {nt} {x} to {t}", k.conv()));
        self.opaque(&r)
    }

    /// `v` (a converted narrow load) behind an empty inline asm, so that to
    /// the optimiser it is an opaque value, as the load of a double is. With
    /// the conversion in sight, LLVM learns facts about the value (an
    /// integer converted to double is never -0.0, for example) that the
    /// load of a double does not give it, and the code it then emits can
    /// differ: on `Normal(a * X * beta + b * y, exp(X * beta))` with float
    /// data the backend fused a different multiply into an add, and the
    /// gradient changed in the last bit. The asm emits no instruction.
    /// (`llvm.arithmetic.fence` instead did not prevent that.)
    pub fn opaque(&mut self, v: &str) -> String {
        let t = self.ty();
        let r = self.reg();
        self.emit(format!("{r} = call {t} asm \"\", \"=x,0\"({t} {v}) nounwind memory(none)"));
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
    for_range_md(cx, lo, hi, None, body)
}

/// for_range with loop metadata `md` (from Module::loop_as_written) on the
/// back edge.
pub fn for_range_md<C: HasFb>(cx: &mut C, lo: &str, hi: &str, md: Option<&str>, body: impl FnOnce(&mut C, &str)) {
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
    match md {
        Some(md) => f.emit(format!("br label %{hdr}, !llvm.loop {md}")),
        None => f.br(&hdr),
    }
    f.start_block(&exit);
}

/// if cond { body } (cond is an i1 register)
pub fn if_then<C: HasFb>(cx: &mut C, cond: &str, body: impl FnOnce(&mut C)) {
    let f = cx.fb();
    let yes = f.label("then");
    let join = f.label("endif");
    f.emit(format!("br i1 {cond}, label %{yes}, label %{join}"));
    f.start_block(&yes);
    body(cx);
    let f = cx.fb();
    f.br(&join);
    f.start_block(&join);
}

/// For i in 0..n: store(i, sum_k M[i*c + k] * v[k]).
/// Rows are processed four at a time so each load of v feeds four FMAs;
/// LLVM vectorises the four independent reductions.
pub fn rows_dot_blocked<C: HasFb>(cx: &mut C, m: &str, v: &str, c: &str, lo: &str, n: &str, store: &dyn Fn(&mut C, &str, &str)) {
    let len = cx.fb().iop("sub nsw", n, lo);
    let nb = cx.fb().iop("sdiv", &len, "4");
    for_range(cx, "0", &nb, |cx, b| {
        let f = cx.fb();
        let i0 = f.imul(b, "4");
        let i0 = f.iadd(&i0, lo);
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
    let done = cx.fb().iadd(&done, lo);
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
pub fn rows_axpy_blocked<C: HasFb>(cx: &mut C, m: &str, c: &str, lo: &str, n: &str, coef: &dyn Fn(&mut C, &str) -> String, g: &str) {
    let len = cx.fb().iop("sub nsw", n, lo);
    let nb = cx.fb().iop("sdiv", &len, "4");
    for_range(cx, "0", &nb, |cx, b| {
        let i0 = cx.fb().imul(b, "4");
        let i0 = cx.fb().iadd(&i0, lo);
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
    let done = cx.fb().iadd(&done, lo);
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
