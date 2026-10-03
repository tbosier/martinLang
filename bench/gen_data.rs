//! Writes synthetic benchmark data in Martin's .f64 format:
//! two little-endian u64 (rows, cols) followed by row-major f64.
//!
//!   gen_data logistic N P SEED PREFIX ALPHA
//!   gen_data linear   N P SEED PREFIX

use std::io::Write;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }
}

fn write(path: &str, rows: usize, cols: usize, data: &[f64]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("create"));
    f.write_all(&(rows as u64).to_le_bytes()).unwrap();
    f.write_all(&(cols as u64).to_le_bytes()).unwrap();
    for x in data {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let kind = a[1].as_str();
    let n: usize = a[2].parse().unwrap();
    let p: usize = a[3].parse().unwrap();
    let mut rng = Rng(a[4].parse().unwrap());
    let prefix = &a[5];
    let x: Vec<f64> = (0..n * p).map(|_| rng.normal()).collect();
    let beta: Vec<f64> = (0..p).map(|_| 0.5 * rng.normal()).collect();
    let mut y = vec![0.0; n];
    let mut truth = String::new();
    match kind {
        "logistic" => {
            let alpha: f64 = a[6].parse().unwrap();
            for i in 0..n {
                let eta = alpha + (0..p).map(|j| x[i * p + j] * beta[j]).sum::<f64>();
                let pr = 1.0 / (1.0 + (-eta).exp());
                y[i] = if rng.uniform() < pr { 1.0 } else { 0.0 };
            }
            truth += &format!("alpha {alpha}\n");
        }
        "linear" => {
            let (alpha, sigma) = (1.5, 0.8);
            for i in 0..n {
                let mu = alpha + (0..p).map(|j| x[i * p + j] * beta[j]).sum::<f64>();
                y[i] = mu + sigma * rng.normal();
            }
            truth += &format!("alpha {alpha}\nsigma {sigma}\n");
        }
        _ => panic!("unknown kind"),
    }
    for (j, b) in beta.iter().enumerate() {
        truth += &format!("beta[{}] {b}\n", j + 1);
    }
    write(&format!("{prefix}_X.f64"), n, p, &x);
    write(&format!("{prefix}_y.f64"), n, 1, &y);
    std::fs::write(format!("{prefix}_truth.txt"), truth).unwrap();
}
