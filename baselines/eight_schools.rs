//! Straightforward Rust: eight schools, non-centred (examples/eight_schools.mint).
//! The log density and its gradient are written by hand; the sampler is the
//! Mint runtime's NUTS. With J = 8 there is nothing to vectorise.
//!
//! theta = [mu, log tau, eta[J]]; tau = exp(theta[1]), with the log Jacobian
//! log tau added (Mint does the same for a Positive parameter).

#[path = "common.rs"]
mod common;

const Y: [f64; 8] = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
const S: [f64; 8] = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];
const J: usize = 8;

extern "C" fn logp(theta: *const f64, grad: *mut f64) -> f64 {
    let th = unsafe { std::slice::from_raw_parts(theta, J + 2) };
    let g = unsafe { std::slice::from_raw_parts_mut(grad, J + 2) };
    let (mu, ltau) = (th[0], th[1]);
    let tau = ltau.exp();
    // mu ~ Normal(0, 5), tau ~ Normal(0, 5) (on tau > 0), Jacobian log tau
    let mut lp = -0.5 * (mu / 5.0) * (mu / 5.0) - 0.5 * (tau / 5.0) * (tau / 5.0) + ltau - 2.0 * 5f64.ln();
    let mut gmu = -mu / 25.0;
    let mut gtau = -tau / 25.0; // d/d tau; times tau below, plus 1 for the Jacobian
    for j in 0..J {
        let eta = th[2 + j];
        // eta ~ Normal(0, 1)
        lp -= 0.5 * eta * eta;
        // y ~ Normal(mu + tau * eta, s)
        let z = (Y[j] - mu - tau * eta) / S[j];
        lp -= 0.5 * z * z + S[j].ln();
        let dm = z / S[j]; // d lp / d mean
        gmu += dm;
        gtau += dm * eta;
        g[2 + j] = -eta + dm * tau;
    }
    g[0] = gmu;
    g[1] = gtau * tau + 1.0;
    lp
}

extern "C" fn constrain(unc: *const f64, out: *mut f64) {
    let u = unsafe { std::slice::from_raw_parts(unc, J + 2) };
    let o = unsafe { std::slice::from_raw_parts_mut(out, J + 2) };
    o.copy_from_slice(u);
    o[1] = u[1].exp();
}

fn main() {
    common::sample_and_print(logp, constrain, J + 2, &["mu", "tau", "eta"], &[-1, -1, J as i64], 4000, 1000, 4, 3);
}
