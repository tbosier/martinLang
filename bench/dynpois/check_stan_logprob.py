"""Check that dynpois.stan's log density matches SPEC's formula.

Stan's log_prob method (jacobian=True; every parameter is unconstrained so the
Jacobian is zero) evaluates with `~` statements, which drop terms that do not
depend on parameters: the -log(scale) terms, 0.5*log(2*pi) and log(y!). So the
absolute values differ by a data-only constant. We check (a) lp differences
between points, absolute tolerance 1e-9, (b) that the offset equals the dropped
-log(scale) terms to 1e-9, and (c) gradients (relative 1e-8, all finite).

Usage: python check_stan_logprob.py [SIZE] [NPOINTS]
"""
import sys

import numpy as np

from spec_logdensity import gradient, log_density, unpack
from stan_common import compile_model, load_data


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "small"
    npoints = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    y, data = load_data(size)
    G, T = y.shape
    model, _ = compile_model()
    rng = np.random.default_rng(7)
    lp_stan, lp_spec, grad_err = [], [], []
    for k in range(npoints):
        theta = np.concatenate([
            [1.5 + 0.3 * rng.normal()],
            1.5 + 0.4 * rng.normal(size=G),
            0.05 * rng.normal(size=T) / np.sqrt(T) * (k + 1),
            0.08 * rng.normal(size=G * T) / np.sqrt(T) * (k + 1),
        ])
        pop, beta, shared, innov = unpack(theta, G, T)
        params = {"pop": pop, "beta": beta.tolist(), "shared": shared.tolist(),
                  "innov": innov.tolist()}
        df = model.log_prob(params=params, data=data, jacobian=True, sig_figs=18)
        row = df.iloc[0].to_numpy(dtype=float)
        lp_stan.append(row[0])
        lp_spec.append(log_density(theta, y))
        g_stan = row[1:]
        g_spec = gradient(theta, y)
        assert g_stan.shape == g_spec.shape, (g_stan.shape, g_spec.shape)
        grad_err.append(np.max(np.abs(g_stan - g_spec) / (1.0 + np.abs(g_spec))))
    lp_stan, lp_spec = np.array(lp_stan), np.array(lp_spec)
    const = lp_spec - lp_stan
    diff_err = np.abs(np.diff(lp_stan) - np.diff(lp_spec))
    rel = diff_err / np.maximum(1.0, np.abs(np.diff(lp_spec)))
    expected_const = -G * np.log(0.4) - T * np.log(0.05) - G * T * np.log(0.08)
    print(f"size={size} points={npoints} D={1 + G + T + G * T}")
    print("lp_spec:", np.array2string(lp_spec, precision=6))
    print("lp_stan:", np.array2string(lp_stan, precision=6))
    print(f"spec - stan constant: {const} (spread {np.ptp(const):.3e})")
    print(f"  expected from dropped -log(scale) terms alone: {expected_const:.6f}")
    print(f"max |d lp_stan - d lp_spec| over consecutive pairs: {diff_err.max():.3e} "
          f"(relative {rel.max():.3e})")
    print(f"max relative gradient error: {np.max(grad_err):.3e}")
    const_err = np.max(np.abs(const - expected_const))
    print(f"max |(spec - stan) - expected constant|: {const_err:.3e}")
    # Absolute tolerances (SPEC: agree to 1e-9). lp values are O(1e4-1e5), so
    # float64 rounding alone is ~1e-11; the constant check shows that the only
    # difference from SPEC's density is the dropped -log(scale) terms.
    grad_err = np.array(grad_err)
    ok = (np.all(np.isfinite(lp_stan)) and np.all(np.isfinite(grad_err))
          and diff_err.max() < 1e-9 and const_err < 1e-9 and grad_err.max() < 1e-8)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
