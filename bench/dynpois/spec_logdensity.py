"""Reference log density and gradient from SPEC.md, in numpy.

theta = [pop, beta[0..G], shared[0..T], innov[0..G*T] (row-major g*T + t)].
Constants dropped: 0.5*log(2*pi) per Normal term and log(y!) per Poisson term;
every Normal term keeps its -log(scale).
"""
import numpy as np

GROUP_SD, SHARED_SD, PROCESS_SD = 0.4, 0.05, 0.08


def unpack(theta, G, T):
    theta = np.asarray(theta, dtype=float)
    assert theta.shape == (1 + G + T + G * T,)
    pop = theta[0]
    beta = theta[1:1 + G]
    shared = theta[1 + G:1 + G + T]
    innov = theta[1 + G + T:].reshape(G, T)
    return pop, beta, shared, innov


def eta_of(theta, G, T):
    pop, beta, shared, innov = unpack(theta, G, T)
    state = np.cumsum(shared[None, :] + innov, axis=1)
    return beta[:, None] + state


def log_density(theta, y):
    G, T = y.shape
    pop, beta, shared, innov = unpack(theta, G, T)
    eta = eta_of(theta, G, T)
    lp = -0.5 * pop ** 2 - np.log(1.0)
    lp += np.sum(-0.5 * ((beta - pop) / GROUP_SD) ** 2 - np.log(GROUP_SD))
    lp += np.sum(-0.5 * (shared / SHARED_SD) ** 2 - np.log(SHARED_SD))
    lp += np.sum(-0.5 * (innov / PROCESS_SD) ** 2 - np.log(PROCESS_SD))
    lp += np.sum(y * eta - np.exp(eta))
    return float(lp)


def gradient(theta, y):
    G, T = y.shape
    pop, beta, shared, innov = unpack(theta, G, T)
    r = y - np.exp(eta_of(theta, G, T))          # d lik / d eta[g, t]
    tail = np.cumsum(r[:, ::-1], axis=1)[:, ::-1]  # sum_{t >= s} r[g, t]
    d_pop = -pop + np.sum(beta - pop) / GROUP_SD ** 2
    d_beta = -(beta - pop) / GROUP_SD ** 2 + r.sum(axis=1)
    d_shared = -shared / SHARED_SD ** 2 + tail.sum(axis=0)
    d_innov = -innov / PROCESS_SD ** 2 + tail
    return np.concatenate([[d_pop], d_beta, d_shared, d_innov.ravel()])
