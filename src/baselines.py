"""Baseline log-determinant estimators.

- stochastic Chebyshev expansion of log (Han, Malioutov, Shin 2015)
- stochastic Lanczos quadrature (Ubaru, Chen, Saad 2017)
- exact Cholesky (ground truth)
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import linalg


def exact_logdet(A: np.ndarray) -> float:
    L = linalg.cholesky(A, lower=True, check_finite=False)
    return float(2.0 * np.sum(np.log(np.diag(L))))


def chebyshev_log_coeffs(m: int, lam_min: float, lam_max: float) -> np.ndarray:
    """Chebyshev interpolation coefficients c_0..c_m of f(x) = log(c + h x) on [-1, 1]."""
    N = m + 1
    j = np.arange(N)
    theta = np.pi * (j + 0.5) / N
    c = 0.5 * (lam_max + lam_min)
    h = 0.5 * (lam_max - lam_min)
    f = np.log(c + h * np.cos(theta))
    k = np.arange(N)
    coef = (2.0 / N) * (np.cos(k[:, None] * theta[None, :]) @ f)
    coef[0] *= 0.5
    return coef


def chebyshev_logdet_from_moments(mu, n: int, lam_min: float, lam_max: float) -> float:
    """log det A ~= n * sum_k c_k mu_k with the same Chebyshev moments MEMe uses."""
    coef = chebyshev_log_coeffs(len(mu) - 1, lam_min, lam_max)
    return float(n * (coef @ np.asarray(mu)))


def lanczos_block(A, Z, m: int):
    """m Lanczos steps for every probe column of Z (full re-orthogonalisation).

    Returns alphas (m, d), betas (m, d), lengths (d,) and cumulative times (m + 1,).
    lengths[c] < m marks an early breakdown (invariant subspace found) for probe c.
    """
    n, d = Z.shape
    m = int(min(m, n))
    Q = np.zeros((m, n, d))
    alphas = np.zeros((m, d))
    betas = np.zeros((m, d))
    lengths = np.full(d, m, dtype=int)
    active = np.ones(d, dtype=bool)
    times = np.zeros(m + 1)
    t0 = perf_counter()
    q = Z / np.linalg.norm(Z, axis=0)
    q_prev = np.zeros_like(q)
    beta_prev = np.zeros(d)
    for j in range(m):
        Q[j] = q
        wv = A @ q
        a = np.sum(q * wv, axis=0)
        alphas[j] = a
        wv = wv - a * q - beta_prev * q_prev
        Qj = Q[: j + 1]
        for _ in range(2):
            coef = np.einsum("jnc,nc->jc", Qj, wv)
            wv = wv - np.einsum("jnc,jc->nc", Qj, coef)
        b = np.linalg.norm(wv, axis=0)
        betas[j] = b
        broke = active & (b < 1e-9 * (np.abs(a) + beta_prev + 1e-300))
        lengths[broke] = j + 1
        active &= ~broke
        q_prev, beta_prev = q, b
        q = wv / np.where(b > 0, b, 1.0)
        q[:, ~active] = 0.0
        times[j + 1] = perf_counter() - t0
    return alphas, betas, lengths, times


def slq_logdet(alphas, betas, lengths, m: int, n: int) -> float:
    """Gauss quadrature of log from the leading m x m Lanczos tridiagonals."""
    d = alphas.shape[1]
    total = 0.0
    for c in range(d):
        k = int(min(m, lengths[c], alphas.shape[0]))
        if k <= 1:
            total += float(np.log(max(alphas[0, c], 1e-300)))
            continue
        theta, vecs = linalg.eigh_tridiagonal(alphas[:k, c], betas[: k - 1, c])
        tau2 = vecs[0, :] ** 2
        total += float(tau2 @ np.log(np.maximum(theta, 1e-300)))
    return n * total / d
