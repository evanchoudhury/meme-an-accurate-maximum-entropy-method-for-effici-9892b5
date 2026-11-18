"""Stochastic spectral moments from matrix-vector products (Hutchinson)."""
from __future__ import annotations

from time import perf_counter

import numpy as np


def rademacher(n: int, d: int, seed: int) -> np.ndarray:
    """n x d matrix of +1/-1 probe vectors."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n, d)).astype(float) * 2.0 - 1.0


def gershgorin_upper(A: np.ndarray) -> float:
    """Upper bound on the largest eigenvalue: maximum absolute row sum."""
    return float(np.abs(A).sum(axis=1).max())


def check_bounds(lam_min: float, lam_max: float):
    if not (np.isfinite(lam_min) and np.isfinite(lam_max)):
        raise ValueError("spectral bounds must be finite")
    if lam_min <= 0:
        raise ValueError("lam_min must be positive (the matrix must be positive definite)")
    if lam_max <= lam_min:
        # degenerate (e.g. a 1x1 matrix): widen slightly so the affine map is defined
        lam_max = lam_min * (1.0 + 1e-6) + 1e-12
    return float(lam_min), float(lam_max)


def _std_err(per: np.ndarray) -> np.ndarray:
    """Standard error of the probe average for each moment (rows = moments)."""
    d = per.shape[1]
    if d < 2:
        return np.zeros(per.shape[0])
    return per.std(axis=1, ddof=1) / np.sqrt(d)


def chebyshev_moments(A, Z, m: int, lam_min: float, lam_max: float,
                      return_std: bool = False):
    """Hutchinson estimates of mu_k = (1/n) tr T_k(B), k = 0..m.

    B = (A - c I) / h maps the spectrum [lam_min, lam_max] onto [-1, 1].
    Uses the three-term recurrence, i.e. m block matrix-vector products.
    Returns (mu, times) where times[k] is the cumulative seconds spent to
    obtain moments 0..k; with return_std=True also the standard error of each
    moment across the d probes (used as the moment-noise level by MEMe).
    """
    lam_min, lam_max = check_bounds(lam_min, lam_max)
    n, d = Z.shape
    c = 0.5 * (lam_max + lam_min)
    h = 0.5 * (lam_max - lam_min)
    per = np.ones((m + 1, d))
    times = np.zeros(m + 1)
    t0 = perf_counter()
    if m >= 1:
        v_prev = Z
        v = (A @ Z - c * Z) / h
        per[1] = np.sum(Z * v, axis=0) / n
        times[1] = perf_counter() - t0
        for k in range(2, m + 1):
            v_next = 2.0 * (A @ v - c * v) / h - v_prev
            per[k] = np.sum(Z * v_next, axis=0) / n
            v_prev, v = v, v_next
            times[k] = perf_counter() - t0
    mu = per.mean(axis=1)
    mu[0] = 1.0
    return (mu, times, _std_err(per)) if return_std else (mu, times)


def power_moments(A, Z, m: int, lam_min: float, lam_max: float, return_std: bool = False):
    """Hutchinson estimates of (1/n) tr S^k, k = 0..m, with S = (A - lam_min I) / range.

    S has its spectrum in [0, 1]. These raw power moments are the ablation that
    shows why orthogonal (Chebyshev) moments are needed.
    """
    lam_min, lam_max = check_bounds(lam_min, lam_max)
    n, d = Z.shape
    rng_ = lam_max - lam_min
    per = np.ones((m + 1, d))
    times = np.zeros(m + 1)
    t0 = perf_counter()
    v = Z
    for k in range(1, m + 1):
        v = (A @ v - lam_min * v) / rng_
        per[k] = np.sum(Z * v, axis=0) / n
        times[k] = perf_counter() - t0
    mu = per.mean(axis=1)
    mu[0] = 1.0
    return (mu, times, _std_err(per)) if return_std else (mu, times)
