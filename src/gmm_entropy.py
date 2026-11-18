"""Differential entropy of 1-D Gaussian mixtures: MEMe vs simple estimators."""
from __future__ import annotations

import numpy as np

from .maxent import basis_matrix, make_grid, solve_maxent


def random_mixture(rng) -> dict:
    k = int(rng.integers(2, 6))
    return {
        "weights": rng.dirichlet(np.ones(k)),
        "means": rng.uniform(-4.0, 4.0, k),
        "stds": rng.uniform(0.3, 1.5, k),
    }


def validate_mixture(mix: dict) -> dict:
    w = np.asarray(mix["weights"], dtype=float).ravel()
    mu = np.asarray(mix["means"], dtype=float).ravel()
    sd = np.asarray(mix["stds"], dtype=float).ravel()
    if not (len(w) == len(mu) == len(sd)) or len(w) == 0:
        raise ValueError("weights, means and stds must have the same non-zero length")
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(sd))):
        raise ValueError("mixture parameters must be finite")
    if np.any(w < 0) or w.sum() <= 0 or np.any(sd <= 0):
        raise ValueError("weights must be non-negative and stds positive")
    return {"weights": w / w.sum(), "means": mu, "stds": sd}


def support(mix: dict, k: float = 7.0):
    lo = float(np.min(mix["means"] - k * mix["stds"]))
    hi = float(np.max(mix["means"] + k * mix["stds"]))
    return lo, hi


def pdf(mix: dict, x) -> np.ndarray:
    x = np.asarray(x, dtype=float)[..., None]
    z = (x - mix["means"]) / mix["stds"]
    comp = np.exp(-0.5 * z * z) / (mix["stds"] * np.sqrt(2.0 * np.pi))
    return comp @ mix["weights"]


def entropy_quadrature(mix: dict, n_points: int = 200001) -> float:
    """Ground truth: dense trapezoid quadrature of -p log p."""
    lo, hi = support(mix, 10.0)
    x = np.linspace(lo, hi, n_points)
    p = pdf(mix, x)
    f = np.where(p > 0, -p * np.log(np.maximum(p, 1e-300)), 0.0)
    return float(np.sum(0.5 * (f[1:] + f[:-1])) * (x[1] - x[0]))


def chebyshev_moments(mix: dict, m: int, lo: float, hi: float) -> np.ndarray:
    """E[T_k(s(x))], k = 0..m, by Gauss-Hermite quadrature per component
    (exact for these polynomial integrands)."""
    t, wt = np.polynomial.hermite.hermgauss(80)
    wt = wt / np.sqrt(np.pi)
    mu = np.zeros(m + 1)
    for wk, mk, sk in zip(mix["weights"], mix["means"], mix["stds"]):
        x = mk + np.sqrt(2.0) * sk * t
        s = (2.0 * x - lo - hi) / (hi - lo)
        V = np.polynomial.chebyshev.chebvander(s, m)
        mu += wk * (wt @ V)
    mu[0] = 1.0
    return mu


def entropy_meme(mix: dict, m: int, max_iter: int = 300) -> dict:
    lo, hi = support(mix)
    mu = chebyshev_moments(mix, m, lo, hi)
    x, w, s = make_grid()
    res = solve_maxent(mu, basis_matrix("chebyshev", x, s, m), w, max_iter=max_iter)
    half = 0.5 * (hi - lo)
    return {
        "entropy": res["entropy"] + float(np.log(half)),
        "converged": res["converged"], "n_iter": res["n_iter"],
        "x": lo + (x + 1.0) * half, "density": res["density"] / half,
    }


def entropy_monte_carlo(mix: dict, n_samples: int, rng) -> float:
    comp = rng.choice(len(mix["weights"]), size=n_samples, p=mix["weights"])
    x = rng.normal(mix["means"][comp], mix["stds"][comp])
    return float(-np.mean(np.log(pdf(mix, x))))


def entropy_moment_matching(mix: dict) -> float:
    mean = float(mix["weights"] @ mix["means"])
    var = float(mix["weights"] @ (mix["stds"] ** 2 + mix["means"] ** 2) - mean ** 2)
    return float(0.5 * np.log(2.0 * np.pi * np.e * var))
