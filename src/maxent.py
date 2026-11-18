"""Maximum-entropy density from moments: convex dual solved by damped Newton.

The density on x in [-1, 1] is p(x) = exp(-sum_k alpha_k phi_k(x)) and alpha
minimises the convex dual

    L(alpha) = integral exp(-sum_k alpha_k phi_k(x)) dx + sum_k alpha_k mu_k

whose gradient is mu_k - E_p[phi_k] and whose Hessian is E_p[phi_j phi_k]
(un-normalised expectations). phi_0 = 1 with mu_0 = 1 enforces normalisation.

Noisy moments. Hutchinson moments carry Monte-Carlo noise, and a noisy moment
vector is often not realisable by any positive density: the hard-constraint
dual then has no minimiser and Newton never converges. MEMe here treats each
moment k >= 1 as a soft constraint with Gaussian noise of standard deviation
s_k (the measured probe standard error). In the dual this adds the term
0.5 * sum_k s_k^2 alpha_k^2, i.e. a Gaussian prior on the Lagrange multipliers
(the constrained variational-inference view of the paper). The dual becomes
strongly convex, so the solve always converges; with s = 0 it reduces exactly
to the original hard-constraint MEMe (kept as the `meme_hard` ablation).
"""
from __future__ import annotations

import functools
import warnings

import numpy as np
from scipy import linalg


@functools.lru_cache(maxsize=8)
def make_grid(n_panels: int = 200, edge_decades: int = 10, per_decade: int = 4,
              order: int = 10):
    """Composite Gauss-Legendre grid on [-1, 1], graded toward the left edge.

    Panels are uniform in theta (x = -cos theta), which resolves Chebyshev
    polynomials of high degree at both ends, plus log-spaced panels next to
    x = -1, where log(lambda) is steep and spectral mass piles up.

    Returns (x, w, s) with s = (x + 1) / 2 kept separately so that values very
    close to the left edge keep full relative precision.
    """
    theta = np.linspace(0.0, np.pi, n_panels + 1)
    s_cheb = 0.5 * (1.0 - np.cos(theta))
    s_log = np.logspace(-edge_decades, 0.0, edge_decades * per_decade + 1)
    br = np.unique(np.concatenate([s_cheb, s_log, [0.0, 1.0]]))
    gx, gw = np.polynomial.legendre.leggauss(order)
    a, b = br[:-1], br[1:]
    half = 0.5 * (b - a)
    s = ((a + half)[:, None] + half[:, None] * gx[None, :]).ravel()
    ws = (half[:, None] * gw[None, :]).ravel()
    x = 2.0 * s - 1.0
    w = 2.0 * ws
    for arr in (x, w, s):
        arr.setflags(write=False)
    return x, w, s


def basis_matrix(kind: str, x: np.ndarray, s: np.ndarray, m: int) -> np.ndarray:
    """Columns phi_0..phi_m evaluated at the grid nodes."""
    k = np.arange(m + 1)
    if kind == "chebyshev":
        theta = np.arccos(np.clip(x, -1.0, 1.0))
        return np.cos(theta[:, None] * k[None, :])
    if kind == "power":
        return s[:, None] ** k[None, :]
    raise ValueError("unknown basis %r" % (kind,))


def solve_maxent(mu, Phi, w, max_iter: int = 200, tol: float = 1e-8,
                 jitter: float = 1e-8, alpha0=None, reg=None) -> dict:
    """Minimise the MaxEnt dual. Never raises on hard problems: it returns the
    best iterate together with `converged`, `n_iter` and the gradient history.

    alpha0 optionally warm-starts the solve (shorter vectors are zero-padded,
    which leaves the density unchanged for nested bases).

    reg optionally holds per-moment noise variances s_k^2 (reg[0] must be 0):
    the soft-constraint penalty 0.5 * sum reg_k alpha_k^2 is added to the dual."""
    mu = np.asarray(mu, dtype=float)
    if not np.all(np.isfinite(mu)):
        raise ValueError("moments contain NaN or infinite values")
    K = mu.shape[0]
    if Phi.shape[1] != K:
        raise ValueError("basis has %d columns but %d moments given" % (Phi.shape[1], K))
    reg = np.zeros(K) if reg is None else np.asarray(reg, dtype=float)[:K].copy()
    if reg.shape[0] != K or np.any(reg < 0) or not np.all(np.isfinite(reg)):
        raise ValueError("reg must hold %d finite non-negative variances" % K)
    reg[0] = 0.0
    alpha = np.zeros(K)
    alpha[0] = np.log(np.sum(w))  # start from the uniform density
    if alpha0 is not None:
        a0 = np.asarray(alpha0, dtype=float)[:K]
        trial = np.zeros(K)
        trial[: len(a0)] = a0
        e0 = -(Phi @ trial)
        if np.all(np.isfinite(e0)) and e0.max() < 690.0:
            alpha = trial

    def objective(a):
        e = -(Phi @ a)
        if not np.all(np.isfinite(e)) or e.max() > 690.0:
            return np.inf, None
        p = np.exp(e)
        return float(w @ p + a @ mu + 0.5 * np.sum(reg * a * a)), p

    L, p = objective(alpha)
    lam = jitter
    eye = np.eye(K)
    history = []
    converged = False
    reason = "max_iter"
    small_steps = 0
    n_iter = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n_iter in range(1, max_iter + 1):
            wp = w * p
            g = mu - Phi.T @ wp + reg * alpha
            gn = float(np.max(np.abs(g)))
            history.append(gn)
            if gn < tol:
                converged, reason = True, "gradient"
                break
            H = (Phi * wp[:, None]).T @ Phi + np.diag(reg)
            accepted = False
            while lam <= 1e10:
                try:
                    step = linalg.solve(H + lam * eye, -g, assume_a="pos")
                except (linalg.LinAlgError, ValueError):
                    lam *= 10.0
                    continue
                slope = float(g @ step)
                if not np.isfinite(slope) or slope >= 0:
                    lam *= 10.0
                    continue
                t = 1.0
                for _ in range(12):
                    L_new, p_new = objective(alpha + t * step)
                    if np.isfinite(L_new) and L_new <= L + 1e-4 * t * slope:
                        accepted = True
                        break
                    t *= 0.5
                if accepted:
                    break
                lam *= 10.0
            if not accepted:
                reason = "stalled"
                break
            decrease = L - L_new
            alpha = alpha + t * step
            L, p = L_new, p_new
            lam = max(jitter, lam / 10.0)
            small_steps = small_steps + 1 if decrease < 1e-14 * max(1.0, abs(L)) else 0
            if small_steps >= 5:
                reason = "stalled"
                break
    wp = w * p
    if not converged:
        gn = float(np.max(np.abs(mu - Phi.T @ wp + reg * alpha)))
        if len(history) == n_iter:  # gradient of the final iterate
            history.append(gn)
        converged = gn < tol
    Z = float(np.sum(wp))
    q = wp / Z  # normalised probability mass per node
    entropy = float(q @ (Phi @ alpha) + np.log(Z))
    return {
        "alpha": alpha, "mass": q, "density": p / Z, "entropy": entropy,
        "converged": bool(converged), "n_iter": int(n_iter), "reason": reason,
        "grad_norm": float(history[-1]), "grad_history": history,
    }


def solve_maxent_continuation(mu, Phi, w, max_iter: int = 200, stage_iter: int = 100,
                              tol: float = 1e-8, reg=None) -> dict:
    """Solve with moment continuation: fit 2, 4, 8, ... moments first and
    warm-start each larger problem from the previous multipliers.

    A cold Newton start from the uniform density is slow when the target is a
    sharp spike (nearly degenerate spectrum); the low-order solves locate the
    spike cheaply. `n_iter` and `grad_history` cover all stages.
    """
    m = len(mu) - 1
    stages = []
    k = 2
    while k < m:
        stages.append(k)
        k *= 2
    stages.append(m)
    alpha, total, history = None, 0, []
    for k in stages:
        last = k == m
        res = solve_maxent(mu[: k + 1], Phi[:, : k + 1], w,
                           max_iter=max_iter if last else stage_iter, tol=tol, alpha0=alpha,
                           reg=None if reg is None else np.asarray(reg)[: k + 1])
        alpha = res["alpha"]
        total += res["n_iter"]
        history.extend(res["grad_history"])
    res["n_iter"] = total
    res["grad_history"] = history
    res["stages"] = stages
    return res


def meme_logdet_from_moments(mu, n: int, lam_min: float, lam_max: float,
                             basis: str = "chebyshev", max_iter: int = 200,
                             grid=None, continuation: bool = True, moment_std=None):
    """log det A = n * E_p[log lambda] with p the MaxEnt spectral density.

    moment_std (standard error of each moment) switches on the soft-constraint
    (noise-aware) dual; None gives the original hard-constraint MaxEnt.

    Returns (logdet, result) where result also holds the grid and the density.
    """
    x, w, s = grid if grid is not None else make_grid()
    Phi = basis_matrix(basis, x, s, len(mu) - 1)
    reg = None
    if moment_std is not None:
        reg = np.asarray(moment_std, dtype=float)[: len(mu)] ** 2
    if continuation:
        res = solve_maxent_continuation(mu, Phi, w, max_iter=max_iter, reg=reg)
    else:
        res = solve_maxent(mu, Phi, w, max_iter=max_iter, reg=reg)
    lam = lam_min + s * (lam_max - lam_min)
    res["lam"] = lam
    res["x"] = x
    res["mean_log"] = float(res["mass"] @ np.log(lam))
    return n * res["mean_log"], res
