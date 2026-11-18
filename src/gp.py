"""RBF kernel and exact Gaussian-process regression (Cholesky)."""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist


def rbf_kernel(X1, X2, ell: float, signal_var: float = 1.0) -> np.ndarray:
    if ell <= 0:
        raise ValueError("length-scale must be positive")
    d2 = cdist(X1, X2, metric="sqeuclidean")
    return signal_var * np.exp(-0.5 * d2 / (ell * ell))


def kernel_matrix(X, ell: float, noise: float, signal_var: float = 1.0) -> np.ndarray:
    """A = K_rbf(X, X) + noise * I (symmetric positive definite for noise > 0)."""
    if noise <= 0:
        raise ValueError("noise variance must be positive")
    A = rbf_kernel(X, X, ell, signal_var)
    A[np.diag_indices_from(A)] += noise
    return A


def log_marginal_likelihood(datafit: float, logdet: float, n: int) -> float:
    """GP LML = -0.5 y^T A^-1 y - 0.5 log det A - n/2 log(2 pi)."""
    return float(-0.5 * datafit - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi))


class ExactGP:
    """Zero-mean GP with an RBF kernel on standardised inputs and targets."""

    def __init__(self, ell: float, noise: float, signal_var: float = 1.0):
        self.ell = float(ell)
        self.noise = float(noise)
        self.signal_var = float(signal_var)
        self.X = None
        self.L = None
        self.alpha = None

    def fit(self, X, z) -> "ExactGP":
        self.X = np.ascontiguousarray(X, dtype=float)
        z = np.asarray(z, dtype=float)
        A = kernel_matrix(self.X, self.ell, self.noise, self.signal_var)
        self.L = linalg.cholesky(A, lower=True, check_finite=False)
        self.alpha = linalg.cho_solve((self.L, True), z, check_finite=False)
        self.datafit = float(z @ self.alpha)
        self.logdet = float(2.0 * np.sum(np.log(np.diag(self.L))))
        return self

    def predict(self, Xs):
        """Latent mean and predictive variance (latent variance + noise), z units."""
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Ks = rbf_kernel(Xs, self.X, self.ell, self.signal_var)
        mean = Ks @ self.alpha
        v = linalg.solve_triangular(self.L, Ks.T, lower=True, check_finite=False)
        var = self.signal_var - np.sum(v * v, axis=0)
        var = np.maximum(var, 0.0) + self.noise
        return mean, var
