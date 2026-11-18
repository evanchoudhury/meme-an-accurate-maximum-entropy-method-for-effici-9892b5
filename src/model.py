"""Baseline and main model.

Log-determinant estimation
    main model : MEMe (MaxEnt spectral density from stochastic Chebyshev moments,
                 soft moment constraints weighted by the measured probe noise)
    ablations  : MEMe with hard constraints (as in the paper), power-moment MaxEnt
    baseline   : stochastic Chebyshev expansion of log, same matrix-vector budget
    reference  : stochastic Lanczos quadrature, power-moment MaxEnt, exact Cholesky

Prediction service
    main model : MemeGPRegressor, an exact GP whose hyper-parameters were chosen by a
                 marginal likelihood that uses a pluggable log-det estimator
    baselines  : MeanPredictor, RidgeBaseline and XGBoostBaseline (gradient-boosted trees)
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV

from . import baselines as bl
from .data import FEATURES, Preprocessor
from .gp import ExactGP, kernel_matrix, log_marginal_likelihood
from .maxent import meme_logdet_from_moments
from .moments import (chebyshev_moments, check_bounds, gershgorin_upper,
                      power_moments, rademacher)

METHODS = ["meme", "meme_hard", "chebyshev", "slq", "maxent_power", "exact"]
METHOD_LABELS = {
    "meme": "MEMe (Chebyshev MaxEnt, noise-aware)",
    "meme_hard": "MEMe, hard constraints (paper)",
    "chebyshev": "Chebyshev expansion",
    "slq": "Stochastic Lanczos (SLQ)",
    "maxent_power": "MaxEnt, power moments",
    "exact": "Exact (Cholesky)",
}


def _check_matrix(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a non-empty square matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("A contains NaN or infinite values")
    return A


def estimate_logdet(A, method: str = "meme", m: int = 30, d: int = 30, seed: int = 0,
                    lam_min: float = None, lam_max: float = None,
                    return_info: bool = False):
    """Estimate log det(A) for a symmetric positive-definite matrix.

    lam_min must be a positive lower bound on the spectrum (for A = K + s2 I
    with K positive semi-definite, pass s2). lam_max defaults to the Gershgorin
    bound. Every stochastic method spends m * d matrix-vector products.
    """
    A = _check_matrix(A)
    n = A.shape[0]
    if method not in METHODS:
        raise ValueError("unknown method %r, choose from %s" % (method, METHODS))
    if m < 1 or d < 1:
        raise ValueError("m and d must be at least 1")
    t0 = perf_counter()
    info = {"method": method, "n": n, "m": int(m), "d": int(d)}
    if method == "exact":
        est = bl.exact_logdet(A)
    else:
        if lam_min is None:
            raise ValueError("lam_min (a positive lower bound on the spectrum) is required")
        if lam_max is None:
            lam_max = gershgorin_upper(A)
        lam_min, lam_max = check_bounds(lam_min, lam_max)
        info.update(lam_min=lam_min, lam_max=lam_max)
        Z = rademacher(n, d, seed)
        if method == "slq":
            al, be, ln, _ = bl.lanczos_block(A, Z, m)
            est = bl.slq_logdet(al, be, ln, m, n)
        elif method == "chebyshev":
            mu, _ = chebyshev_moments(A, Z, m, lam_min, lam_max)
            est = bl.chebyshev_logdet_from_moments(mu, n, lam_min, lam_max)
        else:
            basis = "power" if method == "maxent_power" else "chebyshev"
            mom = power_moments if method == "maxent_power" else chebyshev_moments
            mu, _, sd = mom(A, Z, m, lam_min, lam_max, return_std=True)
            est, res = meme_logdet_from_moments(
                mu, n, lam_min, lam_max, basis=basis,
                moment_std=None if method == "meme_hard" else sd)
            info.update(converged=res["converged"], n_iter=res["n_iter"],
                        grad_norm=res["grad_norm"], entropy=res["entropy"])
            if return_info:
                info.update(lam=res["lam"], mass=res["mass"], density=res["density"])
    info["seconds"] = perf_counter() - t0
    est = float(est)
    return (est, info) if return_info else est


def gp_grid_search(X, z, ells, noises, m: int = 30, d: int = 30, seed: int = 0,
                   methods=("meme", "chebyshev", "slq"), progress=None) -> dict:
    """Log marginal likelihood on an (ell, noise) grid, once per log-det estimator.

    The data-fit term y^T A^-1 y is exact (Cholesky), so the surfaces differ only
    through the log-det term. Returns the surfaces, log-dets and each argmax.
    """
    n = X.shape[0]
    names = ["exact"] + list(methods)
    shape = (len(ells), len(noises))
    logdets = {k: np.zeros(shape) for k in names}
    lml = {k: np.zeros(shape) for k in names}
    seconds = {k: 0.0 for k in names}
    Z = rademacher(n, d, seed)
    for i, ell in enumerate(ells):
        for j, s2 in enumerate(noises):
            A = kernel_matrix(X, ell, s2)
            t0 = perf_counter()
            gp = ExactGP(ell, s2).fit(X, z)
            seconds["exact"] += perf_counter() - t0
            logdets["exact"][i, j] = gp.logdet
            lam_min, lam_max = check_bounds(s2, gershgorin_upper(A))
            mu = None
            for name in methods:
                t0 = perf_counter()
                if name in ("meme", "chebyshev"):
                    if mu is None:
                        mu, _, sd = chebyshev_moments(A, Z, m, lam_min, lam_max,
                                                      return_std=True)
                    if name == "meme":
                        est, _ = meme_logdet_from_moments(mu, n, lam_min, lam_max,
                                                          moment_std=sd)
                    else:
                        est = bl.chebyshev_logdet_from_moments(mu, n, lam_min, lam_max)
                elif name == "slq":
                    al, be, ln, _ = bl.lanczos_block(A, Z, m)
                    est = bl.slq_logdet(al, be, ln, m, n)
                else:
                    raise ValueError("unsupported grid method %r" % (name,))
                seconds[name] += perf_counter() - t0
                logdets[name][i, j] = est
            for name in names:
                lml[name][i, j] = log_marginal_likelihood(gp.datafit, logdets[name][i, j], n)
        if progress is not None:
            progress(i, ell)
    selected = {}
    for name in names:
        i, j = np.unravel_index(int(np.argmax(lml[name])), shape)
        selected[name] = {"ell": float(ells[i]), "noise": float(noises[j]),
                          "i": int(i), "j": int(j), "lml": float(lml[name][i, j])}
    return {"ells": np.asarray(ells, dtype=float), "noises": np.asarray(noises, dtype=float),
            "logdets": logdets, "lml": lml, "selected": selected, "seconds": seconds}


class MemeGPRegressor:
    """Served model: preprocessing + exact GP with fixed (selected) hyper-parameters.

    Inputs and outputs are in original units (target: median house value, $100k).
    """

    def __init__(self, pre: Preprocessor, ell: float, noise: float, meta: dict = None):
        self.pre = pre
        self.ell = float(ell)
        self.noise = float(noise)
        self.meta = dict(meta or {})
        self.gp = None

    def fit_standardised(self, Xs, z) -> "MemeGPRegressor":
        """Fit on already standardised features / targets."""
        self.gp = ExactGP(self.ell, self.noise).fit(Xs, z)
        self._z = np.asarray(z, dtype=float)
        return self

    def fit(self, X_raw, y) -> "MemeGPRegressor":
        return self.fit_standardised(self.pre.transform(X_raw), self.pre.y_to_z(y))

    def predict_standardised(self, Xs):
        mean_z, var_z = self.gp.predict(Xs)
        return self.pre.z_to_y(mean_z), np.sqrt(var_z) * self.pre.y_std

    def predict(self, X_raw):
        """Return (mean, std) in original units for raw feature rows (k, 8)."""
        if self.gp is None:
            raise RuntimeError("model is not fitted")
        return self.predict_standardised(self.pre.transform(X_raw))

    def predict_one(self, row) -> dict:
        mean, std = self.predict(row)
        m, s = float(mean[0]), float(std[0])
        return {"prediction": m, "std": s, "lower_95": m - 1.96 * s, "upper_95": m + 1.96 * s}

    def to_state(self) -> dict:
        return {"format": "meme-gp-v1", "features": list(FEATURES),
                "pre": self.pre.to_state(), "ell": self.ell, "noise": self.noise,
                "X_train": self.gp.X, "z_train": self._z, "meta": self.meta}

    @classmethod
    def from_state(cls, state: dict) -> "MemeGPRegressor":
        if state.get("format") != "meme-gp-v1":
            raise ValueError("unrecognised model file")
        model = cls(Preprocessor.from_state(state["pre"]), state["ell"], state["noise"],
                    state.get("meta"))
        # the Cholesky factor is rebuilt on load: it is deterministic and keeps the file small
        return model.fit_standardised(state["X_train"], state["z_train"])

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.to_state(), path, compress=3)

    @classmethod
    def load(cls, path) -> "MemeGPRegressor":
        return cls.from_state(joblib.load(Path(path)))


class MeanPredictor:
    """Predicts the training mean, with the training std as its uncertainty."""

    def fit(self, Xs, y) -> "MeanPredictor":
        y = np.asarray(y, dtype=float)
        self.mean_, self.std_ = float(y.mean()), float(y.std())
        return self

    def predict(self, Xs):
        k = np.atleast_2d(Xs).shape[0]
        return np.full(k, self.mean_), np.full(k, self.std_)


class XGBoostBaseline:
    """Gradient-boosted trees (xgboost.XGBRegressor) on the standardised features.

    Hyper-parameters are chosen on the validation split only (small grid, early
    stopping on validation RMSE). The test split is never touched here. The
    predictive std is the validation residual std, so NLL and coverage can be
    compared with the GP.
    """

    GRID = [
        {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 1},
        {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 1},
        {"max_depth": 8, "learning_rate": 0.05, "min_child_weight": 3},
    ]

    def __init__(self, grid=None, n_estimators: int = 1500, seed: int = 0, n_jobs: int = 4):
        self.grid = list(self.GRID if grid is None else grid)
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)

    def _make(self, params, n_estimators, early_stopping=None):
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=n_estimators, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, tree_method="hist", random_state=self.seed,
            n_jobs=self.n_jobs, early_stopping_rounds=early_stopping, **params)

    def fit(self, Xs, y, X_val, y_val) -> "XGBoostBaseline":
        Xs, y = np.asarray(Xs, float), np.asarray(y, float)
        X_val, y_val = np.asarray(X_val, float), np.asarray(y_val, float)
        if len(y) < 2 or len(y_val) < 1:
            raise ValueError("XGBoost needs at least 2 training rows and 1 validation row")
        self.search_ = []
        best = None
        for params in self.grid:
            mdl = self._make(params, self.n_estimators, early_stopping=50)
            mdl.fit(Xs, y, eval_set=[(X_val, y_val)], verbose=False)
            hist = mdl.evals_result()["validation_0"]["rmse"]
            n_best = int(np.argmin(hist)) + 1
            val_rmse = float(np.min(hist))
            self.search_.append(dict(params, best_n_estimators=n_best, val_rmse=val_rmse))
            if best is None or val_rmse < best[0]:
                best = (val_rmse, params, n_best, hist)
        self.val_rmse_, self.params_, self.best_n_, self.val_curve_ = best
        self.model_ = self._make(self.params_, self.best_n_).fit(Xs, y, verbose=False)
        self.std_ = float(np.std(y_val - self.model_.predict(X_val)))
        self.std_ = max(self.std_, 1e-6)
        return self

    def predict(self, Xs):
        mean = self.model_.predict(np.atleast_2d(np.asarray(Xs, float))).astype(float)
        return mean, np.full(mean.shape[0], self.std_)

    def feature_importance(self) -> dict:
        gain = self.model_.get_booster().get_score(importance_type="gain")
        vals = np.array([gain.get("f%d" % i, 0.0) for i in range(len(FEATURES))])
        if vals.sum() > 0:
            vals = vals / vals.sum()
        return {f: float(v) for f, v in zip(FEATURES, vals)}


class RidgeBaseline:
    """Ridge regression on the standardised features, alpha chosen by RidgeCV."""

    def __init__(self, alphas=None):
        self.alphas = np.logspace(-3, 3, 13) if alphas is None else np.asarray(alphas)

    def fit(self, Xs, y) -> "RidgeBaseline":
        self.model_ = RidgeCV(alphas=self.alphas).fit(Xs, y)
        self.alpha_ = float(self.model_.alpha_)
        self.std_ = float(np.std(np.asarray(y) - self.model_.predict(Xs)))
        return self

    def predict(self, Xs):
        mean = self.model_.predict(np.atleast_2d(Xs))
        return mean, np.full(mean.shape[0], self.std_)
