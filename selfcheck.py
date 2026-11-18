"""Offline smoke test: no network, no dataset download, finishes in well under 60 s.

Builds a small synthetic sample with the same schema as California Housing and
checks that every part of the pipeline trains, predicts with the right shapes,
is reproducible and rejects bad input. Prints `ok` and exits 0 on success;
prints the failure and exits 1 otherwise.

    python selfcheck.py
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src import gmm_entropy as ge  # noqa: E402
from src.data import FEATURES, make_synthetic, prepare, validate_raw  # noqa: E402
from src.gp import kernel_matrix  # noqa: E402
from src.maxent import meme_logdet_from_moments  # noqa: E402
from src.model import (METHODS, MeanPredictor, MemeGPRegressor, RidgeBaseline,  # noqa: E402
                       XGBoostBaseline, estimate_logdet, gp_grid_search)
from src.moments import chebyshev_moments, rademacher  # noqa: E402


def expect_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except (ValueError, RuntimeError):
        return
    raise AssertionError("expected an error from %s" % getattr(fn, "__name__", fn))


def check_data():
    df = make_synthetic(400, seed=0)
    assert list(df.columns[:8]) == FEATURES and df.shape == (400, 9)
    data = prepare(df)
    n_tr, n_va, n_te = len(data["y_train"]), len(data["y_val"]), len(data["y_test"])
    assert (n_tr, n_va, n_te) == (280, 40, 80), (n_tr, n_va, n_te)
    for split in ("train", "val", "test"):
        assert data["X_" + split].shape == (len(data["y_" + split]), 8)
    idx = np.concatenate([data["idx"][k] for k in ("train", "val", "test")])
    assert len(np.unique(idx)) == 400, "splits overlap"
    assert np.allclose(data["X_train"].mean(axis=0), 0.0, atol=1e-8)
    # same seed -> same split
    assert np.array_equal(prepare(df)["idx"]["test"], data["idx"]["test"])
    # bad input
    expect_error(validate_raw, np.zeros((2, 7)))
    expect_error(validate_raw, np.full((1, 8), np.nan))
    bad = np.ones((1, 8))
    bad[0, FEATURES.index("Population")] = -5.0
    expect_error(validate_raw, bad)
    expect_error(validate_raw, np.zeros((0, 8)))
    assert validate_raw(np.ones(8)).shape == (1, 8)
    return data


def check_logdet(X):
    n = 150
    for ell, tol in ((0.3, 0.3), (1.0, 0.3)):
        A = kernel_matrix(X[:n], ell, 1e-2)
        exact = estimate_logdet(A, "exact")
        for meth in METHODS:
            est, info = estimate_logdet(A, meth, m=20, d=10, seed=0, lam_min=1e-2,
                                        return_info=True)
            assert np.isfinite(est), (meth, est)
            if meth in ("meme", "slq"):
                assert ev.rel_error(est, exact) < tol, (meth, ell, est, exact)
            if meth == "meme":
                assert info["converged"], "noise-aware MEMe did not converge"
                assert np.isclose(info["mass"].sum(), 1.0) and np.all(info["mass"] >= 0)
        # reproducible
        assert estimate_logdet(A, "meme", m=20, d=10, seed=0, lam_min=1e-2) == \
            estimate_logdet(A, "meme", m=20, d=10, seed=0, lam_min=1e-2)
    # near-identity matrix (a spectral spike) and tiny inputs
    A = np.eye(60) + 1e-3 * np.ones((60, 60)) / 60
    est = estimate_logdet(A, "meme", m=30, d=5, lam_min=0.5)
    assert abs(est - estimate_logdet(A, "exact")) < 0.5, est
    assert np.isfinite(estimate_logdet(np.array([[2.0]]), "meme", m=1, d=1, lam_min=1.0))
    # moments: std output and the hard / soft solves
    Z = rademacher(n, 8, 0)
    A = kernel_matrix(X[:n], 0.5, 1e-2)
    mu, times, sd = chebyshev_moments(A, Z, 10, 1e-2, float(np.abs(A).sum(1).max()),
                                      return_std=True)
    assert mu.shape == sd.shape == times.shape == (11,) and mu[0] == 1.0 and sd[0] == 0.0
    ld_soft, _ = meme_logdet_from_moments(mu, n, 1e-2, float(np.abs(A).sum(1).max()),
                                          moment_std=sd)
    assert np.isfinite(ld_soft)
    # bad input
    expect_error(estimate_logdet, np.zeros((3, 4)), "exact")
    expect_error(estimate_logdet, np.eye(3), "meme", lam_min=None)
    expect_error(estimate_logdet, np.eye(3), "meme", lam_min=-1.0)
    expect_error(estimate_logdet, np.eye(3), "nope", lam_min=0.5)
    expect_error(estimate_logdet, np.full((2, 2), np.nan), "exact")


def check_models(data, tmp: Path):
    pre = data["pre"]
    X, y = data["X_train"][:120], data["y_train"][:120]
    grid = gp_grid_search(X, pre.y_to_z(y), [0.5, 2.0], [0.05, 0.3], m=15, d=8, seed=0)
    for name in ("exact", "meme", "chebyshev", "slq"):
        assert grid["lml"][name].shape == (2, 2) and np.all(np.isfinite(grid["lml"][name]))
    sel = grid["selected"]["meme"]
    model = MemeGPRegressor(pre, sel["ell"], sel["noise"]).fit_standardised(X, pre.y_to_z(y))
    mean, std = model.predict_standardised(data["X_test"])
    assert mean.shape == std.shape == data["y_test"].shape and np.all(std > 0)
    met = ev.regression_metrics(data["y_test"], mean, std)
    assert set(met) >= {"rmse", "mae", "r2", "nll", "coverage_95"}
    one = model.predict_one(data["X_raw_test"][0])
    assert one["lower_95"] < one["prediction"] < one["upper_95"]
    path = tmp / "model.joblib"
    model.save(path)
    m2 = MemeGPRegressor.load(path)
    assert np.allclose(m2.predict(data["X_raw_test"])[0], model.predict(data["X_raw_test"])[0])
    expect_error(MemeGPRegressor(pre, 1.0, 0.1).predict, data["X_raw_test"])

    mp = MeanPredictor().fit(data["X_train"], data["y_train"])
    rb = RidgeBaseline().fit(data["X_train"], data["y_train"])
    for mdl in (mp, rb):
        pm, ps = mdl.predict(data["X_test"])
        assert pm.shape == ps.shape == data["y_test"].shape
    grid_x = [{"max_depth": 3, "learning_rate": 0.2, "min_child_weight": 1}]
    xa = XGBoostBaseline(grid=grid_x, n_estimators=40).fit(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    xb = XGBoostBaseline(grid=grid_x, n_estimators=40).fit(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    pa, sa = xa.predict(data["X_test"])
    assert pa.shape == sa.shape == data["y_test"].shape
    assert np.array_equal(pa, xb.predict(data["X_test"])[0]), "XGBoost is not reproducible"
    assert xa.predict(data["X_test"][0])[0].shape == (1,)
    imp = xa.feature_importance()
    assert set(imp) == set(FEATURES) and abs(sum(imp.values()) - 1.0) < 1e-6
    expect_error(XGBoostBaseline(grid=grid_x).fit, data["X_train"][:1], data["y_train"][:1],
                 data["X_val"][:0], data["y_val"][:0])
    # one plot end to end
    ev.plot_xgboost(xa.search_, xa.val_curve_, imp, tmp / "x.png")
    assert (tmp / "x.png").stat().st_size > 1000
    return met


def check_gmm():
    rng = np.random.default_rng(0)
    mix = ge.random_mixture(rng)
    truth = ge.entropy_quadrature(mix)
    res = ge.entropy_meme(mix, 20)
    assert abs(res["entropy"] - truth) / abs(truth) < 0.05, (res["entropy"], truth)
    expect_error(ge.validate_mixture, {"weights": [1.0], "means": [0.0], "stds": [-1.0]})


def main() -> int:
    t0 = perf_counter()
    try:
        data = check_data()
        check_logdet(data["X_train"])
        with tempfile.TemporaryDirectory() as tmp:
            check_models(data, Path(tmp))
        check_gmm()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAILED")
        return 1
    secs = perf_counter() - t0
    if secs > 60:
        print("FAILED: took %.1f s (limit 60 s)" % secs)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
