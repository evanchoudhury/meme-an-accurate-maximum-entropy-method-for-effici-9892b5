"""Metrics and plots. Every figure is written as a PNG with the Agg backend."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from .model import METHOD_LABELS  # noqa: E402

COLORS = {
    "meme": "#d62728", "chebyshev": "#1f77b4", "slq": "#2ca02c",
    "maxent_power": "#9467bd", "exact": "#111111", "meme_hard": "#e377c2",
    "mean": "#8c8c8c", "ridge": "#ff7f0e", "xgboost": "#17becf",
}
MARKERS = {"meme": "o", "meme_hard": "v", "chebyshev": "s", "slq": "^", "maxent_power": "D",
           "exact": "*"}
EST_METHODS = ["meme", "meme_hard", "chebyshev", "slq", "maxent_power"]
SHORT = {"meme": "MEMe", "meme_hard": "MEMe-hard", "maxent_power": "power"}


# ----------------------------------------------------------------- metrics
def rel_error(estimate: float, exact: float) -> float:
    return float(abs(estimate - exact) / max(abs(exact), 1e-300))


def regression_metrics(y, mean, std=None) -> dict:
    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    if y.shape != mean.shape:
        raise ValueError("shape mismatch: %s vs %s" % (y.shape, mean.shape))
    err = y - mean
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else 0.0,
    }
    if std is not None:
        std = np.maximum(np.asarray(std, dtype=float), 1e-12)
        out["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * std ** 2) + 0.5 * (err / std) ** 2))
        out["coverage_95"] = float(np.mean(np.abs(err) <= 1.96 * std))
    return out


def coverage_curve(y, mean, std, levels=None):
    """Empirical coverage of central Gaussian intervals at each nominal level."""
    levels = np.linspace(0.05, 0.95, 19) if levels is None else np.asarray(levels)
    zscore = np.abs(np.asarray(y) - np.asarray(mean)) / np.maximum(np.asarray(std), 1e-12)
    emp = np.array([np.mean(zscore <= stats.norm.ppf(0.5 + lv / 2.0)) for lv in levels])
    return levels, emp


def interval_confidence(std, half_width: float = 0.5) -> np.ndarray:
    """P(|truth - prediction| <= half_width) under the Gaussian predictive."""
    std = np.maximum(np.asarray(std, dtype=float), 1e-12)
    return 2.0 * stats.norm.cdf(half_width / std) - 1.0


# ------------------------------------------------------------------- plots
def _save(fig, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_error_vs_m(bench, path):
    ells = sorted(bench["ell"].unique())
    ncol = min(3, len(ells))
    nrow = int(np.ceil(len(ells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.8 * nrow), squeeze=False,
                             constrained_layout=True)
    for ax, ell in zip(axes.ravel(), ells):
        sub = bench[bench["ell"] == ell]
        for meth in EST_METHODS:
            g = sub[sub["method"] == meth].groupby("m")["rel_err"].median()
            ax.plot(g.index, np.maximum(g.values, 1e-7), marker=MARKERS[meth],
                    color=COLORS[meth], label=METHOD_LABELS[meth], lw=1.6, ms=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("ell = %.3g, cond = %.1e" % (ell, sub["cond"].iloc[0]), fontsize=10)
        ax.set_xlabel("moments / Lanczos steps m")
        ax.set_ylabel("abs. relative error")
        ax.grid(alpha=0.3, which="both")
    for ax in axes.ravel()[len(ells):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Log-det relative error vs m (median over probe seeds, d = %d probes)"
                 % int(bench["d"].iloc[0]))
    return _save(fig, path)


def plot_error_vs_cond(bench, m, path):
    sub = bench[bench["m"] == m]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for meth in EST_METHODS:
        g = sub[sub["method"] == meth].groupby("cond")["rel_err"].median().sort_index()
        ax.plot(g.index, np.maximum(g.values, 1e-7), marker=MARKERS[meth],
                color=COLORS[meth], label=METHOD_LABELS[meth], lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("condition number of A = K + s2 I")
    ax.set_ylabel("abs. relative log-det error")
    ax.set_title("Error vs conditioning at m = %d (equal matrix-vector budget)" % m)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_spectral_density(spectra, path):
    """MEMe spectral CDF and density against the exact eigenvalues."""
    keys = list(spectra.keys())
    fig, axes = plt.subplots(2, len(keys), figsize=(5.0 * len(keys), 7.0), squeeze=False,
                             constrained_layout=True)
    for col, key in enumerate(keys):
        sp = spectra[key]
        eig = np.sort(sp["eig"])
        lam, mass = sp["lam"], sp["mass"]
        ax = axes[0, col]
        ax.step(eig, np.arange(1, len(eig) + 1) / len(eig), where="post",
                color=COLORS["exact"], lw=1.6, label="exact eigenvalues")
        ax.plot(lam, np.cumsum(mass), color=COLORS["meme"], lw=1.6,
                label="MEMe, m = %d" % sp["m"])
        ax.set_xscale("log")
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("cumulative fraction")
        ax.set_title("ell = %.3g: spectral CDF" % sp["ell"], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax = axes[1, col]
        edges = np.logspace(np.log10(sp["lam_min"]) - 0.05, np.log10(sp["lam_max"]), 46)
        hist, _ = np.histogram(eig, bins=edges)
        idx = np.clip(np.searchsorted(edges, lam) - 1, 0, len(edges) - 2)
        binned = np.bincount(idx, weights=mass, minlength=len(edges) - 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
        ax.bar(centers, hist / len(eig), width=np.diff(edges), color="#bbbbbb",
               label="exact (fraction per bin)")
        ax.step(centers, binned, where="mid", color=COLORS["meme"], lw=1.6, label="MEMe mass")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.5)
        ax.set_xlabel("eigenvalue (log bins)")
        ax.set_ylabel("fraction of spectrum")
        ax.set_title("ell = %.3g: histogram" % sp["ell"], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("MEMe maximum-entropy spectral density vs exact spectrum")
    return _save(fig, path)


def plot_entropy_convergence(bench, newton, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    ells = sorted(bench["ell"].unique())
    seed0 = bench["seed"].min()
    for meth, ls in (("meme", "-"), ("maxent_power", "--")):  # entropy panel
        for i, ell in enumerate(ells):
            g = bench[(bench["method"] == meth) & (bench["ell"] == ell) & (bench["seed"] == seed0)]
            g = g.sort_values("m")
            ax.plot(g["m"], g["entropy"], ls=ls, marker=MARKERS[meth], ms=4,
                    color=plt.cm.viridis(i / max(1, len(ells) - 1)),
                    label="ell=%.2g %s" % (ell, SHORT[meth]))
    ax.set_xscale("log")
    ax.set_xlabel("moments m")
    ax.set_ylabel("entropy of MaxEnt density")
    ax.set_title("Entropy vs m (solid: MEMe, dashed: power)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    ax = axes[1]
    for key, hist in newton.items():
        meth = key[0]
        ax.plot(np.arange(len(hist)), np.maximum(hist, 1e-16), color=COLORS[meth],
                ls="-" if key[2] == 30 else ":", lw=1.4,
                label="%s, ell=%.2g, m=%d" % (SHORT[meth], key[1], key[2]))
    ax.axhline(1e-8, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("max |gradient| (moment mismatch)")
    ax.set_title("Newton convergence (dashed line: tolerance)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6)
    ax = axes[2]
    for meth in ("meme", "meme_hard", "maxent_power"):
        g = bench[bench["method"] == meth].groupby("m")["converged"].mean()
        ax.plot(g.index, g.values, marker=MARKERS[meth], color=COLORS[meth],
                label=METHOD_LABELS[meth])
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("moments m")
    ax.set_ylabel("fraction of solves with |grad| < 1e-8")
    ax.set_title("Newton convergence rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_lml_surfaces(grid, path, methods=("exact", "meme", "chebyshev", "slq")):
    ells, noises = grid["ells"], grid["noises"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.2), squeeze=False)
    ref = grid["lml"]["exact"]
    vmax = float(ref.max())
    vmin = float(np.percentile(ref, 20))
    for ax, meth in zip(axes[0], methods):
        surf = grid["lml"][meth]
        im = ax.pcolormesh(np.arange(len(noises) + 1), np.arange(len(ells) + 1),
                           np.clip(surf, vmin, None), cmap="viridis", vmin=vmin,
                           vmax=max(vmax, float(surf.max())))
        sel = grid["selected"][meth]
        ax.plot(sel["j"] + 0.5, sel["i"] + 0.5, marker="*", color="red", ms=16, mec="white")
        ex = grid["selected"]["exact"]
        ax.plot(ex["j"] + 0.5, ex["i"] + 0.5, marker="o", mfc="none", mec="white", ms=18, mew=1.5)
        ax.set_xticks(np.arange(len(noises)) + 0.5)
        ax.set_xticklabels(["%.3g" % v for v in noises], rotation=45, fontsize=7)
        ax.set_yticks(np.arange(len(ells)) + 0.5)
        ax.set_yticklabels(["%.2f" % v for v in ells], fontsize=7)
        ax.set_xlabel("noise variance")
        ax.set_ylabel("length-scale")
        ax.set_title("LML, log-det: %s" % METHOD_LABELS[meth], fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("GP log marginal likelihood surfaces (star: argmax of this surface, "
                 "circle: exact argmax)", y=1.03)
    return _save(fig, path)


def plot_rmse_bars(results, path):
    names = list(results.keys())
    vals = [results[k]["rmse"] for k in names]
    cols = [results[k].get("color", "#555555") for k in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = ax.bar(range(len(names)), vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, "%.4f" % v, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test RMSE ($100k)")
    ax.set_title("Held-out test RMSE: baselines, XGBoost and GPs selected with each log-det estimator")
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)


def plot_xgboost(search, val_curve, importance, path):
    """XGBoost validation learning curve (selected config) and gain feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.plot(np.arange(1, len(val_curve) + 1), val_curve, color=COLORS["xgboost"])
    best = int(np.argmin(val_curve)) + 1
    ax.axvline(best, color="k", ls="--", lw=1, label="best n_estimators = %d" % best)
    ax.set_xlabel("boosting rounds")
    ax.set_ylabel("validation RMSE ($100k)")
    ax.set_title("XGBoost learning curve (selected config)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    txt = "\n".join("depth=%d lr=%.2f mcw=%d: val RMSE %.4f"
                    % (s["max_depth"], s["learning_rate"], s["min_child_weight"], s["val_rmse"])
                    for s in search)
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#cccccc"))
    ax = axes[1]
    items = sorted(importance.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items], color=COLORS["xgboost"])
    ax.set_xlabel("share of total gain")
    ax.set_title("XGBoost feature importance (gain)")
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, path)


def plot_pred_vs_actual(y, mean, std, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.scatter(y, mean, s=5, alpha=0.3, color=COLORS["meme"])
    lim = [min(y.min(), mean.min()), max(y.max(), mean.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("actual median house value ($100k)")
    ax.set_ylabel("predicted")
    ax.set_title("Predicted vs actual (test set)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    levels, emp = coverage_curve(y, mean, std)
    ax.plot(levels, emp, marker="o", color=COLORS["meme"], label="GP predictive intervals")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration of the predictive std")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[2]
    resid = y - mean
    ax.scatter(std, np.abs(resid), s=5, alpha=0.3, color="#1f77b4")
    order = np.argsort(std)
    bins = np.array_split(order, 12)
    ax.plot([std[b].mean() for b in bins], [np.sqrt(np.mean(resid[b] ** 2)) for b in bins],
            color="k", marker="s", label="binned RMSE")
    ax.set_xlabel("predictive std ($100k)")
    ax.set_ylabel("|error| ($100k)")
    ax.set_title("Error analysis: error vs predicted uncertainty")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_gmm(gmm, path):
    cols = [c for c in gmm.columns if c.startswith("ferr_")]
    labels = [c[5:] for c in cols]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    means = [gmm[c].mean() for c in cols]
    meds = [gmm[c].median() for c in cols]
    xs = np.arange(len(cols))
    ax.bar(xs - 0.2, means, 0.4, label="mean", color="#d62728")
    ax.bar(xs + 0.2, meds, 0.4, label="median", color="#1f77b4")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("fractional entropy error")
    ax.set_title("Gaussian-mixture entropy error over %d mixtures" % len(gmm))
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend()
    ax = axes[1]
    ax.boxplot([np.maximum(gmm[c].values, 1e-12) for c in cols])
    ax.set_xticks(np.arange(1, len(cols) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("fractional entropy error")
    ax.set_title("Distribution of the error per mixture")
    ax.grid(alpha=0.3, axis="y", which="both")
    return _save(fig, path)


def plot_wall_time(bench, exact_seconds, path):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for meth in EST_METHODS:
        g = bench[bench["method"] == meth].groupby("m")["seconds"].median()
        ax.plot(g.index, g.values, marker=MARKERS[meth], color=COLORS[meth],
                label=METHOD_LABELS[meth])
    ax.axhline(exact_seconds, color=COLORS["exact"], ls="--",
               label="exact Cholesky (median)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("moments / Lanczos steps m")
    ax.set_ylabel("seconds per estimate (matrix-vector products + solve)")
    ax.set_title("Estimator wall time vs m (n = %d dense matrix)" % int(bench["n"].iloc[0]))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_data_overview(y_train, y_test, conds, ells, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    bins = np.linspace(0, 5.1, 40)
    ax.hist(y_train, bins=bins, alpha=0.6, density=True, label="train (n=%d)" % len(y_train))
    ax.hist(y_test, bins=bins, alpha=0.6, density=True, label="test (n=%d)" % len(y_test))
    ax.set_xlabel("median house value ($100k)")
    ax.set_ylabel("density")
    ax.set_title("Target distribution (capped at 5.0 in the source data)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(ells, conds, marker="o", color="#d62728")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("RBF length-scale (standardised units)")
    ax.set_ylabel("condition number")
    ax.set_title("Benchmark kernel matrices: conditioning vs length-scale")
    ax.grid(alpha=0.3, which="both")
    return _save(fig, path)
