"""Train and evaluate everything end to end.

    python train.py            full run (writes artifacts/)
    python train.py --quick    small smoke run (writes artifacts_quick/)

Steps
  1. load the cached dataset, split 70/10/20, preprocess
  2. log-det benchmark: MEMe vs Chebyshev expansion vs SLQ vs power-moment MaxEnt
  3. GP hyper-parameter selection with each log-det estimator, final GP fit
  4. Gaussian-mixture entropy experiment
  5. write model, metrics.json, CSV tables, plots and sample predictions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import baselines as bl  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import gmm_entropy as ge  # noqa: E402
from src.data import DATASET_NAME, FEATURES, load_dataset, prepare  # noqa: E402
from src.gp import kernel_matrix  # noqa: E402
from src.maxent import meme_logdet_from_moments  # noqa: E402
from src.model import (METHOD_LABELS, MeanPredictor, MemeGPRegressor,  # noqa: E402
                       RidgeBaseline, XGBoostBaseline, gp_grid_search)
from src.moments import (chebyshev_moments, gershgorin_upper, power_moments,  # noqa: E402
                         rademacher)

FULL = dict(
    n_bench=2000, bench_seed=0, bench_noise=1e-3,
    ells=np.logspace(-1, 1, 6), ms=[5, 10, 20, 30, 50, 100], d=30, seeds=[0, 1, 2],
    primary_m=30,
    n_gp=2000, gp_seed=1, gp_ells=np.logspace(np.log10(0.2), np.log10(5.0), 8),
    gp_noises=np.logspace(-3, np.log10(0.5), 6), gp_m=30, gp_d=30, gp_probe_seed=0,
    n_gmm=50, gmm_seed=7, gmm_ms=[10, 30], gmm_mc=100,
    n_latency=200, n_samples=300, xgb_rounds=1500, xgb_grid=None,
)
QUICK = dict(FULL, n_bench=300, ells=np.logspace(-1, 1, 3), ms=[5, 10, 30], seeds=[0],
             n_gp=300, gp_ells=np.logspace(np.log10(0.3), np.log10(3.0), 3),
             gp_noises=np.array([0.01, 0.1, 0.5]), n_gmm=5, n_latency=20, n_samples=50,
             xgb_rounds=200,
             xgb_grid=[{"max_depth": 4, "learning_rate": 0.1, "min_child_weight": 1}])


def log(msg: str) -> None:
    print(msg, flush=True)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


# ---------------------------------------------------------------- benchmark
def run_benchmark(Xb: np.ndarray, cfg: dict):
    n = Xb.shape[0]
    d = cfg["d"]
    m_max = max(cfg["ms"])
    rows, spectra, newton, mats = [], {}, {}, []
    newton_ells = {float(cfg["ells"][len(cfg["ells"]) // 2]), float(cfg["ells"][-1])}
    for ell in cfg["ells"]:
        ell = float(ell)
        t0 = perf_counter()
        A = kernel_matrix(Xb, ell, cfg["bench_noise"])
        tc = perf_counter()
        exact = bl.exact_logdet(A)
        chol_seconds = perf_counter() - tc
        eig = np.linalg.eigvalsh(A)
        cond = float(eig[-1] / eig[0])
        lam_min, lam_max = float(cfg["bench_noise"]), gershgorin_upper(A)
        mats.append({"ell": ell, "cond": cond, "exact_logdet": exact, "lam_min": lam_min,
                     "lam_max": lam_max, "eig_min": float(eig[0]), "eig_max": float(eig[-1]),
                     "cholesky_seconds": chol_seconds})
        for seed in cfg["seeds"]:
            Z = rademacher(n, d, 1000 + seed)
            mu_c, t_c, sd_c = chebyshev_moments(A, Z, m_max, lam_min, lam_max, return_std=True)
            mu_p, t_p, sd_p = power_moments(A, Z, m_max, lam_min, lam_max, return_std=True)
            al, be, ln, t_l = bl.lanczos_block(A, Z, m_max)
            for m in cfg["ms"]:
                base = {"ell": ell, "cond": cond, "n": n, "d": d, "seed": seed, "m": m,
                        "mvms": m * d, "exact": exact}

                def add(method, est, seconds, res=None):
                    row = dict(base, method=method, estimate=float(est),
                               rel_err=ev.rel_error(est, exact),
                               signed_rel_err=float((est - exact) / abs(exact)),
                               abs_err_per_dim=float(abs(est - exact) / n),
                               seconds=float(seconds))
                    if res is not None:
                        row.update(converged=bool(res["converged"]), n_iter=res["n_iter"],
                                   grad_norm=res["grad_norm"], entropy=res["entropy"])
                    rows.append(row)

                ts = perf_counter()
                est, res = meme_logdet_from_moments(mu_c[: m + 1], n, lam_min, lam_max,
                                                    moment_std=sd_c[: m + 1])
                add("meme", est, t_c[m] + perf_counter() - ts, res)
                if seed == cfg["seeds"][0]:
                    if m == cfg["primary_m"]:
                        spectra[ell] = {"ell": ell, "m": m, "eig": eig, "lam": res["lam"],
                                        "mass": res["mass"], "lam_min": lam_min,
                                        "lam_max": lam_max}
                    if ell in newton_ells and m == cfg["primary_m"]:
                        newton[("meme", ell, m)] = res["grad_history"]

                ts = perf_counter()
                est, res = meme_logdet_from_moments(mu_c[: m + 1], n, lam_min, lam_max)
                add("meme_hard", est, t_c[m] + perf_counter() - ts, res)
                if seed == cfg["seeds"][0] and ell in newton_ells and m == cfg["primary_m"]:
                    newton[("meme_hard", ell, m)] = res["grad_history"]

                ts = perf_counter()
                est, res = meme_logdet_from_moments(mu_p[: m + 1], n, lam_min, lam_max,
                                                    basis="power", moment_std=sd_p[: m + 1])
                add("maxent_power", est, t_p[m] + perf_counter() - ts, res)
                if seed == cfg["seeds"][0] and ell in newton_ells and m == cfg["primary_m"]:
                    newton[("maxent_power", ell, m)] = res["grad_history"]

                ts = perf_counter()
                est = bl.chebyshev_logdet_from_moments(mu_c[: m + 1], n, lam_min, lam_max)
                add("chebyshev", est, t_c[m] + perf_counter() - ts)

                ts = perf_counter()
                est = bl.slq_logdet(al, be, ln, m, n)
                add("slq", est, t_l[min(m, len(t_l) - 1)] + perf_counter() - ts)
        sub = pd.DataFrame([r for r in rows if r["ell"] == ell and r["m"] == cfg["primary_m"]])
        med = sub.groupby("method")["rel_err"].median()
        log("  ell=%-6.3g cond=%.2e exact=%11.2f | m=%d rel.err  meme=%.4f  hard=%.4f  "
            "cheb=%.4f  slq=%.4f  power=%.4f  (%.1fs)"
            % (ell, cond, exact, cfg["primary_m"], med["meme"], med["meme_hard"],
               med["chebyshev"], med["slq"], med["maxent_power"], perf_counter() - t0))
    return pd.DataFrame(rows), pd.DataFrame(mats), spectra, newton


def summarise_benchmark(bench: pd.DataFrame, mats: pd.DataFrame, cfg: dict) -> dict:
    pm = cfg["primary_m"]
    at = bench[bench["m"] == pm]
    by_method = {}
    for meth, g in at.groupby("method"):
        by_method[meth] = {
            "median_rel_err": float(g["rel_err"].median()),
            "mean_rel_err": float(g["rel_err"].mean()),
            "max_rel_err": float(g["rel_err"].max()),
            "median_abs_err_per_dim": float(g["abs_err_per_dim"].median()),
            "median_seconds": float(g["seconds"].median()),
        }
    by_m = {}
    for m, g in bench.groupby("m"):
        by_m[int(m)] = {meth: float(v) for meth, v in g.groupby("method")["rel_err"].median().items()}
    by_ell = []
    for _, mat in mats.iterrows():
        g = at[at["ell"] == mat["ell"]]
        entry = {"ell": float(mat["ell"]), "cond": float(mat["cond"]),
                 "exact_logdet": float(mat["exact_logdet"])}
        entry.update({meth: float(v) for meth, v in g.groupby("method")["rel_err"].median().items()})
        by_ell.append(entry)
    # paired comparison on identical probes
    piv = at.pivot_table(index=["ell", "seed"], columns="method", values="rel_err")
    stability = {}
    for meth in ("meme", "meme_hard", "maxent_power"):
        g = bench[bench["method"] == meth]
        mono = []
        for _, run in g.groupby(["ell", "seed"]):
            ent = run.sort_values("m")["entropy"].to_numpy()
            mono.append(bool(np.all(np.diff(ent) <= 1e-6 * np.maximum(1.0, np.abs(ent[:-1])))))
        stability[meth] = {
            "converged_fraction_by_m": {int(m): float(v) for m, v in
                                        g.groupby("m")["converged"].mean().items()},
            "median_grad_norm_by_m": {int(m): float(v) for m, v in
                                      g.groupby("m")["grad_norm"].median().items()},
            "median_newton_iters_by_m": {int(m): float(v) for m, v in
                                         g.groupby("m")["n_iter"].median().items()},
            "entropy_monotone_fraction": float(np.mean(mono)),
        }
    return {
        "primary_m": pm, "d": cfg["d"], "n": int(bench["n"].iloc[0]),
        "n_matrices": int(len(mats)), "n_seeds": len(cfg["seeds"]),
        "at_primary_m": by_method,
        "median_rel_err_by_m": by_m,
        "median_rel_err_by_ell_at_primary_m": by_ell,
        "meme_win_fraction_vs_chebyshev": float(np.mean(piv["meme"] < piv["chebyshev"])),
        "meme_win_fraction_vs_slq": float(np.mean(piv["meme"] < piv["slq"])),
        "meme_win_fraction_vs_maxent_power": float(np.mean(piv["meme"] < piv["maxent_power"])),
        "meme_win_fraction_vs_meme_hard": float(np.mean(piv["meme"] < piv["meme_hard"])),
        "stability": stability,
        "exact_cholesky_median_seconds": float(mats["cholesky_seconds"].median()),
    }


# ----------------------------------------------------------------- GMM
def run_gmm(cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["gmm_seed"])
    rows = []
    for i in range(cfg["n_gmm"]):
        mix = ge.random_mixture(rng)
        truth = ge.entropy_quadrature(mix)
        row = {"mixture": i, "n_components": len(mix["weights"]), "entropy_true": truth}
        for m in cfg["gmm_ms"]:
            res = ge.entropy_meme(mix, m)
            row["entropy_MEMe_m%d" % m] = res["entropy"]
            row["converged_MEMe_m%d" % m] = res["converged"]
        row["entropy_MC%d" % cfg["gmm_mc"]] = ge.entropy_monte_carlo(mix, cfg["gmm_mc"], rng)
        row["entropy_moment_match"] = ge.entropy_moment_matching(mix)
        for key in [k for k in row if k.startswith("entropy_") and k != "entropy_true"]:
            row["ferr_" + key[len("entropy_"):]] = abs(row[key] - truth) / abs(truth)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    args = parser.parse_args(argv)
    cfg = QUICK if args.quick else FULL
    out = ROOT / ("artifacts_quick" if args.quick else "artifacts")
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    t_start = perf_counter()
    timing = {}

    log("=== 1/5 data ===")
    df = load_dataset()
    data = prepare(df)
    pre = data["pre"]
    n_train, n_val, n_test = len(data["y_train"]), len(data["y_val"]), len(data["y_test"])
    log("  rows: train=%d val=%d test=%d, features=%d" % (n_train, n_val, n_test, len(FEATURES)))

    log("=== 2/5 log-det benchmark (n=%d, d=%d probes, m in %s, %d seeds) ==="
        % (cfg["n_bench"], cfg["d"], cfg["ms"], len(cfg["seeds"])))
    t0 = perf_counter()
    idx_b = np.random.default_rng(cfg["bench_seed"]).choice(n_train, cfg["n_bench"], replace=False)
    bench, mats, spectra, newton = run_benchmark(data["X_train"][idx_b], cfg)
    bsum = summarise_benchmark(bench, mats, cfg)
    timing["benchmark_seconds"] = perf_counter() - t0
    value = bsum["at_primary_m"]["meme"]["median_rel_err"]
    baseline = bsum["at_primary_m"]["chebyshev"]["median_rel_err"]
    log("  PRIMARY median rel. error at m=%d: MEMe=%.5f | Chebyshev baseline=%.5f | "
        "SLQ=%.5f | hard MEMe=%.5f | power MaxEnt=%.5f"
        % (cfg["primary_m"], value, baseline, bsum["at_primary_m"]["slq"]["median_rel_err"],
           bsum["at_primary_m"]["meme_hard"]["median_rel_err"],
           bsum["at_primary_m"]["maxent_power"]["median_rel_err"]))

    log("=== 3/5 GP hyper-parameter selection (n=%d, grid %dx%d, m=%d, d=%d) ==="
        % (cfg["n_gp"], len(cfg["gp_ells"]), len(cfg["gp_noises"]), cfg["gp_m"], cfg["gp_d"]))
    t0 = perf_counter()
    idx_g = np.random.default_rng(cfg["gp_seed"]).choice(n_train, cfg["n_gp"], replace=False)
    Xg, zg = data["X_train"][idx_g], pre.y_to_z(data["y_train"][idx_g])
    grid = gp_grid_search(
        Xg, zg, cfg["gp_ells"], cfg["gp_noises"], m=cfg["gp_m"], d=cfg["gp_d"],
        seed=cfg["gp_probe_seed"],
        progress=lambda i, ell: log("  grid row %d/%d (ell=%.3f) done"
                                    % (i + 1, len(cfg["gp_ells"]), ell)))
    timing["gp_grid_seconds"] = perf_counter() - t0
    gp_results, models = {}, {}
    for name, sel in grid["selected"].items():
        model = MemeGPRegressor(pre, sel["ell"], sel["noise"], meta={
            "selected_with": name, "logdet_m": cfg["gp_m"], "logdet_d": cfg["gp_d"],
            "n_train_subset": int(cfg["n_gp"])}).fit_standardised(Xg, zg)
        models[name] = model
        vm, vs = model.predict_standardised(data["X_val"])
        tm, ts_ = model.predict_standardised(data["X_test"])
        ld_err = np.abs(grid["logdets"][name] - grid["logdets"]["exact"]) / np.abs(grid["logdets"]["exact"])
        gp_results[name] = {
            "ell": sel["ell"], "noise": sel["noise"], "lml_at_selection": sel["lml"],
            "exact_lml_at_selection": float(grid["lml"]["exact"][sel["i"], sel["j"]]),
            "same_as_exact_selection": bool(sel["i"] == grid["selected"]["exact"]["i"]
                                            and sel["j"] == grid["selected"]["exact"]["j"]),
            "grid_logdet_median_rel_err": float(np.median(ld_err)),
            "grid_logdet_max_rel_err": float(np.max(ld_err)),
            "grid_logdet_seconds": float(grid["seconds"][name]),
            "val": ev.regression_metrics(data["y_val"], vm, vs),
            "test": ev.regression_metrics(data["y_test"], tm, ts_),
        }
        log("  %-10s -> ell=%.3f noise=%.4f | val RMSE=%.4f | test RMSE=%.4f NLL=%.4f"
            % (name, sel["ell"], sel["noise"], gp_results[name]["val"]["rmse"],
               gp_results[name]["test"]["rmse"], gp_results[name]["test"]["nll"]))
    main_model = models["meme"]
    test_mean, test_std = main_model.predict_standardised(data["X_test"])

    mean_bl = MeanPredictor().fit(data["X_train"], data["y_train"])
    ridge_bl = RidgeBaseline().fit(data["X_train"], data["y_train"])
    bl_results = {}
    for name, mdl in (("mean_predictor", mean_bl), ("ridge", ridge_bl)):
        bl_results[name] = {
            "val": ev.regression_metrics(data["y_val"], *mdl.predict(data["X_val"])),
            "test": ev.regression_metrics(data["y_test"], *mdl.predict(data["X_test"])),
        }
    bl_results["ridge"]["alpha"] = ridge_bl.alpha_
    log("  baselines: mean test RMSE=%.4f | ridge (alpha=%.3g) test RMSE=%.4f"
        % (bl_results["mean_predictor"]["test"]["rmse"], ridge_bl.alpha_,
           bl_results["ridge"]["test"]["rmse"]))

    # XGBoost: tuned on the validation split only; trained (a) on the full training
    # split and (b) on the same 2000-row subset the GP sees (like-for-like data).
    t0 = perf_counter()
    xgb_results = {}
    xgb_full = None
    for tag, Xtr, ytr in (("full_train", data["X_train"], data["y_train"]),
                          ("gp_subset", Xg, data["y_train"][idx_g])):
        mdl = XGBoostBaseline(grid=cfg.get("xgb_grid"), n_estimators=cfg["xgb_rounds"])
        mdl.fit(Xtr, ytr, data["X_val"], data["y_val"])
        xgb_results[tag] = {
            "n_train_rows": int(len(ytr)), "params": mdl.params_,
            "n_estimators": int(mdl.best_n_), "search": mdl.search_,
            "val": ev.regression_metrics(data["y_val"], *mdl.predict(data["X_val"])),
            "test": ev.regression_metrics(data["y_test"], *mdl.predict(data["X_test"])),
        }
        log("  xgboost (%s, %d rows): depth=%d lr=%.2f trees=%d | val RMSE=%.4f | test RMSE=%.4f"
            % (tag, len(ytr), mdl.params_["max_depth"], mdl.params_["learning_rate"],
               mdl.best_n_, xgb_results[tag]["val"]["rmse"], xgb_results[tag]["test"]["rmse"]))
        if tag == "full_train":
            xgb_full = mdl
    xgb_results["full_train"]["feature_importance_gain"] = xgb_full.feature_importance()
    timing["xgboost_seconds"] = perf_counter() - t0

    log("=== 4/5 Gaussian-mixture entropy (%d mixtures) ===" % cfg["n_gmm"])
    t0 = perf_counter()
    gmm = run_gmm(cfg)
    timing["gmm_seconds"] = perf_counter() - t0
    gmm_summary = {}
    for col in [c for c in gmm.columns if c.startswith("ferr_")]:
        gmm_summary[col[5:]] = {"mean_frac_err": float(gmm[col].mean()),
                                "median_frac_err": float(gmm[col].median()),
                                "max_frac_err": float(gmm[col].max())}
        log("  %-14s mean frac. error=%.5f median=%.5f"
            % (col[5:], gmm[col].mean(), gmm[col].median()))
    for m in cfg["gmm_ms"]:
        gmm_summary["MEMe_m%d" % m]["converged_fraction"] = float(gmm["converged_MEMe_m%d" % m].mean())

    log("=== 5/5 artifacts ===")
    model_path = out / "model.joblib"
    main_model.save(model_path)
    reloaded = MemeGPRegressor.load(model_path)
    chk_mean, chk_std = reloaded.predict(data["X_raw_test"][:50])
    if not (np.allclose(chk_mean, test_mean[:50]) and np.allclose(chk_std, test_std[:50])):
        raise RuntimeError("reloaded model does not reproduce the trained model")
    model_mb = model_path.stat().st_size / 1e6
    lat = []
    for i in range(cfg["n_latency"]):
        t0 = perf_counter()
        reloaded.predict_one(data["X_raw_test"][i % n_test])
        lat.append((perf_counter() - t0) * 1000.0)
    service = {"predict_latency_ms_p50": float(np.percentile(lat, 50)),
               "predict_latency_ms_p95": float(np.percentile(lat, 95)),
               "latency_requests": int(cfg["n_latency"]),
               "latency_note": "single-row model call measured in-process (no HTTP overhead)",
               "model_file_mb": float(model_mb)}
    log("  model.joblib %.2f MB, single prediction p50=%.2f ms p95=%.2f ms"
        % (model_mb, service["predict_latency_ms_p50"], service["predict_latency_ms_p95"]))

    bench.to_csv(out / "logdet_benchmark.csv", index=False)
    mats.to_csv(out / "benchmark_matrices.csv", index=False)
    gmm.to_csv(out / "gmm_entropy.csv", index=False)
    grid_rows = []
    for i, ell in enumerate(grid["ells"]):
        for j, s2 in enumerate(grid["noises"]):
            row = {"ell": float(ell), "noise": float(s2)}
            for name in grid["lml"]:
                row["logdet_" + name] = float(grid["logdets"][name][i, j])
                row["lml_" + name] = float(grid["lml"][name][i, j])
            grid_rows.append(row)
    pd.DataFrame(grid_rows).to_csv(out / "gp_grid.csv", index=False)
    np.savez_compressed(
        out / "spectra.npz",
        ells=np.array(sorted(spectra.keys())),
        **{"eig_%d" % i: spectra[k]["eig"] for i, k in enumerate(sorted(spectra))},
        **{"lam_%d" % i: spectra[k]["lam"] for i, k in enumerate(sorted(spectra))},
        **{"mass_%d" % i: spectra[k]["mass"] for i, k in enumerate(sorted(spectra))})

    ns = min(cfg["n_samples"], n_test)
    samp = pd.DataFrame(data["X_raw_test"][:ns], columns=FEATURES)
    samp.insert(0, "row_id", data["idx"]["test"][:ns])
    samp["true_value"] = data["y_test"][:ns]
    samp["prediction"] = test_mean[:ns]
    samp["pred_std"] = test_std[:ns]
    samp["lower_95"] = test_mean[:ns] - 1.96 * test_std[:ns]
    samp["upper_95"] = test_mean[:ns] + 1.96 * test_std[:ns]
    samp["abs_error"] = np.abs(samp["true_value"] - samp["prediction"])
    samp["within_95"] = (samp["abs_error"] <= 1.96 * samp["pred_std"]).astype(int)
    # confidence = P(|truth - prediction| <= 0.5, i.e. $50k) under the Gaussian predictive
    samp["confidence"] = ev.interval_confidence(test_std[:ns], 0.5)
    samp.to_csv(out / "sample_predictions.csv", index=False)

    keys = sorted(spectra.keys())
    pick = [keys[i] for i in sorted({min(1, len(keys) - 1), len(keys) // 2, len(keys) - 1})]
    rmse_bars = {
        "Mean predictor": {"rmse": bl_results["mean_predictor"]["test"]["rmse"], "color": ev.COLORS["mean"]},
        "Ridge (RidgeCV)": {"rmse": bl_results["ridge"]["test"]["rmse"], "color": ev.COLORS["ridge"]},
        "GP, exact log-det": {"rmse": gp_results["exact"]["test"]["rmse"], "color": ev.COLORS["exact"]},
        "GP, MEMe (served)": {"rmse": gp_results["meme"]["test"]["rmse"], "color": ev.COLORS["meme"]},
        "GP, Chebyshev": {"rmse": gp_results["chebyshev"]["test"]["rmse"], "color": ev.COLORS["chebyshev"]},
        "GP, SLQ": {"rmse": gp_results["slq"]["test"]["rmse"], "color": ev.COLORS["slq"]},
        "XGBoost (GP subset)": {"rmse": xgb_results["gp_subset"]["test"]["rmse"],
                                "color": "#9edae5"},
        "XGBoost (full train)": {"rmse": xgb_results["full_train"]["test"]["rmse"],
                                 "color": ev.COLORS["xgboost"]},
    }
    made = [
        ev.plot_data_overview(data["y_train"], data["y_test"], mats["cond"].to_numpy(),
                              mats["ell"].to_numpy(), plots / "00_data_overview.png"),
        ev.plot_error_vs_m(bench, plots / "01_error_vs_moments.png"),
        ev.plot_error_vs_cond(bench, cfg["primary_m"], plots / "02_error_vs_condition.png"),
        ev.plot_spectral_density({k: spectra[k] for k in pick}, plots / "03_spectral_density.png"),
        ev.plot_entropy_convergence(bench, newton, plots / "04_entropy_and_newton.png"),
        ev.plot_lml_surfaces(grid, plots / "05_gp_lml_surfaces.png"),
        ev.plot_rmse_bars(rmse_bars, plots / "06_test_rmse.png"),
        ev.plot_pred_vs_actual(data["y_test"], test_mean, test_std,
                               plots / "07_predicted_vs_actual.png"),
        ev.plot_gmm(gmm, plots / "08_gmm_entropy.png"),
        ev.plot_wall_time(bench, bsum["exact_cholesky_median_seconds"],
                          plots / "09_wall_time.png"),
        ev.plot_xgboost(xgb_full.search_, xgb_full.val_curve_, xgb_full.feature_importance(),
                        plots / "10_xgboost.png"),
    ]
    log("  wrote %d plots" % len(made))

    gp_rmse = gp_results["meme"]["test"]["rmse"]
    checks = {
        "meme_beats_chebyshev_logdet": bool(value < baseline),
        "meme_beats_slq_logdet": bool(value < bsum["at_primary_m"]["slq"]["median_rel_err"]),
        "gp_meme_beats_mean_predictor": bool(gp_rmse < bl_results["mean_predictor"]["test"]["rmse"]),
        "gp_meme_beats_ridge": bool(gp_rmse < bl_results["ridge"]["test"]["rmse"]),
        "gp_meme_within_5pct_of_exact_selected": bool(
            gp_rmse <= 1.05 * gp_results["exact"]["test"]["rmse"]),
        "gp_meme_rmse_ratio_to_exact_selected": float(gp_rmse / gp_results["exact"]["test"]["rmse"]),
        "gp_meme_beats_xgboost_full_train": bool(gp_rmse < xgb_results["full_train"]["test"]["rmse"]),
        "gp_meme_beats_xgboost_gp_subset": bool(gp_rmse < xgb_results["gp_subset"]["test"]["rmse"]),
    }
    best_predictor = min(
        [("gp_meme", gp_rmse), ("gp_exact", gp_results["exact"]["test"]["rmse"]),
         ("ridge", bl_results["ridge"]["test"]["rmse"]),
         ("mean_predictor", bl_results["mean_predictor"]["test"]["rmse"]),
         ("xgboost_full_train", xgb_results["full_train"]["test"]["rmse"]),
         ("xgboost_gp_subset", xgb_results["gp_subset"]["test"]["rmse"])],
        key=lambda kv: kv[1])
    checks["best_test_rmse_predictor"] = best_predictor[0]
    train_seconds = perf_counter() - t_start
    timing["total_seconds"] = train_seconds
    metrics = {
        "task": "log-determinant estimation with MEMe (maximum-entropy spectral density), "
                "used to select and serve a Gaussian-process regressor",
        "dataset": DATASET_NAME,
        "primary_metric": "median_abs_relative_logdet_error_m%d_d%d" % (cfg["primary_m"], cfg["d"]),
        "higher_is_better": False,
        "value": value,
        "baseline": baseline,
        "baseline_name": "Stochastic Chebyshev expansion of log (Han et al. 2015), "
                         "same moments and matrix-vector budget",
        "metrics": {
            "logdet": bsum,
            "gp": {k: gp_results[k] for k in ("meme", "exact", "slq")},
            "gp_served_model": {"selected_with": "meme", "ell": main_model.ell,
                                "noise": main_model.noise, "n_train_subset": int(cfg["n_gp"]),
                                "test": gp_results["meme"]["test"],
                                "val": gp_results["meme"]["val"]},
            "gp_grid": {"ells": grid["ells"], "noises": grid["noises"],
                        "m": cfg["gp_m"], "d": cfg["gp_d"]},
            "xgboost": dict(
                xgb_results["full_train"],
                gp_subset=xgb_results["gp_subset"],
                note="XGBRegressor on the same standardised features, tuned on the validation "
                     "split. It is a predictor, not a log-det estimator, so it is compared on "
                     "the downstream regression metrics (same keys as metrics.gp.*)."),
            "gmm_entropy": gmm_summary,
            "service": service,
            "checks": checks,
            "splits": {"n_train": n_train, "n_val": n_val, "n_test": n_test,
                       "n_benchmark_subset": int(cfg["n_bench"]),
                       "n_gp_train_subset": int(cfg["n_gp"])},
            "timing": timing,
            "quick_mode": bool(args.quick),
        },
        "baseline_metrics": {
            "logdet_chebyshev": bsum["at_primary_m"]["chebyshev"],
            "gp_chebyshev_selected": gp_results["chebyshev"],
            "mean_predictor": bl_results["mean_predictor"],
            "ridge": bl_results["ridge"],
        },
        "n_train": n_train,
        "n_test": n_test,
        "train_seconds": train_seconds,
    }
    with open(out / "metrics.json", "w", encoding="ascii") as fh:
        json.dump(jsonable(metrics), fh, indent=2)

    log("")
    log("RESULT primary (lower is better): MEMe=%.5f vs Chebyshev baseline=%.5f -> %s"
        % (value, baseline, "MEMe wins" if checks["meme_beats_chebyshev_logdet"] else "MEMe LOSES"))
    log("RESULT served GP test RMSE=%.4f (ridge %.4f, mean %.4f, exact-selected GP %.4f)"
        % (gp_rmse, bl_results["ridge"]["test"]["rmse"],
           bl_results["mean_predictor"]["test"]["rmse"], gp_results["exact"]["test"]["rmse"]))
    log("RESULT xgboost test RMSE: full train=%.4f, GP subset=%.4f; best predictor: %s"
        % (xgb_results["full_train"]["test"]["rmse"], xgb_results["gp_subset"]["test"]["rmse"],
           best_predictor[0]))
    for k, v in checks.items():
        log("  check %-40s %s" % (k, v))
    log("done in %.1f s, artifacts in %s/" % (train_seconds, out.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
