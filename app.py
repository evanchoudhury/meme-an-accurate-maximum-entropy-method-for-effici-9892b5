"""Streamlit dashboard for the MEMe project.

Run with:  streamlit run app.py
Everything is read from artifacts/ and docs/ next to this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DOCS = ROOT / "docs"
PLOTS = ART / "plots"
DATA_FILE = ROOT / "data" / "california_housing.csv.gz"
PAPER_URL = "http://arxiv.org/abs/1906.01101v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="MEMe - Maximum Entropy log-determinants", layout="wide")
# dashboard-theme (agent/dashboard_theme.py): glass look, see .streamlit/theme.css
_css = __import__('pathlib').Path(__file__).with_name('.streamlit') / 'theme.css'
if _css.is_file(): st.markdown('<style>' + _css.read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]
FEATURE_HELP = {
    "MedInc": "Median income of the block group (in $10,000s)",
    "HouseAge": "Median house age in the block group (years)",
    "AveRooms": "Average number of rooms per household",
    "AveBedrms": "Average number of bedrooms per household",
    "Population": "Number of people living in the block group",
    "AveOccup": "Average number of people per household",
    "Latitude": "Latitude of the block group (degrees north)",
    "Longitude": "Longitude of the block group (degrees east, negative = west)",
}
METHOD_LABELS = {
    "meme": "MEMe (noise-aware, main model)",
    "meme_hard": "MEMe, hard constraints (paper)",
    "chebyshev": "Chebyshev expansion (baseline)",
    "slq": "Stochastic Lanczos (reference)",
    "maxent_power": "MaxEnt, power moments (ablation)",
    "exact": "Exact (Cholesky)",
}
COLORS = {"meme": "#1f77b4", "meme_hard": "#9467bd", "chebyshev": "#d62728",
          "slq": "#2ca02c", "maxent_power": "#ff7f0e", "exact": "#111111"}


# ------------------------------------------------------------------ loaders
@st.cache_data(show_spinner=False)
def load_metrics():
    p = ART / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_csv(name):
    p = ART / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_doc(name):
    p = DOCS / name
    return p.read_text(encoding="utf-8") if p.exists() else None


@st.cache_data(show_spinner=False)
def load_spectra():
    p = ART / "spectra.npz"
    if not p.exists():
        return None
    try:
        d = np.load(p)
        return {k: d[k] for k in d.files}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATA_FILE.exists():
        return None
    try:
        return pd.read_csv(DATA_FILE)
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading the trained Gaussian process ...")
def load_model():
    p = ART / "model.joblib"
    if not p.exists():
        return None, "artifacts/model.joblib not found"
    try:
        from src.model import MemeGPRegressor
        return MemeGPRegressor.load(p), None
    except Exception as exc:  # keep the app alive
        return None, repr(exc)


def doc_section(text, heading):
    """Return the body of a '## heading' section of a markdown doc."""
    if not text:
        return None
    out, grab = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            if grab:
                break
            grab = line[3:].strip().lower().startswith(heading.lower())
            continue
        if grab:
            out.append(line)
    body = "\n".join(out).strip()
    return body or None


def get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fmt(v, nd=3):
    return "n/a" if v is None else ("%.*f" % (nd, v))


def missing_box(what="This section"):
    st.warning("%s needs the training artifacts, which were not found. "
               "Run `python train.py` in the project folder (about 3 minutes on a laptop CPU), "
               "then refresh this page." % what)


M = load_metrics()
BENCH = load_csv("logdet_benchmark.csv")
MATS = load_csv("benchmark_matrices.csv")
GRID = load_csv("gp_grid.csv")
GMM = load_csv("gmm_entropy.csv")
PRED = load_csv("sample_predictions.csv")

st.title("MEMe: Maximum-Entropy log-determinants for large-scale ML")
st.caption("A reproduction of the idea in Granziol et al., *MEMe* (arXiv:1906.01101), "
           "applied to Gaussian-process model selection on California Housing.")
if M is None:
    st.error("No results found in `artifacts/`. Run `python train.py` first "
             "(it writes artifacts/metrics.json, the CSV files, plots and the model). "
             "The Research tab below still works.")

import ui_rings  # dashboard-rings: headline results as ring gauges
ui_rings.hero()
tabs = st.tabs(["Overview", "Data", "Model", "Results", "Try it", "Research"])

# ================================================================== OVERVIEW
with tabs[0]:
    st.header("The idea in plain words")
    st.markdown(
        """
Many machine-learning models (Gaussian processes, determinantal point processes, Gaussian
graphical models) need the **log-determinant** of a big matrix, `log det(A)`. Computing it
exactly takes a Cholesky factorisation, which costs `O(n^3)` time - far too slow for big `n`.

**MEMe** avoids the factorisation:
1. Multiply the matrix by a few random vectors to estimate the first *m* **moments** of its
   eigenvalue distribution (cheap: only matrix-vector products).
2. Find the **maximum-entropy** distribution that matches those moments - the "least
   opinionated" spectrum consistent with what we measured.
3. Read off `log det(A) = n * E[log lambda]` from that distribution.

The baseline turns the same moments directly into a polynomial (Chebyshev) approximation of
`log`, which struggles when the matrix is badly conditioned. Here we measure both on real
GP kernel matrices, then use the MEMe estimate to choose GP hyper-parameters and serve
house-price predictions.
"""
    )
    obj = doc_section(load_doc("02_problem_definition.md"), "Objectives")
    prob = doc_section(load_doc("02_problem_definition.md"), "Problem statement")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Problem")
        st.markdown(prob or "_docs/02_problem_definition.md not found._")
    with c2:
        st.subheader("Objectives")
        st.markdown(obj or "_docs/02_problem_definition.md not found._")

    st.header("Headline results")
    if M is None:
        missing_box("The headline results")
    else:
        st.markdown(
            "**How to read these numbers.** The primary metric is the *median relative error* of "
            "the log-determinant, `|estimate - exact| / |exact|`, over %s kernel matrices x %s "
            "random seeds, using m = %s moments and d = %s random probe vectors (n = %s). "
            "**Lower is better**; 0.10 means the estimate is typically 10%% off. Both methods "
            "use exactly the same matrix-vector budget."
            % (get(M, "metrics", "logdet", "n_matrices"), get(M, "metrics", "logdet", "n_seeds"),
               get(M, "metrics", "logdet", "primary_m"), get(M, "metrics", "logdet", "d"),
               get(M, "metrics", "logdet", "n")))
        v, b = M.get("value"), M.get("baseline")
        c = st.columns(4)
        c[0].metric("MEMe median rel. error (lower = better)", fmt(v),
                    delta=None if (v is None or b is None) else "%+.3f vs baseline" % (v - b),
                    delta_color="inverse")
        c[1].metric("Chebyshev baseline median rel. error", fmt(b))
        win = get(M, "metrics", "logdet", "meme_win_fraction_vs_chebyshev")
        c[2].metric("Cases where MEMe beats the baseline", "n/a" if win is None else "%.0f%%" % (100 * win))
        slq = get(M, "metrics", "logdet", "at_primary_m", "slq", "median_rel_err")
        c[3].metric("SLQ reference median rel. error", fmt(slq))
        st.caption("Baseline: %s." % M.get("baseline_name", "n/a"))

        st.subheader("Downstream: house-price GP chosen with each estimator")
        gpm = get(M, "metrics", "gp", "meme", "test") or {}
        gpc = get(M, "baseline_metrics", "gp_chebyshev_selected", "test") or {}
        rid = get(M, "baseline_metrics", "ridge", "test") or {}
        c = st.columns(4)
        c[0].metric("GP (MEMe-selected) test RMSE", fmt(gpm.get("rmse")),
                    delta=None if not gpc else "%+.3f vs Chebyshev-selected" % (gpm["rmse"] - gpc["rmse"]),
                    delta_color="inverse")
        c[1].metric("GP (Chebyshev-selected) test RMSE", fmt(gpc.get("rmse")))
        c[2].metric("Ridge regression test RMSE", fmt(rid.get("rmse")))
        c[3].metric("GP (MEMe-selected) test R^2", fmt(gpm.get("r2")))
        st.caption("RMSE is in units of $100,000 of median house value (lower is better). R^2 is "
                   "the share of variance explained (1.0 = perfect). The GP is trained on a "
                   "%s-row subset of the training split." % get(M, "metrics", "gp_served_model", "n_train_subset"))
        checks = get(M, "metrics", "checks") or {}
        if checks:
            st.subheader("Automatic checks written by train.py")
            st.dataframe(pd.DataFrame({"check": list(checks.keys()),
                                       "value": [str(x) for x in checks.values()]}),
                         hide_index=True, width="stretch")
            st.caption("Honest notes: SLQ (a strong Lanczos-based reference) is more accurate than "
                       "MEMe in this setup, and gradient-boosted trees (XGBoost) predict house "
                       "prices better than the GP. MEMe's win is against the Chebyshev baseline "
                       "at equal cost, which is the comparison the paper makes.")

# ================================================================== DATA
with tabs[1]:
    st.header("Dataset")
    ds_doc = load_doc("04_dataset.md")
    with st.expander("Dataset description (docs/04_dataset.md)", expanded=False):
        st.markdown(ds_doc or "_docs/04_dataset.md not found._")
    st.markdown(
        "**California Housing**: one row per census block group in California (1990 census). "
        "8 numeric features, target `MedHouseVal` = median house value in $100,000s "
        "(capped at 5.0). This data is used in two ways: (1) its standardised features build "
        "the RBF kernel matrices whose log-determinants we estimate, and (2) the GP predicts "
        "house value from them.")
    df = load_dataset()
    if df is None:
        st.warning("data/california_housing.csv.gz not found. `python train.py` downloads and "
                   "caches it.")
    else:
        if M is not None:
            sp = get(M, "metrics", "splits") or {}
            c = st.columns(5)
            c[0].metric("Rows", "%d" % len(df))
            c[1].metric("Train", sp.get("n_train", "n/a"))
            c[2].metric("Validation", sp.get("n_val", "n/a"))
            c[3].metric("Test", sp.get("n_test", "n/a"))
            c[4].metric("GP / benchmark subset", sp.get("n_gp_train_subset", "n/a"))
        st.subheader("Filter and explore")
        c1, c2, c3 = st.columns(3)
        inc = c1.slider("Median income range ($10k)", float(df.MedInc.min()), float(df.MedInc.max()),
                        (float(df.MedInc.min()), float(df.MedInc.max())))
        age = c2.slider("House age range (years)", float(df.HouseAge.min()), float(df.HouseAge.max()),
                        (float(df.HouseAge.min()), float(df.HouseAge.max())))
        val = c3.slider("House value range ($100k)", float(df.MedHouseVal.min()),
                        float(df.MedHouseVal.max()),
                        (float(df.MedHouseVal.min()), float(df.MedHouseVal.max())))
        f = df[df.MedInc.between(*inc) & df.HouseAge.between(*age) & df.MedHouseVal.between(*val)]
        st.write("%d of %d rows match the filters." % (len(f), len(df)))
        st.dataframe(f.head(200), width="stretch")
        if len(f):
            c1, c2 = st.columns(2)
            col = c1.selectbox("Distribution of", FEATURES + ["MedHouseVal"], index=8)
            logx = c1.checkbox("Log x-axis", value=col in ("Population", "AveOccup", "AveRooms", "AveBedrms"))
            fig = px.histogram(f, x=col, nbins=60, log_x=logx, title="Histogram of %s" % col)
            c1.plotly_chart(fig, width="stretch")
            c1.caption(FEATURE_HELP.get(col, "Target: median house value ($100k). Note the spike at "
                                             "5.0 caused by the census cap."))
            fx = c2.selectbox("Scatter: x-axis", FEATURES, index=0)
            samp = f.sample(min(3000, len(f)), random_state=0)
            fig = px.scatter(samp, x=fx, y="MedHouseVal", opacity=0.35,
                             title="%s vs house value (up to 3000 points)" % fx)
            c2.plotly_chart(fig, width="stretch")
            st.subheader("Where the houses are")
            fig = px.scatter(samp, x="Longitude", y="Latitude", color="MedHouseVal",
                             color_continuous_scale="Viridis", opacity=0.6,
                             title="Block groups coloured by median house value")
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig, width="stretch")
            st.caption("The coast (Bay Area, Los Angeles) is the most expensive; location is a "
                       "strong signal, which is why Latitude/Longitude rank high in feature importance.")
            st.subheader("Feature correlations")
            st.plotly_chart(px.imshow(f.corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                                      zmin=-1, zmax=1), width="stretch")
    if MATS is not None:
        st.subheader("The benchmark matrices built from this data")
        st.markdown("Each row is an RBF kernel matrix `K + sigma^2 I` on %s standardised training "
                    "rows, with a different length-scale `ell`. Longer length-scales give a larger "
                    "**condition number** (ratio of largest to smallest eigenvalue), which makes "
                    "log-det estimation harder." % get(M, "metrics", "logdet", "n", default="n"))
        st.dataframe(MATS, hide_index=True, width="stretch")

# ================================================================== MODEL
with tabs[2]:
    st.header("Design and pipeline")
    md = load_doc("05_model_design.md")
    st.markdown(doc_section(md, "Components") or "_docs/05_model_design.md not found._")
    pipe = doc_section(md, "Pipeline diagram")
    if pipe:
        st.subheader("Pipeline")
        st.markdown(pipe)
    with st.expander("Hyper-parameters (docs/05_model_design.md)", expanded=True):
        st.markdown(doc_section(md, "Hyper-parameters") or "n/a")
    chg = doc_section(md, "Change made during implementation")
    if chg:
        with st.expander("Change made during implementation"):
            st.markdown(chg)
    if M is not None:
        st.subheader("Values actually used in this run (from metrics.json)")
        sm = get(M, "metrics", "gp_served_model") or {}
        hp = {
            "moments m (primary)": get(M, "metrics", "logdet", "primary_m"),
            "probe vectors d": get(M, "metrics", "logdet", "d"),
            "matrix size n": get(M, "metrics", "logdet", "n"),
            "GP length-scale ell (selected by MEMe)": sm.get("ell"),
            "GP noise variance (selected by MEMe)": sm.get("noise"),
            "GP training rows": sm.get("n_train_subset"),
            "GP grid: ell values": ", ".join("%.3g" % x for x in get(M, "metrics", "gp_grid", "ells", default=[])),
            "GP grid: noise values": ", ".join("%.3g" % x for x in get(M, "metrics", "gp_grid", "noises", default=[])),
            "Ridge alpha": get(M, "baseline_metrics", "ridge", "alpha"),
            "XGBoost params": json.dumps(get(M, "metrics", "xgboost", "params")),
            "XGBoost trees": get(M, "metrics", "xgboost", "n_estimators"),
            "Total training time (s)": get(M, "metrics", "timing", "total_seconds"),
        }
        st.dataframe(pd.DataFrame({"setting": list(hp), "value": [str(x) for x in hp.values()]}),
                     hide_index=True, width="stretch")

    st.header("Training / convergence curves")
    st.markdown("MEMe has no epochs: its 'training' is a Newton solve of the maximum-entropy dual "
                "for every matrix. The curves below show how the solve behaves as more moments "
                "are added - the analogue of a learning curve for this method.")
    if BENCH is None:
        missing_box("The convergence curves")
    else:
        mx = BENCH[BENCH.method.isin(["meme", "meme_hard", "maxent_power"])]
        g = mx.groupby(["method", "m"]).agg(n_iter=("n_iter", "median"),
                                            grad_norm=("grad_norm", "median"),
                                            converged=("converged", lambda s: np.mean(s.astype(str) == "True")),
                                            entropy=("entropy", "median")).reset_index()
        g["label"] = g.method.map(METHOD_LABELS)
        c1, c2 = st.columns(2)
        fig = px.line(g, x="m", y="n_iter", color="label", markers=True, log_x=True,
                      title="Median Newton iterations vs number of moments")
        c1.plotly_chart(fig, width="stretch")
        fig = px.line(g, x="m", y="converged", color="label", markers=True, log_x=True,
                      title="Fraction of solves that converged")
        fig.update_yaxes(range=[-0.05, 1.05])
        c2.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        fig = px.line(g, x="m", y="grad_norm", color="label", markers=True, log_x=True, log_y=True,
                      title="Final gradient norm (lower = moments matched better)")
        c1.plotly_chart(fig, width="stretch")
        ell_pick = c2.selectbox("Entropy curve for length-scale ell", sorted(BENCH.ell.unique()),
                                format_func=lambda x: "%.3g" % x, key="ent_ell")
        e = mx[mx.ell == ell_pick].groupby(["method", "m"]).entropy.median().reset_index()
        e["label"] = e.method.map(METHOD_LABELS)
        fig = px.line(e, x="m", y="entropy", color="label", markers=True, log_x=True,
                      title="Entropy of the fitted spectrum vs m")
        c2.plotly_chart(fig, width="stretch")
        st.caption("Reading the curves: the noise-aware MEMe converges for every m, while the "
                   "hard-constraint and power-moment versions often hit the iteration limit as m "
                   "grows. Entropy should go down (or stay flat) as more moments constrain the "
                   "spectrum.")
    xs = get(M, "metrics", "xgboost", "search") if M else None
    if xs:
        st.subheader("XGBoost baseline: validation search")
        st.dataframe(pd.DataFrame(xs), hide_index=True, width="stretch")

# ================================================================== RESULTS
with tabs[3]:
    st.header("All results")
    if M is None or BENCH is None:
        missing_box("The results page")
    else:
        st.subheader("1. Log-determinant accuracy")
        at = get(M, "metrics", "logdet", "at_primary_m") or {}
        t = pd.DataFrame(at).T
        t.index = [METHOD_LABELS.get(i, i) for i in t.index]
        st.dataframe(t.style.format("{:.4f}"), width="stretch")
        st.caption("Each column is a summary over all matrices and seeds at m = %s. rel_err = "
                   "|estimate - exact| / |exact| (lower is better); seconds = wall time per "
                   "estimate. Exact Cholesky takes %s s on these n = %s matrices - the stochastic "
                   "methods pay off for much larger n, where Cholesky becomes infeasible."
                   % (get(M, "metrics", "logdet", "primary_m"),
                      fmt(get(M, "metrics", "logdet", "exact_cholesky_median_seconds"), 4),
                      get(M, "metrics", "logdet", "n")))

        methods = sorted(BENCH.method.unique())
        sel = st.multiselect("Methods to show", methods, default=methods,
                             format_func=lambda x: METHOD_LABELS.get(x, x))
        ells = sorted(BENCH.ell.unique())
        ell_sel = st.multiselect("Length-scales (ell) to include", ells, default=ells,
                                 format_func=lambda x: "%.3g" % x)
        b = BENCH[BENCH.method.isin(sel) & BENCH.ell.isin(ell_sel)]
        if len(b):
            g = b.groupby(["method", "m"]).rel_err.median().reset_index()
            g["label"] = g.method.map(METHOD_LABELS)
            fig = px.line(g, x="m", y="rel_err", color="label", markers=True, log_x=True, log_y=True,
                          title="Median relative log-det error vs number of moments m",
                          labels={"rel_err": "median relative error", "m": "moments m"})
            st.plotly_chart(fig, width="stretch")
            st.caption("Lower is better. Every point uses m x d matrix-vector products, so at the "
                       "same x-position all methods cost the same.")
            m_pick = st.select_slider("Moments m for the condition-number chart",
                                      sorted(b.m.unique()),
                                      value=get(M, "metrics", "logdet", "primary_m", default=int(b.m.max())))
            c = b[b.m == m_pick].groupby(["method", "cond"]).rel_err.median().reset_index()
            c["label"] = c.method.map(METHOD_LABELS)
            fig = px.line(c, x="cond", y="rel_err", color="label", markers=True, log_x=True, log_y=True,
                          title="Relative error vs condition number (m = %s)" % m_pick,
                          labels={"cond": "condition number of K + sigma^2 I"})
            st.plotly_chart(fig, width="stretch")
            st.caption("Harder (more ill-conditioned) matrices are to the right. The Chebyshev "
                       "polynomial degrades fastest there; that is the regime MEMe was designed for.")
            fig = px.box(b[b.m == m_pick], x="method", y="signed_rel_err", color="method", points="all",
                         title="Signed relative error at m = %s (above 0 = over-estimate)" % m_pick)
            st.plotly_chart(fig, width="stretch")

        sp = load_spectra()
        if sp is not None and "ells" in sp:
            st.subheader("2. What MEMe 'sees': estimated vs true eigenvalue spectrum")
            i = st.selectbox("Kernel matrix (length-scale ell)", list(range(len(sp["ells"]))),
                             format_func=lambda k: "ell = %.3g" % sp["ells"][k], index=min(2, len(sp["ells"]) - 1))
            eig, lam, mass = sp["eig_%d" % i], sp["lam_%d" % i], sp["mass_%d" % i]
            lo = max(min(eig.min(), lam.min()), 1e-6)
            edges = np.logspace(np.log10(lo), np.log10(max(eig.max(), lam.max())), 50)
            h_true = np.histogram(eig, bins=edges)[0] / len(eig)
            h_meme = np.histogram(lam, bins=edges, weights=mass)[0] / max(mass.sum(), 1e-300)
            ctr = np.sqrt(edges[1:] * edges[:-1])
            fig = go.Figure()
            fig.add_bar(x=ctr, y=h_true, name="True eigenvalues (exact)", opacity=0.6)
            fig.add_scatter(x=ctr, y=h_meme, name="MEMe estimated spectrum", mode="lines+markers")
            fig.update_xaxes(type="log", title="eigenvalue (log scale)")
            fig.update_yaxes(title="fraction of eigenvalues in bin")
            fig.update_layout(title="Spectral density, ell = %.3g" % sp["ells"][i])
            st.plotly_chart(fig, width="stretch")
            st.caption("Bars: the real eigenvalues from a full decomposition. Line: the maximum-"
                       "entropy distribution MEMe built from only the stochastic moments. The "
                       "log-det is n times the average of log(eigenvalue) under the line.")

        st.subheader("3. GP model selection on the (ell, noise) grid")
        if GRID is not None:
            cols = [c[4:] for c in GRID.columns if c.startswith("lml_")]
            meth = st.radio("Log-det used inside the marginal likelihood", cols, horizontal=True,
                            format_func=lambda x: METHOD_LABELS.get(x, x))
            piv = GRID.pivot(index="ell", columns="noise", values="lml_" + meth)
            fig = px.imshow(piv.values, x=["%.3g" % v for v in piv.columns], y=["%.3g" % v for v in piv.index],
                            color_continuous_scale="Viridis", aspect="auto", origin="lower",
                            labels={"x": "noise variance", "y": "length-scale ell", "color": "log ML"},
                            title="Log marginal likelihood surface (%s)" % METHOD_LABELS.get(meth, meth))
            r, cidx = np.unravel_index(np.nanargmax(piv.values), piv.shape)
            fig.add_scatter(x=["%.3g" % piv.columns[cidx]], y=["%.3g" % piv.index[r]], mode="markers",
                            marker=dict(symbol="x", size=16, color="red"), name="chosen")
            st.plotly_chart(fig, width="stretch")
            st.caption("Brighter = the data are more likely under those hyper-parameters. The red X is "
                       "what each estimator would choose. A good estimator picks the same cell as "
                       "'exact'.")
        gp_rows = []
        for name, src in (("GP, MEMe-selected", get(M, "metrics", "gp", "meme")),
                          ("GP, exact-selected", get(M, "metrics", "gp", "exact")),
                          ("GP, SLQ-selected", get(M, "metrics", "gp", "slq")),
                          ("GP, Chebyshev-selected", get(M, "baseline_metrics", "gp_chebyshev_selected")),
                          ("Ridge regression", get(M, "baseline_metrics", "ridge")),
                          ("Mean predictor", get(M, "baseline_metrics", "mean_predictor")),
                          ("XGBoost (GP subset)", get(M, "metrics", "xgboost", "gp_subset")),
                          ("XGBoost (full train)", get(M, "metrics", "xgboost"))):
            if src and "test" in src:
                row = {"model": name}
                row.update(src["test"])
                if "ell" in src:
                    row.update(ell=src["ell"], noise=src["noise"])
                gp_rows.append(row)
        if gp_rows:
            gdf = pd.DataFrame(gp_rows)
            st.dataframe(gdf, hide_index=True, width="stretch")
            st.caption("Test-set metrics. rmse/mae in $100k (lower better); r2 (higher better); "
                       "nll = negative log-likelihood of the predictive distribution (lower better); "
                       "coverage_95 = share of test houses inside the 95% interval (ideal 0.95).")
            metric = st.selectbox("Bar chart metric", ["rmse", "mae", "r2", "nll", "coverage_95"])
            fig = px.bar(gdf, x="model", y=metric, color="model", title="Test %s by model" % metric)
            st.plotly_chart(fig, width="stretch")

        if GMM is not None:
            st.subheader("4. Extra: entropy of Gaussian mixtures")
            ferr = [c for c in GMM.columns if c.startswith("ferr_")]
            long = GMM.melt(id_vars=["mixture"], value_vars=ferr, var_name="method", value_name="frac_err")
            long["method"] = long.method.str[5:]
            fig = px.box(long, x="method", y="frac_err", log_y=True, points="all",
                         title="Fractional entropy error over %d random mixtures" % len(GMM))
            st.plotly_chart(fig, width="stretch")
            st.caption("MEMe recovers the entropy of 1-D mixtures from moments much better than "
                       "100-sample Monte Carlo or a single-Gaussian moment match.")

        st.subheader("5. Error analysis of house-price predictions")
        if PRED is None:
            missing_box("Error analysis")
        else:
            st.markdown("`sample_predictions.csv`: %d test houses with the GP's prediction, "
                        "its 95%% interval and the true value." % len(PRED))
            c1, c2 = st.columns(2)
            fig = px.scatter(PRED, x="true_value", y="prediction", color="within_95",
                             error_y=1.96 * PRED.pred_std, opacity=0.6,
                             title="Predicted vs true (error bars = 95% interval)")
            lim = [0, float(max(PRED.true_value.max(), PRED.prediction.max()))]
            fig.add_scatter(x=lim, y=lim, mode="lines", name="perfect", line=dict(dash="dash", color="gray"))
            c1.plotly_chart(fig, width="stretch")
            fig = px.histogram(PRED, x=PRED.prediction - PRED.true_value, nbins=40,
                               title="Residuals (prediction - truth)")
            c2.plotly_chart(fig, width="stretch")
            feat = st.selectbox("Absolute error against feature", FEATURES)
            fig = px.scatter(PRED, x=feat, y="abs_error", color="pred_std", title="Where is the model wrong?")
            st.plotly_chart(fig, width="stretch")
            capped = PRED[PRED.true_value >= 5.0]
            st.markdown("Interval coverage in this sample: **%.1f%%** (target 95%%). %d houses have "
                        "the capped value 5.0; their mean absolute error is %s vs %s for the rest."
                        % (100 * PRED.within_95.mean(), len(capped),
                           fmt(capped.abs_error.mean() if len(capped) else None),
                           fmt(PRED[PRED.true_value < 5.0].abs_error.mean())))
            st.write("Largest 10 errors:")
            st.dataframe(PRED.sort_values("abs_error", ascending=False).head(10),
                         hide_index=True, width="stretch")

        st.subheader("6. Saved figures from train.py")
        pngs = sorted(PLOTS.glob("*.png")) if PLOTS.exists() else []
        for k in range(0, len(pngs), 2):
            cols = st.columns(2)
            for col, p in zip(cols, pngs[k:k + 2]):
                col.image(str(p), caption=p.stem, width="stretch")

# ================================================================== TRY IT
with tabs[4]:
    st.header("Try it")
    mode = st.radio("What do you want to try?",
                    ["Predict a house price (trained GP)", "Estimate a log-determinant live (MEMe vs baselines)"],
                    horizontal=True, key="try_mode")
    if mode.startswith("Predict"):
        model, err = load_model()
        if model is None:
            missing_box("Live prediction")
            if err:
                st.caption("Details: %s" % err)
        else:
            st.markdown("The model is the Gaussian process whose hyper-parameters were chosen with "
                        "the MEMe log-determinant. Set the inputs and it returns a predicted median "
                        "house value with an uncertainty band.")
            src = st.radio("Input", ["Sliders", "Pick a test house", "Upload CSV"], horizontal=True,
                           key="try_src")
            df = load_dataset()
            ref = df if df is not None else (PRED if PRED is not None else None)
            defaults = {"MedInc": 3.87, "HouseAge": 29.0, "AveRooms": 5.4, "AveBedrms": 1.1,
                        "Population": 1200.0, "AveOccup": 3.0, "Latitude": 35.6, "Longitude": -119.6}
            truth = None
            if src == "Pick a test house" and PRED is not None:
                k = st.selectbox("Test house (row id)", list(range(len(PRED))),
                                 format_func=lambda j: "row %d - true value %.2f" % (PRED.row_id[j], PRED.true_value[j]))
                defaults = {f: float(PRED[f][k]) for f in FEATURES}
                truth = float(PRED.true_value[k])
            if src == "Upload CSV":
                up = st.file_uploader("CSV with columns " + ", ".join(FEATURES), type="csv")
                st.caption("Tip: download artifacts/sample_predictions.csv and upload it.")
                if up is not None:
                    try:
                        u = pd.read_csv(up)
                        mean, std = model.predict(u)
                        u["prediction"] = mean
                        u["lower_95"], u["upper_95"] = mean - 1.96 * std, mean + 1.96 * std
                        st.dataframe(u, width="stretch")
                        st.download_button("Download predictions", u.to_csv(index=False), "predictions.csv")
                    except Exception as exc:
                        st.error("Could not score this file: %s" % exc)
            else:
                cols = st.columns(4)
                row = {}
                for j, f in enumerate(FEATURES):
                    if ref is not None and f in ref:
                        lo, hi = float(ref[f].quantile(0.005)), float(ref[f].quantile(0.995))
                    else:
                        lo, hi = defaults[f] * 0.2 - 5, defaults[f] * 2 + 5
                    lo, hi = min(lo, defaults[f]), max(hi, defaults[f])
                    row[f] = cols[j % 4].slider(f, lo, hi, float(defaults[f]), help=FEATURE_HELP[f],
                                                key="sl_%s_%s" % (f, src))
                try:
                    from scipy.stats import norm
                    res = model.predict_one([row[f] for f in FEATURES])
                    conf = float(2 * norm.cdf(0.5 / max(res["std"], 1e-12)) - 1)
                    c = st.columns(4)
                    c[0].metric("Predicted median value", "$%s" % format(int(round(res["prediction"] * 1e5)), ","))
                    c[1].metric("95% interval", "$%sk - $%sk" % (int(round(max(res["lower_95"], 0) * 100)),
                                                              int(round(res["upper_95"] * 100))))
                    c[2].metric("Confidence (within +/- $50k)", "%.0f%%" % (100 * conf))
                    if truth is not None:
                        c[3].metric("True value", "$%s" % format(int(round(truth * 1e5)), ","),
                                    delta="error %+.0fk" % ((res["prediction"] - truth) * 100), delta_color="off")
                    xs = np.linspace(res["prediction"] - 4 * res["std"], res["prediction"] + 4 * res["std"], 200)
                    fig = go.Figure(go.Scatter(x=xs, y=norm.pdf(xs, res["prediction"], res["std"]),
                                               fill="tozeroy", name="predictive density"))
                    if truth is not None:
                        fig.add_vline(x=truth, line_dash="dash", annotation_text="true value")
                    fig.update_layout(title="Predictive distribution (house value, $100k)", height=300)
                    st.plotly_chart(fig, width="stretch")
                    Xs = model.pre.transform([[row[f] for f in FEATURES]])[0]
                    dist = np.sqrt(((model.gp.X - Xs) ** 2).sum(1))
                    near = np.argsort(dist)[:5]
                    st.markdown(
                        "**Why this prediction?** A GP predicts by averaging the training houses that "
                        "are *similar* to yours (similarity = RBF kernel with length-scale ell = %.2f in "
                        "standardised units). The nearest training house is %.2f units away. %s "
                        "Confidence is the probability (under the GP's own Gaussian) that the true "
                        "value is within $50k of the prediction."
                        % (model.ell, dist[near[0]],
                           "Your inputs are close to the training data, so the interval is narrow."
                           if dist[near[0]] < model.ell else
                           "Your inputs are far from any training house, so the model falls back "
                           "towards the average and widens the interval."))
                    nb = pd.DataFrame(model.pre.z_to_y(model._z)[near],
                                      columns=["training house value ($100k)"])
                    nb["distance"] = dist[near]
                    st.dataframe(nb, hide_index=True)
                except Exception as exc:
                    st.error("Invalid input: %s" % exc)
    else:
        st.markdown("Build a GP kernel matrix `K + sigma^2 I` from real data and estimate its "
                    "log-determinant with each method, using the same budget of m x d "
                    "matrix-vector products. The exact Cholesky value is the ground truth.")
        df = load_dataset()
        c = st.columns(4)
        n = c[0].slider("Matrix size n", 100, 800, 400, 50)
        ell = c[1].select_slider("Length-scale ell", [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0], value=1.0)
        s2 = c[2].select_slider("Noise sigma^2", [1e-3, 1e-2, 0.05, 0.1, 0.5], value=1e-2)
        m = c[3].slider("Moments m", 5, 60, 30, 5)
        d = st.slider("Random probe vectors d", 5, 50, 30, 5)
        if st.button("Run estimators", type="primary", key="run_ld") or "ld_res" in st.session_state:
            try:
                from src.gp import kernel_matrix
                from src.model import estimate_logdet
                if df is not None:
                    X = df[FEATURES].to_numpy(float)[np.random.default_rng(0).permutation(len(df))[:n]]
                    X[:, [2, 3, 4, 5]] = np.log1p(X[:, [2, 3, 4, 5]])
                else:
                    X = np.random.default_rng(0).normal(size=(n, 8))
                X = (X - X.mean(0)) / X.std(0)
                A = kernel_matrix(X, ell, s2)
                rows, dens = [], None
                for meth in ["exact", "meme", "meme_hard", "chebyshev", "slq", "maxent_power"]:
                    est, info = estimate_logdet(A, meth, m=m, d=d, seed=0, lam_min=s2, return_info=True)
                    rows.append({"method": METHOD_LABELS[meth], "key": meth, "estimate": est,
                                 "seconds": info["seconds"], "converged": info.get("converged")})
                    if meth == "meme":
                        dens = info
                r = pd.DataFrame(rows)
                ex = float(r.estimate[0])
                r["rel_error"] = (r.estimate - ex).abs() / max(abs(ex), 1e-300)
                eig = np.linalg.eigvalsh(A)
                st.session_state["ld_res"] = True
                st.metric("Condition number of this matrix", "%.3g" % (eig[-1] / eig[0]))
                st.dataframe(r.drop(columns="key"), hide_index=True, width="stretch")
                fig = px.bar(r[r.key != "exact"], x="method", y="rel_error", color="method", log_y=True,
                             title="Relative error vs exact (lower is better)")
                st.plotly_chart(fig, width="stretch")
                best = r[r.key != "exact"].sort_values("rel_error").iloc[0]
                st.markdown("**Result:** the most accurate estimator here is **%s** (relative error "
                            "%.3g). MEMe: %.3g, Chebyshev baseline: %.3g. Try a longer length-scale "
                            "and smaller noise to make the matrix ill-conditioned and see the "
                            "Chebyshev expansion break down."
                            % (best.method, best.rel_error, r.rel_error[r.key == "meme"].iloc[0],
                               r.rel_error[r.key == "chebyshev"].iloc[0]))
                if dens is not None and "lam" in dens:
                    lam, mass = dens["lam"], dens["mass"]
                    edges = np.logspace(np.log10(max(eig.min(), 1e-8)), np.log10(eig.max() * 1.01), 40)
                    ctr = np.sqrt(edges[1:] * edges[:-1])
                    fig = go.Figure()
                    fig.add_bar(x=ctr, y=np.histogram(eig, edges)[0] / n, name="true eigenvalues", opacity=0.6)
                    fig.add_scatter(x=ctr, y=np.histogram(lam, edges, weights=mass)[0] / max(mass.sum(), 1e-300),
                                    name="MEMe spectrum", mode="lines+markers")
                    fig.update_xaxes(type="log", title="eigenvalue")
                    fig.update_layout(title="True vs MEMe-estimated spectrum")
                    st.plotly_chart(fig, width="stretch")
            except Exception as exc:
                st.error("Estimation failed: %s" % exc)
        else:
            st.info("Choose the settings and press **Run estimators** (takes a few seconds).")

# ================================================================== RESEARCH
with tabs[5]:
    st.header("Research documents")
    st.markdown("Source paper: Granziol, Ru, Zohren, Dong, Osborne, Roberts. *MEMe: An Accurate "
                "Maximum Entropy Method for Efficient Approximations in Large-Scale Machine "
                "Learning*. [%s](%s)" % (PAPER_URL, PAPER_URL))
    for title, name in (("Literature survey", "01_literature_survey.md"),
                        ("Problem definition", "02_problem_definition.md"),
                        ("Solution strategy", "03_solution_strategy.md"),
                        ("Dataset", "04_dataset.md"),
                        ("Model design", "05_model_design.md"),
                        ("Results", "06_results.md")):
        txt = load_doc(name)
        with st.expander(title, expanded=(name == "01_literature_survey.md")):
            st.markdown(txt or "_docs/%s not found._" % name)
