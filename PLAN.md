# PLAN - MEMe log-determinant estimation, GP service and dashboard

## Files to create (phases 2-4)
- `meme/__init__.py`, `meme/moments.py`, `meme/maxent.py`, `meme/baselines.py`, `meme/gp.py`, `meme/gmm_entropy.py`, `meme/data.py`
- `train.py`: runs all experiments (`--quick` flag for a fast smoke run) and writes `artifacts/`
- `serve.py`: `http.server` JSON API with `GET /health`, `GET /info`, `POST /predict`, `POST /logdet`
- `dashboard/app.py`: Streamlit app
- `tests/test_meme.py`, `tests/test_service.py`: run with `python -m unittest`
- `Dockerfile` (default CMD `python serve.py`), `k8s/deployment.yaml` (readiness and liveness probes on `/health`), `k8s/service.yaml`
- `requirements.txt`, `README.md`, `artifacts/model_card.md` (generated)

## Dataset
`sklearn.datasets.fetch_california_housing(as_frame=True)`: 20,640 x 8. `log1p` on skewed columns, StandardScaler, split 70/10/20 with seed 42.

## Models
- **MEMe**: Hutchinson Chebyshev moments, then a Newton MaxEnt spectral density, then the log-det.
- Ablation: MaxEnt with power moments.
- Baselines: stochastic Chebyshev log expansion (primary), SLQ (reference), and exact Cholesky as ground truth.
- Downstream: exact GP (RBF), with hyper-parameters picked by grid-search LML using each log-det estimator. Compared with a mean predictor, RidgeCV and the exact-selected GP.
- Extra: 1-D Gaussian-mixture entropy (MEMe m = 10 and m = 30, MC-100, moment matching) against quadrature.

## Primary metric and baseline
Median absolute relative log-det error at m = 30 and d = 30 over 6 length-scales and 3 seeds. MEMe must beat the stochastic Chebyshev expansion at an equal MVM budget. Service check: the MEMe-selected GP test RMSE must be below Ridge and the mean predictor, and within 5 % of the exact-selected GP.

## Plots (artifacts/plots/*.png)
1. Relative error vs m per method (log scale), one panel per length-scale
2. Relative error vs condition number at m = 30
3. MEMe density vs exact eigenvalue histogram (3 length-scales)
4. MaxEnt entropy vs m (MEMe vs power moments) and Newton convergence
5. GP LML surface (exact vs MEMe) with the chosen points
6. Test RMSE bar chart: mean, Ridge, GP-exact-selected, GP-MEMe, GP-Chebyshev, GP-SLQ
7. Predicted vs actual on the test set and calibration of the predictive std
8. GMM entropy fractional error by method
9. Estimator wall time vs m

## Dashboard pages (Streamlit)
1. **Overview**: problem, headline metrics from `metrics.json`, pipeline diagram
2. **Spectrum Explorer**: choose ell, sigma^2, m and d on a small live matrix (n <= 800). Shows the MEMe density vs the exact histogram and every estimator's log-det vs the exact value.
3. **Log-det Benchmark**: interactive Plotly charts from the CSV (filter by method, m and ell)
4. **GP Model**: LML surfaces, selected hyper-parameters, test metrics vs baselines, predicted vs actual
5. **GMM Entropy**: result table, and a live mixture you can edit
6. **Live Prediction**: feature sliders that call the model in-process (or a running API URL), showing the mean and 95 % interval
7. **Service and Deploy**: endpoints, example curl calls, latency metrics, Docker and k8s notes

## Runtime budget
Benchmark sweep about 3-5 min, GP grid about 3-5 min, GMM under 1 min. Target under 12 min in total, with the measured time recorded.
