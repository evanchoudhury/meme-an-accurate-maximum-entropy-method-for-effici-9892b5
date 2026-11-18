# 02 - Problem Definition

## Problem statement

Many large-scale ML models need `log det(A)` of a large symmetric positive-definite matrix. Examples are
Gaussian processes, determinantal point processes and Gaussian graphical models. The exact value
requires a Cholesky factorisation, which costs `O(n^3)` time and `O(n^2)` memory. Stochastic estimators
reduce the cost to matrix-vector products. Polynomial (Chebyshev, Taylor) and Lanczos estimators lose
accuracy when the matrix is ill-conditioned, and GP kernel matrices with long length-scales and small
noise are exactly that case.

This project implements **MEMe**, a maximum-entropy spectral density estimator built from stochastic
Chebyshev moments. It measures MEMe's accuracy against the exact log determinant and against standard
estimators. It then uses MEMe to select GP regression hyper-parameters on a real housing dataset and
serves the resulting GP as an HTTP prediction service.

## Motivation

- GP marginal-likelihood training needs `log det(K + sigma^2 I)` for every candidate hyper-parameter
  setting. Cheap, accurate estimates make kernel learning scale.
- MEMe always returns a valid (positive, normalised) density, so the log-det estimate cannot come from a
  surrogate that is negative somewhere.
- The source paper does not test downstream impact or more than 30 moments. Both are cheap to test.

## Objectives (numbered, measurable)

1. **O1 - Log-det accuracy.** Report the absolute relative error `|est - exact| / |exact|` of MEMe,
   Chebyshev expansion, SLQ and power-moment MaxEnt. Use RBF kernel matrices (n = 2000) from the dataset
   at 6 length-scales and moment counts m in {5, 10, 20, 30, 50, 100}, averaged over 3 probe seeds.
2. **O2 - Conditioning regimes.** Report each matrix's condition number (from exact eigenvalues) and the
   per-regime error of every method.
3. **O3 - Moment stability.** Record whether each MaxEnt variant's Newton solve converges and whether its
   entropy decreases monotonically in m, up to m = 100.
4. **O4 - Downstream GP quality.** Select (length-scale, noise) by maximising the GP log marginal
   likelihood on a grid. Do this once with the exact log-det and once with each estimator. Report test
   RMSE, test NLL and the chosen hyper-parameters.
5. **O5 - Gaussian-mixture entropy.** On random 1-D Gaussian mixtures, report the mean fractional
   entropy error of MEMe (m = 10, 30), Monte Carlo (100 samples) and moment matching against dense
   quadrature.
6. **O6 - Service.** An HTTP JSON API with `/health` and `/predict`. The p50 latency of a single
   prediction is measured and written to `artifacts/metrics.json`.
7. **O7 - Runtime.** The full `python train.py` must finish in under 15 minutes on a CPU laptop. The
   wall-clock time is recorded in `metrics.json`.

## Scope

In scope: dense RBF kernel matrices up to a few thousand rows, Hutchinson moments, MaxEnt dual solved
with Newton, Chebyshev and SLQ baselines, exact GP regression on a subset, grid-based hyper-parameter
selection, 1-D GMM entropy, an HTTP service, Docker and k8s manifests, and a Streamlit dashboard.

## Non-goals

- Reproducing the FITBO Bayesian-optimisation benchmarks (Michalewicz-5D, Hartmann-6D).
- Sparse matrices with millions of rows. The UF Ecology matrix needs a manual download.
- Gradient-based kernel learning, sparse or inducing-point GPs, and GPU acceleration.
- Beating state-of-the-art predictive accuracy on California Housing. The GP is a vehicle for testing
  the log-det estimator.

## Success criteria

- **Primary metric:** the median absolute relative log-det error at m = 30 moments and d = 30 probes,
  across all length-scales and seeds.
- **Simple baseline the method must beat:** the **stochastic Chebyshev expansion** of Han et al. (2015)
  with the *same* number of matrix-vector products (m terms times d probes). MEMe succeeds if its median
  error is lower. SLQ is reported as a stronger reference, and we will report honestly whether MEMe
  beats it.
- **Service-level check:** the GP whose hyper-parameters were selected with MEMe must have lower test
  RMSE than (a) a mean predictor and (b) a Ridge linear regression baseline. Its test RMSE must also be
  within 5 % of the GP selected with the exact log-det.
- All numbers are produced by `train.py` and stored in `artifacts/metrics.json`. None are hand-written.
