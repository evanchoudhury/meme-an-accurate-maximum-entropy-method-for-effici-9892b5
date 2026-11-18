# 03 - Solution Strategy

## Approach and rationale

We estimate `log det(A)` through the matrix's **spectral density** `p(lambda)`. Then
`log det A = n * integral p(lambda) log(lambda) d lambda`. MEMe obtains `p` as the maximum-entropy density
consistent with m stochastic moments of the spectrum:

1. **Scale.** Bound the spectrum. The upper bound `lambda_u` is the Gershgorin maximum absolute row sum.
   The lower bound `lambda_l` is the noise variance `sigma^2`, because `A = K + sigma^2 I` with `K` PSD.
   Map `A` affinely to `B` with spectrum in [-1, 1].
2. **Moments.** Use Hutchinson estimates of `mu_k = (1/n) tr T_k(B)` for k = 0..m, where `T_k` are
   Chebyshev polynomials. The three-term recurrence `T_{k+1}(B) z = 2 B T_k(B) z - T_{k-1}(B) z` gives
   all m moments from m block matrix-vector products with d Rademacher probes.
3. **MaxEnt.** Solve `p(x) = exp(-sum_k alpha_k T_k(x))` on [-1, 1] by minimising the convex dual
   `L(alpha) = integral exp(-sum alpha_k T_k) dx + sum alpha_k mu_k`. Use Newton's method with the
   analytic gradient and Hessian, 1e-8 Hessian jitter, a backtracking line search, and Gauss-Legendre or
   dense-grid quadrature. The grid is graded toward the left edge, where `log` is steep.
4. **Read out.** Compute `log det A = n * E_p[log(lambda(x))]`, where `lambda(x)` maps back to the
   original scale.

The paper itself reports that orthogonal (Chebyshev or Legendre) moments are much better conditioned
than power moments. This is the property that should let MEMe use tens to hundreds of moments.

## Alternatives considered

| Alternative | Why not the main method |
|---|---|
| Exact Cholesky | It is the ground truth, but cubic cost is the problem we want to avoid. We use it only for evaluation. |
| Chebyshev expansion of log (Han et al. 2015) | Kept as the **primary baseline**. It is simple, but its polynomial fit of log is poor near 0. |
| Stochastic Lanczos quadrature (Ubaru et al. 2017) | Kept as a strong reference baseline. |
| MaxEnt with power moments (Fitzsimons et al. 2017) | Kept as an ablation, to show the conditioning problem MEMe fixes. |
| Taylor series of log | The paper argues it is unsound, because its implied density can be negative. Omitted. |
| Inducing-point or sparse GP | Changes the model, not the log-det. Out of scope. |

## Pipeline (step by step)

1. Load California Housing with `sklearn.datasets.fetch_california_housing`. Split into train, validation
   and test with seed 42. Standardise using train statistics only.
2. **Benchmark matrices.** Take a fixed random subset of n = 2000 training rows. For each length-scale in
   a log-spaced grid, build `A = K_rbf(X, X; ell) + sigma^2 I` and compute the exact log-det and
   eigenvalues (`numpy.linalg.eigvalsh`) for the ground truth and condition number.
3. **Estimator sweep.** For each matrix, each m in {5, 10, 20, 30, 50, 100} and each of 3 seeds, run MEMe
   (Chebyshev moments), MaxEnt with power moments, the Chebyshev expansion and SLQ with d = 30 probes.
   All methods share the same probes per seed and the same budget of m * d MVMs. Record the estimate,
   relative error, wall time, Newton convergence and entropy.
4. **GP hyper-parameter selection.** On a GP training subset (n = 2000), evaluate the log marginal
   likelihood over a length-scale by noise grid. The data-fit term `y^T A^{-1} y` is computed exactly by
   Cholesky solve, so only the log-det term differs between estimators. Pick the argmax per estimator.
5. **Final model.** Fit an exact GP with the MEMe-selected hyper-parameters. Evaluate RMSE, MAE, R^2 and
   NLL on the test set. Compare with the exact-selected GP, a mean predictor and Ridge regression.
6. **GMM entropy.** On 50 random 1-D mixtures, compare the entropy errors of MEMe (m = 10, 30), MC-100 and
   moment matching against dense quadrature.
7. **Artifacts.** Write `model.joblib` (GP state and scalers), `metrics.json`, CSV result tables, PNG
   plots and a model card.
8. **Serve.** `serve.py` (standard-library `http.server`) loads the artifact and exposes `/health`,
   `/predict`, `/info` and `/logdet`. The `/logdet` endpoint runs MEMe on a kernel matrix built from
   posted rows.
9. **Explore.** A Streamlit dashboard reads the artifacts and runs MEMe live on small matrices.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Newton diverges for large m or extreme spectra | Hessian jitter, backtracking line search, gradient-norm stop. If it still fails, fall back to the best iterate and record `converged=false` in the results. Divergence is reported, not hidden. |
| Spectrum mass piles up at the lower edge (`log` singularity) | Use the exact lower bound `sigma^2`, a quadrature grid graded toward -1, and an explicit integral of log over the grid. |
| Hutchinson moments violate realisability (noisy high-order moments) | Damp high-order moments with Jackson weights as an option. Report results both with and without damping if it is used. |
| Runtime over 15 minutes | n = 2000 dense matrices (eigvalsh takes seconds), a capped grid, and a `--quick` flag for CI. Total time is recorded. |
| Dataset download fails offline | `fetch_california_housing` caches to the scikit-learn data home. A clear error message tells the user to retry with network access. |
| GP exact solve memory | n = 2000 gives 32 MB per matrix, which is well within 16 GB. |
| MEMe does not beat SLQ | That is acceptable. We report it honestly. The success criterion is defined against the Chebyshev baseline. |
