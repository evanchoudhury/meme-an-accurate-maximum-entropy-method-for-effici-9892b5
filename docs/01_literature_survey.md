# 01 - Literature Survey

## Source paper

**MEMe: An Accurate Maximum Entropy Method for Efficient Approximations in Large-Scale Machine Learning**
D. Granziol, B. Ru, S. Zohren, X. Dong, M. Osborne, S. Roberts. arXiv:1906.01101 (June 2019); published in *Entropy* 21(6), 551 (2019).
Links: [arXiv abstract](https://arxiv.org/abs/1906.01101v1), [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/1906.01101)

MEMe recovers a probability density from a set of moments by maximising Shannon entropy under moment
constraints. It solves the convex dual for the Lagrange multipliers with a Newton-type method (SciPy
Newton-CG, analytic gradient and Hessian, 1e-8 diagonal jitter, integrals evaluated by quadrature). The
authors show that this is equivalent to constrained Bayesian variational inference. They apply it to two
problems:
1. **Log-determinant estimation.** The matrix is scaled by a Gershgorin bound so its spectrum lies in
   [0, 1]. Moments of the spectrum come from Hutchinson stochastic trace estimation using matrix-vector
   products. MEMe fits the spectral density, and the estimate is `log det A = n * E_p[log lambda] + n * log lambda_u`.
2. **Entropy of Gaussian mixtures** inside an information-theoretic Bayesian optimisation acquisition
   function (FITBO).

## Related work

| Paper | Year | Method | Dataset | Key result (as stated by the authors) | Limitation |
|---|---|---|---|---|---|
| [Hutchinson, "A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines"](https://doi.org/10.1080/03610919008812866) | 1990 | Unbiased trace estimator `tr(A) ~ mean(z^T A z)` with Rademacher (+1/-1) probes | Simulations for GCV with smoothing splines | Minimum-variance estimator among the class considered; close to exact traces for samples of a few hundred | Gives traces (moments) only, not a density or `tr f(A)`; variance falls only as 1/sqrt(probes). (Bibliographic indexes disagree on volume and year; the DOI record says 1990, vol. 19(2).) |
| [Han, Malioutov, Shin, "Large-scale log-determinant computation through stochastic Chebyshev expansions"](https://arxiv.org/abs/1503.06394) (ICML 2015) | 2015 | Chebyshev polynomial expansion of `log(x)` plus Hutchinson trace estimation | Large sparse matrices (tens of millions of variables) | Linear-time randomised algorithm; high accuracy and orders of magnitude faster than Cholesky | The polynomial approximation of `log` degrades when eigenvalues sit near 0 (ill-conditioned matrices) and then needs many terms |
| [Ubaru, Chen, Saad, "Fast estimation of tr(f(A)) via stochastic Lanczos quadrature"](https://www-users.cse.umn.edu/~saad/PDF/ys-2016-04.pdf) (SIMAX 38(4)) | 2017 | Stochastic Lanczos quadrature (SLQ): Hutchinson probes, Lanczos tridiagonalisation, Gauss quadrature | Test matrices for log-det, Schatten norms, Estrada index, trace of inverse | Faster convergence than Chebyshev or Taylor expansions; error bounds provided | Lanczos loses orthogonality in finite precision; each probe needs its own Krylov run |
| [Fitzsimons, Granziol, Cutajar, Osborne, Filippone, Roberts, "Entropic trace estimates for log determinants"](https://arxiv.org/abs/1704.07223) (ECML-PKDD 2017) | 2017 | Maximum-entropy spectral density from stochastic power moments, then log-det | SuiteSparse matrices; a Markov random field application | Significant improvement over state-of-the-art alternatives on SuiteSparse | Uses raw power moments, which are badly conditioned, so only a few moments are usable. MEMe is the follow-up that fixes this. |
| [Dong, Eriksson, Nickisch, Bindel, Wilson, "Scalable log determinants for Gaussian process kernel learning"](https://arxiv.org/abs/1711.03481) (NeurIPS 2017) | 2017 | Chebyshev, Lanczos and surrogate-model estimators of log-det and its derivatives, using only MVMs | GP kernel-learning benchmarks | Lanczos generally beats Chebyshev for kernel learning; surrogates are efficient for popular kernels | Stochastic estimates add noise to hyper-parameter optimisation |
| [Gardner, Pleiss, Bindel, Weinberger, Wilson, "GPyTorch: Blackbox matrix-matrix GP inference with GPU acceleration"](https://arxiv.org/abs/1809.11165) (NeurIPS 2018) | 2018 | BBMM: batched (modified) conjugate gradients yielding solves and Lanczos log-det terms; pivoted-Cholesky preconditioner | GP regression benchmarks | Exact GP inference cost reduced from cubic to quadratic in n | Built for GPUs; log-det is still a Lanczos-type stochastic estimate |
| [Lin, Saad, Yang, "Approximating spectral densities of large matrices"](https://arxiv.org/abs/1308.5467) (SIAM Review 58(1)) | 2016 | Survey of kernel polynomial (Chebyshev) and Lanczos-based density-of-states methods; proposes the Delta-Gauss-Legendre variant | Matrices from physics and numerical linear algebra | Introduces an accuracy metric for comparing spectral density estimates | The density estimates are smoothed or polynomial and can be negative or oscillate (Gibbs effects) |
| [Bandyopadhyay, Bhattacharya, Biswas, Drabold, "Maximum entropy and the problem of moments: a stable algorithm"](https://arxiv.org/abs/cond-mat/0412717) | 2005 | Discretised MaxEnt solved iteratively in the dual space without matrix inversion (OMxnt in the MEMe paper) | Numerically difficult test functions | Stable reconstruction for the classical moment problem | In MEMe's tests (Ecology matrix), OMxnt's error rose after about 3 moments and its entropy was not monotone in m |
| [Ru, McLeod, Granziol, Osborne, "Fast information-theoretic Bayesian optimisation" (FITBO)](https://arxiv.org/abs/1711.00673) (ICML 2018) | 2018 | Information-theoretic acquisition based on the global minimum *value*, using a single extra hyper-parameter sample | BO benchmarks | Performs as well as or better than PES and MES, at lower cost | Needs the entropy of a Gaussian mixture, which has no closed form. MEMe addresses this. |

All links above were found and opened (abstract or listing pages) during this survey. I read full text
only for the source paper, through its ar5iv HTML page. For the other works, the "Key result" column
paraphrases each abstract and adds no numbers.

## Critical review of the source paper

**Strengths**
- The paper rests on a sound idea: use orthogonal-polynomial moments (Chebyshev or Legendre) instead of
  power moments. This removes the conditioning problem that limited earlier MaxEnt methods, Fitzsimons
  et al. 2017 and OMxnt.
- The dual problem is convex. With an analytic gradient and Hessian, Newton's method converges in few
  iterations, and the output is always a valid, positive density. Taylor or Chebyshev surrogates of
  `log` do not guarantee this.
- It needs only matrix-vector products, so the cost is `O(d * m * nnz)`.
- It is honest about where it helps. On the synthetic RBF-kernel experiment, MEMe's gains are large for
  ill-conditioned matrices (for example, relative error 0.00048 vs 0.0838 for Lanczos at condition
  number about 4e9). Lanczos is better on a well-conditioned case (0.0024 vs 0.0522 at about 1.1e3).
  These are the paper's numbers and we do not reuse them as our results.

**Weaknesses**
- The title claims "hundreds of moments", but the reported experiments use m = 10 and m = 30 only.
- The log-det experiments are small: one synthetic n = 1000 kernel, plus the Ecology matrix compared
  only against OMxnt. The synthetic setup has an apparent typo ("66 dimensional").
- There is no end-to-end test of whether a better log-det estimate gives better *downstream* models, for
  example GP hyper-parameters chosen by an approximate marginal likelihood.
- Quadrature resolution and jitter affect accuracy on ill-conditioned problems. The paper does not
  explore these sensitivities systematically.
- Runtime tables come from MATLAB with no units stated.

**What can be reproduced on a laptop (CPU, 16 GB, under 15 minutes)**
- The full log-det pipeline on RBF kernel matrices with n of about 2000-3000, where exact Cholesky
  ground truth costs seconds. Sweeping the length-scale gives condition numbers over many orders of
  magnitude.
- A comparison of MEMe (Chebyshev moments), MaxEnt with raw power moments, a Chebyshev expansion
  (Han et al.) and SLQ (Ubaru et al.) at *equal matrix-vector-product budgets*.
- A moments sweep up to m = 100 or more, to test the "hundreds of moments" claim directly.
- The 1-D Gaussian-mixture entropy experiment, which is cheap. Ground truth comes from dense quadrature,
  and MEMe is compared with Monte Carlo and single-Gaussian moment matching.
- The full FITBO Bayesian-optimisation benchmarks (Michalewicz-5D, Hartmann-6D, 10 seeds) are *not*
  reproduced, because they are too costly and need FITBO's own code.

## Gap this project addresses

1. **Downstream validation.** We use MEMe-estimated log determinants inside the GP log marginal
   likelihood for hyper-parameter selection on a real dataset. We then measure whether the selected
   model predicts as well as one selected with the exact log-det.
2. **The moment-count claim.** We sweep the number of moments well past 30 and report where MEMe stays
   stable and where power-moment MaxEnt breaks down.
3. **Equal-budget, multi-regime comparison.** We compare all estimators at the same number of
   matrix-vector products, across condition numbers from well- to ill-conditioned, on kernels built from
   real data rather than only synthetic Gaussian inputs.
4. **Deployability.** We package the result as an HTTP prediction service (Docker plus Kubernetes) with
   an interactive dashboard. No existing open implementation we found provides this.
