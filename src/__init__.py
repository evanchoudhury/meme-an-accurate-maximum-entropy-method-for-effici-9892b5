"""MEMe: maximum-entropy log-determinant estimation, GP model and helpers.

Modules
-------
data          dataset loading (cached), preprocessing, splits, synthetic sample
moments       Hutchinson probes, spectral bounds, Chebyshev and power moments
maxent        MaxEnt dual solved with damped Newton on a graded quadrature grid
baselines     stochastic Chebyshev expansion, stochastic Lanczos quadrature, exact
gp            RBF kernel and exact Gaussian-process regression
gmm_entropy   entropy of 1-D Gaussian mixtures (MEMe, Monte Carlo, moment matching)
model         baseline + main model: log-det estimators and the served GP regressor
evaluate      metrics and plots
"""

__version__ = "1.0.0"
