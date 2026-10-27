"""Covariance estimators. The optimizer is only as good as this matrix.

- sample: plain covariance of daily log returns, annualized.
- ledoit_wolf: shrinkage toward the constant-correlation target (Ledoit and Wolf 2004, "Honey, I
  shrunk the sample covariance matrix"), with the analytical shrinkage intensity. On 50 names and
  one year of data the sample matrix has 1275 free parameters for 252 observations: the shrunk
  one is what a risk desk would use.
- ewma: exponentially weighted, RiskMetrics style, for a faster-reacting estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * ANN


def ewma_cov(returns: pd.DataFrame, halflife: int = 63) -> pd.DataFrame:
    lam = 0.5 ** (1 / halflife)
    x = returns.values - returns.values.mean(axis=0)
    n = len(x)
    w = lam ** np.arange(n - 1, -1, -1)
    w = w / w.sum()
    cov = (x * w[:, None]).T @ x
    return pd.DataFrame(cov * ANN, index=returns.columns, columns=returns.columns)


def ledoit_wolf_cov(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Constant-correlation shrinkage. Returns (covariance, shrinkage intensity)."""
    x = returns.values - returns.values.mean(axis=0)
    t, n = x.shape
    s = x.T @ x / t
    var = np.diag(s)
    sd = np.sqrt(var)
    corr = s / np.outer(sd, sd)
    rbar = (corr.sum() - n) / (n * (n - 1))
    f = rbar * np.outer(sd, sd)
    np.fill_diagonal(f, var)
    # pi: asymptotic variance of the sample covariance entries
    y = x**2
    pi_mat = (y.T @ y) / t - s**2
    pi_hat = pi_mat.sum()
    # rho: covariance between sample cov and target entries
    term = ((x**3).T @ x) / t  # theta_ii,ij
    theta = term - var[:, None] * s
    np.fill_diagonal(theta, 0)
    rho_hat = np.diag(pi_mat).sum() + rbar * ((1 / sd)[:, None] * sd[None, :] * theta).sum()
    gamma = ((f - s) ** 2).sum()
    kappa = (pi_hat - rho_hat) / gamma if gamma > 0 else 0.0
    delta = float(np.clip(kappa / t, 0.0, 1.0))
    sigma = delta * f + (1 - delta) * s
    return pd.DataFrame(sigma * ANN, index=returns.columns, columns=returns.columns), delta


ESTIMATORS = {"Ledoit-Wolf": lambda r: ledoit_wolf_cov(r)[0], "Sample": sample_cov, "EWMA (63d)": ewma_cov}


def corr_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    sd = np.sqrt(np.diag(cov.values))
    return pd.DataFrame(cov.values / np.outer(sd, sd), index=cov.index, columns=cov.columns)
