"""Long-only weighting schemes, all driven by the covariance matrix only (no return forecasts:
expected returns are where estimation error does the most damage, so none of these use them).

- equal_weight
- inverse_vol: w_i ∝ 1/σ_i
- min_variance: argmin w'Σw, long-only, per-name cap
- risk_parity (ERC): every name contributes the same share of portfolio variance
- max_diversification: maximize (w'σ) / sqrt(w'Σw), the diversification ratio (Choueifaty 2008)
- hrp: hierarchical risk parity (Lopez de Prado 2016), no optimizer, recursive bisection on the
  correlation-distance tree
Constraints: weights sum to one, 0 ≤ w_i ≤ max_weight, optional cap on the total weight of a group
(e.g. the Mag 7, or one sector).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


def _bounds(n: int, max_weight: float):
    return [(0.0, max_weight)] * n


def _group_constraints(tickers, group_caps: dict[str, tuple[list[str], float]] | None):
    cons = []
    if not group_caps:
        return cons
    for _, (members, cap) in group_caps.items():
        idx = [i for i, t in enumerate(tickers) if t in members]
        if idx:
            cons.append({"type": "ineq", "fun": lambda w, idx=idx, cap=cap: cap - w[idx].sum()})
    return cons


def equal_weight(cov: pd.DataFrame, **_) -> pd.Series:
    n = len(cov)
    return pd.Series(1 / n, index=cov.index)


def inverse_vol(cov: pd.DataFrame, max_weight: float = 1.0, **_) -> pd.Series:
    iv = 1 / np.sqrt(np.diag(cov.values))
    w = iv / iv.sum()
    w = np.minimum(w, max_weight)
    return pd.Series(w / w.sum(), index=cov.index)


def _solve(objective, cov, max_weight, group_caps, x0=None, extra_cons=()):
    n = len(cov)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}] + _group_constraints(list(cov.index), group_caps) + list(extra_cons)
    x0 = np.full(n, 1 / n) if x0 is None else x0
    res = minimize(objective, x0, method="SLSQP", bounds=_bounds(n, max_weight), constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=cov.index)


def min_variance(cov: pd.DataFrame, max_weight: float = 0.1, group_caps=None, **_) -> pd.Series:
    S = cov.values
    return _solve(lambda w: w @ S @ w, cov, max_weight, group_caps)


def risk_parity(cov: pd.DataFrame, max_weight: float = 1.0, group_caps=None, **_) -> pd.Series:
    S = cov.values
    n = len(S)

    def obj(w):
        port_var = w @ S @ w
        rc = w * (S @ w) / port_var
        return ((rc - 1 / n) ** 2).sum() * 1e4

    iv = inverse_vol(cov).values
    return _solve(obj, cov, max_weight, group_caps, x0=iv)


def max_diversification(cov: pd.DataFrame, max_weight: float = 0.1, group_caps=None, **_) -> pd.Series:
    S = cov.values
    sd = np.sqrt(np.diag(S))
    return _solve(lambda w: -(w @ sd) / np.sqrt(w @ S @ w), cov, max_weight, group_caps)


def hrp(cov: pd.DataFrame, max_weight: float = 1.0, **_) -> pd.Series:
    corr = cov.values / np.outer(np.sqrt(np.diag(cov.values)), np.sqrt(np.diag(cov.values)))
    dist = np.sqrt(0.5 * (1 - np.clip(corr, -1, 1)))
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method="single")
    order = list(leaves_list(link))
    w = pd.Series(1.0, index=cov.index[order])
    clusters = [order]

    def cluster_var(idx):
        sub = cov.values[np.ix_(idx, idx)]
        ivp = 1 / np.diag(sub)
        ivp = ivp / ivp.sum()
        return ivp @ sub @ ivp

    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            half = len(c) // 2
            a, b = c[:half], c[half:]
            va, vb = cluster_var(a), cluster_var(b)
            alpha = 1 - va / (va + vb)
            w[cov.index[a]] *= alpha
            w[cov.index[b]] *= 1 - alpha
            nxt += [a, b]
        clusters = nxt
    w = w.reindex(cov.index)
    w = np.minimum(w, max_weight)
    return w / w.sum()


METHODS = {
    "Equal weight": equal_weight,
    "Inverse vol": inverse_vol,
    "Min variance": min_variance,
    "Risk parity (ERC)": risk_parity,
    "Max diversification": max_diversification,
    "HRP": hrp,
}
