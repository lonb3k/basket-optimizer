"""Forward-looking covariance: let the option market set the level of correlation.

A covariance matrix estimated on the last year assumes next month's correlation will look like last
year's. The option market publishes its own view every day: the CBOE one-month implied correlation
(COR1M). It cannot be plugged in raw, for two reasons:
- it carries a premium: implied correlation sits about 12 points above the correlation realized
  next month on average (see the dispersion project), so raw COR1M overstates risk all the time;
- it is defined against the index (SPX vol vs the top-50 vols), not as the average pairwise
  correlation of the basket's names, which is what a covariance matrix needs.
So COR1M is turned into a forecast first: at each rebalance date, an OLS fitted only on past data maps
COR1M and trailing realized correlation to the average pairwise correlation realized over the next
month. The forecast then sets the correlation level of the covariance in one of two ways:
- "LW, implied target": Ledoit-Wolf with the forecast as the constant-correlation target (the
  analytical intensity is small, ~6%, so this moves the matrix a little);
- "Implied level": the Ledoit-Wolf matrix with its average pairwise correlation moved to the
  forecast, keeping the historical structure (which names move together) and the historical vols.
Vols are never touched, so any difference comes from the correlation level alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import members_at
from .covariance import ANN, ledoit_wolf_cov


def avg_pairwise_corr(returns: pd.DataFrame) -> float:
    """Equal-weighted mean of the off-diagonal correlations (names with a full window only)."""
    r = returns.dropna(axis=1, how="any")
    n = r.shape[1]
    if n < 2:
        return np.nan
    c = np.corrcoef(r.values, rowvar=False)
    return float((c.sum() - n) / (n * (n - 1)))


def correlation_panel(returns: pd.DataFrame, cor: pd.Series, rebalance: int = 21, horizon: int = 21,
                      windows=(21, 63, 252), first: int = 252, members: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per rebalance date t: trailing average pairwise correlations, COR1M at t (as a fraction),
    and the realized average pairwise correlation over the next `horizon` days (t+1 .. t+horizon),
    computed on the names that have a full trailing window at t."""
    rows = []
    cor = cor.reindex(returns.index).ffill() / 100
    for end in range(first, len(returns), rebalance):
        block = returns.iloc[end - max(windows):end]
        eligible = members_at(members, returns.index[end - 1])
        if eligible is not None:
            block = block[[c for c in eligible if c in block.columns]]
        names = block.dropna(axis=1, how="any").columns
        if len(names) < 5:
            continue
        row = {"date": returns.index[end - 1], "cor1m": cor.iloc[end - 1], "n": len(names)}
        for wdw in windows:
            row[f"trail_{wdw}"] = avg_pairwise_corr(returns[names].iloc[end - wdw:end])
        fwd = returns[names].iloc[end:end + horizon]
        row["realized_fwd"] = avg_pairwise_corr(fwd) if len(fwd) == horizon else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def forecast_race(panel: pd.DataFrame, min_obs: int = 36, rolling_obs: int = 60) -> pd.DataFrame:
    """Out-of-sample forecasts of next-month average pairwise correlation.

    At row k only rows < k (whose forward window has closed by date k) are used to fit; the first
    `min_obs` rows get no regression forecast. "rolling" models refit on the last `rolling_obs` rows only
    (60 monthly rows = 5 years), so the mapping can follow a change of regime such as the record-low
    correlation since 2024. Columns: the realized target and one column per model.
    """
    p = panel.dropna(subset=["cor1m", "trail_63", "trail_252"]).copy()
    out = pd.DataFrame(index=p.index)
    out["realized_fwd"] = p["realized_fwd"]
    out["Last year (sample)"] = p["trail_252"]
    out["Last quarter"] = p["trail_63"]
    out["COR1M raw"] = p["cor1m"]
    specs = {"COR1M, debiased": (["cor1m"], None), "COR1M + last quarter": (["cor1m", "trail_63"], None),
             "COR1M + last quarter, 5y rolling": (["cor1m", "trail_63"], rolling_obs),
             "Last quarter, debiased": (["trail_63"], None)}
    for name, (cols, roll) in specs.items():
        fc = np.full(len(p), np.nan)
        for k in range(len(p)):
            train = p.iloc[:k].dropna(subset=["realized_fwd"])
            if roll is not None:
                train = train.iloc[-roll:]
            # the forward window of row k-1 closes at date k: it is known at k, nothing later is
            if len(train) < min_obs:
                continue
            beta = _ols(train[cols].values, train["realized_fwd"].values)
            fc[k] = beta[0] + p.iloc[k][cols].values @ beta[1:]
        out[name] = np.clip(fc, -0.05, 0.95)
    return out


def forecast_scores(race: pd.DataFrame, start: str | None = None) -> pd.DataFrame:
    """RMSE, mean error (bias) and correlation with the realized target, on rows where every model has a forecast."""
    d = race.dropna()
    if start is not None:
        d = d[d.index >= pd.Timestamp(start)]
    y = d["realized_fwd"]
    rows = {}
    for c in d.columns.drop("realized_fwd"):
        e = d[c] - y
        rows[c] = {"RMSE (pts)": float(np.sqrt((e**2).mean()) * 100), "bias (pts)": float(e.mean() * 100),
                   "corr with realized": float(np.corrcoef(d[c], y)[0, 1]), "obs": int(len(d))}
    return pd.DataFrame(rows).T.sort_values("RMSE (pts)")


def shift_correlation_level(cov: pd.DataFrame, rho_target: float) -> pd.DataFrame:
    """Move the average off-diagonal correlation of `cov` to `rho_target`, keep vols and the structure.
    Negative eigenvalues created by the shift are clipped (rarely needed)."""
    sd = np.sqrt(np.diag(cov.values))
    c = cov.values / np.outer(sd, sd)
    n = len(sd)
    rbar = (c.sum() - n) / (n * (n - 1))
    c2 = c + (rho_target - rbar) * (np.ones((n, n)) - np.eye(n))
    vals, vecs = np.linalg.eigh(c2)
    if vals.min() < 1e-8:
        c2 = vecs @ np.diag(np.clip(vals, 1e-8, None)) @ vecs.T
        d = np.sqrt(np.diag(c2))
        c2 = c2 / np.outer(d, d)
    return pd.DataFrame(c2 * np.outer(sd, sd), index=cov.index, columns=cov.columns)


def implied_estimators(rho_hat: pd.Series) -> dict:
    """Covariance estimators that read the correlation forecast in force at the end of the window.
    `rho_hat` is indexed by forecast date; the value used is the last one dated on or before the window's
    last day, so nothing from the future leaks in."""
    rho_hat = rho_hat.dropna().sort_index()

    def _rho(hist: pd.DataFrame) -> float | None:
        s = rho_hat[rho_hat.index <= hist.index[-1]]
        return None if s.empty else float(s.iloc[-1])

    def lw_target(hist: pd.DataFrame) -> pd.DataFrame:
        r = _rho(hist)
        return ledoit_wolf_cov(hist)[0] if r is None else ledoit_wolf_cov(hist, r)[0]

    def level(hist: pd.DataFrame) -> pd.DataFrame:
        base = ledoit_wolf_cov(hist)[0]
        r = _rho(hist)
        return base if r is None else shift_correlation_level(base, r)

    return {"LW, implied target": lw_target, "Implied level": level}


def risk_forecast_test(returns: pd.DataFrame, estimators: dict, weights_fn, window: int = 252,
                       rebalance: int = 21, horizon: int = 21, start: int | None = None,
                       own_weights: bool = False, members: pd.DataFrame | None = None) -> pd.DataFrame:
    """At each rebalance date: forecast the vol of a portfolio with each estimator, then measure the vol
    it realizes over the next `horizon` days. With `own_weights=False`, `weights_fn(hist)` gives one
    portfolio for all estimators (e.g. equal weight), so only the risk forecast differs. With
    `own_weights=True`, `weights_fn(cov)` builds each estimator's own portfolio (e.g. min variance), the
    case where an optimizer exploits the errors of its own matrix. One row per (date, estimator)."""
    rows = []
    first = window if start is None else max(window, start)
    for end in range(first, len(returns) - horizon, rebalance):
        hist = returns.iloc[end - window:end]
        eligible = members_at(members, returns.index[end - 1])
        if eligible is not None:
            hist = hist[[c for c in eligible if c in hist.columns]]
        hist = hist.dropna(axis=1, how="any")
        if hist.shape[1] < 5:
            continue
        fwd = returns[hist.columns].iloc[end:end + horizon].fillna(0)
        w_common = None if own_weights else weights_fn(hist)
        for name, est in estimators.items():
            cov = est(hist)
            w = weights_fn(cov).reindex(hist.columns).fillna(0) if own_weights else w_common
            realized = float(np.sqrt((fwd.values @ w.values).var(ddof=1) * ANN))
            fc = float(np.sqrt(w.values @ cov.values @ w.values))
            rows.append({"date": returns.index[end - 1], "estimator": name, "forecast": fc, "realized": realized})
    return pd.DataFrame(rows)


def risk_scores(test: pd.DataFrame) -> pd.DataFrame:
    """Per estimator: mean forecast and realized vol, bias ratio (realized / forecast, 1 is unbiased),
    QLIKE loss (lower is better; the standard loss for variance forecasts), share of months where the
    realized vol came out above 1.5x the forecast (the risk surprises that hurt)."""
    t = test.copy()
    ratio = (t["realized"] / t["forecast"]) ** 2
    t["qlike"] = ratio - np.log(ratio) - 1
    t["surprise"] = t["realized"] > 1.5 * t["forecast"]
    t["bias"] = t["realized"] / t["forecast"]
    g = t.groupby("estimator")
    return pd.DataFrame({"forecast vol %": g["forecast"].mean() * 100, "realized vol %": g["realized"].mean() * 100,
                         "bias ratio": g["bias"].mean(), "QLIKE": g["qlike"].mean(),
                         "surprises > 1.5x, %": g["surprise"].mean() * 100}).sort_values("QLIKE")
