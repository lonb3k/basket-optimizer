"""Walk-forward backtest of weighting schemes with turnover costs and a vol-target overlay.

Every `rebalance` trading days: estimate the covariance on the trailing `window` days (no
look-ahead), compute the weights, trade from the drifted holdings to the new weights at `cost_bps`
per unit of turnover, hold. The vol-target overlay scales the whole basket by
target / forecast_vol, capped at `max_leverage`, where the forecast is one of:
- "realized": trailing 22-day realized vol of the basket itself
- "vix_scaled": trailing realized vol times VIX / realized vol of the index (uses the market's
  own forecast of the regime)
Leverage changes are traded daily at the same cost. Returns daily net returns per scheme.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .covariance import ESTIMATORS
from .optimizers import METHODS

ANN = 252


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def walk_forward_weights(returns: pd.DataFrame, method: str, estimator: str = "Ledoit-Wolf", window: int = 252,
                         rebalance: int = 21, max_weight: float = 0.1, group_caps=None, start: int | None = None) -> pd.DataFrame:
    """Target weights at each rebalance date (indexed by the date they take effect, i.e. next day)."""
    fn, est = METHODS[method], ESTIMATORS[estimator]
    rows, idx = [], []
    first = window if start is None else max(window, start)
    for end in range(first, len(returns), rebalance):
        hist = returns.iloc[end - window:end].dropna(axis=1, how="any")
        if hist.shape[1] < 5:
            continue
        cov = est(hist)
        w = fn(cov, max_weight=max_weight, group_caps=group_caps)
        rows.append(w.reindex(returns.columns).fillna(0))
        idx.append(returns.index[end])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def simulate(returns: pd.DataFrame, targets: pd.DataFrame, cost_bps: float = 5.0) -> pd.DataFrame:
    """Daily basket returns from target weights: holdings drift with returns between rebalances,
    turnover is charged at each rebalance. Columns: ret, turnover, cost, gross_ret."""
    simple = np.exp(returns.fillna(0)) - 1
    dates = simple.index
    rebal = set(targets.index)
    holdings = None
    out = []
    for d in dates:
        if d in rebal:
            new = targets.loc[d].values
            if holdings is None:
                turnover = 1.0
            else:
                turnover = float(np.abs(new - holdings).sum())
            holdings = new.copy()
            cost = turnover * cost_bps / 1e4
        else:
            if holdings is None:
                continue
            turnover, cost = 0.0, 0.0
        r = simple.loc[d].values
        gross = float(holdings @ r)
        holdings = holdings * (1 + r)
        s = holdings.sum()
        holdings = holdings / s if s > 0 else holdings
        out.append({"date": d, "gross_ret": gross, "turnover": turnover, "cost": cost, "ret": gross - cost})
    return pd.DataFrame(out).set_index("date")


def vol_target_overlay(basket: pd.DataFrame, target: float = 0.10, forecast: str = "realized", max_leverage: float = 2.0,
                       cost_bps: float = 5.0, index_ret: pd.Series | None = None, vix: pd.Series | None = None,
                       window: int = 22) -> pd.DataFrame:
    """Scale the basket to a constant vol target. Leverage set on day t from information up to t-1."""
    r = basket["ret"]
    rv = r.rolling(window).std() * np.sqrt(ANN)
    if forecast == "vix_scaled" and index_ret is not None and vix is not None:
        idx_rv = index_ret.rolling(window).std() * np.sqrt(ANN)
        ratio = (vix.reindex(r.index).ffill() / 100) / idx_rv
        fc = rv * ratio.clip(0.5, 3.0)
    else:
        fc = rv
    lev = (target / fc).clip(upper=max_leverage).shift(1).fillna(1.0)
    dlev = lev.diff().abs().fillna(0)
    out = pd.DataFrame({"ret": lev * r - dlev * cost_bps / 1e4, "leverage": lev, "forecast_vol": fc.shift(1)})
    return out


def stats(ret: pd.Series, name: str = "") -> dict:
    r = ret.dropna()
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    years = len(r) / ANN
    monthly = r.resample("ME").apply(lambda s: (1 + s).prod() - 1)
    return {"name": name, "ann_return_%": float((cum.iloc[-1] ** (1 / years) - 1) * 100), "ann_vol_%": float(r.std() * np.sqrt(ANN) * 100),
            "sharpe": float(r.mean() / r.std() * np.sqrt(ANN)), "max_dd_%": float(dd.min() * 100),
            "worst_month_%": float(monthly.min() * 100), "calmar": float(((cum.iloc[-1] ** (1 / years) - 1) / -dd.min())) if dd.min() < 0 else np.nan,
            "days": int(len(r))}


def run_race(returns: pd.DataFrame, methods: list[str], estimator: str = "Ledoit-Wolf", window: int = 252, rebalance: int = 21,
             max_weight: float = 0.1, cost_bps: float = 5.0, group_caps=None, start: int | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """Returns (daily returns per method, stats per method, weights per method)."""
    rets, st, wts = {}, {}, {}
    for m in methods:
        tw = walk_forward_weights(returns, m, estimator, window, rebalance, max_weight, group_caps, start)
        sim = simulate(returns, tw, cost_bps)
        rets[m] = sim["ret"]
        s = stats(sim["ret"], m)
        s["avg_turnover_%"] = float(sim.loc[sim["turnover"] > 0, "turnover"].mean() * 100)
        s["cost_drag_%"] = float(sim["cost"].sum() / (len(sim) / ANN) * 100)
        st[m] = s
        wts[m] = tw
    return pd.DataFrame(rets), st, wts
