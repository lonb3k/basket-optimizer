"""Risk decomposition of a basket: contributions, concentration, effective number of bets,
clusters, tracking error against the index."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ANN = 252


def portfolio_vol(w: pd.Series, cov: pd.DataFrame) -> float:
    w = w.reindex(cov.index).fillna(0).values
    return float(np.sqrt(w @ cov.values @ w))


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Share of portfolio variance attributable to each name (sums to one)."""
    wv = w.reindex(cov.index).fillna(0).values
    var = wv @ cov.values @ wv
    rc = wv * (cov.values @ wv) / var
    return pd.Series(rc, index=cov.index)


def effective_number_of_bets(rc: pd.Series) -> float:
    """Entropy of the risk contributions: 1 = all risk in one name, N = perfectly balanced."""
    p = rc.clip(lower=1e-12)
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def herfindahl(w: pd.Series) -> float:
    return float((w**2).sum())


def diversification_ratio(w: pd.Series, cov: pd.DataFrame) -> float:
    wv = w.reindex(cov.index).fillna(0).values
    sd = np.sqrt(np.diag(cov.values))
    return float((wv @ sd) / np.sqrt(wv @ cov.values @ wv))


def clusters_from_cov(cov: pd.DataFrame, n_clusters: int = 6) -> pd.Series:
    corr = cov.values / np.outer(np.sqrt(np.diag(cov.values)), np.sqrt(np.diag(cov.values)))
    dist = np.sqrt(0.5 * (1 - np.clip(corr, -1, 1)))
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(link, n_clusters, criterion="maxclust")
    return pd.Series(labels, index=cov.index, name="cluster")


def group_shares(w: pd.Series, rc: pd.Series, groups: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"weight": w, "risk": rc, "group": groups.reindex(w.index)})
    return df.groupby("group")[["weight", "risk"]].sum().sort_values("risk", ascending=False)


def tracking_stats(basket_ret: pd.Series, index_ret: pd.Series) -> dict:
    d = pd.DataFrame({"b": basket_ret, "i": index_ret}).dropna()
    active = d["b"] - d["i"]
    beta = float(np.cov(d["b"], d["i"])[0, 1] / d["i"].var())
    return {"tracking_error": float(active.std() * np.sqrt(ANN) * 100), "beta": beta,
            "correlation": float(d.corr().iloc[0, 1]), "active_return": float(active.mean() * ANN * 100)}
