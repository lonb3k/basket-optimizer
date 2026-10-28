"""Morning sheet: today's basket, today's leverage, and which basket history says to hold this month.

`build_sheet()` runs the existing functions (covariance, optimizers, risk, backtest) and returns
a plain dict: tiles with sparklines, sections of sentences with the numbers filled in, three
ideas ranked by historical conditional expectation against historical worst case, and the
existing Plotly figures as JSON for an annex. Serialized to `sheet/latest.json` by
`scripts/build_sheet.py`. No financial logic lives here.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.io as pio

from . import backtest as bt, charts, covariance as cv, optimizers as opt, risk as rk
from .data import load_cboe_index, load_prices, point_in_time_universe
from .universe import MAG7, load_weights, sector_of

REPO_URL = "https://github.com/lonb3k/basket-optimizer"
SPARK_POINTS = 60
CHART_YEARS = 5
MAX_POINTS = 500
DEFAULTS = dict(estimator="Ledoit-Wolf", window=252, max_weight=0.10, mag7_cap=0.30, cost_bps=5.0, years=11,
                target=0.10, max_leverage=2.0, scheme="Risk parity (ERC)")
HORIZON = 21


# ---------- helpers (same layout as the sister projects) ----------

def _num(x: Any, digits: int = 2) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _spark(s: pd.Series, points: int = SPARK_POINTS) -> list[float]:
    s = s.dropna()
    if s.empty:
        return []
    tail = s.iloc[-points * 2:]
    if len(tail) > points:
        idx = np.linspace(0, len(tail) - 1, points).round().astype(int)
        tail = tail.iloc[idx]
    return [round(float(v), 3) for v in tail.values]


def tile(label: str, value, unit: str = "", sub: str = "", spark: pd.Series | None = None, tone: str = "neutral", digits: int = 1) -> dict:
    return {"label": label, "value": _num(value, digits), "unit": unit, "sub": sub, "tone": tone,
            "spark": _spark(spark) if spark is not None else []}


def _compact(obj):
    if isinstance(obj, list):
        if obj and all(isinstance(v, str) and v.endswith("T00:00:00") for v in obj):
            return [v[:10] for v in obj]
        if obj and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            return [round(v, 3) if isinstance(v, float) else v for v in obj]
        return [_compact(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items()}
    return obj


def _fig(fig, chart_id: str, title: str | None = None) -> dict:
    d = _compact(json.loads(pio.to_json(fig, validate=False)))
    d["layout"].pop("template", None)  # the front end applies its own theme
    return {"id": chart_id, "title": title or fig.layout.title.text, "figure": d}


def _recent(s, years: int = CHART_YEARS, max_points: int = MAX_POINTS):
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    s = s[s.index >= cutoff]
    if len(s) > max_points:
        step = math.ceil(len(s) / max_points)
        s = s.iloc[list(range(len(s) - 1, -1, -step))[::-1]]
    return s


def _pct_rank(s: pd.Series, lookback: int | None = None) -> float:
    w = s.dropna() if lookback is None else s.dropna().iloc[-lookback:]
    return float((w < w.iloc[-1]).mean() * 100) if not w.empty else float("nan")


def _pct_word(p: float) -> str:
    return "very high" if p >= 90 else "high" if p >= 70 else "mid-range" if p >= 30 else "low" if p >= 10 else "very low"


def _score(expected, worst) -> float:
    if expected is None or worst is None or worst >= 0:
        return expected or -1e9
    return expected / abs(worst)


def _forward_return(ret: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Simple return over the next `horizon` days, aligned on the start date, in %."""
    cum = (1 + ret).cumprod()
    return (cum.shift(-horizon) / cum - 1).rename("fwd") * 100


# ---------- the sheet ----------

def build_sheet(offline: bool = False, top_n: int = 50, **overrides) -> dict:
    p = {**DEFAULTS, **overrides}
    today = pd.Timestamp.today().normalize()
    warnings_: list[str] = []
    sections: list[dict] = []
    ideas: list[dict] = []
    chart_list: list[dict] = []

    w = load_weights(top_n)
    w["sector"] = [sector_of(t) for t in w["ticker"]]
    tickers = list(w["ticker"])
    px = load_prices(tickers + ["SPY", "RSP"], start="2005-01-01", offline=offline)
    if not {"SPY", "RSP"} <= set(px.columns):  # a cache written without the benchmarks: refresh it once
        px = load_prices(tickers + ["SPY", "RSP"], start="2005-01-01", offline=False)
    as_of = px.index[-1]
    r = bt.log_returns(px[tickers])
    caps = {"mag7": (MAG7, p["mag7_cap"])} if p["mag7_cap"] < 1 else None
    vix = load_cboe_index("VIX", offline=offline)
    spy_ret = np.exp(bt.log_returns(px[["SPY"]])["SPY"]) - 1
    rsp_ret = np.exp(bt.log_returns(px[["RSP"]])["RSP"]) - 1

    # ---------- today's basket ----------
    r_hist = r.iloc[-p["window"]:].dropna(axis=1)
    cov = cv.ESTIMATORS[p["estimator"]](r_hist)
    lw, delta = cv.ledoit_wolf_cov(r_hist)
    wt = opt.METHODS[p["scheme"]](cov, max_weight=p["max_weight"], group_caps=caps)
    rc = rk.risk_contributions(wt, cov)
    ex_ante = rk.portfolio_vol(wt, cov) * 100
    enb = rk.effective_number_of_bets(rc)
    dr = rk.diversification_ratio(wt, cov)
    mag7_w = float(wt[wt.index.isin(MAG7)].sum() * 100)
    mag7_r = float(rc[rc.index.isin(MAG7)].sum() * 100)
    sec = pd.Series({t: s for t, s in zip(w["ticker"], w["sector"])})
    by_sector = rk.group_shares(wt, rc, sec)
    avg_corr = float((cv.corr_from_cov(lw).values.sum() - len(lw)) / (len(lw) * (len(lw) - 1)))
    cond_sample, cond_lw = float(np.linalg.cond(cv.sample_cov(r_hist).values)), float(np.linalg.cond(lw.values))

    # ---------- the race and the overlay ----------
    # the backtest runs on the point-in-time top N: only names that were then among the largest are eligible at each rebalance
    pit_px, pit_w, _ = point_in_time_universe(offline=offline, top_n=top_n)
    r_pit = bt.log_returns(pit_px)
    rets, stats_, wts = bt.run_race(r_pit, list(opt.METHODS), p["estimator"], p["window"], 21, p["max_weight"], p["cost_bps"], caps,
                                    start=len(r_pit) - 252 * p["years"], members=pit_w.reindex(r_pit.index))
    overlays = {m_: bt.vol_target_overlay(pd.DataFrame({"ret": rets[m_]}), p["target"], "realized", p["max_leverage"], p["cost_bps"], spy_ret, vix)
                for m_ in rets.columns}
    table = pd.DataFrame(stats_).T.drop(columns=["name"]).astype(float)
    bench = pd.DataFrame({"SPY": spy_ret, "RSP": rsp_ret}).reindex(rets.index)
    for b in ("SPY", "RSP"):
        s = bt.stats(bench[b].dropna(), b)
        s.pop("name")
        table.loc[b] = s
    vt_rows = []
    for m_ in rets.columns:
        raw, vt = bt.stats(rets[m_], m_), bt.stats(overlays[m_]["ret"], m_)
        vt_rows.append({"scheme": m_, "sharpe_raw": raw["sharpe"], "sharpe_vt": vt["sharpe"], "maxdd_raw": raw["max_dd_%"], "maxdd_vt": vt["max_dd_%"],
                        "vol_raw": raw["ann_vol_%"], "vol_vt": vt["ann_vol_%"], "avg_leverage": float(overlays[m_]["leverage"].mean())})
    vt_tbl = pd.DataFrame(vt_rows).set_index("scheme")
    gain = float((vt_tbl["sharpe_vt"] - vt_tbl["sharpe_raw"]).mean())

    # today's leverage for the chosen scheme: target / trailing 22d realized vol of the basket, capped
    scheme = p["scheme"]
    base = rets[scheme]
    rv22 = (base.rolling(22).std() * np.sqrt(252)).dropna()
    rv_now = float(rv22.iloc[-1] * 100)
    rv_pct = _pct_rank(rv22)
    lev_now = float(min(p["target"] / (rv22.iloc[-1]), p["max_leverage"]))
    lev_hist = overlays[scheme]["leverage"].dropna()
    lev_avg = float(lev_hist.mean())
    vix_now = float(vix.iloc[-1])
    basket_1d = float(base.iloc[-1] * 100)

    # ---------- tiles ----------
    tiles = [
        tile("VIX", vix_now, "", f"{_pct_rank(vix, 252):.0f}th pct 1y", vix, "neg" if _pct_rank(vix, 252) > 70 else "neutral"),
        tile(f"{scheme} realized 22d", rv_now, "%", f"{rv_pct:.0f}th pct since {rv22.index[0].year}, {_pct_word(rv_pct)}", rv22 * 100, "neutral"),
        tile("Vol-target leverage", lev_now, "x", f"10% target, avg {lev_avg:.2f}x", lev_hist, "pos" if lev_now >= 1 else "neg", 2),
        tile("Ex-ante basket vol", ex_ante, "%", f"{scheme}, Ledoit-Wolf {p['window']}d", None, "neutral"),
        tile("Effective bets", enb, "", f"of {len(wt)} names, DR {dr:.2f}", None, "neutral"),
        tile("Mag 7 weight", mag7_w, "%", f"{mag7_r:.0f}% of risk, cap {p['mag7_cap']*100:.0f}%", None, "neutral"),
        tile("Largest weight", wt.max() * 100, "%", wt.idxmax(), None, "neutral"),
        tile("Shrinkage intensity", delta, "", f"condition {cond_sample:,.0f} -> {cond_lw:,.0f}", None, "neutral", 3),
    ]

    # ---------- sections ----------
    top = wt.sort_values(ascending=False).head(10)
    sections.append({
        "title": "Today's basket",
        "paragraphs": [
            f"{scheme} on the {len(wt)} largest S&P 500 names, Ledoit-Wolf covariance on {p['window']} days (shrinkage intensity {delta:.3f}, "
            f"average pairwise correlation {avg_corr:.2f}), {p['max_weight']*100:.0f}% cap per name and {p['mag7_cap']*100:.0f}% cap on the Mag 7. "
            f"Ex-ante vol {ex_ante:.1f}%, {enb:.1f} effective bets, diversification ratio {dr:.2f}.",
            f"The Mag 7 take {mag7_w:.0f}% of the weight and {mag7_r:.0f}% of the risk; cap-weighting would hold them at "
            f"{w.loc[w['ticker'].isin(MAG7), 'weight'].sum()*100:.0f}%. Largest position {wt.idxmax()} at {wt.max()*100:.1f}%. "
            f"Largest sector by risk: {by_sector.index[0]} ({by_sector.iloc[0]['risk']*100:.0f}% of risk for {by_sector.iloc[0]['weight']*100:.0f}% of weight).",
            f"The sample covariance of {r_hist.shape[1]} names on {p['window']} days has a condition number of {cond_sample:,.0f}; shrinking toward the "
            f"constant-correlation target brings it to {cond_lw:,.0f}. That is why min variance holds {int((opt.min_variance(cov, max_weight=p['max_weight'], group_caps=caps) > 1e-4).sum())} names instead of five.",
        ],
        "table": {"columns": ["Name", "Weight %", "Risk share %", "Sector"],
                  "rows": [[t, _num(v * 100, 2), _num(rc[t] * 100, 2), sec.get(t, "")] for t, v in top.items()]},
    })
    sections.append({
        "title": "Leverage and regime",
        "paragraphs": [
            f"The {scheme} basket realized {rv_now:.1f}% over the last 22 sessions, the {rv_pct:.0f}th percentile since {rv22.index[0].year} ({_pct_word(rv_pct)}); "
            f"VIX at {vix_now:.1f}. A {p['target']*100:.0f}% vol target sets tomorrow's leverage at {lev_now:.2f}x (cap {p['max_leverage']:.1f}x, "
            f"average {lev_avg:.2f}x over the backtest): {'add exposure, the regime is quiet' if lev_now > 1.1 else 'de-lever, realized vol is above target' if lev_now < 0.9 else 'roughly fully invested'}.",
            f"Over {p['years']} years the overlay added {gain:+.2f} of Sharpe on average across the six schemes and cut the average worst drawdown from "
            f"{vt_tbl['maxdd_raw'].mean():.0f}% to {vt_tbl['maxdd_vt'].mean():.0f}%. The spread of Sharpe between the six schemes is "
            f"{table['sharpe'].drop(['SPY', 'RSP']).max() - table['sharpe'].drop(['SPY', 'RSP']).min():.2f}: the vol target moves the result more than the weights.",
            f"On {as_of.strftime('%d %b')} the basket returned {basket_1d:+.2f}% against {float(spy_ret.iloc[-1]*100):+.2f}% for SPY and {float(rsp_ret.iloc[-1]*100):+.2f}% for RSP, "
            f"the equal-weight S&P 500 ETF ({table.loc['RSP', 'ann_return_%']:.0f}% a year over the window against "
            f"{table.loc['Equal weight', 'ann_return_%']:.0f}% for equal weight on the point-in-time top {top_n}).",
        ],
        "table": {"columns": ["Scheme", "Sharpe", "Sharpe vol-targeted", "Max DD %", "Max DD vol-targeted %", "Avg leverage"],
                  "rows": [[m_, _num(rw["sharpe_raw"], 2), _num(rw["sharpe_vt"], 2), _num(rw["maxdd_raw"], 1), _num(rw["maxdd_vt"], 1), _num(rw["avg_leverage"], 2)]
                           for m_, rw in vt_tbl.iterrows()]},
    })

    # ---------- ideas: which basket to hold this month, conditional on the vol regime ----------
    reg_edges = [0, 25, 50, 75, 100.01]
    reg_labels = ["calm (bottom quartile)", "quiet (2nd quartile)", "busy (3rd quartile)", "stressed (top quartile)"]
    rv_pct_series = rv22.rank(pct=True) * 100
    regime_now = str(pd.cut([rv_pct], reg_edges, labels=reg_labels)[0])
    regimes = pd.cut(rv_pct_series, reg_edges, labels=reg_labels).rename("regime")

    def conditional(ret: pd.Series, label: str, instrument: str, thesis: str, direction: str = "long basket") -> dict | None:
        fwd = _forward_return(ret).to_frame().join(regimes, how="inner").dropna()
        in_r = fwd[fwd["regime"].astype(str) == regime_now]
        if len(in_r) < 60:
            return None
        exp_, worst = float(in_r["fwd"].mean()), float(in_r["fwd"].min())
        return {"title": label, "direction": direction, "instrument": instrument,
                "condition": f"basket realized vol in the {regime_now} regime ({rv_pct:.0f}th percentile, {rv_now:.1f}%)",
                "thesis": thesis.format(exp=exp_, pos=float((in_r['fwd'] > 0).mean() * 100), worst=worst, all=float(fwd['fwd'].mean())),
                "stats": {"expected": _num(exp_, 2), "expected_unit": "% over the next 21 days", "hit": _num((in_r["fwd"] > 0).mean() * 100, 0),
                          "worst": _num(worst, 1), "worst_unit": "% over 21 days", "worst_date": in_r["fwd"].idxmin().strftime("%b %Y"),
                          "n": int(len(in_r)), "sample": f"{len(in_r)} overlapping 21-day windows in this regime since {fwd.index[0].year}",
                          "unconditional": f"all regimes: {float(fwd['fwd'].mean()):+.2f}% avg, worst {float(fwd['fwd'].min()):+.1f}% ({fwd['fwd'].idxmin().strftime('%b %Y')})"}}

    cand = [
        conditional(overlays[scheme]["ret"], f"{scheme} with the 10% vol target ({lev_now:.2f}x today)",
                    f"{len(wt)} names, {scheme.lower()} weights, leverage reset daily to 10% / trailing vol",
                    "In this regime the vol-targeted basket returned {exp:+.2f}% over the following month on average ({pos:.0f}% positive, worst {worst:+.1f}%), "
                    "against {all:+.2f}% across all regimes."),
        conditional(rets[scheme], f"{scheme}, unlevered", f"{len(wt)} names, {scheme.lower()} weights, fully invested",
                    "Same weights without the overlay: {exp:+.2f}% over the following month in this regime ({pos:.0f}% positive, worst {worst:+.1f}%)."),
        conditional(rets["Equal weight"], "Equal weight, unlevered", f"{len(wt)} names at {100/len(wt):.0f}% each, rebalanced monthly",
                    "The naive basket: {exp:+.2f}% over the following month in this regime ({pos:.0f}% positive, worst {worst:+.1f}%). "
                    "Backtests use a point-in-time universe: on today's survivors equal weight would have made about 26% a year instead of 15%."),
        conditional(rets["Min variance"], "Min variance, unlevered", f"the quietest {int((wts['Min variance'].iloc[-1] > 1e-4).sum())} names, highest turnover",
                    "The defensive basket: {exp:+.2f}% over the following month in this regime ({pos:.0f}% positive, worst {worst:+.1f}%)."),
    ]
    ideas = [c for c in cand if c]
    for i in ideas:
        i["score"] = _num(_score(i["stats"].get("expected"), i["stats"].get("worst")), 3)
    ideas.sort(key=lambda d: d["score"] if d["score"] is not None else -1e9, reverse=True)
    ideas = ideas[:3]
    for k, i in enumerate(ideas, 1):
        i["rank"] = k

    # ---------- charts ----------
    allr = pd.concat([rets, bench], axis=1)
    chart_list.append(_fig(charts.cumulative(_recent(allr, years=p["years"]), "Growth of 1, log scale (net of costs)"), "growth"))
    chart_list.append(_fig(charts.drawdowns(_recent(allr, years=p["years"]), "Drawdowns"), "drawdowns"))
    chart_list.append(_fig(charts.sharpe_before_after(vt_tbl), "sharpe_vt"))
    chart_list.append(_fig(charts.leverage_path(_recent(overlays[scheme], years=p["years"]), f"{scheme}: leverage set by the vol target"), "leverage"))
    chart_list.append(_fig(charts.weights_bar(wt, rc, f"{scheme}: weight vs share of risk"), "weights"))
    chart_list.append(_fig(charts.group_bars(by_sector, "By sector"), "sectors"))
    order = rk.clusters_from_cov(lw, 6).sort_values().index
    chart_list.append(_fig(charts.heatmap(cv.corr_from_cov(lw).loc[order, order], "Shrunk correlation matrix, names ordered by cluster"), "corr_heatmap"))

    lede = (f"{as_of.strftime('%A %d %B %Y')}. {scheme} basket: {ex_ante:.1f}% ex-ante vol, {enb:.0f} effective bets, Mag 7 {mag7_w:.0f}% of weight "
            f"for {mag7_r:.0f}% of risk. Realized 22d vol {rv_now:.1f}% ({_pct_word(rv_pct)}), VIX {vix_now:.1f}: the 10% vol target sets leverage at {lev_now:.2f}x.")
    if ideas:
        lede += f" Top idea: {ideas[0]['title'][0].lower() + ideas[0]['title'][1:]}."

    return {
        "project": "basket_optimizer", "slug": "basket", "title": "Basket", "subtitle": "Risk-based weights and vol targeting on the S&P 500 top-50",
        "as_of": as_of.strftime("%Y-%m-%d"), "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "headline": lede, "tiles": tiles, "sections": sections, "ideas": ideas, "charts": chart_list,
        "method": "Ideas are the baskets to hold this month, ranked by historical conditional expectation (next 21 days) divided by the historical "
                  "worst case in the same regime. Conditioning variable: the quartile of the basket's trailing 22-day realized vol.",
        "caveats": [
            "Point-in-time universe: monthly top N by today's float shares times that month's price; share-count changes and names delisted since are not captured. Today's basket (first section) uses today's names.",
            "Conditional returns are averages over overlapping 21-day windows; they describe the regime, not a forecast.",
            "Costs: 5 bps per unit of turnover, including leverage changes. No financing cost on leverage above 1.",
        ],
        "warnings": warnings_,
        "links": {"repo": REPO_URL, "streamlit": "https://lonb3k-basket-optimizer.streamlit.app"},
    }
