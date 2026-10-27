"""Vol-targeted basket optimizer on the S&P 500 top-50.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from basket import backtest as bt, charts, covariance as cv, optimizers as opt, risk as rk  # noqa: E402
from basket.data import load_cboe_index, load_prices  # noqa: E402
from basket.universe import MAG7, load_weights, sector_of  # noqa: E402

st.set_page_config(page_title="Basket Optimizer", page_icon="🧺", layout="wide")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def universe(top_n: int) -> pd.DataFrame:
    w = load_weights(top_n)
    w["sector"] = [sector_of(t) for t in w["ticker"]]
    return w


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def prices(tickers: tuple[str, ...], offline: bool) -> pd.DataFrame:
    return load_prices(list(tickers), start="2005-01-01", offline=offline)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def race(tickers: tuple[str, ...], methods: tuple[str, ...], estimator: str, window: int, max_weight: float,
         mag7_cap: float, cost_bps: float, years: int, offline: bool):
    px = load_prices(list(tickers), start="2005-01-01", offline=offline)
    r = bt.log_returns(px[list(tickers)])
    caps = {"mag7": (MAG7, mag7_cap)} if mag7_cap < 1 else None
    rets, stats_, wts = bt.run_race(r, list(methods), estimator, window, 21, max_weight, cost_bps, caps, start=len(r) - 252 * years)
    return rets, stats_, wts


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cap_sweep(tickers: tuple[str, ...], method: str, estimator: str, window: int, max_weight: float, cost_bps: float, years: int, offline: bool):
    px = load_prices(list(tickers), start="2005-01-01", offline=offline)
    r = bt.log_returns(px[list(tickers)])
    rows = []
    for cap in (0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0):
        caps = {"mag7": (MAG7, cap)} if cap < 1 else None
        tw = bt.walk_forward_weights(r, method, estimator, window, 21, max_weight, caps, start=len(r) - 252 * years)
        sim = bt.simulate(r, tw, cost_bps)
        s = bt.stats(sim["ret"])
        enb = np.mean([rk.effective_number_of_bets(rk.risk_contributions(tw.loc[d], cv.ESTIMATORS[estimator](r.loc[:d].iloc[-window:].dropna(axis=1))))
                       for d in tw.index[::6]])
        rows.append({"mag7_cap": cap, "Sharpe": s["sharpe"], "Ann. return %": s["ann_return_%"], "Max drawdown %": s["max_dd_%"],
                     "Worst month %": s["worst_month_%"], "ENB": enb, "avg Mag 7 weight": float(tw[[c for c in tw.columns if c in MAG7]].sum(axis=1).mean())})
    return pd.DataFrame(rows)


def fmt(x, suffix="%", d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}{suffix}"


with st.sidebar:
    st.title("Basket Optimizer")
    st.caption("Risk-based weights on the S&P 500 top-50, walk-forward, with costs and a vol target.")
    top_n = st.slider("Universe size", 20, 50, 50, 5)
    estimator = st.selectbox("Covariance", list(cv.ESTIMATORS), 0)
    window = st.select_slider("Estimation window (days)", [126, 252, 504], value=252)
    max_weight = st.slider("Max weight per name", 0.03, 0.25, 0.10, 0.01)
    mag7_cap = st.slider("Cap on Mag 7 total weight", 0.10, 1.0, 0.30, 0.05)
    cost_bps = st.slider("Cost per unit turnover (bps)", 0.0, 20.0, 5.0, 1.0)
    years = st.slider("Backtest years", 5, 18, 11)
    offline = st.toggle("Offline", value=False)

w = universe(top_n)
tickers = tuple(w["ticker"])
px = prices(tickers + ("SPY", "RSP"), offline)
caps = {"mag7": (MAG7, mag7_cap)} if mag7_cap < 1 else None

st.title("Where does a basket's performance come from?")
st.caption(f"{top_n} largest S&P 500 names, {w['index_weight'].sum()*100:.0f}% of the index. Last price {px.index[-1].date()}.")
st.warning("**Survivorship, stated upfront.** Today's top 50 are the winners of the last decade, so every basket built on them beats SPY. "
           "Comparisons between schemes are fair (same universe); comparisons with the index are not. RSP (equal-weight S&P 500 ETF) is "
           "shown as the honest equal-weight benchmark without survivorship.", icon="⚠️")

tabs = st.tabs(["Build today", "Race", "Vol targeting", "Concentration", "Diagnostics"])

# ---------- Build today ----------
with tabs[0]:
    method = st.selectbox("Weighting scheme", list(opt.METHODS), 3)
    r_hist = bt.log_returns(px[list(tickers)]).iloc[-window:].dropna(axis=1)
    cov = cv.ESTIMATORS[estimator](r_hist)
    wt = opt.METHODS[method](cov, max_weight=max_weight, group_caps=caps)
    wt_free = opt.METHODS[method](cov, max_weight=1.0, group_caps=None)
    rc = rk.risk_contributions(wt, cov)
    c = st.columns(6)
    c[0].metric("Basket vol (ex ante)", fmt(rk.portfolio_vol(wt, cov) * 100), f"unconstrained {rk.portfolio_vol(wt_free, cov)*100:.1f}%", delta_color="off")
    c[1].metric("Effective number of bets", fmt(rk.effective_number_of_bets(rc), "", 1), f"of {len(wt)} names", delta_color="off")
    c[2].metric("Diversification ratio", fmt(rk.diversification_ratio(wt, cov), "", 2))
    c[3].metric("Largest weight", fmt(wt.max() * 100), wt.idxmax(), delta_color="off")
    c[4].metric("Mag 7 weight", fmt(wt[wt.index.isin(MAG7)].sum() * 100), f"risk {rc[rc.index.isin(MAG7)].sum()*100:.0f}%", delta_color="off")
    c[5].metric("Names held", int((wt > 1e-4).sum()))
    st.plotly_chart(charts.weights_bar(wt, rc, f"{method}: weight vs share of risk"), use_container_width=True)
    sec = pd.Series({t: s for t, s in zip(w["ticker"], w["sector"])})
    clus = rk.clusters_from_cov(cov, 6)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.group_bars(rk.group_shares(wt, rc, sec), "By sector"), use_container_width=True)
    c2.plotly_chart(charts.group_bars(rk.group_shares(wt, rc, clus.astype(str)), "By correlation cluster (6 clusters from the covariance)"), use_container_width=True)
    st.markdown(
        "**Weight is not risk.** A 10% weight in a 50-vol name that moves with everything else is a far bigger bet than 10% in a 15-vol staple. "
        "The orange bars are the shares of portfolio variance; the effective number of bets is the entropy of those shares. "
        "Equal weight on this universe is about 30 effective bets; a cap-weighted basket is under 10."
    )
    with st.expander("Weights"):
        st.dataframe(pd.DataFrame({"weight": wt, "risk_share": rc, "sector": sec.reindex(wt.index), "cluster": clus}).sort_values("weight", ascending=False).round(4), use_container_width=True)

# ---------- Race ----------
with tabs[1]:
    with st.spinner("Walk-forward backtest of six schemes..."):
        rets, stats_, wts = race(tickers, tuple(opt.METHODS), estimator, window, max_weight, mag7_cap, cost_bps, years, offline)
    bench = (np.exp(bt.log_returns(px[["SPY", "RSP"]])) - 1).reindex(rets.index)
    allr = pd.concat([rets, bench], axis=1)
    table = pd.DataFrame(stats_).T.drop(columns=["name"])
    for b in ("SPY", "RSP"):
        s = bt.stats(bench[b].dropna(), b)
        s.pop("name")
        table.loc[b] = s
    st.dataframe(table.astype(float).style.format("{:.2f}").highlight_max(subset=["sharpe", "calmar"], color="#cde2fb"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.cumulative(allr, "Growth of 1, log scale (net of costs)"), use_container_width=True)
    c2.plotly_chart(charts.drawdowns(allr, "Drawdowns"), use_container_width=True)
    best = table["sharpe"].drop(["SPY", "RSP"]).idxmax()
    st.markdown(
        f"**Reading the race.** Best Sharpe among the schemes: **{best}**. The spread between the six schemes is small next to the gap between "
        "any of them and the cap-weighted index: on this universe the choice of risk-based weights matters less than the decision not to "
        "cap-weight. Min variance has the lowest vol and the highest turnover (it chases the quietest names), HRP and risk parity get most of "
        "the diversification at a fraction of the trading. RSP shows what equal weight did without survivorship: the honest number."
    )
    by_year = allr.groupby(allr.index.year).apply(lambda d: (1 + d).prod() - 1) * 100
    with st.expander("Returns by year, %"):
        st.dataframe(by_year.round(1), use_container_width=True)

# ---------- Vol targeting ----------
with tabs[2]:
    with st.spinner("Overlays..."):
        rets, stats_, wts = race(tickers, tuple(opt.METHODS), estimator, window, max_weight, mag7_cap, cost_bps, years, offline)
    target = st.slider("Vol target", 0.05, 0.20, 0.10, 0.01)
    max_lev = st.slider("Max leverage", 1.0, 3.0, 2.0, 0.25)
    forecast = st.radio("Vol forecast", ["realized", "vix_scaled"], horizontal=True, format_func=lambda s: {"realized": "Trailing 22d realized", "vix_scaled": "Realized × VIX / index realized"}[s])
    spy_ret = (np.exp(bt.log_returns(px[["SPY"]])["SPY"]) - 1)
    vix = load_cboe_index("VIX", offline=offline)
    rows, paths = [], {}
    for m in rets.columns:
        raw = bt.stats(rets[m], m)
        ov = bt.vol_target_overlay(pd.DataFrame({"ret": rets[m]}), target, forecast, max_lev, cost_bps, spy_ret, vix)
        vt = bt.stats(ov["ret"], m)
        rows.append({"scheme": m, "sharpe_raw": raw["sharpe"], "sharpe_vt": vt["sharpe"], "vol_raw": raw["ann_vol_%"], "vol_vt": vt["ann_vol_%"],
                     "maxdd_raw": raw["max_dd_%"], "maxdd_vt": vt["max_dd_%"], "worst_month_raw": raw["worst_month_%"], "worst_month_vt": vt["worst_month_%"],
                     "avg_leverage": float(ov["leverage"].mean())})
        paths[m] = ov
    tbl = pd.DataFrame(rows).set_index("scheme")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.sharpe_before_after(tbl), use_container_width=True)
    pick = c2.selectbox("Show overlay for", list(rets.columns), 3)
    c2.plotly_chart(charts.leverage_path(paths[pick], f"{pick}: leverage set by the vol target"), use_container_width=True)
    st.dataframe(tbl.round(2), use_container_width=True)
    both = pd.DataFrame({f"{pick} raw": rets[pick], f"{pick} vol-targeted": paths[pick]["ret"]})
    st.plotly_chart(charts.drawdowns(both, "Drawdowns, raw vs vol-targeted"), use_container_width=True)
    gain = (tbl["sharpe_vt"] - tbl["sharpe_raw"]).mean()
    st.markdown(
        f"**This is where the performance comes from.** Vol targeting adds **{gain:+.2f}** of Sharpe on average across the six schemes and cuts "
        f"the worst drawdown from {tbl['maxdd_raw'].mean():.0f}% to {tbl['maxdd_vt'].mean():.0f}%, more than any choice of weights. "
        "The mechanism: vol clusters and spikes before the worst returns, so de-levering when realized vol rises sidesteps part of every crash "
        "and re-levering in calm markets harvests the quiet trend. The VIX-scaled forecast reacts faster than trailing realized vol because "
        "implied vol jumps on the first bad day. Costs of changing leverage are charged daily."
    )

# ---------- Concentration ----------
with tabs[3]:
    m_sw = st.selectbox("Scheme for the sweep", list(opt.METHODS), 3, key="sweep")
    with st.spinner("Sweeping the Mag 7 cap..."):
        sweep = cap_sweep(tickers, m_sw, estimator, window, max_weight, cost_bps, years, offline)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.sweep_lines(sweep, "mag7_cap", ["Sharpe"], "Sharpe vs cap on Mag 7 weight"), use_container_width=True)
    c2.plotly_chart(charts.sweep_lines(sweep, "mag7_cap", ["Max drawdown %", "Worst month %"], "Tail vs cap on Mag 7 weight", "%"), use_container_width=True)
    st.dataframe(sweep.round(2), use_container_width=True)
    st.markdown(
        "**What a concentration limit costs.** The cap binds only when the scheme wants more Mag 7 than allowed (cap-weighting would hold 54%). "
        "Risk-based schemes already hold 12-16% because the giants are the most volatile names, so a 30% cap costs them nothing. "
        "The sweep is the argument to bring to a risk committee: the return given up per point of drawdown saved."
    )

# ---------- Diagnostics ----------
with tabs[4]:
    r_hist = bt.log_returns(px[list(tickers)]).iloc[-window:].dropna(axis=1)
    lw, delta = cv.ledoit_wolf_cov(r_hist)
    c1, c2, c3 = st.columns(3)
    c1.metric("Ledoit-Wolf shrinkage intensity", fmt(delta, "", 3), f"{window} days, {r_hist.shape[1]} names", delta_color="off")
    c2.metric("Average pairwise correlation", fmt(float((cv.corr_from_cov(lw).values.sum() - len(lw)) / (len(lw) * (len(lw) - 1))), "", 2))
    c3.metric("Condition number (sample)", f"{np.linalg.cond(cv.sample_cov(r_hist).values):,.0f}", f"shrunk {np.linalg.cond(lw.values):,.0f}", delta_color="off")
    order = rk.clusters_from_cov(lw, 6).sort_values().index
    st.plotly_chart(charts.heatmap(cv.corr_from_cov(lw).loc[order, order], "Shrunk correlation matrix, names ordered by cluster"), use_container_width=True)
    st.markdown(
        "The sample covariance of 50 names on 252 days is badly conditioned; the optimizer amplifies its noise into extreme weights. "
        "Shrinking toward a constant-correlation target (Ledoit-Wolf 2004) with the analytical intensity above is the standard fix, "
        "and the reason min variance here holds 30 names rather than 5."
    )
