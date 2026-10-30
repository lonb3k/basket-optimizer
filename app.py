"""Vol-targeted basket optimizer on the S&P 500 top-50.

Run:  streamlit run app.py
"""
from __future__ import annotations

import functools
import hashlib
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from basket import backtest as bt, charts, covariance as cv, implied_corr as ic, optimizers as opt, risk as rk  # noqa: E402
from basket.data import load_cboe_index, load_prices, point_in_time_universe  # noqa: E402
from basket.universe import MAG7, load_weights, sector_of  # noqa: E402

st.set_page_config(page_title="Basket Optimizer", page_icon="🧺", layout="wide")

# Shared hosts get rate-limited by Yahoo, so there the app opens on the CSV mirrors committed to the repo.
# Streamlit Community Cloud mounts the repo under /mount/src, Hugging Face Spaces sets SPACE_ID.
HOSTED = (str(Path(__file__).resolve()).startswith("/mount/src/") or "SPACE_ID" in os.environ
          or os.environ.get("OFFLINE_DEFAULT") == "1")

# The walk-forward race and the cap sweep take one to two minutes. On offline data they are deterministic, so their
# results are pickled to data/cache/, keyed on the arguments and the price file, and committed for the default settings:
# the hosted app's cold start reads them instead of re-running the optimizers.
CACHE_DIR = Path(__file__).parent / "data" / "cache"
PRICES_FILE = Path(__file__).parent / "data" / "prices_2005-01-01.csv"
PIT_FILE = Path(__file__).parent / "data" / "pit_adj.csv.gz"


def returns_and_members(tickers: tuple[str, ...], pit: bool, offline: bool):
    """Log returns of the backtest universe and its point-in-time membership (None for the fixed basket of today's names)."""
    if pit:
        P, W, _ = point_in_time_universe(offline=offline, top_n=len(tickers))
        r = bt.log_returns(P)
        return r, W.reindex(r.index)
    px = load_prices(list(tickers), start="2005-01-01", offline=offline)
    return bt.log_returns(px[list(tickers)]), None


def offline_disk_cache(fn):
    @functools.wraps(fn)
    def wrapper(*args):
        if not args[-1] or not PRICES_FILE.exists():  # last argument is `offline`
            return fn(*args)
        sig = hashlib.sha1(PRICES_FILE.read_bytes() + (PIT_FILE.read_bytes() if PIT_FILE.exists() else b"")).hexdigest()
        key = hashlib.sha1(repr((fn.__name__, args, sig)).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"{fn.__name__}_{key}.pkl"
        if path.exists():
            return pd.read_pickle(path)
        out = fn(*args)
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            pd.to_pickle(out, path)
        except OSError:  # read-only host
            pass
        return out
    return wrapper


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def universe(top_n: int) -> pd.DataFrame:
    w = load_weights(top_n)
    w["sector"] = [sector_of(t) for t in w["ticker"]]
    return w


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def prices(tickers: tuple[str, ...], offline: bool) -> pd.DataFrame:
    return load_prices(list(tickers), start="2005-01-01", offline=offline)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
@offline_disk_cache
def race(tickers: tuple[str, ...], methods: tuple[str, ...], estimator: str, window: int, max_weight: float,
         mag7_cap: float, cost_bps: float, years: int, pit: bool, offline: bool):
    r, members = returns_and_members(tickers, pit, offline)
    caps = {"mag7": (MAG7, mag7_cap)} if mag7_cap < 1 else None
    rets, stats_, wts = bt.run_race(r, list(methods), estimator, window, 21, max_weight, cost_bps, caps,
                                    start=len(r) - 252 * years, members=members)
    return rets, stats_, wts


@st.cache_data(ttl=6 * 3600, show_spinner=False)
@offline_disk_cache
def cap_sweep(tickers: tuple[str, ...], method: str, estimator: str, window: int, max_weight: float, cost_bps: float, years: int,
              pit: bool, offline: bool):
    r, members = returns_and_members(tickers, pit, offline)
    rows = []
    for cap in (0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0):
        caps = {"mag7": (MAG7, cap)} if cap < 1 else None
        tw = bt.walk_forward_weights(r, method, estimator, window, 21, max_weight, caps, start=len(r) - 252 * years, members=members)
        sim = bt.simulate(r, tw, cost_bps)
        s = bt.stats(sim["ret"])
        enb = np.mean([rk.effective_number_of_bets(rk.risk_contributions(tw.loc[d], cv.ESTIMATORS[estimator](r.loc[:d].iloc[-window:].dropna(axis=1))))
                       for d in tw.index[::6]])
        rows.append({"mag7_cap": cap, "Sharpe": s["sharpe"], "Ann. return %": s["ann_return_%"], "Max drawdown %": s["max_dd_%"],
                     "Worst month %": s["worst_month_%"], "ENB": enb, "avg Mag 7 weight": float(tw[[c for c in tw.columns if c in MAG7]].sum(axis=1).mean())})
    return pd.DataFrame(rows)


FORECAST = "COR1M + last quarter, 5y rolling"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
@offline_disk_cache
def implied_risk(tickers: tuple[str, ...], window: int, max_weight: float, cost_bps: float, pit: bool, offline: bool):
    """Correlation forecast race, risk-forecast tests and min-variance walk-forward per covariance estimator."""
    r, members = returns_and_members(tickers, pit, offline)
    r = r.iloc[1:]
    members = None if members is None else members.reindex(r.index)
    cor = load_cboe_index("COR1M", offline=offline)
    race = ic.forecast_race(ic.correlation_panel(r, cor, first=window, members=members))
    fc = race[FORECAST]
    start = int(np.searchsorted(r.index, fc.dropna().index[0])) + 1
    ests = {"Sample": cv.sample_cov, "Ledoit-Wolf": lambda h: cv.ledoit_wolf_cov(h)[0], **ic.implied_estimators(fc),
            "Implied level, raw COR1M": ic.implied_estimators(cor / 100)["Implied level"]}
    ew = ic.risk_scores(ic.risk_forecast_test(r, ests, lambda h: pd.Series(1 / h.shape[1], index=h.columns), window, start=start,
                                              members=members))
    mv = ic.risk_scores(ic.risk_forecast_test(r, ests, lambda c: opt.min_variance(c, max_weight=max_weight), window,
                                              start=start, own_weights=True, members=members))
    rows = {}
    for name, est in ests.items():
        sim = bt.simulate(r, bt.walk_forward_weights(r, "Min variance", est, window, 21, max_weight, None, start, members), cost_bps)
        s = bt.stats(sim["ret"], name)
        s["avg_turnover_%"] = float(sim.loc[sim["turnover"] > 0, "turnover"].mean() * 100)
        rows[name] = s
    return race, ew, mv, pd.DataFrame(rows).T, r.index[start]


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
    universe_mode = st.radio("Backtest universe", ["Point-in-time top N", "Today's top N (survivorship bias)"], 0,
                             help="Point in time: at each rebalance, only the N largest names of that month are eligible (today's float "
                                  "shares times that month's price). Today's top N: the naive fixed basket of today's winners, kept to show the bias.")
    pit = universe_mode.startswith("Point")
    offline = st.toggle("Offline", value=HOSTED, help="Off: live Yahoo prices. On: the CSV extracts in data/.")
    if HOSTED and offline:
        st.caption("Hosted demo: opens on the price snapshot committed to the repo. Switch offline off for live data.")

w = universe(top_n)
tickers = tuple(w["ticker"])
px = prices(tickers + ("SPY", "RSP"), offline)
caps = {"mag7": (MAG7, mag7_cap)} if mag7_cap < 1 else None

st.title("Where does a basket's performance come from?")
st.caption(f"{top_n} largest S&P 500 names, {w['index_weight'].sum()*100:.0f}% of the index. Last price {px.index[-1].date()}.")
if pit:
    st.info("**Survivorship, handled.** Backtests run on a point-in-time universe: at each monthly rebalance only the names that were then "
            "among the largest are eligible (today's float shares times that month's price; names delisted since cannot enter). On today's "
            "winners equal weight made 26% a year since 2015; point in time it made 15%, next to SPY's 15%. *Build today* uses today's names.", icon="ℹ️")
else:
    st.warning("**Survivorship bias on.** This mode backtests today's top names, the winners of the last decade, as if they had been chosen "
               "back then. Every basket built on them beats SPY. Switch the sidebar back to point in time.", icon="⚠️")

tabs = st.tabs(["Build today", "Race", "Vol targeting", "Concentration", "Forward-looking risk", "Diagnostics"])

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
        rets, stats_, wts = race(tickers, tuple(opt.METHODS), estimator, window, max_weight, mag7_cap, cost_bps, years, pit, offline)
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
        f"**Reading the race.** Best Sharpe among the schemes: **{best}**. The six schemes span a Sharpe of "
        f"{table['sharpe'].drop(['SPY', 'RSP']).min():.2f} to {table['sharpe'].drop(['SPY', 'RSP']).max():.2f}, against "
        f"{table.loc['SPY', 'sharpe']:.2f} for SPY. "
        + ("Point in time, risk-based weights on the largest names roughly match the cap-weighted index: the choice of weights is not "
           "where the performance comes from. " if pit else
           "On today's survivors every scheme seems to crush the index: that gap is survivorship, not skill. ")
        + "Min variance has the lowest vol and the highest turnover (it chases the quietest names), HRP and risk parity get most of "
        "the diversification at a fraction of the trading. RSP is equal weight across all 500 names, a broader and smaller-cap version of the same idea."
    )
    by_year = allr.groupby(allr.index.year).apply(lambda d: (1 + d).prod() - 1) * 100
    with st.expander("Returns by year, %"):
        st.dataframe(by_year.round(1), use_container_width=True)

# ---------- Vol targeting ----------
with tabs[2]:
    with st.spinner("Overlays..."):
        rets, stats_, wts = race(tickers, tuple(opt.METHODS), estimator, window, max_weight, mag7_cap, cost_bps, years, pit, offline)
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
        sweep = cap_sweep(tickers, m_sw, estimator, window, max_weight, cost_bps, years, pit, offline)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.sweep_lines(sweep, "mag7_cap", ["Sharpe"], "Sharpe vs cap on Mag 7 weight"), use_container_width=True)
    c2.plotly_chart(charts.sweep_lines(sweep, "mag7_cap", ["Max drawdown %", "Worst month %"], "Tail vs cap on Mag 7 weight", "%"), use_container_width=True)
    st.dataframe(sweep.round(2), use_container_width=True)
    st.markdown(
        "**What a concentration limit costs.** The cap binds only when the scheme wants more Mag 7 than allowed (cap-weighting would hold 54%). "
        "Risk-based schemes already hold about 11% because the giants are the most volatile names, so a cap above 15% costs them nothing. "
        "The sweep is the argument to bring to a risk committee: the return given up per point of drawdown saved."
    )

# ---------- Forward-looking risk ----------
with tabs[4]:
    with st.spinner("Racing correlation forecasts and testing five covariance matrices (about two minutes the first time)..."):
        race_c, ew_s, mv_s, mv_bt, t0 = implied_risk(tickers, window, max_weight, cost_bps, pit, offline)
    st.markdown(
        "**Question.** Every covariance matrix in this app is estimated on the past year, which assumes next month's correlation "
        "will look like last year's. The option market publishes its own view every day, the CBOE one-month implied correlation "
        "(COR1M). Its level is off in both directions (a premium of about 8 points when VIX is above 30, below realized in calm markets, "
        "see the dispersion project) and it is defined against "
        "the index rather than as the average pairwise correlation of the names, so it is first turned into a forecast: an OLS "
        "refitted each month on the previous five years only, mapping COR1M and last quarter's correlation to next month's."
    )
    sc = ic.forecast_scores(race_c)
    m = st.columns(3)
    m[0].metric("Forecast error, last year's correlation", f"{sc.loc['Last year (sample)', 'RMSE (pts)']:.1f} pts",
                f"what the optimizer assumes, corr {sc.loc['Last year (sample)', 'corr with realized']:.2f}", delta_color="off")
    m[1].metric("Forecast error, COR1M-based", f"{sc.loc[FORECAST, 'RMSE (pts)']:.1f} pts",
                f"corr {sc.loc[FORECAST, 'corr with realized']:.2f} with realized", delta_color="off")
    m[2].metric("Next month, forecast", f"{race_c[FORECAST].iloc[-1] * 100:.1f}%",
                f"COR1M {race_c['COR1M raw'].iloc[-1] * 100:.1f}%, last quarter {race_c['Last quarter'].iloc[-1] * 100:.1f}%", delta_color="off")
    st.plotly_chart(charts.corr_forecasts(race_c[race_c.index >= t0], ["Last year (sample)", "COR1M raw", FORECAST]), use_container_width=True)
    st.dataframe(sc.style.format({"RMSE (pts)": "{:.1f}", "bias (pts)": "{:+.1f}", "corr with realized": "{:.2f}", "obs": "{:.0f}"}),
                 use_container_width=True)
    st.markdown(
        "**Reading.** Last year's average correlation, the level built into every historical covariance matrix, is the worst "
        "forecaster of next month's. Raw COR1M already tracks it better (it moves first, before a sell-off shows up in returns); "
        "debiased and refitted on a rolling window it cuts the error by about a fifth."
    )
    st.subheader("Does the covariance forecast risk better?")
    st.markdown(
        "Five matrices, same vols, same history: only the correlation level differs. *LW, implied target* shrinks toward the "
        "forecast instead of last year's average; *Implied level* keeps the Ledoit-Wolf structure (which names move together) and "
        "moves its average correlation to the forecast. Each month the matrix forecasts the vol of a portfolio, which is then "
        "compared with the vol realized over the next month. QLIKE is the standard loss for variance forecasts: lower is better."
    )
    fmt_s = {"forecast vol %": "{:.1f}", "realized vol %": "{:.1f}", "bias ratio": "{:.3f}", "QLIKE": "{:.3f}", "surprises > 1.5x, %": "{:.1f}"}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Equal-weight basket** (same weights for all, pure risk forecast)")
        st.dataframe(ew_s.style.format(fmt_s), use_container_width=True)
    with c2:
        st.markdown("**Min variance** (each matrix builds its own portfolio)")
        st.dataframe(mv_s.style.format(fmt_s), use_container_width=True)
    st.dataframe(mv_bt[["ann_return_%", "ann_vol_%", "sharpe", "max_dd_%", "worst_month_%", "avg_turnover_%"]].astype(float)
                 .style.format("{:.2f}"), use_container_width=True)
    st.markdown(
        "**Reading.** Moving the correlation level to the option-implied forecast lowers the QLIKE loss by 12% to 15% on both "
        "portfolios: the matrix knows more about next month's risk. Plugging raw COR1M in instead gives the worst risk forecast "
        "for min variance (its premium makes the matrix see risk that never shows up), while its Sharpe stays within noise of the "
        "others: the debiasing step is what makes the signal usable. The min-variance portfolio itself barely changes (realized vol and Sharpe within noise of Ledoit-Wolf): "
        "its weights depend on which names are relatively cheap in risk, not on the average correlation level. The option market "
        "helps size risk, not pick names."
    )

# ---------- Diagnostics ----------
with tabs[5]:
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
