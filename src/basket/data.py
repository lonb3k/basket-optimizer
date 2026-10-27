"""Data access: constituent prices, CBOE correlation index history, option chains.

Prices go through yfinance and are mirrored to data/prices.csv. The CBOE implied correlation
indices come straight from the CBOE public CSV endpoint.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from .universe import DATA_DIR

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{index}_History.csv"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    return df.droplevel(1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df


def load_prices(tickers: list[str], start: str = "2010-01-01", offline: bool = False) -> pd.DataFrame:
    """Adjusted closes, one column per ticker. Cached per start date in data/prices_<start>.csv so a
    short request (the daily snapshot needs 2024+) never clobbers the long history."""
    path = DATA_DIR / f"prices_{start}.csv"
    if offline and path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        have = [t for t in tickers if t in cached]
        return cached[have]
    try:
        raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    except Exception:
        raw = pd.DataFrame()
    if raw.empty:
        return pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()
    close = raw["Close"] if "Close" in raw else _flatten(raw)
    close = close.dropna(how="all")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "Date"
    close.to_csv(path)
    return close


def load_cboe_index(index: str = "COR1M", offline: bool = False) -> pd.Series:
    """Daily close of a CBOE index (COR1M, COR3M, VIX...) since inception."""
    path = DATA_DIR / f"cboe_{index}.csv"
    if not offline or not path.exists():
        try:
            r = requests.get(CBOE_URL.format(index=index), timeout=30, allow_redirects=True)
            r.raise_for_status()
            path.write_bytes(r.content)
        except Exception:
            pass
    if not path.exists():
        return pd.Series(dtype=float, name=index)
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.set_index("DATE")["CLOSE"].rename(index)


def option_chain(ticker: str, target_days: int = 30) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    """Calls and puts for the listed expiry closest to `target_days`, with spot.
    Same layout as the vol-dashboard project: strike, type, last, iv_yahoo, bid, ask, volume, openInterest."""
    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        return pd.DataFrame(), pd.NaT, float("nan")
    today = pd.Timestamp.today().normalize()
    exp_dates = [pd.Timestamp(e) for e in expiries]
    candidates = [e for e in exp_dates if (e - today).days >= 7] or exp_dates
    expiry = min(candidates, key=lambda e: abs((e - today).days - target_days))
    chain = tk.option_chain(expiry.strftime("%Y-%m-%d"))
    cols = ["strike", "lastPrice", "impliedVolatility", "bid", "ask", "volume", "openInterest"]
    df = pd.concat([chain.calls[cols].assign(type="call"), chain.puts[cols].assign(type="put")], ignore_index=True)
    df = df.rename(columns={"impliedVolatility": "iv_yahoo", "lastPrice": "last"})
    hist = tk.history(period="5d")
    spot = float(hist["Close"].iloc[-1]) if not hist.empty else float("nan")
    return df, expiry, spot


def next_earnings(ticker: str) -> pd.Timestamp | None:
    """Next earnings date if Yahoo knows it. Single-name IV is inflated into earnings, which matters for dispersion."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            d = pd.Timestamp(dates[0])
            return d if d >= pd.Timestamp.today().normalize() else None
    except Exception:
        return None
    return None
