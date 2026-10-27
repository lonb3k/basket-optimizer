"""S&P 500 constituents and weights from the SPY holdings file (State Street).

The CBOE implied correlation indices (COR1M, COR3M) use the 50 largest S&P 500 components,
re-weighted to sum to one. We do the same, so our implied correlation is comparable to the
CBOE print.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)
HOLDINGS_URL = "https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
HOLDINGS_FILE = DATA_DIR / "spy_holdings.xlsx"
WEIGHTS_FILE = DATA_DIR / "weights_top50.csv"

# Yahoo uses '-' where SSGA uses '.' (BRK.B -> BRK-B)
YAHOO_FIX = {"BRK.B": "BRK-B", "BF.B": "BF-B"}


def download_holdings(force: bool = False) -> Path:
    if HOLDINGS_FILE.exists() and not force:
        return HOLDINGS_FILE
    r = requests.get(HOLDINGS_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    HOLDINGS_FILE.write_bytes(r.content)
    return HOLDINGS_FILE


def load_weights(top_n: int = 50, force: bool = False) -> pd.DataFrame:
    """Top-N constituents by SPY weight, re-normalized. Columns: ticker, name, sector, weight, index_weight."""
    path = download_holdings(force)
    raw = pd.read_excel(path, header=None)
    # find the header row (contains 'Ticker')
    hdr = raw.index[raw.apply(lambda r: r.astype(str).str.strip().eq("Ticker").any(), axis=1)][0]
    df = pd.read_excel(path, header=hdr)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Weight": "index_weight"})
    df = df[["Name", "Ticker", "Sector", "index_weight"]].dropna(subset=["Ticker", "index_weight"])
    df = df[df["Ticker"].astype(str).str.match(r"^[A-Z.]+$")]  # drop cash and futures lines
    df["index_weight"] = pd.to_numeric(df["index_weight"], errors="coerce") / 100
    df = df.dropna().sort_values("index_weight", ascending=False).head(top_n)
    df["ticker"] = df["Ticker"].map(lambda t: YAHOO_FIX.get(t, t))
    df["weight"] = df["index_weight"] / df["index_weight"].sum()
    out = df.rename(columns={"Name": "name", "Sector": "sector"})[["ticker", "name", "sector", "weight", "index_weight"]]
    out = out.reset_index(drop=True)
    out.to_csv(WEIGHTS_FILE, index=False)
    return out


# GICS sector for the usual top-50 names; anything missing is looked up on Yahoo once and cached.
SECTORS = {
    "NVDA": "Technology", "AAPL": "Technology", "MSFT": "Technology", "AVGO": "Technology", "ORCL": "Technology",
    "AMD": "Technology", "MU": "Technology", "PLTR": "Technology", "CSCO": "Technology", "INTC": "Technology",
    "PANW": "Technology", "AMAT": "Technology", "LRCX": "Technology", "TXN": "Technology", "CRWD": "Technology",
    "IBM": "Technology", "CRM": "Technology", "ADBE": "Technology", "QCOM": "Technology", "NOW": "Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "GOOGL": "Communication", "GOOG": "Communication", "META": "Communication", "NFLX": "Communication", "DIS": "Communication",
    "BRK-B": "Financials", "JPM": "Financials", "V": "Financials", "MA": "Financials", "BAC": "Financials", "GS": "Financials",
    "WFC": "Financials", "MS": "Financials", "AXP": "Financials", "BLK": "Financials",
    "LLY": "Health Care", "JNJ": "Health Care", "ABBV": "Health Care", "UNH": "Health Care", "MRK": "Health Care",
    "TMO": "Health Care", "AMGN": "Health Care", "PFE": "Health Care", "ABT": "Health Care", "ISRG": "Health Care",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "WMT": "Consumer Staples", "COST": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples", "PM": "Consumer Staples",
    "CAT": "Industrials", "GE": "Industrials", "RTX": "Industrials", "GEV": "Industrials", "HON": "Industrials", "UNP": "Industrials", "BA": "Industrials",
    "LIN": "Materials", "NEE": "Utilities", "T": "Communication", "VZ": "Communication",
}
MAG7 = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA"]


def sector_of(ticker: str) -> str:
    if ticker in SECTORS:
        return SECTORS[ticker]
    try:
        import yfinance as yf
        s = yf.Ticker(ticker).info.get("sector")
        return s or "Other"
    except Exception:
        return "Other"
