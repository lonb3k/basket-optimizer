"""Print today's target weights for one scheme and the trades versus the last saved sheet.

    python scripts/rebalance_sheet.py --method "Risk parity (ERC)" --mag7-cap 0.3
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from basket import backtest as bt, covariance as cv, optimizers as opt, risk as rk  # noqa: E402
from basket.data import load_prices  # noqa: E402
from basket.universe import DATA_DIR, MAG7, load_weights  # noqa: E402

SHEET = DATA_DIR / "last_rebalance.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="Risk parity (ERC)")
    ap.add_argument("--estimator", default="Ledoit-Wolf")
    ap.add_argument("--window", type=int, default=252)
    ap.add_argument("--max-weight", type=float, default=0.10)
    ap.add_argument("--mag7-cap", type=float, default=0.30)
    args = ap.parse_args()

    w = load_weights(50)
    px = load_prices(list(w["ticker"]), start="2005-01-01")
    r = bt.log_returns(px).iloc[-args.window:].dropna(axis=1)
    cov = cv.ESTIMATORS[args.estimator](r)
    caps = {"mag7": (MAG7, args.mag7_cap)} if args.mag7_cap < 1 else None
    wt = opt.METHODS[args.method](cov, max_weight=args.max_weight, group_caps=caps)
    rc = rk.risk_contributions(wt, cov)
    sheet = pd.DataFrame({"weight": wt, "risk_share": rc}).sort_values("weight", ascending=False)
    if SHEET.exists():
        prev = pd.read_csv(SHEET, index_col=0)["weight"].reindex(sheet.index).fillna(0)
        sheet["trade"] = sheet["weight"] - prev
        print(f"turnover vs last sheet: {sheet['trade'].abs().sum()*100:.1f}%")
    print(f"{args.method}, ex-ante vol {rk.portfolio_vol(wt, cov)*100:.1f}%, ENB {rk.effective_number_of_bets(rc):.1f}")
    print(sheet.round(4).to_string())
    sheet.to_csv(SHEET)


if __name__ == "__main__":
    main()
