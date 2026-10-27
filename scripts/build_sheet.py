"""Write the morning sheet to sheet/latest.json. Run after the US close.

    python scripts/build_sheet.py [--offline]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from basket.sheet import build_sheet  # noqa: E402

OUT = ROOT / "sheet" / "latest.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use the cached CSV extracts only")
    args = ap.parse_args()
    sheet = build_sheet(offline=args.offline)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(sheet, indent=None, separators=(",", ":"), allow_nan=False))
    print(f"{OUT} ({OUT.stat().st_size / 1024:.0f} KB), as of {sheet['as_of']}, {len(sheet['ideas'])} ideas, "
          f"{len(sheet['charts'])} charts, warnings: {sheet['warnings'] or 'none'}")
    print(sheet["headline"])
    for i in sheet["ideas"]:
        s = i["stats"]
        print(f"  {i['rank']}. {i['title']}  [{i['condition']}]  E={s['expected']} {s['expected_unit']}, worst={s['worst']} ({s.get('worst_date')}), score={i['score']}")


if __name__ == "__main__":
    main()
