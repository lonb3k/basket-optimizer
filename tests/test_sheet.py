import json

import pytest

from basket import sheet


def test_helpers_are_json_safe():
    assert sheet._num(float("nan")) is None
    assert sheet._num(1.23456, 2) == 1.23
    import pandas as pd
    s = pd.Series(range(500), dtype=float)
    assert len(sheet._spark(s)) == sheet.SPARK_POINTS
    assert sheet._compact(["2024-01-02T00:00:00"]) == ["2024-01-02"]


@pytest.mark.slow
def test_build_sheet_offline_has_the_contract():
    out = sheet.build_sheet(offline=True)
    json.dumps(out, allow_nan=False)  # no NaN leaks
    for key in ("project", "as_of", "headline", "tiles", "sections", "ideas", "charts", "caveats"):
        assert key in out
    assert out["tiles"] and all({"label", "value", "spark"} <= set(t) for t in out["tiles"])
    assert out["ideas"] and [i["rank"] for i in out["ideas"]] == list(range(1, len(out["ideas"]) + 1))
    for i in out["ideas"]:
        assert {"expected", "worst", "n"} <= set(i["stats"])
