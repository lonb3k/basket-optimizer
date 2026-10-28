"""Forward-looking covariance: level shift, generalized Ledoit-Wolf, no look-ahead in the forecasts."""
import numpy as np
import pandas as pd

from basket import covariance as cv, implied_corr as ic


def _returns(n_days=1500, n=8, rho=0.3, seed=3):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n_days)
    e = rng.standard_normal((n_days, n))
    vols = np.linspace(0.01, 0.02, n)
    r = (np.sqrt(rho) * f[:, None] + np.sqrt(1 - rho) * e) * vols
    return pd.DataFrame(r, index=pd.bdate_range("2010-01-01", periods=n_days), columns=[f"S{i}" for i in range(n)])


def test_level_shift_hits_target_keeps_vols_and_stays_psd():
    cov = cv.ledoit_wolf_cov(_returns().iloc[-252:])[0]
    for target in (0.05, 0.6):
        out = ic.shift_correlation_level(cov, target)
        c = cv.corr_from_cov(out).values
        n = len(c)
        assert abs((c.sum() - n) / (n * (n - 1)) - target) < 1e-6
        assert np.allclose(np.diag(out.values), np.diag(cov.values))
        assert np.linalg.eigvalsh(out.values).min() > 0


def test_ledoit_wolf_with_its_own_average_as_target_is_unchanged():
    r = _returns().iloc[-252:]
    base, d0 = cv.ledoit_wolf_cov(r)
    x = r.values - r.values.mean(axis=0)
    s = x.T @ x / len(x)
    sd = np.sqrt(np.diag(s))
    c = s / np.outer(sd, sd)
    rbar = (c.sum() - len(c)) / (len(c) * (len(c) - 1))
    same, d1 = cv.ledoit_wolf_cov(r, rbar)
    assert np.allclose(base.values, same.values) and abs(d0 - d1) < 1e-12


def test_forecasts_do_not_look_ahead():
    r = _returns()
    cor = pd.Series(30.0, index=r.index) + np.random.default_rng(0).normal(0, 3, len(r))
    panel = ic.correlation_panel(r, cor)
    race = ic.forecast_race(panel, min_obs=12, rolling_obs=24)
    k = 40
    tampered = panel.copy()
    tampered.iloc[k:, tampered.columns.get_loc("realized_fwd")] = 0.99  # change everything from row k on
    race2 = ic.forecast_race(tampered, min_obs=12, rolling_obs=24)
    cols = ["COR1M, debiased", "COR1M + last quarter", "COR1M + last quarter, 5y rolling"]
    assert np.allclose(race[cols].iloc[: k + 1].fillna(-1), race2[cols].iloc[: k + 1].fillna(-1))


def test_risk_forecast_is_unbiased_on_a_stationary_market():
    r = _returns(n_days=2500)
    test = ic.risk_forecast_test(r, {"Sample": cv.sample_cov}, lambda h: pd.Series(1 / h.shape[1], index=h.columns))
    sc = ic.risk_scores(test)
    assert 0.9 < sc.loc["Sample", "bias ratio"] < 1.1


def test_walk_forward_only_holds_point_in_time_members():
    from basket import backtest as bt
    r = _returns(n_days=600)
    members = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    members.loc[:, ["S0", "S1", "S2", "S3", "S4"]] = 1.0         # five names (the backtest needs at least five)
    members.loc[r.index[400]:, "S5"] = 1.0                      # S5 joins the universe on day 400
    tw = bt.walk_forward_weights(r, "Equal weight", "Ledoit-Wolf", 252, 21, 1.0, None, None, members)
    before = tw[tw.index <= r.index[400]]
    after = tw[tw.index > r.index[401]]
    assert len(before) > 0 and (before[["S5", "S6", "S7"]] == 0).all().all()
    assert (after["S5"] > 0).all() and (after[["S6", "S7"]] == 0).all().all()
    assert np.allclose(tw.sum(axis=1), 1.0)
