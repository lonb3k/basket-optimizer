"""Optimizers against closed forms, shrinkage sanity, simulation accounting, vol targeting."""
import numpy as np
import pandas as pd

from basket import backtest as bt, covariance as cv, optimizers as opt, risk as rk


def _returns(n_days=500, n=8, seed=0, rho=0.3):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n_days)
    e = rng.standard_normal((n_days, n))
    vols = np.linspace(0.15, 0.45, n) / np.sqrt(252)
    r = (np.sqrt(rho) * f[:, None] + np.sqrt(1 - rho) * e) * vols
    return pd.DataFrame(r, index=pd.bdate_range("2020-01-01", periods=n_days), columns=[f"S{i}" for i in range(n)])


def test_ledoit_wolf_between_sample_and_target_and_psd():
    r = _returns()
    lw, delta = cv.ledoit_wolf_cov(r)
    assert 0 <= delta <= 1
    assert np.all(np.linalg.eigvalsh(lw.values) > 0)
    assert np.allclose(np.diag(lw.values), np.diag(r.cov(ddof=0).values) * 252)  # diagonal preserved (LW uses 1/T)


def test_min_variance_two_assets_closed_form():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    w = opt.min_variance(cov, max_weight=1.0)
    assert abs(w["A"] - 0.2) < 1e-4 and abs(w["B"] - 0.8) < 1e-4  # w_i ∝ 1/var_i when uncorrelated


def test_risk_parity_equalizes_risk_contributions():
    r = _returns()
    cov = cv.sample_cov(r)
    w = opt.risk_parity(cov)
    rc = rk.risk_contributions(w, cov)
    assert rc.max() - rc.min() < 0.01
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()


def test_constraints_are_respected():
    r = _returns(n=10)
    cov = cv.sample_cov(r)
    caps = {"g": (["S0", "S1", "S2"], 0.2)}
    for name in ("Min variance", "Risk parity (ERC)", "Max diversification"):
        w = opt.METHODS[name](cov, max_weight=0.15, group_caps=caps)
        assert w.max() <= 0.15 + 1e-4, name
        assert w[["S0", "S1", "S2"]].sum() <= 0.2 + 1e-4, name  # SLSQP tolerance


def test_hrp_sums_to_one_and_prefers_low_vol():
    r = _returns()
    cov = cv.sample_cov(r)
    w = opt.hrp(cov)
    assert abs(w.sum() - 1) < 1e-12 and (w > 0).all()
    assert w["S0"] > w["S7"]  # lowest vol name gets more than the highest


def test_effective_number_of_bets_bounds():
    rc = pd.Series([0.25, 0.25, 0.25, 0.25])
    assert abs(rk.effective_number_of_bets(rc) - 4) < 1e-9
    assert abs(rk.effective_number_of_bets(pd.Series([1.0, 0.0, 0.0])) - 1) < 1e-6


def test_simulation_accounting():
    r = _returns(n_days=100, n=3)
    targets = pd.DataFrame([[1 / 3] * 3] * 2, index=[r.index[10], r.index[60]], columns=r.columns)
    sim = bt.simulate(r, targets, cost_bps=10)
    assert sim.index[0] == r.index[10]
    assert abs(sim["turnover"].iloc[0] - 1.0) < 1e-12           # first allocation is full turnover
    assert sim.loc[r.index[60], "turnover"] < 0.2               # back to equal weight after drift: small
    assert (sim["ret"] <= sim["gross_ret"] + 1e-15).all()


def test_vol_target_scales_volatility():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 0.02, 2000), index=pd.bdate_range("2015-01-01", periods=2000))
    ov = bt.vol_target_overlay(pd.DataFrame({"ret": r}), target=0.10, max_leverage=5.0, cost_bps=0)
    vol = ov["ret"].iloc[50:].std() * np.sqrt(252)
    assert 0.085 < vol < 0.125
