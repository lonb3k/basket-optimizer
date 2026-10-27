# Basket Optimizer: where does a basket's performance come from?

Streamlit app that builds risk-based baskets on the 50 largest S&P 500 names, tests six
weighting schemes walk-forward with turnover costs, adds a volatility-target overlay, and sweeps
a concentration limit on the Mag 7. The question it answers is not "which optimizer is best"
but "which decision actually moves the Sharpe": the weights, the vol target, or the cap.

Third project in the series after the [volatility dashboard](https://github.com/lonb3k/vol-dashboard)
and the [dispersion](https://github.com/lonb3k/dispersion) study, on the same universe and data.

## Survivorship, stated upfront

Today's top 50 are the winners of the last decade. Every basket built on them beats SPY by a
wide margin, and that gap is not alpha. Comparisons *between* schemes are fair (same universe,
same dates); comparisons with the index are not. RSP, the equal-weight S&P 500 ETF, is shown as
the honest equal-weight benchmark: it did about 12% a year over the backtest window against 26%
for equal weight on today's survivors. The difference is the survivorship bias, measured.

## Three results worth defending

**1. Vol targeting moves the Sharpe more than the choice of weights.**
Six schemes (equal weight, inverse vol, min variance, risk parity, max diversification, HRP),
walk-forward since 2015 with Ledoit-Wolf covariance, monthly rebalancing and 5 bps per unit of
turnover: the spread of Sharpe ratios between them is about 0.4. A 10% vol target on top of any
of them adds about 0.2 of Sharpe and cuts the worst drawdown from 30% to 13%. Vol clusters and
spikes before the worst returns, so de-levering on rising realized vol sidesteps part of every
crash. That is the mechanism, and it is worth more than the optimizer.

**2. Weight is not risk.**
On this universe the Mag 7 are 54% of the cap weight and 66% of the variance. Risk-based schemes
hold them at 12-16% because they are the most volatile names, so a 30% cap on Mag 7 weight costs
those schemes nothing. The app reports the effective number of bets (entropy of risk
contributions): about 30 for equal weight, 48 for risk parity, under 10 for cap weight.

**3. The covariance matrix is the product.**
50 names on 252 days is 1275 parameters for 252 observations. The sample matrix's condition
number runs in the thousands and an optimizer amplifies its noise into five-name portfolios.
Ledoit-Wolf shrinkage toward a constant-correlation target, with the analytical intensity,
is why min variance here holds 30 names.

## What the app shows

| Tab | Content |
|---|---|
| **Build today** | Weights for any scheme under the sidebar constraints, ex-ante vol, effective number of bets, diversification ratio, weight vs risk by name, by sector, by correlation cluster |
| **Race** | Six schemes walk-forward, net of costs, against SPY and RSP: return, vol, Sharpe, drawdown, worst month, turnover, cost drag, growth and drawdown charts, returns by year |
| **Vol targeting** | Overlay at a chosen target and max leverage, forecast from trailing realized vol or VIX-scaled, Sharpe before and after per scheme, leverage path, drawdown comparison |
| **Concentration** | Sweep of the Mag 7 weight cap: Sharpe, drawdown, worst month, effective number of bets, average Mag 7 weight |
| **Diagnostics** | Shrinkage intensity, condition numbers, clustered correlation heatmap |

## Decisions, and why

- **No expected returns.** Every scheme uses the covariance only. Expected-return estimates are
  where estimation error does the most damage (Michaud 1989); risk-based weights are what
  survives out of sample.
- **Walk-forward, no look-ahead.** Covariance from the trailing window ending the day before the
  rebalance; weights take effect the next day; holdings drift between rebalances and turnover
  is what it takes to get back to target.
- **Costs on every trade, including leverage changes** in the vol-target overlay. Min variance
  and max diversification turn over 20-25% a month; HRP and risk parity 7%. That difference is
  in the numbers.
- **Risk parity by optimization, HRP by recursion.** Both aim at balanced risk; HRP never
  inverts the covariance, which is why it is stable when the matrix is not.
- **Caps as constraints in the solver**, not as an after-the-fact clip, so the solution stays
  optimal under the constraint.

## Tests

```bash
.venv/bin/python -m pytest -q tests
```

Eight tests: shrinkage intensity in [0, 1] with a positive-definite result; min variance matches
the closed form on two uncorrelated assets; risk parity equalizes risk contributions; caps and
group caps are respected by every solver; HRP sums to one and prefers low-vol names; effective
number of bets hits its bounds; the simulation charges full turnover on the first allocation and
small turnover after drift; the vol-target overlay delivers the target.

## Rebalance sheet

```bash
.venv/bin/python scripts/rebalance_sheet.py --method "Risk parity (ERC)" --mag7-cap 0.3
```

Prints today's weights and risk shares and the trades against the last saved sheet.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Layout

```
app.py                     Streamlit UI
src/basket/
  universe.py              SPY holdings, sectors, Mag 7
  data.py                  prices and CBOE indices with CSV cache
  covariance.py            sample, EWMA, Ledoit-Wolf constant-correlation shrinkage
  optimizers.py            equal weight, inverse vol, min variance, risk parity, max diversification, HRP
  risk.py                  risk contributions, effective number of bets, clusters, tracking stats
  backtest.py              walk-forward weights, simulation with costs, vol-target overlay, stats
  charts.py                Plotly figures
scripts/rebalance_sheet.py
tests/
```

## Next

- Historical constituents to remove the survivorship bias from the index comparison.
- A long-short extension: the dispersion book from the companion project as an overlay.
- Transaction-cost model by name (spread and ADV) instead of a flat bps.
