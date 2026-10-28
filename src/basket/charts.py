"""Plotly figures. One y-axis per chart, thin marks, categorical colors in fixed order."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Validated categorical palette (light surface), fixed slot order
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"], C["red"]]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, Helvetica, Arial, sans-serif", size=12, color=TEXT),
    margin=dict(l=48, r=24, t=84, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=MUTED)),
    xaxis=dict(showgrid=False, zeroline=False, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix="%"),
)


def _base(title: str, height: int = 380, **overrides) -> go.Figure:
    fig = go.Figure()
    layout = {**LAYOUT, **overrides}
    fig.update_layout(title=dict(text=title, x=0, y=0.98, yanchor="top", font=dict(size=14)), height=height, **layout)
    return fig


# ---------- time series ----------

def realized_vs_implied(rv: pd.DataFrame, implied: pd.Series | None, implied_name: str | None) -> go.Figure:
    fig = _base("Realized vol (annualized) vs 30d implied")
    for i, col in enumerate(rv.columns):
        fig.add_trace(go.Scatter(x=rv.index, y=rv[col], name=col, mode="lines",
                                 line=dict(width=1.5, color=SERIES[i]),
                                 hovertemplate="%{y:.1f}%"))
    if implied is not None and not implied.empty:
        fig.add_trace(go.Scatter(x=implied.index, y=implied, name=implied_name or "Implied", mode="lines",
                                 line=dict(width=2, color=C["red"]), hovertemplate="%{y:.1f}%"))


def weights_bar(w: pd.Series, rc: pd.Series, title: str) -> go.Figure:
    d = pd.DataFrame({"weight": w, "risk": rc}).sort_values("weight", ascending=False)
    d = d[d["weight"] > 1e-4]
    fig = _base(title, height=380, hovermode="x unified", xaxis=dict(showgrid=False, type="category"),
                yaxis=dict(gridcolor=GRID, zeroline=False, tickformat=".0%", ticksuffix=""))
    fig.add_trace(go.Bar(x=d.index, y=d["weight"], name="Weight", marker=dict(color=C["blue"], line=dict(width=0)), hovertemplate="%{y:.1%}"))
    fig.add_trace(go.Bar(x=d.index, y=d["risk"], name="Share of risk", marker=dict(color=C["orange"], line=dict(width=0)), hovertemplate="%{y:.1%}"))
    fig.update_layout(barmode="group", bargap=0.25)
    return fig


def group_bars(gs: pd.DataFrame, title: str) -> go.Figure:
    fig = _base(title, height=320, hovermode="x unified", xaxis=dict(showgrid=False, type="category"),
                yaxis=dict(gridcolor=GRID, zeroline=False, tickformat=".0%", ticksuffix=""))
    fig.add_trace(go.Bar(x=gs.index.astype(str), y=gs["weight"], name="Weight", marker=dict(color=C["blue"], line=dict(width=0)), hovertemplate="%{y:.1%}"))
    fig.add_trace(go.Bar(x=gs.index.astype(str), y=gs["risk"], name="Share of risk", marker=dict(color=C["orange"], line=dict(width=0)), hovertemplate="%{y:.1%}"))
    fig.update_layout(barmode="group", bargap=0.3)
    return fig


def cumulative(rets: pd.DataFrame, title: str, log: bool = True) -> go.Figure:
    fig = _base(title, height=400, yaxis=dict(gridcolor=GRID, zeroline=False, type="log" if log else "linear", ticksuffix=""))
    for i, col in enumerate(rets.columns):
        cum = (1 + rets[col].fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name=col, mode="lines", line=dict(width=1.5 if i < 6 else 1.2, color=SERIES[i % 8],
                                 dash="solid" if i < 6 else "dot"), hovertemplate="%{y:.2f}x"))
    return fig


def drawdowns(rets: pd.DataFrame, title: str) -> go.Figure:
    fig = _base(title, height=320, yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, tickformat=".0%", ticksuffix=""))
    for i, col in enumerate(rets.columns):
        cum = (1 + rets[col].fillna(0)).cumprod()
        dd = cum / cum.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=col, mode="lines", line=dict(width=1.3, color=SERIES[i % 8]), hovertemplate="%{y:.1%}"))
    return fig


def leverage_path(ov: pd.DataFrame, title: str) -> go.Figure:
    fig = _base(title, height=300, yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix="x"))
    fig.add_trace(go.Scatter(x=ov.index, y=ov["leverage"], name="Leverage", mode="lines", line=dict(width=1.5, color=C["blue"]), hovertemplate="%{y:.2f}x"))
    fig.add_hline(y=1.0, line=dict(color=MUTED, width=1, dash="dot"))
    return fig


def sweep_lines(sweep: pd.DataFrame, x: str, cols: list[str], title: str, suffix: str = "") -> go.Figure:
    fig = _base(title, height=340, hovermode="x unified", xaxis=dict(title=x, showgrid=False, tickformat=".0%"),
                yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix=suffix))
    for i, c in enumerate(cols):
        fig.add_trace(go.Scatter(x=sweep[x], y=sweep[c], name=c, mode="lines+markers", line=dict(width=2, color=SERIES[i]), marker=dict(size=7),
                                 hovertemplate="%{y:.2f}" + suffix))
    return fig


def heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    fig = _base(title, height=560, hovermode="closest", xaxis=dict(showgrid=False, tickangle=45), yaxis=dict(showgrid=False, autorange="reversed", ticksuffix=""))
    fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-0.2, zmax=1,
                             colorscale=[[0, "#fcfcfb"], [0.17, "#fcfcfb"], [1, SEQ[6]]], colorbar=dict(title="rho", thickness=12),
                             hovertemplate="%{y} / %{x}: %{z:.2f}"))
    fig.update_layout(margin=dict(l=60, r=24, t=60, b=80))
    return fig


def sharpe_before_after(table: pd.DataFrame) -> go.Figure:
    fig = _base("Sharpe ratio: raw basket vs vol-targeted at 10%", height=340, hovermode="x unified",
                xaxis=dict(showgrid=False, type="category"), yaxis=dict(gridcolor=GRID, zeroline=False, ticksuffix=""))
    fig.add_trace(go.Bar(x=table.index, y=table["sharpe_raw"], name="Raw", marker=dict(color=C["blue"], line=dict(width=0)), hovertemplate="%{y:.2f}"))
    fig.add_trace(go.Bar(x=table.index, y=table["sharpe_vt"], name="Vol-targeted", marker=dict(color=C["orange"], line=dict(width=0)), hovertemplate="%{y:.2f}"))
    fig.update_layout(barmode="group", bargap=0.3)
    return fig


def corr_forecasts(race: pd.DataFrame, models: list[str]) -> go.Figure:
    """Next-month average pairwise correlation: realized vs forecasts, out of sample."""
    fig = _base("Average pairwise correlation next month: realized vs forecasts", height=400)
    fig.add_trace(go.Scatter(x=race.index, y=race["realized_fwd"] * 100, name="Realized next month", mode="lines",
                             line=dict(width=1.4, color=TEXT), hovertemplate="%{y:.1f}%"))
    for i, m in enumerate(models):
        fig.add_trace(go.Scatter(x=race.index, y=race[m] * 100, name=m, mode="lines",
                                 line=dict(width=1.8, color=SERIES[i]), hovertemplate="%{y:.1f}%"))
    return fig
