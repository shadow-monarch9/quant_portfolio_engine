"""Streamlit dashboard for the quant portfolio engine. Run: ``streamlit run app.py``."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st

from src.backtest.engine import run_strategy
from src.backtest.metrics import calculate_performance_metrics
from src.config import get_settings
from src.data.yahoo_loader import YahooDataLoader
from src.portfolio.optimizer import WeightingScheme

FREQUENCIES: list[str] = ["daily", "weekly", "monthly", "quarterly"]
SCHEMES: list[str] = ["equal", "inverse_volatility"]


@st.cache_data(show_spinner="Loading prices...")
def load_prices() -> pd.DataFrame:
    return YahooDataLoader.from_settings(get_settings()).load()


def equity_chart(strategy: pd.Series, benchmark: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy.index, y=strategy, name="Strategy", line={"width": 2.5}))
    fig.add_trace(
        go.Scatter(x=benchmark.index, y=benchmark, name="Equal-weight benchmark", line={"width": 2})
    )
    fig.update_layout(
        title="Equity curve (growth of 1.0)",
        xaxis_title=None,
        yaxis_title="Equity",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
    )
    return fig


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Quant Portfolio Engine", layout="wide")
    st.title("Quant Portfolio Engine")

    prices = load_prices()
    default_weights = settings.factor_weights

    with st.sidebar:
        st.header("Factor weights")
        w_mom = st.slider("Momentum", 0.0, 1.0, float(default_weights.momentum), 0.05)
        w_vol = st.slider("Volatility", 0.0, 1.0, float(default_weights.volatility), 0.05)
        w_z = st.slider("Z-Score", 0.0, 1.0, float(default_weights.zscore), 0.05)

        st.header("Portfolio")
        top_k = st.slider("Top-K holdings", 1, 5, min(max(settings.portfolio.top_k, 1), 5))
        scheme = st.selectbox(
            "Weighting scheme", SCHEMES, index=SCHEMES.index(settings.portfolio.weighting_scheme)
        )
        frequency = st.selectbox(
            "Rebalance frequency",
            FREQUENCIES,
            index=FREQUENCIES.index(settings.rebalance.frequency),
        )
        cost_bps = st.slider(
            "Transaction cost (bps)",
            0.0,
            50.0,
            float(min(settings.rebalance.transaction_cost_bps, 50.0)),
            0.5,
        )

    total = w_mom + w_vol + w_z
    if total <= 0:
        st.error("Set at least one factor weight above zero.")
        st.stop()
    factor_weights = {"momentum": w_mom / total, "volatility": w_vol / total, "zscore": w_z / total}
    st.sidebar.caption(
        "Normalised: "
        + ", ".join(f"{name} {weight:.0%}" for name, weight in factor_weights.items())
    )

    if top_k > prices.shape[1]:
        st.error(f"Top-K ({top_k}) exceeds the universe size ({prices.shape[1]}).")
        st.stop()

    scores, weights, result = run_strategy(
        prices=prices,
        factor_weights=factor_weights,
        top_k=top_k,
        weighting_scheme=scheme_cast(scheme),
        rebalance_frequency=frequency,
        transaction_cost_bps=cost_bps,
    )
    if result.strategy_returns.empty:
        st.warning("Not enough history to clear the 252-day burn-in.")
        st.stop()
    stats = calculate_performance_metrics(result.strategy_returns, result.benchmark_returns)

    cols = st.columns(4)
    cols[0].metric("Sharpe Ratio", f"{stats['sharpe']:.2f}", f"{stats['benchmark_sharpe']:.2f} bench")
    cols[1].metric(
        "CAGR", f"{stats['cagr_pct']:.1f}%", f"{stats['excess_cagr_pct']:+.1f}% vs bench"
    )
    cols[2].metric("Max Drawdown", f"{stats['max_drawdown_pct']:.1f}%")
    cols[3].metric("Win Rate", f"{stats['win_rate_pct']:.1f}%")

    st.plotly_chart(equity_chart(result.strategy_equity, result.benchmark_equity), width="stretch")
    start = result.strategy_returns.index[0].date()
    st.caption(
        f"Backtest from {start} (after the 252-day factor burn-in). Weights are lagged one day; "
        f"costs apply to turnover. Sortino: {stats['sortino']:.2f}."
    )

    latest = scores.dropna(how="all").index[-1]
    table = pd.DataFrame(
        {
            "Alpha score": scores.loc[latest],
            "Target weight": weights.loc[latest],
        }
    ).sort_values(["Target weight", "Alpha score"], ascending=False)
    table.insert(0, "Rank", range(1, len(table) + 1))
    table.index.name = "Ticker"
    st.subheader(f"Top-ranked tickers and target allocation ({latest.date()})")
    st.dataframe(
        table.style.format({"Alpha score": "{:.3f}", "Target weight": "{:.1%}"}),
        width="stretch",
    )


def scheme_cast(value: str) -> WeightingScheme:
    return "inverse_volatility" if value == "inverse_volatility" else "equal"


main()
