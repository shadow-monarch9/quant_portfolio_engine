"""Streamlit dashboard for the quant portfolio engine. Run: ``streamlit run app.py``."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st

from src.agent.factor_generator import GenerationResult
from src.agent.orchestrator import AgentOrchestrator, AgentTurn, BacktestParams
from src.agent.regime import RegimeDetector, equal_weight_benchmark
from src.backtest.engine import run_strategy
from src.backtest.metrics import calculate_performance_metrics
from src.config import Settings, get_settings
from src.data.yahoo_loader import YahooDataLoader
from src.portfolio.optimizer import WeightingScheme

FREQUENCIES: list[str] = ["daily", "weekly", "monthly", "quarterly"]
SCHEMES: list[str] = ["equal", "inverse_volatility"]


@st.cache_data(show_spinner="Loading prices...")
def load_prices() -> pd.DataFrame:
    return YahooDataLoader.from_settings(get_settings()).load()


def init_slider_state(settings: Settings) -> None:
    """Seed slider session-state keys once so presets can overwrite them before widgets render."""
    weights = settings.factor_weights
    defaults: dict[str, float | int] = {
        "momentum_weight": float(weights.momentum),
        "volatility_weight": float(weights.volatility),
        "zscore_weight": float(weights.zscore),
        "top_k": min(max(settings.portfolio.top_k, 1), 5),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_regime_preset(preset: dict[str, float]) -> None:
    """Button callback: runs before the rerun, so it may set keys of widgets already rendered."""
    st.session_state["momentum_weight"] = float(preset["momentum"])
    st.session_state["volatility_weight"] = float(preset["volatility"])
    st.session_state["zscore_weight"] = float(preset["zscore"])
    st.session_state["top_k"] = int(preset["top_k"])


def render_regime_banner(prices: pd.DataFrame) -> None:
    analysis = RegimeDetector.analyze(equal_weight_benchmark(prices))
    preset: dict[str, float] = analysis["preset"]
    with st.container(border=True):
        st.subheader(f"Market regime: {analysis['regime']}")
        st.write(analysis["summary"])
        if not preset:
            return
        cols = st.columns(3)
        cols[0].metric("Benchmark level", f"{analysis['current_price']:.2f}")
        cols[1].metric("200-day SMA", f"{analysis['sma_200']:.2f}")
        cols[2].metric("30-day volatility (ann.)", f"{analysis['rolling_vol']:.1%}")
        st.info(
            f"{analysis['suggestion']}\n\n"
            f"Preset: momentum {preset['momentum']:.0%}, volatility {preset['volatility']:.0%}, "
            f"z-score {preset['zscore']:.0%}, top-K {int(preset['top_k'])}"
        )
        st.button(
            "Apply Recommended Factor Preset",
            on_click=apply_regime_preset,
            args=(preset,),
        )


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


def render_metrics(stats: dict[str, float]) -> None:
    cols = st.columns(4)
    cols[0].metric("Sharpe Ratio", f"{stats['sharpe']:.2f}", f"{stats['benchmark_sharpe']:.2f} bench")
    cols[1].metric(
        "CAGR", f"{stats['cagr_pct']:.1f}%", f"{stats['excess_cagr_pct']:+.1f}% vs bench"
    )
    cols[2].metric("Max Drawdown", f"{stats['max_drawdown_pct']:.1f}%")
    cols[3].metric("Win Rate", f"{stats['win_rate_pct']:.1f}%")


def render_dashboard(
    prices: pd.DataFrame, factor_weights: dict[str, float], params: BacktestParams
) -> None:
    scores, weights, result = run_strategy(
        prices=prices,
        factor_weights=factor_weights,
        top_k=params.top_k,
        weighting_scheme=params.weighting_scheme,
        rebalance_frequency=params.rebalance_frequency,
        transaction_cost_bps=params.transaction_cost_bps,
    )
    if result.strategy_returns.empty or scores.dropna(how="all").empty:
        st.warning("Not enough history to clear the 252-day burn-in.")
        return
    stats = calculate_performance_metrics(result.strategy_returns, result.benchmark_returns)

    render_metrics(stats)
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


def get_orchestrator() -> AgentOrchestrator:
    """One orchestrator (and factor registry) per browser session."""
    if "orchestrator" not in st.session_state:
        st.session_state["orchestrator"] = AgentOrchestrator(config=get_settings().agent)
        st.session_state["chat"] = []
    orchestrator: AgentOrchestrator = st.session_state["orchestrator"]
    return orchestrator


def render_turn(turn: AgentTurn) -> None:
    """Render one agent reply: factor source, validation attempts and backtest results."""
    generation = turn.generation
    if turn.error:
        st.error(turn.error)
    if generation.factor is not None:
        st.success(f"Factor **{generation.factor.name}**: {generation.factor.description}")
    if generation.source:
        with st.expander(f"Generated code ({len(generation.attempts)} attempt(s))"):
            st.code(generation.source, language="python")
            for attempt in generation.attempts:
                if attempt.error:
                    st.caption(f"Attempt {attempt.attempt} failed at '{attempt.stage}': {attempt.error}")
    if turn.report is not None:
        render_metrics(turn.report.stats)
        result = turn.report.result
        st.plotly_chart(
            equity_chart(result.strategy_equity, result.benchmark_equity), width="stretch"
        )


def render_agent_tab(prices: pd.DataFrame, params: BacktestParams) -> None:
    orchestrator = get_orchestrator()
    chat: list[dict[str, Any]] = st.session_state["chat"]

    names = orchestrator.registry.names()
    with st.expander(f"Factor registry ({len(names)})"):
        if not names:
            st.caption("No factors yet - describe one in the chat below.")
        selected = st.multiselect("Blend registered factors (equal weight)", names)
        if st.button("Backtest selection", disabled=not selected):
            try:
                report = orchestrator.backtest(prices, selected, params)
            except ValueError as exc:
                st.error(str(exc))
            else:
                chat.append(
                    {
                        "role": "assistant",
                        "text": f"Backtest of blend: {', '.join(selected)}",
                        "turn": AgentTurn(GenerationResult(True, "", "", None), report),
                    }
                )

    if not chat:
        st.info(
            'Describe a factor, e.g. *"14-day RSI mean reversion"* or *"price distance from the '
            '50-day moving average"*. The agent writes it, validates it (AST checks, look-ahead '
            "test, sandboxed pytest) and backtests it with the sidebar settings."
        )
    for message in chat:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message.get("turn") is not None:
                render_turn(message["turn"])

    prompt = st.chat_input("Describe the factor to build and backtest...")
    if not prompt:
        return

    chat.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Generating, validating and backtesting factor..."):
                turn = orchestrator.handle_request(prompt, prices, params)
        except Exception as exc:  # noqa: BLE001  missing `anthropic` package / API key, network errors, ...
            error_text = f"Agent unavailable: {exc}"
            st.error(error_text)
            chat.append({"role": "assistant", "text": error_text})
            return
        text = "Factor built and backtested." if turn.report else "I couldn't complete that request."
        st.markdown(text)
        render_turn(turn)
    chat.append({"role": "assistant", "text": text, "turn": turn})


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Quant Portfolio Engine", layout="wide")
    st.title("Quant Portfolio Engine")

    prices = load_prices()
    init_slider_state(settings)

    with st.sidebar:
        st.header("Factor weights")
        w_mom = st.slider("Momentum", 0.0, 1.0, step=0.05, key="momentum_weight")
        w_vol = st.slider("Volatility", 0.0, 1.0, step=0.05, key="volatility_weight")
        w_z = st.slider("Z-Score", 0.0, 1.0, step=0.05, key="zscore_weight")

        st.header("Portfolio")
        top_k = st.slider("Top-K holdings", 1, 5, key="top_k")
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
    factor_weights = (
        {"momentum": w_mom / total, "volatility": w_vol / total, "zscore": w_z / total}
        if total > 0
        else {}
    )
    if factor_weights:
        st.sidebar.caption(
            "Normalised: "
            + ", ".join(f"{name} {weight:.0%}" for name, weight in factor_weights.items())
        )
    params = BacktestParams(
        top_k=top_k,
        weighting_scheme=scheme_cast(scheme),
        rebalance_frequency=frequency,
        transaction_cost_bps=cost_bps,
    )

    render_regime_banner(prices)

    dashboard_tab, agent_tab = st.tabs(
        ["📊 Quant Engine Dashboard", "🤖 AI Factor Studio & Agent Chat"]
    )

    # st.stop() would also halt the other tab, so each tab reports problems locally.
    with dashboard_tab:
        if not factor_weights:
            st.error("Set at least one factor weight above zero.")
        elif top_k > prices.shape[1]:
            st.error(f"Top-K ({top_k}) exceeds the universe size ({prices.shape[1]}).")
        else:
            render_dashboard(prices, factor_weights, params)

    with agent_tab:
        render_agent_tab(prices, params)


def scheme_cast(value: str) -> WeightingScheme:
    return "inverse_volatility" if value == "inverse_volatility" else "equal"


main()
