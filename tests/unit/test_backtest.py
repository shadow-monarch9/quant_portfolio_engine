"""Unit tests for the optimizer, backtest engine and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import calculate_performance_metrics
from src.portfolio.optimizer import TopKOptimizer


@pytest.fixture
def scores() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=3, name="Date")
    return pd.DataFrame(
        [[np.nan] * 4, [0.1, 0.9, 0.5, 0.7], [1.0, 0.0, 0.4, 0.6]],
        index=index,
        columns=list("ABCD"),
    )


def test_topk_equal_weights(scores: pd.DataFrame) -> None:
    weights = TopKOptimizer(k=2, weighting_scheme="equal").optimize(scores)
    assert (weights.iloc[0] == 0.0).all()  # burn-in row: no positions
    assert weights.iloc[1].to_dict() == {"A": 0.0, "B": 0.5, "C": 0.0, "D": 0.5}
    assert weights.iloc[2].to_dict() == {"A": 0.5, "B": 0.0, "C": 0.0, "D": 0.5}
    assert weights.sum(axis=1).tolist() == [0.0, 1.0, 1.0]


def test_topk_inverse_volatility_prefers_calm_assets(scores: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 1, size=(80, 4)) * np.array([0.01, 0.05, 0.01, 0.02])
    prices = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), columns=list("ABCD"))
    prices.index = pd.bdate_range("2024-01-01", periods=80, name="Date")
    alpha = pd.DataFrame(
        np.tile([0.9, 0.8, 0.1, 0.0], (80, 1)), index=prices.index, columns=prices.columns
    )
    weights = TopKOptimizer(k=2, weighting_scheme="inverse_volatility").optimize(alpha, prices)
    last = weights.iloc[-1]
    assert last.sum() == pytest.approx(1.0)
    assert last["A"] > last["B"] > 0.0
    assert last["C"] == 0.0 and last["D"] == 0.0


def test_optimizer_validation(scores: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        TopKOptimizer(k=0)
    with pytest.raises(ValueError, match="exceeds"):
        TopKOptimizer(k=9).optimize(scores)
    with pytest.raises(ValueError, match="prices are required"):
        TopKOptimizer(k=2, weighting_scheme="inverse_volatility").optimize(scores)


def test_engine_lags_weights_and_charges_costs() -> None:
    index = pd.bdate_range("2024-01-01", periods=4, name="Date")
    prices = pd.DataFrame({"A": [100.0, 110.0, 110.0, 110.0], "B": [100.0] * 4}, index=index)
    weights = pd.DataFrame({"A": [1.0] * 4, "B": [0.0] * 4}, index=index)

    result = BacktestEngine(transaction_cost_bps=10.0).run(weights, prices)

    # Weights set at day 0 first earn day 1's return, so the day-0 return is never captured
    # and trimming starts at day 1 (the first invested day).
    assert result.strategy_returns.index[0] == index[1]
    assert result.strategy_returns.iloc[0] == pytest.approx(0.10 - 0.0010)  # 10% less 10 bps entry
    assert result.strategy_returns.iloc[1] == pytest.approx(0.0)  # no turnover afterwards
    assert result.benchmark_returns.iloc[0] == pytest.approx(0.05)
    assert result.strategy_equity.iloc[-1] == pytest.approx(1.0 + 0.10 - 0.0010)


def test_metrics_values() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.0] * 63)
    out = calculate_performance_metrics(returns, returns)
    expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert out["sharpe"] == pytest.approx(expected_sharpe)
    assert out["win_rate_pct"] == pytest.approx(50.0)
    assert out["max_drawdown_pct"] == pytest.approx(-1.0)
    assert out["cagr_pct"] == pytest.approx((np.prod(1 + returns.to_numpy()) ** (252 / len(returns)) - 1) * 100)
    assert out["sortino"] > out["sharpe"]
    assert out["excess_cagr_pct"] == pytest.approx(0.0)
