"""Look-ahead bias guard: future prices must never influence past weights or returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.factors.momentum import MomentumFactor
from src.factors.volatility import VolatilityFactor
from src.factors.zscore import ZScoreFactor
from src.portfolio.optimizer import TopKOptimizer, WeightingScheme
from src.portfolio.ranker import CrossSectionalRanker

N_DAYS = 400
T = 330  # positional index of "time T", past the 252-day burn-in


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2021-01-04", periods=N_DAYS, name="Date")
    steps = rng.normal(0.0004, 0.02, size=(N_DAYS, 6))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=list("ABCDEF")
    )


def _weights(prices: pd.DataFrame, scheme: WeightingScheme) -> pd.DataFrame:
    ranker = CrossSectionalRanker(
        factors=[MomentumFactor(), VolatilityFactor(), ZScoreFactor()],
        weights={"momentum": 0.5, "volatility": 0.3, "zscore": 0.2},
    )
    return TopKOptimizer(k=2, weighting_scheme=scheme).optimize(ranker.score(prices), prices)


def _perturb_future(prices: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``prices`` with everything from T+1 onwards drastically changed."""
    altered = prices.copy()
    future = altered.iloc[T + 1 :].to_numpy()
    altered.iloc[T + 1 :] = future * np.random.default_rng(1).uniform(0.2, 5.0, size=future.shape)
    return altered


@pytest.mark.parametrize("scheme", ["equal", "inverse_volatility"])
def test_weights_at_t_unchanged_by_prices_after_t(
    prices: pd.DataFrame, scheme: WeightingScheme
) -> None:
    base = _weights(prices, scheme)
    altered = _weights(_perturb_future(prices), scheme)

    assert base.iloc[T].sum() == pytest.approx(1.0)  # T is past burn-in, so this is non-trivial
    pd.testing.assert_frame_equal(base.iloc[: T + 1], altered.iloc[: T + 1], check_exact=True)


def test_single_price_shock_at_t_plus_1_leaves_weights_at_t_unchanged(prices: pd.DataFrame) -> None:
    shocked = prices.copy()
    shocked.iloc[T + 1, 0] = prices.to_numpy()[T + 1, 0] * 10.0
    base = _weights(prices, "equal")
    after = _weights(shocked, "equal")
    pd.testing.assert_series_equal(base.iloc[T], after.iloc[T], check_exact=True)


@pytest.mark.parametrize("frequency", ["daily", "monthly"])
def test_backtest_returns_up_to_t_unchanged_by_prices_after_t(
    prices: pd.DataFrame, frequency: str
) -> None:
    engine = BacktestEngine(transaction_cost_bps=10.0, rebalance_frequency=frequency)
    base_result = engine.run(_weights(prices, "equal"), prices)

    altered_prices = _perturb_future(prices)
    altered_result = engine.run(_weights(altered_prices, "equal"), altered_prices)

    cutoff = prices.index[T]
    pd.testing.assert_series_equal(
        base_result.strategy_returns.loc[:cutoff],
        altered_result.strategy_returns.loc[:cutoff],
        check_exact=True,
    )
