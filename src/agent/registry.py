"""Registry of validated factors and a helper to backtest them through the standard pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.factors.base import AbstractFactor
from src.portfolio.optimizer import TopKOptimizer, WeightingScheme
from src.portfolio.ranker import CrossSectionalRanker


class FactorRegistry:
    """In-memory store of factors that passed validation, keyed by ``factor.name``."""

    def __init__(self) -> None:
        self._factors: dict[str, AbstractFactor] = {}

    def register(self, factor: AbstractFactor, overwrite: bool = False) -> None:
        if factor.name in self._factors and not overwrite:
            raise ValueError(f"factor '{factor.name}' is already registered")
        self._factors[factor.name] = factor

    def get(self, name: str) -> AbstractFactor:
        return self._factors[name]

    def names(self) -> list[str]:
        return sorted(self._factors)

    def __contains__(self, name: object) -> bool:
        return name in self._factors


def backtest_factors(
    prices: pd.DataFrame,
    factors: Sequence[AbstractFactor],
    weights: Mapping[str, float],
    top_k: int,
    weighting_scheme: WeightingScheme,
    rebalance_frequency: str,
    transaction_cost_bps: float,
) -> BacktestResult:
    """Rank -> top-K weights -> backtest for an arbitrary factor set (built-in or generated)."""
    scores = CrossSectionalRanker(factors=factors, weights=weights).score(prices)
    target = TopKOptimizer(k=top_k, weighting_scheme=weighting_scheme).optimize(scores, prices)
    engine = BacktestEngine(
        transaction_cost_bps=transaction_cost_bps, rebalance_frequency=rebalance_frequency
    )
    return engine.run(target, prices)
