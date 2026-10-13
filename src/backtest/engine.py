"""Vectorised backtest engine with execution lag and turnover-based transaction costs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from src.backtest.metrics import calculate_performance_metrics
from src.portfolio.optimizer import TopKOptimizer, WeightingScheme
from src.portfolio.ranker import CrossSectionalRanker

_PERIOD_CODES: dict[str, str] = {"weekly": "W", "monthly": "M", "quarterly": "Q"}


@dataclass(frozen=True)
class BacktestResult:
    """Outputs of a backtest, all indexed from the first invested day."""

    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    strategy_equity: pd.Series
    benchmark_equity: pd.Series
    positions: pd.DataFrame  # weights actually held each day (target weights lagged 1 day)
    turnover: pd.Series
    costs: pd.Series


class BacktestEngine:
    """Applies lagged target weights to daily returns and charges turnover costs."""

    def __init__(
        self,
        transaction_cost_bps: float = 10.0,
        rebalance_frequency: str = "daily",
        trim_burn_in: bool = True,
    ) -> None:
        if transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative")
        if rebalance_frequency != "daily" and rebalance_frequency not in _PERIOD_CODES:
            raise ValueError(f"unknown rebalance frequency: {rebalance_frequency!r}")
        self.cost_rate: float = transaction_cost_bps / 10_000.0
        self.rebalance_frequency = rebalance_frequency
        self.trim_burn_in = trim_burn_in

    def rebalance_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        """True on the last trading day of each rebalance period (every day if daily)."""
        if self.rebalance_frequency == "daily":
            return pd.Series(True, index=index)
        period = pd.Series(index.to_period(_PERIOD_CODES[self.rebalance_frequency]), index=index)
        return period.ne(period.shift(-1))

    def run(self, target_weights: pd.DataFrame, prices: pd.DataFrame) -> BacktestResult:
        """Simulate the strategy.

        Target weights decided at close ``T`` are sampled on rebalance dates, held until the
        next rebalance, and lagged one day (``shift(1)``) so they earn the return of
        ``T+1`` onwards. Costs are ``cost_rate * sum(|delta weight|)`` on the day the
        position changes; intra-period weight drift is ignored.
        """
        prices = prices.sort_index()
        weights = target_weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
        returns = prices.pct_change()

        held = weights.where(self.rebalance_mask(pd.DatetimeIndex(prices.index)), other=float("nan")).ffill().fillna(0.0)
        positions = held.shift(1).fillna(0.0)

        gross = (positions * returns.fillna(0.0)).sum(axis=1)
        turnover = positions.diff().abs().sum(axis=1)
        costs = turnover * self.cost_rate
        strategy_returns = (gross - costs).rename("strategy")
        benchmark_returns = returns.mean(axis=1).fillna(0.0).rename("benchmark")

        if self.trim_burn_in:
            invested = positions.abs().sum(axis=1) > 0
            if invested.any():
                start = invested.idxmax()
                strategy_returns = strategy_returns.loc[start:]
                benchmark_returns = benchmark_returns.loc[start:]
                positions = positions.loc[start:]
                turnover = turnover.loc[start:]
                costs = costs.loc[start:]

        return BacktestResult(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            strategy_equity=(1.0 + strategy_returns).cumprod().rename("strategy"),
            benchmark_equity=(1.0 + benchmark_returns).cumprod().rename("benchmark"),
            positions=positions,
            turnover=turnover.rename("turnover"),
            costs=costs.rename("costs"),
        )


def run_strategy(
    prices: pd.DataFrame,
    factor_weights: Mapping[str, float],
    top_k: int,
    weighting_scheme: WeightingScheme,
    rebalance_frequency: str,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, BacktestResult]:
    """Full pipeline: factors -> alpha scores -> top-K weights -> backtest.

    Returns ``(alpha_scores, target_weights, result)``.
    """
    from src.factors.momentum import MomentumFactor
    from src.factors.volatility import VolatilityFactor
    from src.factors.zscore import ZScoreFactor

    ranker = CrossSectionalRanker(
        factors=[MomentumFactor(), VolatilityFactor(), ZScoreFactor()],
        weights=factor_weights,
    )
    scores = ranker.score(prices)
    weights = TopKOptimizer(k=top_k, weighting_scheme=weighting_scheme).optimize(scores, prices)
    engine = BacktestEngine(
        transaction_cost_bps=transaction_cost_bps, rebalance_frequency=rebalance_frequency
    )
    return scores, weights, engine.run(weights, prices)


if __name__ == "__main__":
    from src.config import get_settings
    from src.data.yahoo_loader import YahooDataLoader
    from src.utils.logger import logger, setup_logger

    settings = get_settings()
    setup_logger(settings.logging.level)

    price_matrix = YahooDataLoader.from_settings(settings).load()  # uses Parquet cache
    _, _, outcome = run_strategy(
        prices=price_matrix,
        factor_weights=settings.factor_weights.model_dump(),
        top_k=settings.portfolio.top_k,
        weighting_scheme=settings.portfolio.weighting_scheme,
        rebalance_frequency=settings.rebalance.frequency,
        transaction_cost_bps=settings.rebalance.transaction_cost_bps,
    )
    stats = calculate_performance_metrics(outcome.strategy_returns, outcome.benchmark_returns)

    logger.info(
        "Backtest {} -> {} ({} days, K={}, {}, {} rebalance, {:.1f} bps)",
        outcome.strategy_returns.index[0].date(),
        outcome.strategy_returns.index[-1].date(),
        len(outcome.strategy_returns),
        settings.portfolio.top_k,
        settings.portfolio.weighting_scheme,
        settings.rebalance.frequency,
        settings.rebalance.transaction_cost_bps,
    )
    logger.success("Sharpe Ratio : {:.2f}", stats["sharpe"])
    logger.success("CAGR         : {:.2f}%", stats["cagr_pct"])
    logger.success("Max Drawdown : {:.2f}%", stats["max_drawdown_pct"])
    logger.info(
        "Benchmark    : Sharpe {:.2f}, CAGR {:.2f}%, MaxDD {:.2f}%",
        stats["benchmark_sharpe"],
        stats["benchmark_cagr_pct"],
        stats["benchmark_max_drawdown_pct"],
    )
