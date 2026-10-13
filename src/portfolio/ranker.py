"""Cross-sectional ranking and composite alpha scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from src.factors.base import AbstractFactor


class CrossSectionalRanker:
    """Blends per-factor daily percentile ranks into a composite alpha score in [0, 1]."""

    def __init__(self, factors: Sequence[AbstractFactor], weights: Mapping[str, float]) -> None:
        names = [factor.name for factor in factors]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate factor names: {names}")
        missing = [name for name in names if name not in weights]
        if missing:
            raise ValueError(f"no weight configured for factors: {missing}")
        if any(weights[name] < 0 for name in names):
            raise ValueError("factor weights must be non-negative")
        if sum(weights[name] for name in names) <= 0:
            raise ValueError("factor weights must sum to a positive value")

        self.factors: dict[str, AbstractFactor] = {f.name: f for f in factors}
        self.weights: dict[str, float] = {name: float(weights[name]) for name in names}

    def rank_factor(self, factor: AbstractFactor, values: pd.DataFrame) -> pd.DataFrame:
        """Daily cross-sectional percentile rank, inverted when lower values are better."""
        ranks = values.rank(axis=1, pct=True)
        return ranks if factor.higher_is_better else 1.0 - ranks

    def composite_score(self, factor_values: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Weighted blend of factor ranks, min-max normalised per date to [0, 1]."""
        missing = [name for name in self.factors if name not in factor_values]
        if missing:
            raise ValueError(f"missing factor values for: {missing}")

        weighted_sum: pd.DataFrame | None = None
        complete: pd.DataFrame | None = None
        for name, factor in self.factors.items():
            ranks = self.rank_factor(factor, factor_values[name])
            contribution = ranks * self.weights[name]
            if weighted_sum is None or complete is None:
                weighted_sum = contribution
                complete = ranks.notna()
            else:
                weighted_sum = weighted_sum + contribution
                complete = complete & ranks.notna()

        assert weighted_sum is not None and complete is not None
        # Strict burn-in: a score exists only where *every* factor is available, so the
        # warm-up of the slowest factor (252d momentum) defines the first scored row.
        blended = weighted_sum.where(complete) / sum(self.weights.values())

        row_min = blended.min(axis=1)
        row_range = blended.max(axis=1) - row_min
        normalised = blended.sub(row_min, axis=0).div(row_range.where(row_range > 0), axis=0)
        # Rows where every asset ties get a neutral 0.5 rather than NaN.
        tied = row_range.eq(0) & blended.notna().any(axis=1)
        normalised.loc[tied] = blended.loc[tied].notna().astype(float) * 0.5
        normalised = normalised.where(blended.notna())
        return normalised.clip(lower=0.0, upper=1.0)

    def score(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute every factor from ``prices`` and return the composite alpha score."""
        values = {name: factor.compute(prices) for name, factor in self.factors.items()}
        return self.composite_score(values)


if __name__ == "__main__":
    from src.config import get_settings
    from src.data.yahoo_loader import YahooDataLoader
    from src.factors.momentum import MomentumFactor
    from src.factors.volatility import VolatilityFactor
    from src.factors.zscore import ZScoreFactor
    from src.utils.logger import logger, setup_logger

    settings = get_settings()
    setup_logger(settings.logging.level)

    prices = YahooDataLoader.from_settings(settings).load()  # uses Sprint 1 Parquet cache
    ranker = CrossSectionalRanker(
        factors=[MomentumFactor(), VolatilityFactor(), ZScoreFactor()],
        weights=settings.factor_weights.model_dump(),
    )
    composite = ranker.score(prices)

    latest = composite.dropna(how="all").index[-1]
    top3 = composite.loc[latest].dropna().sort_values(ascending=False).head(3)
    logger.info("Top 3 tickers on {}:", latest.date())
    for position, (ticker, alpha) in enumerate(top3.items(), start=1):
        logger.info("  #{} {} (alpha={:.3f})", position, ticker, alpha)
