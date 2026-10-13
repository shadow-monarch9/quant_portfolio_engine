"""Top-K portfolio construction from composite alpha scores."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.factors.volatility import VolatilityFactor

WeightingScheme = Literal["equal", "inverse_volatility"]


class TopKOptimizer:
    """Holds the ``k`` highest-scoring assets each day, long-only and fully invested."""

    def __init__(self, k: int = 3, weighting_scheme: WeightingScheme = "equal") -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if weighting_scheme not in ("equal", "inverse_volatility"):
            raise ValueError(f"unknown weighting scheme: {weighting_scheme!r}")
        self.k = k
        self.weighting_scheme: WeightingScheme = weighting_scheme

    def optimize(self, scores: pd.DataFrame, prices: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return target weights (dates x tickers).

        Each row sums to 1.0, or to 0.0 where the alpha score is NaN (burn-in). Weights on
        date T use only information available at T; execution lag is the engine's job.
        ``prices`` is required for ``inverse_volatility``.
        """
        if self.k > scores.shape[1]:
            raise ValueError(f"k={self.k} exceeds the number of assets ({scores.shape[1]})")

        # Rank 1 = best score; ties resolved by column order so the result is deterministic.
        position = scores.rank(axis=1, ascending=False, method="first")
        selected = position.le(self.k) & scores.notna()
        raw = selected.astype(float)

        if self.weighting_scheme == "inverse_volatility":
            if prices is None:
                raise ValueError("prices are required for inverse_volatility weighting")
            vol = VolatilityFactor().compute(prices).reindex(index=scores.index, columns=scores.columns)
            inverse = (1.0 / vol.where(vol > 0)).where(selected)
            # Fall back to equal weights on rows where volatility is unavailable.
            usable = inverse.notna().sum(axis=1).eq(selected.sum(axis=1))
            raw = inverse.where(usable, raw).where(selected, 0.0)

        total = raw.sum(axis=1)
        weights = raw.div(total.where(total > 0), axis=0).fillna(0.0)
        return weights.replace([np.inf, -np.inf], 0.0)
