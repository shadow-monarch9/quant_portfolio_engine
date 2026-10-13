"""Rolling price z-score (mean-reversion) factor."""

from __future__ import annotations

import pandas as pd

from src.factors.base import AbstractFactor


class ZScoreFactor(AbstractFactor):
    """Distance of price from its 20-day mean in units of 20-day standard deviation."""

    def __init__(self, window: int = 20) -> None:
        super().__init__(
            name="zscore",
            description=f"(price - {window}d mean) / {window}d stdev",
            higher_is_better=False,
        )
        self.window = window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        rolling = prices.rolling(self.window)
        return (prices - rolling.mean()) / rolling.std()
