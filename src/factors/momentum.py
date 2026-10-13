"""12-1 month momentum factor."""

from __future__ import annotations

import pandas as pd

from src.factors.base import AbstractFactor


class MomentumFactor(AbstractFactor):
    """Trailing 252-day return excluding the most recent 21 days."""

    def __init__(self, long_window: int = 252, short_window: int = 21) -> None:
        super().__init__(
            name="momentum",
            description=f"{long_window}d return minus {short_window}d return",
            higher_is_better=True,
        )
        self.long_window = long_window
        self.short_window = short_window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change(self.long_window) - prices.pct_change(self.short_window)
