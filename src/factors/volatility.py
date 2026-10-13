"""Rolling realised-volatility factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import AbstractFactor


class VolatilityFactor(AbstractFactor):
    """30-day rolling annualised standard deviation of daily returns."""

    def __init__(self, window: int = 30, trading_days: int = 252) -> None:
        super().__init__(
            name="volatility",
            description=f"{window}d rolling annualised stdev of daily returns",
            higher_is_better=False,
        )
        self.window = window
        self.trading_days = trading_days

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        return returns.rolling(self.window).std() * np.sqrt(self.trading_days)
