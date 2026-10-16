"""Deterministic market-regime detection that maps a benchmark series to a factor preset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MIN_HISTORY = 200
SMA_WINDOW = 200
VOL_WINDOW = 30
HIGH_VOL_THRESHOLD = 0.22  # annualised


def equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight index level (starting at 1.0) built from a dates x tickers price matrix."""
    index: pd.Series = (1.0 + prices.pct_change().mean(axis=1).fillna(0.0)).cumprod()
    return index.rename("benchmark")


class RegimeDetector:
    """Classifies the market as bull / high-volatility / neutral from trend and variance."""

    @staticmethod
    def analyze(benchmark_prices: pd.Series) -> dict[str, Any]:
        """Return the regime, commentary, recommended factor preset and supporting metrics."""
        prices = benchmark_prices.dropna()
        if len(prices) < MIN_HISTORY:
            return {
                "regime": "Insufficient History",
                "summary": f"Regime detection needs at least {MIN_HISTORY} observations "
                f"(got {len(prices)}).",
                "suggestion": "Keep the current factor weights.",
                "preset": {},
                "current_price": float(prices.iloc[-1]) if len(prices) else float("nan"),
                "sma_200": float("nan"),
                "rolling_vol": float("nan"),
            }

        current_price = float(prices.iloc[-1])
        sma_200 = float(prices.rolling(SMA_WINDOW).mean().iloc[-1])
        rolling_vol = float(prices.pct_change().rolling(VOL_WINDOW).std().iloc[-1] * np.sqrt(252))

        is_bull = current_price > sma_200
        is_high_vol = rolling_vol > HIGH_VOL_THRESHOLD

        if is_high_vol:
            regime = {
                "regime": "High Volatility / Bear Risk",
                "summary": "Portfolio variance is elevated relative to historical baseline.",
                "suggestion": "Increase Volatility dampening and holdings count to preserve "
                "capital and smooth drawdowns.",
                "preset": {"momentum": 0.20, "volatility": 0.60, "zscore": 0.20, "top_k": 4},
            }
        elif is_bull:
            regime = {
                "regime": "Strong Bull Market (Low Volatility)",
                "summary": "Benchmark is trending above its 200-day SMA with low market variance.",
                "suggestion": "Focus allocation heavily on Momentum to maximize CAGR during "
                "upward trend.",
                "preset": {"momentum": 0.80, "volatility": 0.00, "zscore": 0.20, "top_k": 2},
            }
        else:
            regime = {
                "regime": "Neutral / Sideways Market",
                "summary": "Price is below the 200-day SMA but volatility remains bounded.",
                "suggestion": "Maintain a balanced multi-factor weighting across momentum and "
                "risk indicators.",
                "preset": {"momentum": 0.40, "volatility": 0.40, "zscore": 0.20, "top_k": 3},
            }

        return {
            **regime,
            "current_price": current_price,
            "sma_200": sma_200,
            "rolling_vol": rolling_vol,
        }
