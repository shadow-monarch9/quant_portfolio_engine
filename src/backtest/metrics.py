"""Performance statistics for daily strategy and benchmark return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


def _cagr(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    growth = float(np.prod(1.0 + returns.to_numpy(dtype=float)))
    if growth <= 0:
        return -100.0
    return (growth ** (TRADING_DAYS / len(returns)) - 1.0) * 100.0


def _sharpe(returns: pd.Series) -> float:
    std = float(returns.std(ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return float("nan")
    return float(returns.mean()) / std * float(np.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series) -> float:
    downside = float(np.sqrt((np.minimum(returns, 0.0) ** 2).mean()))
    if not np.isfinite(downside) or downside == 0.0:
        return float("nan")
    return float(returns.mean()) / downside * float(np.sqrt(TRADING_DAYS))


def _max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough decline of the equity curve, in percent (<= 0)."""
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return float((equity / peak - 1.0).min()) * 100.0


def calculate_performance_metrics(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> dict[str, float]:
    """Annualised performance statistics (0% risk-free rate, 252 trading days).

    Strategy metrics use the plain keys; the benchmark's CAGR, Sharpe and drawdown are
    returned alongside with a ``benchmark_`` prefix for comparison.
    """
    strategy = strategy_returns.dropna()
    benchmark = benchmark_returns.dropna()
    strategy_cagr = _cagr(strategy)
    benchmark_cagr = _cagr(benchmark)
    return {
        "cagr_pct": strategy_cagr,
        "sharpe": _sharpe(strategy),
        "sortino": _sortino(strategy),
        "max_drawdown_pct": _max_drawdown(strategy),
        "win_rate_pct": float((strategy > 0).mean()) * 100.0 if len(strategy) else float("nan"),
        "benchmark_cagr_pct": benchmark_cagr,
        "benchmark_sharpe": _sharpe(benchmark),
        "benchmark_max_drawdown_pct": _max_drawdown(benchmark),
        "excess_cagr_pct": strategy_cagr - benchmark_cagr,
    }
