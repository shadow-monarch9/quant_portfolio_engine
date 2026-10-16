"""Unit tests for RegimeDetector using deterministic synthetic price series (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agent.regime import RegimeDetector

N_DAYS = 300
VOL_THRESHOLD = 0.22
REQUIRED_KEYS = {
    "regime",
    "summary",
    "suggestion",
    "preset",
    "current_price",
    "sma_200",
    "rolling_vol",
}


def _prices_from_returns(daily_returns: np.ndarray) -> pd.Series:
    index = pd.bdate_range("2022-01-03", periods=len(daily_returns))
    return pd.Series(100.0 * np.cumprod(1.0 + daily_returns), index=index, name="benchmark")


@pytest.fixture
def short_prices() -> pd.Series:
    np.random.seed(0)
    return _prices_from_returns(np.random.normal(0.001, 0.005, 100))


@pytest.fixture
def bull_prices() -> pd.Series:
    """Steady uptrend (+0.1%/day) with ~0.5% daily noise -> ~8% annualised volatility."""
    np.random.seed(1)
    return _prices_from_returns(np.random.normal(0.001, 0.005, N_DAYS))


@pytest.fixture
def high_vol_prices() -> pd.Series:
    """Uptrend, then +/-3% daily swings over the last 30 days -> ~48% annualised volatility."""
    np.random.seed(2)
    returns = np.random.normal(0.001, 0.005, N_DAYS)
    returns[-30:] = np.random.choice([-0.03, 0.03], size=30)
    return _prices_from_returns(returns)


@pytest.fixture
def neutral_prices() -> pd.Series:
    """Rally for 200 days, then a quiet decline that leaves price below its 200-day SMA."""
    np.random.seed(3)
    returns = np.concatenate(
        [np.random.normal(0.001, 0.005, 200), np.random.normal(-0.003, 0.005, 100)]
    )
    return _prices_from_returns(returns)


def test_insufficient_data(short_prices: pd.Series) -> None:
    result = RegimeDetector.analyze(short_prices)
    assert result["regime"] == "Insufficient History"
    assert result["preset"] == {}
    assert "200" in result["summary"]


def test_strong_bull_market(bull_prices: pd.Series) -> None:
    result = RegimeDetector.analyze(bull_prices)
    assert "Strong Bull" in result["regime"]
    assert result["preset"] == {"momentum": 0.80, "volatility": 0.00, "zscore": 0.20, "top_k": 2}
    assert result["current_price"] > result["sma_200"]
    assert result["rolling_vol"] <= VOL_THRESHOLD


def test_high_volatility_bear_risk(high_vol_prices: pd.Series) -> None:
    result = RegimeDetector.analyze(high_vol_prices)
    assert "High Volatility" in result["regime"]
    assert result["preset"] == {"momentum": 0.20, "volatility": 0.60, "zscore": 0.20, "top_k": 4}
    assert result["rolling_vol"] > VOL_THRESHOLD


def test_neutral_sideways_market(neutral_prices: pd.Series) -> None:
    result = RegimeDetector.analyze(neutral_prices)
    assert "Neutral" in result["regime"] or "Sideways" in result["regime"]
    assert result["preset"] == {"momentum": 0.40, "volatility": 0.40, "zscore": 0.20, "top_k": 3}
    assert result["current_price"] < result["sma_200"]
    assert result["rolling_vol"] <= VOL_THRESHOLD


@pytest.mark.parametrize("fixture_name", ["bull_prices", "high_vol_prices", "neutral_prices"])
def test_output_dictionary_keys(fixture_name: str, request: pytest.FixtureRequest) -> None:
    result = RegimeDetector.analyze(request.getfixturevalue(fixture_name))
    assert REQUIRED_KEYS <= result.keys()


def test_insufficient_data_still_has_required_keys(short_prices: pd.Series) -> None:
    assert REQUIRED_KEYS <= RegimeDetector.analyze(short_prices).keys()


def test_metrics_match_spec_formulas(bull_prices: pd.Series) -> None:
    result = RegimeDetector.analyze(bull_prices)
    expected_sma = bull_prices.rolling(200).mean().iloc[-1]
    expected_vol = bull_prices.pct_change().rolling(30).std().iloc[-1] * np.sqrt(252)
    assert result["current_price"] == pytest.approx(bull_prices.iloc[-1])
    assert result["sma_200"] == pytest.approx(expected_sma)
    assert result["rolling_vol"] == pytest.approx(expected_vol)


def test_exactly_200_points_is_enough() -> None:
    np.random.seed(4)
    result = RegimeDetector.analyze(_prices_from_returns(np.random.normal(0.001, 0.005, 200)))
    assert result["regime"] != "Insufficient History"


# --- Boundary behaviour ------------------------------------------------------------------


def test_history_boundary_199_vs_200() -> None:
    np.random.seed(5)
    returns = np.random.normal(0.001, 0.005, 200)
    assert RegimeDetector.analyze(_prices_from_returns(returns[:199]))["regime"] == (
        "Insufficient History"
    )
    assert RegimeDetector.analyze(_prices_from_returns(returns))["regime"] != "Insufficient History"


def test_price_equal_to_sma_is_not_bull() -> None:
    flat = pd.Series(100.0, index=pd.bdate_range("2022-01-03", periods=N_DAYS))
    result = RegimeDetector.analyze(flat)
    assert result["current_price"] == result["sma_200"]
    assert result["rolling_vol"] == 0.0
    assert "Neutral" in result["regime"]


def test_volatility_exactly_at_threshold_is_not_high(
    bull_prices: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    vol = RegimeDetector.analyze(bull_prices)["rolling_vol"]
    monkeypatch.setattr("src.agent.regime.HIGH_VOL_THRESHOLD", vol)
    assert "Strong Bull" in RegimeDetector.analyze(bull_prices)["regime"]  # strict ">"
    monkeypatch.setattr("src.agent.regime.HIGH_VOL_THRESHOLD", vol - 1e-9)
    assert "High Volatility" in RegimeDetector.analyze(bull_prices)["regime"]


def test_nan_values_are_ignored(bull_prices: pd.Series) -> None:
    with_nan = bull_prices.copy()
    with_nan.iloc[[10, 50, 120]] = np.nan
    assert RegimeDetector.analyze(with_nan)["regime"] != "Insufficient History"


def test_empty_series_returns_fallback() -> None:
    result = RegimeDetector.analyze(pd.Series(dtype=float))
    assert result["regime"] == "Insufficient History"
    assert np.isnan(result["current_price"])
