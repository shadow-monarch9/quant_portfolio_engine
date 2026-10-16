"""Streamlit AppTest coverage for the regime banner and its Apply Preset button."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.agent.regime import RegimeDetector, equal_weight_benchmark
from src.config import get_settings
from src.data.yahoo_loader import YahooDataLoader

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APPLY_LABEL = "Apply Recommended Factor Preset"
SLIDER_KEYS = ["momentum_weight", "volatility_weight", "zscore_weight", "top_k"]


def _synthetic_prices(n_days: int) -> pd.DataFrame:
    """Seeded uptrend with low noise for every configured ticker (no network)."""
    np.random.seed(11)
    tickers = get_settings().universe.tickers
    returns = np.random.normal(0.001, 0.005, (n_days, len(tickers)))
    index = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=n_days), name="Date")
    return pd.DataFrame(100.0 * np.cumprod(1.0 + returns, axis=0), index=index, columns=tickers)


@pytest.fixture(autouse=True)
def _clear_streamlit_cache() -> Iterator[None]:
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _run_app(monkeypatch: pytest.MonkeyPatch, prices: pd.DataFrame) -> AppTest:
    monkeypatch.setattr(YahooDataLoader, "load", lambda self, force_refresh=False: prices)
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception
    return at


def _apply_buttons(at: AppTest) -> list[object]:
    return [b for b in at.button if b.label == APPLY_LABEL]


def _slider_values(at: AppTest) -> list[float]:
    return [at.session_state[key] for key in SLIDER_KEYS]


def test_sliders_start_at_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run_app(monkeypatch, _synthetic_prices(400))
    weights = get_settings().factor_weights
    assert _slider_values(at)[:3] == [weights.momentum, weights.volatility, weights.zscore]
    assert at.session_state["top_k"] == get_settings().portfolio.top_k


def test_apply_button_sets_sliders_to_recommended_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = _synthetic_prices(400)
    preset = RegimeDetector.analyze(equal_weight_benchmark(prices))["preset"]
    at = _run_app(monkeypatch, prices)
    assert _slider_values(at) != [preset[k.removesuffix("_weight")] for k in SLIDER_KEYS]

    next(b for b in at.button if b.label == APPLY_LABEL).click().run()

    assert not at.exception
    assert _slider_values(at) == [
        preset["momentum"],
        preset["volatility"],
        preset["zscore"],
        preset["top_k"],
    ]
    # The widgets themselves (not only session state) reflect the preset.
    assert [s.value for s in at.slider[:4]] == _slider_values(at)


def test_apply_button_overrides_manual_slider_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = _synthetic_prices(400)
    preset = RegimeDetector.analyze(equal_weight_benchmark(prices))["preset"]
    at = _run_app(monkeypatch, prices)
    at.slider(key="momentum_weight").set_value(0.05).run()
    assert at.session_state["momentum_weight"] == 0.05

    next(b for b in at.button if b.label == APPLY_LABEL).click().run()

    assert at.session_state["momentum_weight"] == preset["momentum"]
    assert not at.exception


def test_banner_shows_regime_name(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = _synthetic_prices(400)
    regime = RegimeDetector.analyze(equal_weight_benchmark(prices))["regime"]
    at = _run_app(monkeypatch, prices)
    assert any(regime in sub.value for sub in at.subheader)


def test_no_apply_button_when_history_is_insufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run_app(monkeypatch, _synthetic_prices(150))
    assert any("Insufficient History" in sub.value for sub in at.subheader)
    assert _apply_buttons(at) == []
