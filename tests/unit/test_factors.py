"""Unit tests for factors and the cross-sectional ranker."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.base import AbstractFactor
from src.factors.momentum import MomentumFactor
from src.factors.volatility import VolatilityFactor
from src.factors.zscore import ZScoreFactor
from src.portfolio.ranker import CrossSectionalRanker


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2022-01-03", periods=320, name="Date")
    steps = rng.normal(0.0005, 0.02, size=(320, 5))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(steps, axis=0)),
        index=index,
        columns=["A", "B", "C", "D", "E"],
    )


@pytest.fixture
def ranker() -> CrossSectionalRanker:
    return CrossSectionalRanker(
        factors=[MomentumFactor(), VolatilityFactor(), ZScoreFactor()],
        weights={"momentum": 0.5, "volatility": 0.3, "zscore": 0.2},
    )


def test_abstract_factor_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        AbstractFactor("x", "y", True)  # type: ignore[abstract]


@pytest.mark.parametrize("factor_cls", [MomentumFactor, VolatilityFactor, ZScoreFactor])
def test_output_shape_matches_input(factor_cls: type[AbstractFactor], prices: pd.DataFrame) -> None:
    out = factor_cls().compute(prices)  # type: ignore[call-arg]
    assert out.shape == prices.shape
    assert out.index.equals(prices.index)
    assert list(out.columns) == list(prices.columns)


def test_factor_metadata() -> None:
    assert MomentumFactor().higher_is_better is True
    assert VolatilityFactor().higher_is_better is False
    assert ZScoreFactor().higher_is_better is False
    for factor in (MomentumFactor(), VolatilityFactor(), ZScoreFactor()):
        assert factor.name and factor.description


def test_momentum_warmup_nans(prices: pd.DataFrame) -> None:
    out = MomentumFactor().compute(prices)
    assert out.iloc[:252].isna().all().all()
    assert out.iloc[252:].notna().all().all()


def test_volatility_warmup_nans(prices: pd.DataFrame) -> None:
    out = VolatilityFactor().compute(prices)
    # first return is NaN, so the first valid 30-obs window ends at row 30
    assert out.iloc[:30].isna().all().all()
    assert out.iloc[30:].notna().all().all()


def test_zscore_warmup_nans(prices: pd.DataFrame) -> None:
    out = ZScoreFactor().compute(prices)
    assert out.iloc[:19].isna().all().all()
    assert out.iloc[19:].notna().all().all()


def test_momentum_formula(prices: pd.DataFrame) -> None:
    out = MomentumFactor().compute(prices)
    expected = prices.pct_change(252) - prices.pct_change(21)
    pd.testing.assert_frame_equal(out, expected)


def test_volatility_annualisation(prices: pd.DataFrame) -> None:
    out = VolatilityFactor().compute(prices)
    daily = prices.pct_change().rolling(30).std()
    pd.testing.assert_frame_equal(out, daily * np.sqrt(252))


def test_ranks_bounded_in_unit_interval(prices: pd.DataFrame, ranker: CrossSectionalRanker) -> None:
    for factor in ranker.factors.values():
        ranks = ranker.rank_factor(factor, factor.compute(prices)).stack().dropna()
        assert not ranks.empty
        assert ((ranks >= 0.0) & (ranks <= 1.0)).all()


def test_composite_bounded_in_unit_interval(
    prices: pd.DataFrame, ranker: CrossSectionalRanker
) -> None:
    composite = ranker.score(prices)
    assert composite.shape == prices.shape
    values = composite.stack().dropna()
    assert not values.empty
    assert ((values >= 0.0) & (values <= 1.0)).all()
    assert composite.iloc[-1].min() == pytest.approx(0.0)
    assert composite.iloc[-1].max() == pytest.approx(1.0)


def test_composite_nan_only_during_warmup(prices: pd.DataFrame, ranker: CrossSectionalRanker) -> None:
    composite = ranker.score(prices)
    # strict burn-in: momentum (252d) is the slowest factor, so the composite is NaN until
    # every factor is available (rows 0..251) and fully populated afterwards.
    assert composite.iloc[:252].isna().all().all()
    assert composite.iloc[252:].notna().all().all()


def test_composite_nan_if_any_factor_nan(prices: pd.DataFrame, ranker: CrossSectionalRanker) -> None:
    values = {name: f.compute(prices) for name, f in ranker.factors.items()}
    values["zscore"].iloc[-1, 0] = float("nan")
    composite = ranker.composite_score(values)
    assert composite.iloc[-1, 0] != composite.iloc[-1, 0]  # NaN
    assert composite.iloc[-1, 1:].notna().all()


def test_rank_inversion_for_volatility_and_zscore(ranker: CrossSectionalRanker) -> None:
    index = pd.bdate_range("2024-01-01", periods=1)
    values = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]], index=index, columns=["A", "B", "C", "D"])

    for name in ("volatility", "zscore"):
        factor = ranker.factors[name]
        raw = values.rank(axis=1, pct=True)
        inverted = ranker.rank_factor(factor, values)
        pd.testing.assert_frame_equal(inverted, 1.0 - raw)
        # lowest raw value must receive the best (highest) rank
        assert inverted.idxmax(axis=1).iloc[0] == "A"
        assert inverted.idxmin(axis=1).iloc[0] == "D"


def test_no_inversion_for_momentum(ranker: CrossSectionalRanker) -> None:
    index = pd.bdate_range("2024-01-01", periods=1)
    values = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]], index=index, columns=["A", "B", "C", "D"])
    ranks = ranker.rank_factor(ranker.factors["momentum"], values)
    assert ranks.idxmax(axis=1).iloc[0] == "D"


def test_ranker_rejects_missing_weight() -> None:
    with pytest.raises(ValueError, match="no weight"):
        CrossSectionalRanker([MomentumFactor(), ZScoreFactor()], {"momentum": 1.0})
