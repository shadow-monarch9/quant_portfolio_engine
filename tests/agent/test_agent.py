"""Tests for the factor compiler, sandbox and self-healing generator (LLM stubbed)."""

from __future__ import annotations

import pytest

from src.agent.factor_compiler import (
    FactorCompileError,
    compile_factor,
    inspect_factor_source,
    instantiate_factor,
    make_synthetic_prices,
    validate_factor_output,
)
from src.agent.factor_generator import FactorGenerator, extract_code
from src.agent.registry import FactorRegistry, backtest_factors
from src.factors.momentum import MomentumFactor

GOOD = '''
import numpy as np
import pandas as pd
from src.factors.base import AbstractFactor


class RsiMeanReversion(AbstractFactor):
    def __init__(self) -> None:
        super().__init__(name="rsi14", description="14d RSI", higher_is_better=False)

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        return 100 - 100 / (1 + gain / loss)
'''

LOOKAHEAD = GOOD.replace("prices.diff()", "prices.shift(-1) - prices").replace("rsi14", "peek")
BAD_IMPORT = GOOD.replace("import numpy as np", "import os\nimport numpy as np")
BAD_SHAPE = GOOD.replace("return 100 - 100 / (1 + gain / loss)", "return (100 - 100 / (1 + gain / loss)).iloc[1:]")


class ScriptedClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        self.calls.append(list(messages))
        return f"```python\n{self.replies[len(self.calls) - 1]}```"


def test_valid_source_compiles_and_validates() -> None:
    factor = instantiate_factor(compile_factor(GOOD))
    values = validate_factor_output(factor, make_synthetic_prices())
    assert values.shape == (400, 6)


@pytest.mark.parametrize(
    "snippet",
    [
        "import os",
        "import subprocess",
        "from requests import get",
        "import sys",
        "from src.factors.momentum import MomentumFactor",
    ],
)
def test_forbidden_imports_rejected(snippet: str) -> None:
    report = inspect_factor_source(snippet + "\n" + GOOD)
    assert not report.ok and "not allowed" in report.errors[0]


@pytest.mark.parametrize(
    "payload",
    ["exec('1')", "open('x')", "getattr(prices, 'a')", "prices.__class__", "pd.read_csv('x')", "prices.to_csv('x')"],
)
def test_dangerous_constructs_rejected(payload: str) -> None:
    source = GOOD.replace("delta = prices.diff()", f"{payload}\n        delta = prices.diff()")
    assert not inspect_factor_source(source).ok


def test_missing_compute_and_base_rejected() -> None:
    no_base = GOOD.replace("(AbstractFactor)", "")
    assert not inspect_factor_source(no_base).ok
    assert any("compute" in e for e in inspect_factor_source(GOOD.replace("def compute", "def calc")).errors)


def test_runtime_import_blocked() -> None:
    sneaky = GOOD.replace(
        "delta = prices.diff()", "import importlib\n        delta = prices.diff()"
    )
    assert not inspect_factor_source(sneaky).ok


def test_shape_mismatch_detected() -> None:
    factor = instantiate_factor(compile_factor(BAD_SHAPE))
    with pytest.raises(FactorCompileError, match="shape"):
        validate_factor_output(factor, make_synthetic_prices())


def test_extract_code_handles_fenced_and_raw() -> None:
    assert extract_code("text\n```python\nx = 1\n```\nbye") == "x = 1\n"
    assert extract_code("x = 1") == "x = 1\n"


def test_self_healing_repairs_ast_then_lookahead() -> None:
    client = ScriptedClient([BAD_IMPORT, LOOKAHEAD, GOOD])
    registry = FactorRegistry()
    result = FactorGenerator(client, registry=registry, max_attempts=3).generate("14-day RSI")

    assert result.success
    assert [a.stage for a in result.attempts] == ["ast", "pytest", "ok"]
    assert "look-ahead" in (result.attempts[1].error or "")
    assert "rsi14" in registry
    # the repair prompt carried the previous failure back to the model
    assert "failed the 'ast' check" in client.calls[1][-1]["content"]


def test_gives_up_after_max_attempts() -> None:
    result = FactorGenerator(ScriptedClient([BAD_IMPORT, BAD_IMPORT]), max_attempts=2).generate("x")
    assert not result.success and result.factor is None and len(result.attempts) == 2


def test_generated_factor_runs_through_backtest() -> None:
    factor = instantiate_factor(compile_factor(GOOD))
    prices = make_synthetic_prices()
    result = backtest_factors(
        prices, [MomentumFactor(), factor], {"momentum": 0.5, "rsi14": 0.5},
        top_k=2, weighting_scheme="equal", rebalance_frequency="monthly", transaction_cost_bps=10.0,
    )
    assert len(result.strategy_returns) > 0
