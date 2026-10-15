"""LLM-driven factor generation with AST checks, sandboxed pytest and a self-healing loop."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

from src.agent.factor_compiler import (
    FactorCompileError,
    compile_factor,
    instantiate_factor,
    make_synthetic_prices,
    validate_factor_output,
)
from src.agent.registry import FactorRegistry
from src.factors.base import AbstractFactor
from src.utils.logger import logger

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT: str = '''You write alpha factors for a quantitative research engine.

Reply with ONE python code block and nothing else. The module must contain exactly one class
that subclasses AbstractFactor, defined in src/factors/base.py as:

```python
class AbstractFactor(ABC):
    name: str
    description: str
    higher_is_better: bool

    def __init__(self, name: str, description: str, higher_is_better: bool) -> None: ...

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame: ...
```

Rules:
- Imports allowed: pandas, numpy, math, and `from src.factors.base import AbstractFactor`. Nothing else.
- The class must be constructible with NO arguments and call
  `super().__init__(name=<snake_case identifier>, description=<str>, higher_is_better=<bool>)`.
- `compute(self, prices: pd.DataFrame) -> pd.DataFrame` receives a (T dates x N tickers) price
  matrix and returns a DataFrame with the identical index and columns.
- No look-ahead: the value on row t may only use rows <= t. Never use shift(-n), center=True
  rolling windows, or full-sample statistics (e.g. prices.mean()).
- Leading NaNs for the warm-up window are fine; entire columns must not be NaN and values
  must not be +/-inf (guard divisions, e.g. `.replace(0, np.nan)`).
- No file/network access, no exec/eval/getattr, no dunder attributes, no module-level code
  other than imports, constants and the class.
'''


class LLMClient(Protocol):
    """Anything that can answer a chat conversation with plain text."""

    def complete(self, system: str, messages: list[dict[str, str]]) -> str: ...


class AnthropicClient:
    """``LLMClient`` backed by the Anthropic Messages API (``pip install anthropic``)."""

    def __init__(self, model: str, max_tokens: int = 2000) -> None:
        import anthropic  # type: ignore[import-not-found]  # imported lazily: only needed when a live model is used

        load_dotenv(PROJECT_ROOT / ".env")
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key or key == "YOUR_KEY_HERE":
            raise RuntimeError("ANTHROPIC_API_KEY is not set; add your key to .env")
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        response = self._client.messages.create(
            model=self._model, max_tokens=self._max_tokens, system=system, messages=messages  # type: ignore[arg-type]
        )
        return "".join(block.text for block in response.content if block.type == "text")


_SANDBOX_TEST: str = '''
import sys
from pathlib import Path

sys.path.insert(0, __ROOT__)

import numpy as np
import pandas as pd
import pytest

from src.agent.factor_compiler import (
    check_no_lookahead, compile_factor, instantiate_factor,
    make_synthetic_prices, validate_factor_output,
)

SOURCE = Path(__file__).with_name("factor.py").read_text(encoding="utf-8")
PRICES = make_synthetic_prices()


def _new():
    return instantiate_factor(compile_factor(SOURCE))


def test_no_lookahead_bias():
    check_no_lookahead(_new, PRICES)


def test_output_is_t_by_n_without_nan_columns_or_inf():
    validate_factor_output(_new(), PRICES)


def test_has_signal_after_warmup():
    values = validate_factor_output(_new(), PRICES)
    assert values.iloc[-1].notna().all(), "last row must be fully populated"
    assert values.iloc[-1].nunique() > 1, "factor is constant across assets"


def test_does_not_mutate_input():
    prices = PRICES.copy()
    _new().compute(prices)
    pd.testing.assert_frame_equal(prices, PRICES)


def test_deterministic():
    pd.testing.assert_frame_equal(_new().compute(PRICES.copy()), _new().compute(PRICES.copy()))
'''.replace("__ROOT__", repr(str(PROJECT_ROOT)))


def run_sandboxed_tests(source: str, timeout_s: int = 60) -> tuple[bool, str]:
    """Run the validation suite against ``source`` in an isolated pytest subprocess.

    This gives process isolation, a timeout and a scratch directory; it is not an OS-level
    security boundary, so the AST inspection remains the primary defence.
    """
    with tempfile.TemporaryDirectory(prefix="factor_sandbox_") as tmp:
        directory = Path(tmp)
        (directory / "factor.py").write_text(source, encoding="utf-8")
        (directory / "test_factor.py").write_text(_SANDBOX_TEST, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "test_factor.py"],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, f"sandbox timed out after {timeout_s}s"
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output[-4000:]


def extract_code(reply: str) -> str:
    """Pull the python source out of a fenced block, or fall back to the raw reply."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", reply, flags=re.DOTALL)
    return (match.group(1) if match else reply).strip() + "\n"


@dataclass(frozen=True)
class GenerationAttempt:
    attempt: int
    source: str
    stage: str  # "ast" | "load" | "shape" | "pytest" | "ok"
    error: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    success: bool
    request: str
    source: str
    factor: AbstractFactor | None
    attempts: list[GenerationAttempt] = field(default_factory=list)


class FactorGenerator:
    """Turns a natural-language request into a validated, registered ``AbstractFactor``."""

    def __init__(
        self,
        client: LLMClient,
        registry: FactorRegistry | None = None,
        max_attempts: int = 3,
        sandbox_timeout_s: int = 60,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.client = client
        self.registry = registry
        self.max_attempts = max_attempts
        self.sandbox_timeout_s = sandbox_timeout_s

    def _validate(self, source: str) -> tuple[str, str | None, AbstractFactor | None]:
        """Return ``(stage, error, factor)``; error is None when every check passed."""
        try:
            cls = compile_factor(source)
        except FactorCompileError as exc:
            stage = "load" if exc.errors and exc.errors[0].startswith("loading failed") else "ast"
            return stage, "\n".join(exc.errors), None
        try:
            factor = instantiate_factor(cls)
            validate_factor_output(factor, make_synthetic_prices())
        except FactorCompileError as exc:
            return "shape", "\n".join(exc.errors), None

        passed, report = run_sandboxed_tests(source, self.sandbox_timeout_s)
        if not passed:
            return "pytest", report, None
        return "ok", None, factor

    def generate(self, request: str) -> GenerationResult:
        """Generate, validate and (on success) register a factor, repairing on failure."""
        messages: list[dict[str, str]] = [{"role": "user", "content": f"Build this factor: {request}"}]
        attempts: list[GenerationAttempt] = []
        source = ""

        for number in range(1, self.max_attempts + 1):
            source = extract_code(self.client.complete(SYSTEM_PROMPT, messages))
            stage, error, factor = self._validate(source)
            attempts.append(GenerationAttempt(number, source, stage, error))

            if factor is not None:
                logger.success("Factor '{}' passed all checks on attempt {}", factor.name, number)
                if self.registry is not None:
                    self.registry.register(factor)
                return GenerationResult(True, request, source, factor, attempts)

            logger.warning("Attempt {}/{} failed at stage '{}': {}", number, self.max_attempts, stage, error)
            messages += [
                {"role": "assistant", "content": f"```python\n{source}```"},
                {
                    "role": "user",
                    "content": (
                        f"That code failed the '{stage}' check:\n{error}\n\n"
                        "Fix it and reply with the complete corrected module in one python block."
                    ),
                },
            ]

        logger.error("Giving up on request {!r} after {} attempts", request, self.max_attempts)
        return GenerationResult(False, request, source, None, attempts)
