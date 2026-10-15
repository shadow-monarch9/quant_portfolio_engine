"""Ties factor generation to backtesting: one chat prompt -> validated factor -> performance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from src.agent.factor_generator import (
    AnthropicClient,
    FactorGenerator,
    GenerationResult,
    LLMClient,
)
from src.agent.registry import FactorRegistry, backtest_factors
from src.backtest.engine import BacktestResult
from src.backtest.metrics import calculate_performance_metrics
from src.config import AgentConfig
from src.portfolio.optimizer import WeightingScheme
from src.utils.logger import logger


@dataclass(frozen=True)
class BacktestParams:
    top_k: int
    weighting_scheme: WeightingScheme
    rebalance_frequency: str
    transaction_cost_bps: float


@dataclass(frozen=True)
class BacktestReport:
    factor_names: list[str]
    result: BacktestResult
    stats: dict[str, float]


@dataclass(frozen=True)
class AgentTurn:
    """Outcome of one chat prompt. ``report`` is None when generation failed."""

    generation: GenerationResult
    report: BacktestReport | None = None
    error: str | None = None


class AgentOrchestrator:
    """Generates factors from natural language and backtests them via the standard pipeline."""

    def __init__(
        self,
        registry: FactorRegistry | None = None,
        client: LLMClient | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.registry = registry if registry is not None else FactorRegistry()
        self._client = client

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = AnthropicClient(self.config.model, self.config.max_tokens)
        return self._client

    def backtest(
        self, prices: pd.DataFrame, factor_names: Sequence[str], params: BacktestParams
    ) -> BacktestReport:
        """Equal-weight blend of the named registered factors, run through the engine."""
        if not factor_names:
            raise ValueError("select at least one factor to backtest")
        factors = [self.registry.get(name) for name in factor_names]
        weights = {name: 1.0 for name in factor_names}
        result = backtest_factors(
            prices=prices,
            factors=factors,
            weights=weights,
            top_k=min(params.top_k, prices.shape[1]),
            weighting_scheme=params.weighting_scheme,
            rebalance_frequency=params.rebalance_frequency,
            transaction_cost_bps=params.transaction_cost_bps,
        )
        if result.strategy_returns.empty:
            raise ValueError("not enough history to clear the factor burn-in")
        stats = calculate_performance_metrics(result.strategy_returns, result.benchmark_returns)
        return BacktestReport(list(factor_names), result, stats)

    def handle_request(
        self, request: str, prices: pd.DataFrame, params: BacktestParams
    ) -> AgentTurn:
        """Generate a factor for ``request``, register it and backtest it on its own."""
        generator = FactorGenerator(
            client=self.client,
            registry=self.registry,
            max_attempts=self.config.max_attempts,
            sandbox_timeout_s=self.config.sandbox_timeout_s,
        )
        try:
            generation = generator.generate(request)
        except ValueError as exc:  # e.g. the generated factor name is already registered
            logger.warning("Generation rejected: {}", exc)
            empty = GenerationResult(False, request, "", None)
            return AgentTurn(empty, error=f"{exc}. Ask for a differently named factor.")
        if not generation.success or generation.factor is None:
            return AgentTurn(generation, error="Could not produce a factor that passed validation.")
        try:
            report = self.backtest(prices, [generation.factor.name], params)
        except Exception as exc:  # noqa: BLE001  surface any pipeline failure to the chat instead of crashing
            logger.exception("Backtest of generated factor failed")
            return AgentTurn(generation, error=f"Backtest failed: {exc}")
        return AgentTurn(generation, report)
