"""Typed application configuration loaded from ``config/default_config.yaml``."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "default_config.yaml"


class UniverseConfig(BaseModel):
    tickers: list[str] = Field(min_length=1)
    start_date: date
    end_date: date | None = None
    cache_dir: Path

    @field_validator("tickers")
    @classmethod
    def _normalise_tickers(cls, value: list[str]) -> list[str]:
        cleaned = [ticker.strip().upper() for ticker in value]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("tickers must be unique")
        return cleaned

    @model_validator(mode="after")
    def _check_dates(self) -> UniverseConfig:
        if self.end_date is not None and self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self

    @property
    def cache_path(self) -> Path:
        """Cache directory resolved against the project root if relative."""
        if self.cache_dir.is_absolute():
            return self.cache_dir
        return PROJECT_ROOT / self.cache_dir


class FactorWeights(BaseModel):
    momentum: float = Field(ge=0.0, le=1.0)
    volatility: float = Field(ge=0.0, le=1.0)
    zscore: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sum(self) -> FactorWeights:
        total = self.momentum + self.volatility + self.zscore
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"factor weights must sum to 1.0, got {total:.4f}")
        return self


class RebalanceConfig(BaseModel):
    frequency: Literal["daily", "weekly", "monthly", "quarterly"]
    lookback_days: int = Field(gt=0)
    transaction_cost_bps: float = Field(ge=0.0)


class PortfolioConfig(BaseModel):
    top_k: int = Field(ge=1)
    weighting_scheme: Literal["equal", "inverse_volatility"]


class LoggingConfig(BaseModel):
    level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class AgentConfig(BaseModel):
    model: str = "claude-sonnet-5-5"
    max_tokens: int = Field(default=2000, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    sandbox_timeout_s: int = Field(default=60, gt=0)


class Settings(BaseSettings):
    """Root settings object populated from the YAML config file."""

    model_config = SettingsConfigDict(yaml_file=DEFAULT_CONFIG_PATH, extra="forbid")

    universe: UniverseConfig
    factor_weights: FactorWeights
    rebalance: RebalanceConfig
    portfolio: PortfolioConfig = PortfolioConfig(top_k=3, weighting_scheme="equal")
    logging: LoggingConfig = LoggingConfig()
    agent: AgentConfig = AgentConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, YamlConfigSettingsSource(settings_cls))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, validated application settings."""
    return Settings()  # type: ignore[call-arg]


if __name__ == "__main__":
    from src.utils.logger import logger, setup_logger

    setup_logger()
    logger.info("Loaded settings:\n{}", get_settings().model_dump_json(indent=2))
