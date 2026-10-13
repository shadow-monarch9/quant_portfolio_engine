"""Yahoo Finance loader: fetch, clean, align, cache and validate adjusted prices."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

from src.config import Settings, get_settings
from src.data.schemas import PriceMatrixSchema
from src.utils.logger import logger, setup_logger


class YahooDataLoader:
    """Loads an adjusted close price matrix (dates x tickers) from Yahoo Finance."""

    def __init__(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date | None,
        cache_dir: Path,
    ) -> None:
        self.tickers: list[str] = tickers
        self.start_date: date = start_date
        self.end_date: date | None = end_date
        self.cache_dir: Path = cache_dir

    @classmethod
    def from_settings(cls, settings: Settings) -> YahooDataLoader:
        universe = settings.universe
        return cls(
            tickers=universe.tickers,
            start_date=universe.start_date,
            end_date=universe.end_date,
            cache_dir=universe.cache_path,
        )

    @property
    def cache_file(self) -> Path:
        key = f"{sorted(self.tickers)}|{self.start_date}|{self.end_date}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / f"prices_{digest}.parquet"

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """Return the validated price matrix, using the Parquet cache when present."""
        path = self.cache_file
        if path.exists() and not force_refresh:
            logger.info("Loading cached prices from {}", path)
            prices = pd.read_parquet(path)
        else:
            ohlcv = self._fetch()
            prices = self._clean(ohlcv)
            self._write_cache(prices, path)
        validated: pd.DataFrame = PriceMatrixSchema.validate(prices)
        logger.success(
            "Price matrix validated: {} rows x {} tickers ({} -> {})",
            validated.shape[0],
            validated.shape[1],
            validated.index.min().date(),
            validated.index.max().date(),
        )
        return validated

    def _fetch(self) -> pd.DataFrame:
        logger.info(
            "Fetching OHLCV for {} tickers from {} to {}",
            len(self.tickers),
            self.start_date,
            self.end_date or "latest",
        )
        raw = yf.download(
            tickers=self.tickers,
            start=self.start_date.isoformat(),
            end=self.end_date.isoformat() if self.end_date else None,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw is None or raw.empty:
            raise RuntimeError("yfinance returned no data for the requested universe")
        logger.debug("Raw OHLCV shape: {}", raw.shape)
        return raw

    def _clean(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Extract adjusted closes, forward-fill gaps and align dates across assets."""
        close = ohlcv["Close"]
        prices: pd.DataFrame = (
            close.to_frame(name=self.tickers[0]) if isinstance(close, pd.Series) else close.copy()
        )

        missing = [t for t in self.tickers if t not in prices.columns]
        if missing:
            raise RuntimeError(f"No data returned for tickers: {missing}")

        prices = prices[self.tickers]
        index = pd.DatetimeIndex(prices.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        index.name = "Date"
        prices.index = index
        prices = prices.sort_index().astype(float)

        prices = prices.dropna(how="all").ffill()

        # Align: drop leading dates until every asset has a price (common history).
        before = len(prices)
        prices = prices.dropna(how="any")
        if len(prices) < before:
            logger.warning(
                "Dropped {} leading rows to align asset histories", before - len(prices)
            )
        return prices

    @staticmethod
    def _write_cache(prices: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(path, compression="zstd")
        logger.info("Cached prices to {} ({:.1f} KB)", path, path.stat().st_size / 1024)


if __name__ == "__main__":
    settings = get_settings()
    setup_logger(settings.logging.level)

    loader = YahooDataLoader.from_settings(settings)
    df = loader.load(force_refresh=True)  # live fetch + validate + cache write
    logger.info("Head:\n{}", df.head())
    logger.info("Tail:\n{}", df.tail())

    cached = pd.read_parquet(loader.cache_file)
    PriceMatrixSchema.validate(cached)
    logger.success(
        "Parquet round-trip OK: {} ({} rows x {} cols)",
        loader.cache_file,
        *cached.shape,
    )
