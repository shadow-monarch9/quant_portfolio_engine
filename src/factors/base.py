"""Abstract interface shared by all alpha factors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class AbstractFactor(ABC):
    """A factor maps a price matrix (dates x tickers) to a same-shaped factor matrix."""

    name: str
    description: str
    higher_is_better: bool

    def __init__(self, name: str, description: str, higher_is_better: bool) -> None:
        self.name = name
        self.description = description
        self.higher_is_better = higher_is_better

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return raw factor values with the same index and columns as ``prices``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, higher_is_better={self.higher_is_better})"
