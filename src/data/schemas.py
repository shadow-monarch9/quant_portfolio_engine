"""Pandera schemas validating market data frames."""

from __future__ import annotations

import pandera.pandas as pa

# Adjusted price matrix: DatetimeIndex rows (dates) x ticker columns, float prices.
PriceMatrixSchema: pa.DataFrameSchema = pa.DataFrameSchema(
    columns={
        r".*": pa.Column(
            float,
            checks=pa.Check.gt(0.0, error="prices must be strictly positive"),
            nullable=False,
            regex=True,
        )
    },
    index=pa.Index(pa.DateTime, unique=True, nullable=False, name="Date"),
    checks=pa.Check(
        lambda df: df.index.is_monotonic_increasing,
        error="index must be sorted ascending",
    ),
    strict=True,
    coerce=False,
    name="PriceMatrixSchema",
)
