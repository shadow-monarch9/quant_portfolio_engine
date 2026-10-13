# Quant Portfolio Engine Guidelines

## Tech Stack
- Python 3.11+
- Configuration: pydantic-settings, pyyaml
- Data Validation & Storage: pandera, pyarrow (Parquet), yfinance
- Logging & Observability: loguru
- Quality Tools: ruff, mypy, pytest

## Engineering Principles
1. Strict Type Hinting: All function signatures must include type annotations.
2. Production Logging: Use `loguru` instead of `print()` statements.
3. Schema Validation: Validate incoming DataFrames against Pandera schemas in `src/data/schemas.py`.
4. Configuration Decoupling: Load parameters dynamically from `config/default_config.yaml` via `src/config.py`.
5. Disk Caching: Store downloaded price matrices as Parquet files in `data/cache/`.
