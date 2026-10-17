# Quant Portfolio Engine ("QuantAgent")

A multi-factor, long-only equity portfolio engine with a Streamlit dashboard, a market-regime
detector and an **AI factor agent** that turns plain-English ideas ("14-day RSI mean reversion")
into validated, backtested alpha factors.

Everything is vectorised pandas, fully typed, config-driven and covered by tests. Look-ahead bias
is treated as a first-class bug: it is guarded in the engine, tested explicitly, and checked on
every AI-generated factor before it is allowed into the system.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Project layout](#project-layout)
4. [Walkthrough](#walkthrough)
   - [1. Configuration](#1-configuration)
   - [2. Data layer](#2-data-layer)
   - [3. Factors](#3-factors)
   - [4. Ranking and portfolio construction](#4-ranking-and-portfolio-construction)
   - [5. Backtest engine and metrics](#5-backtest-engine-and-metrics)
   - [6. Market regime detector](#6-market-regime-detector)
   - [7. AI factor agent](#7-ai-factor-agent)
   - [8. Streamlit dashboard](#8-streamlit-dashboard)
5. [Getting started](#getting-started)
6. [Testing and code quality](#testing-and-code-quality)
7. [Deployment](#deployment)
8. [Design decisions and known limitations](#design-decisions-and-known-limitations)
9. [Development history](#development-history)

---

## What it does

1. **Downloads** adjusted close prices from Yahoo Finance, validates them with Pandera and caches
   them as Parquet.
2. **Scores** every asset each day with three built-in factors (momentum, volatility, z-score),
   converts each to a cross-sectional percentile rank and blends them into one alpha score.
3. **Builds a portfolio** by holding the top-K assets (equal or inverse-volatility weighted).
4. **Backtests** it with a one-day execution lag, periodic rebalancing and turnover-based
   transaction costs, and reports Sharpe, Sortino, CAGR, max drawdown and win rate against an
   equal-weight benchmark.
5. **Detects the market regime** (bull / high-volatility / neutral) and recommends a factor preset
   you can apply with one click.
6. **Generates new factors with an LLM**, then validates them (AST security checks, look-ahead
   test, sandboxed pytest, self-healing retries) before backtesting them through the same pipeline.

## Architecture

```
                 config/default_config.yaml
                            |
                      src/config.py  (pydantic-settings)
                            |
 Yahoo Finance --> YahooDataLoader --> Pandera schema --> Parquet cache (data/cache/)
                            |
                     price matrix (dates x tickers)
                            |
        +-------------------+--------------------+
        |                                        |
  Built-in factors                      AI Factor Agent
  momentum / volatility / zscore        prompt -> LLM -> AST check -> load -> shape check
        |                               -> sandboxed pytest -> FactorRegistry
        +-------------------+--------------------+
                            |
              CrossSectionalRanker  (percentile ranks -> composite alpha in [0, 1])
                            |
                  TopKOptimizer  (equal / inverse-volatility weights)
                            |
                  BacktestEngine (1-day lag, rebalancing, costs)
                            |
                  performance metrics  -->  Streamlit app (app.py)
                                                  ^
                                   RegimeDetector --+  (banner + preset)
```

## Project layout

```
quant_portfolio_engine/
├── app.py                         # Streamlit dashboard + AI agent chat tab
├── config/default_config.yaml     # all tunable parameters
├── Procfile                       # Railway start command
├── pytest.ini
├── requirements.txt
├── src/
│   ├── config.py                  # typed Settings (pydantic-settings + YAML)
│   ├── data/
│   │   ├── schemas.py             # Pandera PriceMatrixSchema
│   │   └── yahoo_loader.py        # fetch, clean, align, cache, validate
│   ├── factors/
│   │   ├── base.py                # AbstractFactor interface
│   │   ├── momentum.py            # 12-1 month momentum
│   │   ├── volatility.py          # 30d annualised realised volatility
│   │   └── zscore.py              # 20d price z-score (mean reversion)
│   ├── portfolio/
│   │   ├── ranker.py              # CrossSectionalRanker (composite alpha)
│   │   └── optimizer.py           # TopKOptimizer
│   ├── backtest/
│   │   ├── engine.py              # BacktestEngine + run_strategy()
│   │   └── metrics.py             # Sharpe, Sortino, CAGR, drawdown, win rate
│   ├── agent/
│   │   ├── factor_generator.py    # LLM client, prompt, self-healing loop, sandbox
│   │   ├── factor_compiler.py     # AST inspector, restricted loader, output checks
│   │   ├── registry.py            # FactorRegistry + backtest_factors()
│   │   ├── orchestrator.py        # prompt -> factor -> backtest report
│   │   └── regime.py              # RegimeDetector
│   └── utils/logger.py            # loguru setup
└── tests/                         # unit, look-ahead, agent, regime and UI tests
```

## Walkthrough

### 1. Configuration

`src/config.py` loads `config/default_config.yaml` into a typed `Settings` object (cached via
`get_settings()`). Nothing is hard-coded in the pipeline; the defaults are:

| Section          | Defaults                                                                          |
|------------------|-----------------------------------------------------------------------------------|
| `universe`       | AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA from 2020-01-01; cache in `data/cache`  |
| `factor_weights` | momentum 0.50, volatility 0.30, zscore 0.20 (must sum to 1.0)                     |
| `rebalance`      | monthly, 252-day lookback, 10 bps transaction cost                                |
| `portfolio`      | top-K = 3, `equal` weighting (or `inverse_volatility`)                            |
| `logging`        | INFO                                                                              |
| `agent`          | model `claude-sonnet-5-5`, 2000 max tokens, 3 attempts, 60 s sandbox timeout      |

### 2. Data layer

`YahooDataLoader` (`src/data/yahoo_loader.py`) produces a clean **dates x tickers** matrix of
adjusted closes:

- Fetches with `yfinance` (up to 3 attempts; an empty or incomplete matrix is never cached or
  returned). Downloads run with `threads=False` to avoid yfinance's SQLite *"database is locked"*
  error.
- Cleans and aligns the data, then validates it against `PriceMatrixSchema`
  (`src/data/schemas.py`): float columns, strictly positive prices, unique and sorted
  `DatetimeIndex`, no nulls, strict columns.
- Caches to `data/cache/prices_<hash>.parquet`. The hash is derived from tickers and dates, so
  changing the universe invalidates the cache automatically. A corrupt cache falls back to a live
  fetch; `load(force_refresh=True)` forces one.

### 3. Factors

All factors implement `AbstractFactor` (`name`, `description`, `higher_is_better`,
`compute(prices) -> DataFrame` of the same shape):

| Factor       | Definition                                              | Better when |
|--------------|---------------------------------------------------------|-------------|
| `momentum`   | 252-day return minus 21-day return (12-1 momentum)     | higher      |
| `volatility` | 30-day rolling stdev of daily returns, x sqrt(252)     | lower       |
| `zscore`     | (price - 20d mean) / 20d stdev (mean-reversion signal) | lower       |

### 4. Ranking and portfolio construction

**`CrossSectionalRanker`** (`src/portfolio/ranker.py`):

1. Each factor is converted to a daily cross-sectional percentile rank, inverted when
   `higher_is_better` is false.
2. Ranks are blended by factor weight, then min-max normalised per date to an alpha score in
   `[0, 1]` (all-tied rows get a neutral 0.5).
3. **Strict burn-in:** a score exists only where *every* factor is available, so the slowest
   factor (252-day momentum) defines the first scored day.

**`TopKOptimizer`** (`src/portfolio/optimizer.py`) holds the `k` best-scoring assets, long-only
and fully invested. `equal` gives 1/k each; `inverse_volatility` weights by 1/volatility and falls
back to equal weights on rows where volatility is unavailable. Ties are broken deterministically.

### 5. Backtest engine and metrics

`BacktestEngine` (`src/backtest/engine.py`) is fully vectorised:

- **Rebalancing:** `daily`, `weekly`, `monthly` or `quarterly`. Target weights are sampled on the
  last trading day of each period and held until the next one.
- **No look-ahead:** weights decided at close `T` are lagged one day (`shift(1)`), so they only
  earn returns from `T+1`.
- **Costs:** `cost_rate x sum(|delta weight|)` on days the position changes.
- **Benchmark:** equal-weight buy-and-hold of the whole universe.
- Returns a `BacktestResult` with returns, equity curves, positions, turnover and costs, trimmed
  to the first invested day.

`run_strategy()` wires the whole pipeline (factors, ranker, optimizer, engine). Running
`python -m src.backtest.engine` prints a backtest summary from the config.

`calculate_performance_metrics` (`src/backtest/metrics.py`) returns CAGR, Sharpe, Sortino, max
drawdown and win rate, plus benchmark CAGR / Sharpe / drawdown and excess CAGR. It assumes a 0%
risk-free rate and 252 trading days.

### 6. Market regime detector

`RegimeDetector.analyze()` (`src/agent/regime.py`) classifies an equal-weight benchmark index
using its 200-day SMA and 30-day annualised volatility (threshold 22%):

| Regime                               | Condition                              | Preset (mom / vol / z / top-K) |
|--------------------------------------|----------------------------------------|--------------------------------|
| High Volatility / Bear Risk          | vol > 22%                              | 0.20 / 0.60 / 0.20 / 4         |
| Strong Bull Market (Low Volatility)  | price > SMA200 and vol <= 22%          | 0.80 / 0.00 / 0.20 / 2         |
| Neutral / Sideways Market            | price <= SMA200 and vol <= 22%         | 0.40 / 0.40 / 0.20 / 3         |
| Insufficient History                 | fewer than 200 observations            | none (keep current weights)    |

High volatility takes priority over the trend check. Boundaries are strict: price equal to the SMA
is *not* bull, and volatility exactly at the threshold is *not* high.

### 7. AI factor agent

The agent (`src/agent/`) lets you describe a factor in natural language. The LLM's code is treated
as untrusted and has to pass several gates before it is registered:

1. **Generation** (`factor_generator.py`): a system prompt specifies the `AbstractFactor`
   contract (allowed imports, no look-ahead, no I/O, no inf or all-NaN columns). The default
   backend is the Anthropic Messages API behind a small `LLMClient` protocol, so tests use a fake
   client.
2. **AST inspection** (`factor_compiler.py`): rejects imports outside `pandas`, `numpy`, `math` and
   `src.factors.base`; calls such as `exec`, `eval`, `open`, `getattr`, `__import__`; I/O methods
   like `to_csv` or `load`; dunder access; `global`; and disallowed top-level statements. It also
   checks the class subclasses `AbstractFactor` and has the right `compute` signature.
3. **Restricted loading:** the code is executed with a whitelist of builtins and a restricted
   import hook. The class must be constructible with no arguments.
4. **Output validation:** the factor runs on synthetic prices and must return the same index and
   columns, no all-NaN columns and no infinities.
5. **Sandboxed pytest:** a subprocess (`python -I -m pytest`, temp directory, timeout) runs five
   checks: no look-ahead bias, valid output, signal in the last row, input not mutated, and
   determinism.
6. **Self-healing:** if any stage fails, the error is fed back to the model for a corrected
   module, up to `agent.max_attempts` times.

Validated factors go into the `FactorRegistry`. The `AgentOrchestrator`
(`orchestrator.py`) then backtests the new factor on its own, or an equal-weight blend of several
registered factors, through the same ranker, optimizer and engine as the built-ins.

### 8. Streamlit dashboard

`app.py` is the user interface:

- **Sidebar:** factor weight sliders (normalised automatically), top-K, weighting scheme,
  rebalance frequency and transaction cost.
- **Regime banner:** shows the detected regime, benchmark level, 200-day SMA and 30-day
  volatility, with an **Apply Recommended Factor Preset** button. Once the sliders match the
  preset, the banner switches to an "aligned" state.
- **Quant Engine Dashboard tab:** Sharpe, CAGR, max drawdown and win rate; an equity-curve chart
  against the benchmark; and a table of the latest alpha scores and target weights.
- **AI Factor Studio & Agent Chat tab:** describe a factor in chat, view the generated code and
  each validation attempt, see the backtest result, and blend registered factors.

## Getting started

**Requirements:** Python 3.11+ (the pinned `requirements.txt` was generated on a recent
environment, so you may need a current Python and pip).

```bash
# 1. create an environment and install dependencies
python -m venv .venv
.venv\Scripts\activate            # Windows (use source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. (AI agent only) install the Anthropic SDK and add your key
pip install anthropic
echo ANTHROPIC_API_KEY=your_key_here > .env

# 3. launch the dashboard
streamlit run app.py
```

`anthropic` is imported lazily and is not pinned in `requirements.txt`, so the dashboard works
without it. Only the agent tab needs it, and without it that tab shows "Agent unavailable".

Command-line entry points:

```bash
python -m src.data.yahoo_loader      # live fetch + schema validation + cache write
python -m src.portfolio.ranker       # print today's top-3 tickers
python -m src.backtest.engine        # run the configured backtest and log metrics
```

To change the universe, dates, weights, costs or agent settings, edit
`config/default_config.yaml`.

## Testing and code quality

```bash
pytest          # tests live in tests/, pythonpath is set in pytest.ini
ruff check .
mypy src
```

| Test module                       | Covers                                                                    |
|-----------------------------------|---------------------------------------------------------------------------|
| `tests/unit/test_factors.py`      | factor formulas, warm-up NaNs, ranking bounds and inversion, burn-in      |
| `tests/unit/test_backtest.py`     | top-K weights, execution lag, cost charging, metric values                |
| `tests/test_lookahead_bias.py`    | weights and returns at `T` are unchanged when later prices are altered    |
| `tests/agent/test_agent.py`       | AST rejections, restricted imports, shape checks, self-healing, end-to-end |
| `tests/test_regime_detector.py`   | regimes, thresholds, 199/200 boundary, NaN and empty input                |
| `tests/test_regime_banner.py`     | Streamlit banner, Apply button and aligned state (via Streamlit testing)  |

Engineering conventions (from `CLAUDE.md`): strict type hints on every signature, `loguru` instead
of `print`, Pandera validation of incoming DataFrames, parameters loaded from config, and Parquet
caching of downloaded prices.

## Deployment

The app is configured for [Railway](https://railway.app). The `Procfile` runs:

```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Set `ANTHROPIC_API_KEY` as an environment variable on the host to enable the agent tab.

## Design decisions and known limitations

- **Look-ahead safety by construction:** the one-day weight lag, strict burn-in and dedicated
  tests exist to keep backtests honest.
- **Rank-based scoring:** percentile ranks make factors with different scales comparable and
  robust to outliers.
- **Sandbox scope:** the pytest subprocess provides process isolation and a timeout, but it is
  *not* an OS-level security boundary. The AST inspection is the primary defence, so run the agent
  tab only on infrastructure you control.
- **Backtest simplifications:** long-only, 0% risk-free rate, no intra-period weight drift, no
  slippage or borrowing model, and a small 7-stock default universe with survivorship bias.
  Treat results as research output, not investment advice.
- **Registry is in-memory** and per browser session, so generated factors are lost on restart.
- **Regime presets are heuristic**, based on a simple SMA and volatility threshold, not a fitted
  model.
- **Top-K slider** in the UI is capped at 5.

## Development history

| Commit    | Milestone                                                              |
|-----------|------------------------------------------------------------------------|
| `7811114` | Initial commit: data layer, factors, ranker, optimizer, backtest, app  |
| `118ac8a` | AI factor agent, orchestrator and Streamlit agent chat tab             |
| `cdbaf6d` | Railway deployment configuration                                       |
| `06bbba5` | Fix yfinance "database is locked" failure; add Procfile start command  |
| `96c515c` | Market regime detector and preset banner                               |
| `a117a1b` | Regime boundary and Apply-preset tests; fix crash on short history     |
| `6262ae3` | Show "aligned" state in the regime banner once a preset is applied     |
