# Fantasy Baseball Manager

Player projection, valuation, keeper analysis, and draft tools for fantasy baseball. Generates projections from historical stats using configurable pipelines, computes z-score valuations, ranks keeper candidates by surplus value, and backtests accuracy against actuals.

## Setup

Requires Python 3.13+. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```
uv sync
```

## CLI Usage

All commands are available via `fantasy-baseball-manager` (or `uv run fantasy-baseball-manager`).

### Projections

```sh
# Generate batting and pitching projections for 2026
fantasy-baseball-manager players project 2026

# Batting only, sorted by OBP
fantasy-baseball-manager players project 2026 --batting --sort-by obp --top 50

# Use a specific projection engine
fantasy-baseball-manager players project 2026 --engine marcel_gb
```

### Valuations

```sh
# Z-score valuations (default categories from config)
fantasy-baseball-manager players valuate 2026

# Custom categories
fantasy-baseball-manager players valuate 2026 --categories hr,sb,obp,r,rbi
```

### Keeper Analysis

```sh
# Rank keeper candidates by surplus value
fantasy-baseball-manager keeper rank 2026 --candidates "id1,id2,id3"

# Find optimal keeper combination
fantasy-baseball-manager keeper optimize 2026 --yahoo

# Analyze all teams in a league
fantasy-baseball-manager keeper league 2026
```

### Evaluation (Backtesting)

```sh
# Evaluate projection accuracy against 2024 actuals
fantasy-baseball-manager evaluate 2024

# Compare multiple engines side-by-side
fantasy-baseball-manager evaluate 2024 --engine marcel --engine marcel_gb

# Multi-year backtest with JSON output
fantasy-baseball-manager evaluate 2024 --years 2021,2022,2023,2024 --json results.json
```

### Chat Interface

```sh
# Interactive chat with the fantasy baseball assistant
fantasy-baseball-manager chat
```

## Architecture

For a comprehensive walkthrough of the codebase — module map, data flow diagrams, modeling approaches, and architecture improvement opportunities — see **[IMPLEMENTATION.md](./IMPLEMENTATION.md)**.

### Composable Projection Pipeline

Projections flow through a pipeline of independent, swappable stages. Each stage transforms a list of `PlayerRates` (per-PA or per-out rates with metadata):

```
StatsDataSource
    |
    v
RateComputer          -- weighted rates + regression from raw season data
    |
    v
RateAdjuster (chain)  -- rebaselining, aging, park factors, Statcast, etc.
    |
    v
PlayingTimeProjector  -- sets projected PA or IP
    |
    v
ProjectionFinalizer   -- rates x opportunities -> BattingProjection / PitchingProjection
```

Stages are composed into named pipelines via a registry:

```python
# pipeline/presets.py
PIPELINES = {
    "marcel_classic": ...,  # Original Marcel with uniform aging
    "marcel": ...,          # Per-stat regression + component aging
    "marcel_full": ...,     # + park factors, Statcast, BABIP adjustments
    "marcel_gb": ...,       # + gradient boosting residual corrections
}
```

New projection variants are created by mixing stages. The `evaluate` command runs any registered pipeline through the evaluation harness for A/B comparison.

### Key Modules

```
src/fantasy_baseball_manager/
    pipeline/           # Composable pipeline framework
        types.py        #   PlayerRates intermediate dataclass
        protocols.py    #   Stage protocols (RateComputer, RateAdjuster, ...)
        engine.py       #   ProjectionPipeline orchestrator
        presets.py      #   PIPELINES registry
        stages/         #   Stage implementations
    marcel/             # MARCEL utilities (weights, league averages, aging)
    valuation/          # Z-score and SGP player valuations
    keeper/             # Keeper surplus value and optimization
    draft/              # Draft state, simulation, and rankings
    ros/                # Rest-of-season projection blending
    evaluation/         # Backtesting harness (RMSE, MAE, correlation, Spearman)
    agent/              # LLM agent tools and chat interface
    ml/                 # Machine learning models (gradient boosting)
    services/           # Dependency injection container
```

## Development

```sh
uv run pytest                # run tests
uv run ruff check src tests  # lint
uv run black src tests       # format
uv run ty check src tests    # type check
```
