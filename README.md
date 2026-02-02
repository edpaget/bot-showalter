# Fantasy Baseball Manager

Player projection, valuation, and evaluation tools for fantasy baseball. Generates MARCEL projections from historical stats, computes z-score valuations, and backtests accuracy against actuals.

## Setup

Requires Python 3.13+. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```
uv sync
```

## CLI Usage

All commands are available via `fantasy-baseball-manager` (or `uv run python -m fantasy_baseball_manager`).

### Projections

```sh
# Generate batting and pitching projections for 2026
fantasy-baseball-manager players project 2026

# Batting only, sorted by OBP
fantasy-baseball-manager players project 2026 --batting --sort-by obp --top 50
```

### Valuations

```sh
# Z-score valuations (default categories: HR, SB, OBP / K, ERA, WHIP)
fantasy-baseball-manager players valuate 2026

# Custom categories
fantasy-baseball-manager players valuate 2026 --categories hr,sb,obp,r,rbi
```

### Evaluation (Backtesting)

```sh
# Evaluate MARCEL accuracy against 2024 actuals
fantasy-baseball-manager evaluate 2024

# Compare multiple engines side-by-side
fantasy-baseball-manager evaluate 2024 --engine marcel --engine enhanced

# Multi-year backtest with JSON output
fantasy-baseball-manager evaluate 2024 --years 2021,2022,2023,2024 --json results.json
```

## Architecture

### Composable Projection Pipeline

Projections flow through a pipeline of independent, swappable stages. Each stage transforms a list of `PlayerRates` (per-PA or per-out rates with metadata):

```
StatsDataSource
    |
    v
RateComputer          -- weighted rates + regression from raw season data
    |
    v
RateAdjuster (chain)  -- rebaselining, aging, park factors, etc.
    |
    v
PlayingTimeProjector  -- sets projected PA or outs
    |
    v
ProjectionFinalizer   -- rates x opportunities -> BattingProjection / PitchingProjection
```

Stages are composed into named pipelines via a registry:

```python
# pipeline/presets.py
PIPELINES = {
    "marcel": marcel_pipeline,     # weights (5,4,3), flat 1200 PA regression, linear aging
    # "enhanced": enhanced_pipeline,  # stat-specific regression, FIP-based ERA, ...
}
```

New projection variants are created by mixing stages. The `evaluate` command runs any registered pipeline through the evaluation harness for A/B comparison.

### MARCEL as Default Configuration

The MARCEL pipeline uses these stages:

| Stage | Implementation | What it does |
|---|---|---|
| `RateComputer` | `MarcelRateComputer` | 5/4/3 year weights (batting), 3/2/1 (pitching), flat regression to league mean |
| `RateAdjuster` | `RebaselineAdjuster` | Scales rates to most recent year's league environment |
| `RateAdjuster` | `MarcelAgingAdjuster` | +0.6%/yr below 29, -0.3%/yr above 29 |
| `PlayingTimeProjector` | `MarcelPlayingTime` | 0.5 * y1 + 0.1 * y2 + base (200 PA / 60 IP starter / 25 IP reliever) |
| `ProjectionFinalizer` | `StandardFinalizer` | Counting stats, derives AB/H/ERA/WHIP/NSVH |

Stage implementations reuse utility functions from `marcel/` (e.g. `weighted_rate`, `rebaseline`, `age_multiplier`).

### Key Modules

```
src/fantasy_baseball_manager/
    pipeline/           # Composable pipeline framework
        types.py        #   PlayerRates intermediate dataclass
        protocols.py    #   Stage protocols (RateComputer, RateAdjuster, ...)
        engine.py       #   ProjectionPipeline orchestrator
        presets.py      #   PIPELINES registry
        source.py       #   PipelineProjectionSource (evaluation adapter)
        stages/         #   Stage implementations
    marcel/             # MARCEL utilities (weights, league averages, aging)
    evaluation/         # Backtesting harness (RMSE, MAE, correlation, Spearman)
    valuation/          # Z-score and SGP player valuations
```

## Development

```sh
uv run pytest              # run tests
uv run ruff check src tests  # lint
uv run black src tests     # format
```
