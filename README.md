# Fantasy Baseball Manager

[![CI](https://github.com/edpaget/bot-showalter/actions/workflows/ci.yml/badge.svg)](https://github.com/edpaget/bot-showalter/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/edpaget/COVERAGE_GIST_ID/raw/coverage-badge.json)](https://github.com/edpaget/bot-showalter/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A projection-driven fantasy baseball toolkit. Statistical models forecast player performance to support drafts, trades, and lineup decisions.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Getting Started

```bash
uv sync          # install dependencies
fbm list         # show registered models
fbm info marcel  # show model details and supported operations
```

## Project Structure

```
src/fantasy_baseball_manager/
├── cli/          # Typer CLI — the `fbm` command
├── config.py     # fbm.toml config loading
├── domain/       # Core data classes (Player, BattingStats, Projection, etc.)
├── repos/        # SQLite repository layer
├── db/           # Connection pooling and migrations
├── ingest/       # Data ingestion from FanGraphs, Baseball Reference, Statcast, MLB API
├── features/     # Feature engineering — assembler, transforms, SQL-backed feature sets
├── models/       # Projection models (see below)
└── services/     # Evaluation, lookup, valuation, and reporting services
```

## Projection Models

| Model | Description |
|---|---|
| `marcel` | Weighted averages, regression to the mean, and aging curves |
| `statcast-gbm` | Gradient-boosted model using Statcast features |
| `statcast-gbm-preseason` | Preseason variant using lagged Statcast features |
| `playing_time` | PA/IP projections via OLS regression |
| `mle` | Minor league equivalency projections |
| `composite` | Rate projections using an external playing-time model |
| `ensemble` | Weighted-average ensemble of multiple systems |
| `zar` | Z-Score Above Replacement valuation for H2H categories leagues |

## CLI Usage

The `fbm` command is the main entry point. Core model operations:

```bash
fbm prepare <model> --season 2024 2025    # materialize feature sets
fbm train <model> --season 2024 2025      # train a model
fbm predict <model> --season 2026         # generate projections
fbm evaluate <model> --season 2025        # evaluate against actuals
fbm tune <model>                          # hyperparameter search
fbm ablate <model>                        # feature ablation study
```

Data ingestion:

```bash
fbm ingest players                              # Chadwick register
fbm ingest batting --season 2024 --source fangraphs
fbm ingest pitching --season 2024
fbm ingest statcast --season 2024               # pitch-level Statcast data
fbm ingest il --season 2024                     # IL stints from MLB API
fbm ingest milb-batting --season 2024           # minor league stats
fbm import steamer batting.csv --season 2026 --version pre --player-type batter
```

Comparing and exploring projections:

```bash
fbm compare marcel/latest steamer/pre --season 2025 --stat wOBA
fbm compare marcel/latest statcast-gbm/latest --season 2025 --stratify age
fbm projections lookup "Ohtani" --season 2026
fbm projections systems --season 2026
```

Valuations:

```bash
fbm valuations lookup "Soto" --season 2026
fbm valuations rankings --season 2026 --top 50
fbm valuations evaluate --season 2025 --league default
```

Reports:

```bash
fbm report overperformers statcast-gbm/latest --season 2025 --player-type batter
fbm report talent-delta statcast-gbm/latest --season 2025 --player-type batter --top 20
```

Dataset and run management:

```bash
fbm datasets list
fbm datasets rebuild statcast-gbm
fbm runs list --model statcast-gbm
fbm runs show statcast-gbm/v1
```

## Configuration

Optional `fbm.toml` in the project root sets defaults:

```toml
[common]
data_dir = "./data"
artifacts_dir = "./artifacts"
seasons = [2021, 2022, 2023, 2024, 2025]

[models.statcast-gbm]
version = "v1"

[models.statcast-gbm.params]
n_estimators = 200
```

CLI flags override TOML values.

## Development

```bash
uv run pytest                # run tests (parallel, randomized)
uv run ruff check src tests  # lint
uv run ruff format src tests # format
uv run ty check src tests    # type check
```

Pre-commit hooks run the full quality gate (format, lint, type check, tests) on every commit.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
