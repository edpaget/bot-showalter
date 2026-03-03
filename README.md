# Fantasy Baseball Manager

[![CI](https://github.com/edpaget/bot-showalter/actions/workflows/ci.yml/badge.svg)](https://github.com/edpaget/bot-showalter/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/edpaget/c65024c26aba685cdc7f8c65839236cc/raw/coverage-badge.json)](https://github.com/edpaget/bot-showalter/actions/workflows/ci.yml)
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
├── cli/              # Typer CLI — the `fbm` command
├── config.py         # fbm.toml config loading
├── config_league.py  # League settings config
├── config_yahoo.py   # Yahoo Fantasy API config
├── domain/           # Core data classes (Player, BattingStats, Projection, etc.)
├── repos/            # SQLite repository layer
├── db/               # Connection pooling and migrations
├── ingest/           # Data ingestion from FanGraphs, Lahman, Statcast, MLB API, FantasyPros
├── features/         # Feature engineering — assembler, transforms, SQL-backed feature sets
├── models/           # Projection models (see below)
├── services/         # Evaluation, lookup, valuation, and reporting services
├── agent/            # LangGraph-based LLM agent (Claude) for conversational access
├── tools/            # Agent tool definitions (ADP, player, projection, valuation lookups)
├── discord_bot/      # Discord bot for chat-based access to projections and valuations
└── yahoo/            # Yahoo Fantasy API OAuth client and league sync
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
fbm sweep <model>                         # meta-parameter sweep (e.g. weight transforms)
fbm ablate <model>                        # feature ablation study
fbm finetune <model>                      # fine-tune a trained model
fbm gate <model>                          # multi-season regression gate
fbm quick-eval <model>                    # single-target quick evaluation
fbm marginal-value <model>               # RMSE improvement from candidate features
fbm compare-features <model>             # compare two feature sets on identical splits
fbm list                                  # show registered models
fbm info <model>                          # show model details and supported operations
fbm features <model>                      # list declared features for a model
```

Data ingestion:

```bash
fbm ingest players                              # Chadwick register
fbm ingest bio                                  # Lahman biographical enrichment
fbm ingest batting --season 2024                # FanGraphs batting stats
fbm ingest pitching --season 2024               # FanGraphs pitching stats
fbm ingest statcast --season 2024               # pitch-level Statcast data
fbm ingest sprint-speed --season 2024           # Baseball Savant sprint speed
fbm ingest il --season 2024                     # IL stints from MLB API
fbm ingest appearances --season 2024            # position appearances from Lahman
fbm ingest roster --season 2024                 # roster stints from Lahman
fbm ingest milb-batting --season 2024           # minor league stats from MLB API
fbm ingest adp data/adp.csv --season 2026       # ADP from FantasyPros CSV
fbm ingest adp-bulk data/adp/                   # bulk ingest ADP CSVs
fbm ingest adp-fetch --season 2026              # live ADP fetch from FantasyPros
fbm import steamer batting.csv --season 2026 --version pre --player-type batter
```

Compute derived data:

```bash
fbm compute league-env --season 2024            # league environment aggregates
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

Draft board:

```bash
fbm draft board --season 2026                   # terminal draft board
fbm draft export --season 2026 --output board.csv
fbm draft export --season 2026 --output board.html --format html
fbm draft live --season 2026                    # live Flask server with auto-refresh
fbm draft start --season 2026                   # interactive draft session (REPL)
fbm draft report draft.json                     # post-draft analysis
fbm draft tiers --season 2026                   # position-grouped tier assignments
fbm draft tier-summary --season 2026            # cross-position tier summary matrix
fbm draft needs --season 2026                   # category weakness recommendations
fbm draft pick-values --season 2026             # draft pick value curve
fbm draft trade-picks --season 2026             # evaluate draft pick trades
```

Reports:

```bash
fbm report overperformers statcast-gbm/latest --season 2025 --player-type batter
fbm report underperformers statcast-gbm/latest --season 2025 --player-type batter
fbm report talent-delta statcast-gbm/latest --season 2025 --player-type batter --top 20
fbm report talent-quality statcast-gbm/latest --season 2024 2025
fbm report residual-persistence statcast-gbm/latest --season 2024 2025
fbm report residual-analysis statcast-gbm/latest --season 2024 2025
fbm report value-over-adp --season 2026
fbm report adp-accuracy --season 2025 --league default
fbm report adp-movers --season 2026 --window 14
fbm report projection-confidence --season 2026
fbm report variance-targets --season 2026
fbm report system-disagreements "Ohtani" --season 2026
```

Keeper league management:

```bash
fbm keeper import keepers.csv --season 2026     # import keeper costs
fbm keeper set "Soto" 45 --season 2026          # set a single keeper cost
fbm keeper decisions --season 2026              # rank players by surplus value
fbm keeper adjusted-rankings --season 2026      # post-keeper adjusted rankings
fbm keeper trade-eval --season 2026             # evaluate a trade
fbm keeper optimize --season 2026               # find optimal keeper set
fbm keeper scenario --season 2026               # compare keeper scenarios
fbm keeper trade-impact --season 2026           # how trades change optimal keepers
```

Experiment journal:

```bash
fbm experiment log                              # log a trial
fbm experiment search --tag statcast            # search by target, tag, model, feature
fbm experiment summary statcast-gbm batter      # exploration summary
fbm experiment show <id>                        # full experiment details
fbm experiment checkpoint save <name>           # save/restore feature set checkpoints
```

Data profiling:

```bash
fbm profile columns --season 2024               # statcast column distributions
fbm profile correlate --season 2024             # correlate columns against targets
fbm profile stability --season 2022 2023 2024   # temporal stability of correlations
```

Dataset and run management:

```bash
fbm datasets list
fbm datasets rebuild statcast-gbm
fbm runs list --model statcast-gbm
fbm runs show statcast-gbm/v1
fbm runs delete statcast-gbm/v1
```

Conversational access:

```bash
fbm chat                                        # interactive LLM chat session
fbm discord                                     # run the Discord bot
```

Yahoo Fantasy integration:

```bash
fbm yahoo auth                                  # OAuth authentication flow
fbm yahoo sync --league keeper                  # sync league metadata
fbm yahoo map-player <yahoo-key> <player-name>  # manual player mapping
fbm yahoo rosters --league keeper               # display all teams' rosters
fbm yahoo my-roster --league keeper             # your roster with projections
fbm yahoo draft-results --league keeper         # fetch draft results
fbm yahoo draft-live --league keeper            # live draft session with auto-pick
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

[leagues.h2h]
teams = 12
budget = 260
batting_stats = ["R", "HR", "RBI", "SB", "AVG", "OPS"]
pitching_stats = ["W", "SV", "K", "ERA", "WHIP", "QS"]

[yahoo]
client_id = "..."
client_secret = "..."
default_league = "keeper"

[yahoo.leagues.keeper]
league_id = 12345
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
