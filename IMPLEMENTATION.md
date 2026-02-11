# Implementation Guide

High-level architecture of the Fantasy Baseball Manager: how each subsystem works, how they connect, and how data flows from raw stats to draft-day decisions.

## System Overview

```
                          CLI (Typer)
                              |
            +---------+-------+-------+---------+
            |         |       |       |         |
        players    keeper   draft  evaluate   chat
            |         |       |       |         |
            v         v       v       v         v
     +------------+  +-----------+  +----------+  +-------+
     | Projection |  | Valuation |  | Draft    |  | Agent |
     | Pipeline   |->| (Z/SGP)   |->| Sim &    |  | (LLM) |
     +------------+  +-----------+  | Rankings |  +-------+
            |                       +----------+
            |
    +-------+--------+--------+--------+
    |       |        |        |        |
  Marcel  Statcast  ML/MTL  MLE     Contextual
  Engine  Blending  Models  (Minors) Transformer
```

The system is organized around a **composable projection pipeline** that feeds into downstream consumers (valuation, keeper analysis, draft tools). Each box above is an independent module with protocol-defined boundaries.

## Core Pipeline Architecture

The projection pipeline is the heart of the system. It follows a linear stage-based architecture where each stage transforms a list of `PlayerRates` — a dataclass carrying per-PA (or per-out) rates, projected opportunities, and incrementally-populated metadata.

```
  DataSource[BattingSeasonStats]        DataSource[PitchingSeasonStats]
            \                                      /
             \                                    /
              v                                  v
        +------------------------------------------+
        |            RateComputer                  |
        |  (3-year weighted rates + regression)    |
        +------------------------------------------+
                          |
                   list[PlayerRates]
                          |
        +------------------------------------------+
        |         RateAdjuster chain               |
        |                                          |
        |  1. PlayerIdentityEnricher               |
        |  2. ParkFactorAdjuster                   |
        |  3. PitcherNormalizationAdjuster         |
        |  4. PitcherStatcastAdjuster              |
        |  5. StatcastRateAdjuster                 |
        |  6. BatterBabipAdjuster                  |
        |  7. GBResidualAdjuster (ML)              |
        |  8. MTLBlender / ContextualBlender       |
        |  8b. EnsembleAdjuster (multi-model)      |
        |  9. RebaselineAdjuster                   |
        | 10. ComponentAgingAdjuster               |
        +------------------------------------------+
                          |
                   list[PlayerRates]
                          |
        +------------------------------------------+
        |        PlayingTimeProjector              |
        |  (sets projected PA or IP)               |
        +------------------------------------------+
                          |
                   list[PlayerRates]
                          |
        +------------------------------------------+
        |        ProjectionFinalizer               |
        |  (rates x opportunities -> projections)  |
        +------------------------------------------+
                          |
              list[BattingProjection]
              list[PitchingProjection]
```

### Stage Protocols

Each stage is defined as a Python `Protocol`, not a base class. This enables structural subtyping — any object with the right method signature satisfies the contract:

| Protocol | Method | Role |
|----------|--------|------|
| `RateComputer` | `compute_batting_rates()` / `compute_pitching_rates()` | Convert raw season stats into per-PA rates with regression |
| `RateAdjuster` | `adjust(list[PlayerRates]) -> list[PlayerRates]` | Transform rates in-place (park, Statcast, aging, ML corrections) |
| `PlayingTimeProjector` | `project(list[PlayerRates]) -> list[PlayerRates]` | Set the `opportunities` field (projected PA or IP) |
| `ProjectionFinalizer` | `finalize_batting()` / `finalize_pitching()` | Multiply rates by opportunities to produce counting stats |

### Pipeline Composition

Pipelines are composed via `ProjectionPipeline`, a frozen dataclass:

```python
@dataclass(frozen=True)
class ProjectionPipeline:
    name: str
    rate_computer: RateComputer
    adjusters: tuple[RateAdjuster, ...]
    playing_time: PlayingTimeProjector
    finalizer: ProjectionFinalizer
    years_back: int = 3
```

New projection variants are created by mixing stages. The `PipelineBuilder` provides a fluent API that auto-wires data sources, caching, and adjuster ordering:

```python
pipeline = (
    PipelineBuilder("marcel_gb", config=cfg)
    .with_park_factors()
    .with_pitcher_normalization()
    .with_statcast()
    .with_batter_babip()
    .with_gb_residual(gb_config)
    .build()
)
```

The builder handles dependency resolution internally — e.g., `.with_statcast()` and `.with_batter_babip()` share a single cached `StatcastDataSource`.

### Registered Pipelines

Named pipelines are registered in `pipeline/presets.py`:

| Pipeline | Approach | Key Stages |
|----------|----------|------------|
| `marcel_classic` | Original Marcel | Uniform aging, flat regression |
| `marcel` | Modern baseline | Per-stat regression, component aging |
| `marcel_full` | All adjustments | + park factors, Statcast, BABIP, pitcher normalization |
| `marcel_gb` | Best accuracy | + gradient boosting residual corrections |
| `marcel_gb_mle` | Best + rookies | + minor league equivalencies for call-ups |
| `marcel_mtl` | Pitcher-focused | + neural network blend (70/30 Marcel/MTL) |
| `mtl` | Standalone NN | MTL rate computer replaces Marcel |
| `mle` | Rookies only | MLE rate computer for thin MLB histories |
| `contextual` | Transformer | Pitch-sequence model replaces Marcel |
| `marcel_contextual` | Transformer blend | Marcel + contextual transformer blend |
| `marcel_ensemble` | Best combined | Ensemble: 50/25/25 Marcel/MTL/contextual + GB residual |
| `steamer` / `zips` | External | FanGraphs projection systems (fetched or CSV) |

## Module Map

```
src/fantasy_baseball_manager/
│
├── pipeline/                  # Core projection framework
│   ├── protocols.py           #   Stage protocol definitions
│   ├── engine.py              #   ProjectionPipeline orchestrator
│   ├── presets.py             #   Named pipeline registry + factories
│   ├── builder.py             #   PipelineBuilder (fluent composition)
│   ├── types.py               #   PlayerRates + PlayerMetadata
│   ├── source.py              #   PipelineProjectionSource wrapper
│   ├── statcast_data.py       #   Statcast xStats data source
│   ├── park_factors.py        #   Park factor data source
│   ├── skill_data.py          #   Swing decision + sprint speed data
│   ├── batted_ball_data.py    #   Pitcher batted ball profiles
│   └── stages/                #   40+ stage implementations
│       ├── rate_computers.py          MarcelRateComputer
│       ├── stat_specific_rate_*.py    Per-stat regression constants
│       ├── platoon_rate_computer.py   Platoon split rates
│       ├── adjusters.py               Rebaseline, uniform aging
│       ├── component_aging.py         Per-stat aging curves
│       ├── park_factor_adjuster.py    Home/away park effects
│       ├── pitcher_normalization.py   BABIP/LOB% regression
│       ├── statcast_adjuster.py       xStats blending (batters)
│       ├── pitcher_statcast_*.py      xStats blending (pitchers)
│       ├── batter_babip_adjuster.py   BABIP skill adjustment
│       ├── pitcher_babip_skill_*.py   GB%-based BABIP adjustment
│       ├── gb_residual_adjuster.py    LightGBM residual corrections
│       ├── mtl_blender.py             Neural network blend
│       ├── mtl_rate_computer.py       Standalone MTL rates
│       ├── contextual_blender.py      Transformer blend
│       ├── contextual_rate_*.py       Transformer rates
│       ├── prediction_source.py       PredictionSource protocol + implementations
│       ├── ensemble.py                EnsembleAdjuster (multi-model blending)
│       ├── skill_change_adjuster.py   Year-over-year skill deltas
│       ├── identity_enricher.py       Player ID resolution
│       ├── playing_time.py            Marcel playing time
│       ├── enhanced_playing_time.py   + injury/age/volatility
│       ├── finalizers.py              Rates -> counting stats
│       └── regression_config.py       Tunable regression constants
│
├── marcel/                    # MARCEL projection core
│   ├── models.py              #   BattingSeasonStats, PitchingSeasonStats,
│   │                          #   BattingProjection, PitchingProjection
│   ├── weights.py             #   5-4-3 weighting, PA/IP formulas
│   ├── league_averages.py     #   League rate calculations
│   ├── age_adjustment.py      #   Age curve factors
│   ├── data_source.py         #   DataSource factories for pybaseball
│   └── cli.py                 #   `players project` command
│
├── ml/                        # Machine learning models
│   ├── mtl/                   #   Multi-Task Learning (PyTorch)
│   │   ├── model.py           #     Shared trunk + stat-specific heads
│   │   ├── trainer.py         #     Training loop (uncertainty-weighted loss)
│   │   ├── dataset.py         #     Feature/label datasets
│   │   ├── config.py          #     Architecture + training + blender config
│   │   └── persistence.py     #     Model save/load
│   ├── residual_model.py      #   LightGBM residual model (train + predict)
│   ├── features.py            #   Feature engineering (shared across ML models)
│   ├── training.py            #   ResidualModelTrainer
│   ├── validation.py          #   Cross-validation utilities
│   └── cli.py                 #   `ml train` / `ml train-mtl` commands
│
├── contextual/                # Transformer-based contextual model
│   ├── model/                 #   Model architecture (PyTorch)
│   │   ├── model.py           #     ContextualPerformanceModel
│   │   ├── embedder.py        #     EventEmbedder (pitch types, results)
│   │   ├── transformer.py     #     GamestateTransformer (multi-head attention)
│   │   ├── heads.py           #     MaskedGamestateHead, PerformancePredictionHead
│   │   ├── mask.py            #     Attention masking (player/game boundaries)
│   │   ├── positional.py      #     Sinusoidal positional encoding
│   │   ├── tensorizer.py      #     Data -> tensor conversion
│   │   └── config.py          #     ModelConfig
│   ├── training/              #   Training pipeline
│   │   ├── pretrain.py        #     Masked Gamestate Model pretraining
│   │   ├── finetune.py        #     Performance prediction finetuning
│   │   ├── dataset.py         #     Training datasets
│   │   └── config.py          #     Training + blender config
│   ├── data/                  #   Data pipeline
│   │   ├── source.py          #     Gamestate data fetching
│   │   ├── builder.py         #     GameSequenceBuilder
│   │   ├── models.py          #     Domain models
│   │   ├── vocab.py           #     Token vocabularies
│   │   └── cache.py           #     Data caching
│   ├── predictor.py           #   Inference wrapper
│   ├── adapter.py             #   Predictions -> projections
│   └── cli.py                 #   Contextual model commands
│
├── minors/                    # Minor League Equivalencies (MLE)
│   ├── model.py               #   MLEStatModel (LightGBM per stat)
│   ├── training.py            #   MLE training pipeline
│   ├── features.py            #   MLE feature engineering
│   ├── rate_computer.py       #   MLERateComputer, MLEAugmentedRateComputer
│   ├── data_source.py         #   MiLB data sources (MLB Stats API)
│   └── evaluation.py          #   MLE validation
│
├── valuation/                 # Player value calculations
│   ├── models.py              #   StatCategory, LeagueSettings, PlayerValue
│   ├── zscore.py              #   Z-score valuation (count + ratio stats)
│   ├── sgp.py                 #   Standings Gain Points valuation
│   ├── stat_extractors.py     #   Extract category values from projections
│   └── cli.py                 #   `players valuate` command
│
├── keeper/                    # Keeper analysis
│   ├── models.py              #   KeeperCandidate, KeeperSurplus
│   ├── surplus.py             #   Surplus value over replacement
│   ├── replacement.py         #   Replacement-level calculations
│   ├── yahoo_source.py        #   Yahoo league integration
│   └── cli.py                 #   `keeper rank/optimize/league`
│
├── draft/                     # Draft tools
│   ├── models.py              #   RosterSlot, RosterConfig, DraftRanking
│   ├── state.py               #   DraftState (roster needs tracking)
│   ├── simulation.py          #   Snake draft simulation engine
│   ├── strategy_presets.py    #   Predefined team strategies
│   └── cli.py                 #   `players draft-rank/draft-simulate`
│
├── evaluation/                # Backtesting harness
│   ├── models.py              #   StatAccuracy, RankAccuracy, SourceEvaluation
│   ├── harness.py             #   Evaluation engine (multi-year, stratified)
│   ├── metrics.py             #   RMSE, MAE, Pearson r, Spearman rho
│   ├── actuals.py             #   Load actual season stats
│   └── cli.py                 #   `evaluate` command
│
├── ros/                       # Rest-of-season blending
│   ├── protocol.py            #   ProjectionBlender protocol
│   ├── blender.py             #   BayesianBlender (preseason + actuals)
│   ├── projector.py           #   ROSProjector
│   └── cli.py                 #   `players ros-project`
│
├── agent/                     # LLM chat assistant
│   ├── core.py                #   LangGraph agent (ReAct loop)
│   ├── tools.py               #   Tool definitions (project, lookup, rank)
│   ├── formatters.py          #   Output formatting for LLM consumption
│   ├── player_lookup.py       #   Fuzzy name matching
│   └── cli.py                 #   `chat start` (interactive REPL)
│
├── projections/               # External projection sources
│   ├── fangraphs.py           #   FanGraphsProjectionSource (API)
│   ├── csv_source.py          #   CSVProjectionSource (historical files)
│   ├── csv_resolver.py        #   Resolve CSV file paths
│   ├── data_source.py         #   DataSource wrappers
│   ├── adapter.py             #   ExternalProjectionAdapter
│   └── models.py              #   ProjectionSystem enum, ProjectionData
│
├── statcast/                  # Statcast data management
│   ├── models.py              #   DownloadConfig, SeasonManifest
│   ├── downloader.py          #   Download from Chadwick Bureau
│   ├── store.py               #   Parquet file storage
│   ├── fetcher.py             #   Fetch + cache management
│   ├── chunker.py             #   Date-based chunking
│   ├── calendar.py            #   Season date ranges
│   └── cli.py                 #   `statcast` commands
│
├── cache/                     # Caching layer
│   ├── protocol.py            #   CacheStore protocol
│   ├── sqlite_store.py        #   SQLite-backed persistent cache
│   ├── wrapper.py             #   cached() decorator, cached_call()
│   ├── serialization.py       #   DataclassListSerializer, etc.
│   └── factory.py             #   Cache store creation
│
├── league/                    # Yahoo Fantasy league integration
│   ├── models.py              #   TeamRoster, LeagueRosters, TeamProjection
│   ├── roster.py              #   RosterSource protocol
│   ├── projections.py         #   League projection aggregation
│   └── cli.py                 #   League commands
│
├── data/protocol.py           # Unified DataSource[T] protocol
├── player/identity.py         # Player canonical identity
├── player_id/mapper.py        # SFBB ID mapping (Yahoo/FanGraphs/MLBAM)
├── result.py                  # Result[T, E] type (Ok/Err)
├── context.py                 # Ambient execution context (year, cache, db)
├── config.py                  # YAML + env config loading
├── services/container.py      # ServiceContainer (DI for CLI)
├── yahoo_api.py               # Yahoo Fantasy API client
├── engines.py                 # Pipeline name validation
└── cli.py / main.py           # Typer CLI entry point
```

## Data Flow: End to End

### Projection -> Valuation -> Draft

```
config.yaml (league settings)
       |
       v
  LeagueSettings
  (team_count, categories, scoring_style)
       |
       v
+-----------------+     +------------------+     +----------------+
| Pipeline        |     | Valuation        |     | Draft          |
|                 |     |                  |     |                |
| BattingStats  --+--->-| zscore_batting() |--->-| DraftRanking   |
| PitchingStats --+--->-| zscore_pitching()|--->-| (value x pos   |
|                 |     |                  |     |  scarcity)     |
+-----------------+     +------------------+     +----------------+
                              |                        |
                              v                        v
                        PlayerValue list         DraftRanking list
                              |                        |
                              v                        v
                        +----------+            +------------+
                        | Keeper   |            | Simulation |
                        | Surplus  |            | Engine     |
                        +----------+            +------------+
```

### Evaluation (Backtesting)

```
Historical years (2021-2024)
       |
       v
  For each year Y:
       |
  +----+----+
  |         |
  v         v
Pipeline   Actuals
project(Y) load(Y)
  |         |
  +----+----+
       |
       v
  EvaluationHarness
  (RMSE, MAE, Pearson r, Spearman rho, top-N precision)
       |
       v
  SourceEvaluation
  (per-stat accuracy + rank accuracy, stratified by PA/IP/age)
```

### Rest-of-Season Update

```
Preseason projection (from pipeline)
       +
In-season actuals (partial season stats)
       |
       v
  ProjectionBlender
  (Bayesian weighting: more games played -> more weight on actuals)
       |
       v
  Updated full-season projection
```

## Cross-Cutting Infrastructure

### DataSource Protocol

All data fetching goes through `DataSource[T]`, a callable protocol with overloaded return types:

```python
class DataSource(Protocol[T]):
    @overload
    def __call__(self, query: type[AllPlayers]) -> Ok[list[T]] | Err[DataSourceError]: ...
    @overload
    def __call__(self, query: Player) -> Ok[T] | Err[DataSourceError]: ...
    @overload
    def __call__(self, query: list[Player]) -> Ok[list[T]] | Err[DataSourceError]: ...
```

This gives a unified interface for stats, projections, ADP, minor league data, and ID mappings. The `cached()` decorator wraps any `DataSource` with SQLite-backed caching.

### Result Type

The codebase uses `Ok[T] | Err[E]` instead of exceptions for recoverable errors, borrowed from Rust:

```python
result = batting_source(ALL_PLAYERS)
match result:
    case Ok(players): ...
    case Err(error): ...
```

### Context Variables

The `Context` dataclass provides ambient state (year, cache settings, database path) via Python `ContextVar`. Pipeline stages access the current year without explicit parameter threading:

```python
with new_context(year=2026):
    players = rate_computer.compute_batting_rates(...)
```

### Service Container

`ServiceContainer` provides lazy dependency injection for the CLI layer. It constructs data sources, mappers, and caches on first access and supports explicit injection for testing:

```python
container = ServiceContainer(
    config=ServiceConfig(no_cache=True),
    batting_source=mock_source,
)
```

### Caching

The `cached()` decorator wraps `DataSource` instances. `cached_call()` wraps arbitrary callables. Both use `CacheStore` (SQLite-backed) with namespace/key/TTL semantics:

```python
source = cached(
    create_batting_source(),
    namespace="stats_batting",
    ttl_seconds=30 * 86400,
    serializer=DataclassListSerializer(BattingSeasonStats),
)
```

### Player Identity

`Player` is the canonical identity object carrying cross-system IDs (Yahoo, FanGraphs, MLBAM). The `PlayerIdentityEnricher` pipeline stage stamps `Player` objects onto `PlayerRates` early in the adjuster chain so downstream stages can access MLBAM IDs for Statcast lookups without additional mapper calls.

## Modeling Approaches

### Marcel (Statistical Baseline)

The foundation. Three-year weighted averages (5-4-3 weights for most recent to oldest) with regression to league mean, per-stat aging curves, and MARCEL playing time formulas. No external data or ML — pure historical stats.

### Statcast Blending

Blends observed batting stats with Baseball Savant expected stats (xBA, xSLG, xwOBA). A batter with a low BABIP but high expected stats gets a boost; one with a lucky BABIP gets regressed. Controlled by a `blend_weight` that balances signal vs. noise.

### Gradient Boosting Residuals

LightGBM models trained on historical Marcel errors. Features include Statcast metrics (barrel rate, exit velocity), swing decisions (chase rate, whiff rate), and demographics. Applied in conservative mode (HR/SB only for batters) to avoid degrading BABIP-sensitive stats like OBP.

### Multi-Task Learning

PyTorch neural network with shared hidden layers and stat-specific output heads. Learns cross-stat correlations (barrel rate drives HR and 2B; sprint speed drives SB and triples). Uses uncertainty-weighted loss to auto-balance stats of different scales. Best for pitcher ERA/WHIP.

### Contextual Transformer

Transformer encoder trained on pitch-by-pitch game sequences. Pretrained with masked gamestate prediction (predict masked pitch types/results), then finetuned on season-long stat prediction. Captures situational factors (vs LHP, pitcher quality, game state) that aggregate stats miss.

### Minor League Equivalencies

Per-stat LightGBM models that translate MiLB rates to MLB equivalents, conditioned on age, level, and sample size. Used for rookies and call-ups with fewer than 200 MLB plate appearances. Can wrap any rate computer via `MLEAugmentedRateComputer`.

## Testing

Tests mirror `src/` structure under `tests/`. Shared fixtures live in `tests/conftest.py`:

- `test_context`: Initialized context with temp database
- `no_cache_context`: Cache disabled
- `mock_container`: Mock `ServiceContainer`
- `reset_service_container`: Cleanup after test

Tests use `pytest` with parametrized test cases. Pipeline stage tests construct stages with fake data sources and verify `PlayerRates` transformations. Integration tests exercise full pipeline presets against known inputs.

## Configuration

**`config.default.yaml`** defines league settings (team count, scoring style, categories), Yahoo OAuth credentials, cache TTLs, and database paths. Environment variable overrides follow the `FANTASY__section__key` convention.

**`RegressionConfig`** controls pipeline-specific tuning: regression constants per stat, pitcher normalization parameters, Statcast blending weights, and platoon split ratios.

---

## Modeling Architecture: Best Practices & Improvement Opportunities

This section catalogs patterns and improvements that could make the modeling approach more flexible, composable, and easier to extend with new model types.

### Current Strengths

1. **Protocol-based stages**: `RateComputer`, `RateAdjuster`, `PlayingTimeProjector`, and `ProjectionFinalizer` are all protocols. Any object with the right methods can slot in without inheritance hierarchies.

2. **Composable pipeline builder**: The `PipelineBuilder` fluent API makes it easy to construct new pipeline variants by mixing and matching stages.

3. **Evaluation harness**: Any pipeline can be backtested against historical actuals with consistent metrics, enabling rigorous A/B comparison.

4. **Conservative ML application**: The gradient boosting model only adjusts stats where it has demonstrated predictive value (HR, SB), avoiding degradation of other stats.

5. **Incremental metadata**: `PlayerMetadata` is a TypedDict with optional fields, populated by each stage. Downstream stages can inspect what upstream stages contributed.

6. **Formal ensemble framework**: The `EnsembleAdjuster` combines multiple prediction sources (MTL, contextual, GB residual) with per-stat-per-model weights and pluggable blending strategies. Sources declare their prediction mode (RATE vs RESIDUAL) via a common `PredictionSource` protocol, and weights are automatically renormalized when a source lacks coverage for a stat.

### Improvement Opportunities

#### 1. Unified Model Registry & Discovery

**Current state**: Pipeline presets are defined as individual factory functions in `presets.py`. ML models (GB, MTL, contextual) have separate persistence and loading patterns. Adding a new model type requires touching the preset file, builder, and potentially the CLI.

**Improvement**: Create a model registry that decouples model definition from pipeline wiring:

```python
@register_model("gb_residual", version="v2")
class GBResidualModelV2:
    """Auto-discovered by registry. Metadata describes inputs/outputs."""
    inputs: tuple[str, ...] = ("statcast", "skill_data", "batted_ball")
    outputs: tuple[str, ...] = ("hr_residual", "sb_residual")
```

This would allow new models to be added by dropping a file into a directory without modifying central registry code. Models could declare their data dependencies, enabling the pipeline to auto-resolve data sources.

#### 2. Formal Ensemble Framework (**Implemented**)

**Implemented** in `pipeline/stages/prediction_source.py` and `pipeline/stages/ensemble.py`. A `PredictionSource` protocol unifies all model types (MTL, contextual, GB residual) behind a common interface. Each source declares a `PredictionMode` — `RATE` (absolute predictions blended via weighted average with Marcel) or `RESIDUAL` (additive corrections applied after blending).

`EnsembleAdjuster` orchestrates multiple sources with per-stat-per-model weights and a pluggable `BlendingStrategy`:

```python
config = EnsembleConfig(
    default_weights={"marcel": 0.5, "mtl": 0.25, "contextual": 0.25},
    weights={"hr": {"marcel": 0.4, "mtl": 0.3, "contextual": 0.3}},
)
ensemble = EnsembleAdjuster(sources=[mtl_source, ctx_source, gb_source], config=config)
```

When a source doesn't produce a prediction for a stat, its weight is excluded and remaining weights are renormalized. The `marcel_ensemble` preset wires all three sources with default 50/25/25 weights plus GB residual corrections. Future strategies (stacking, rank averaging) can be added by implementing the `BlendingStrategy` protocol.

#### 3. Prediction Intervals & Uncertainty

**Current state**: All projections are point estimates. No confidence intervals, percentile ranges, or uncertainty quantification.

**Improvement**: Extend `PlayerRates` and projections to carry uncertainty:

```python
@dataclass
class ProjectedStat:
    mean: float
    std: float           # or percentiles
    p10: float
    p90: float
```

Sources of uncertainty:
- **Marcel**: Regression amount inversely relates to certainty. Players regressed heavily have wider intervals.
- **GB/MTL**: MC Dropout or prediction intervals from the model.
- **Playing time**: Injury risk, roster competition.

This enables risk-aware drafting (high-variance players as upside picks, low-variance as safe floors).

#### 4. Feature Store Abstraction

**Current state**: Each model computes its own features. `ml/features.py` has shared feature engineering, but the contextual model, MTL, and GB models each have independent feature pipelines. Statcast data, skill data, and batted ball data are resolved separately by each consumer.

**Improvement**: Create a centralized feature store that computes features once and serves them to all models:

```python
feature_store = FeatureStore(
    sources=[statcast_source, skill_source, batted_ball_source],
    year=2026,
)
# Each model requests the features it needs
gb_features = feature_store.get(player, features=["barrel_rate", "chase_rate", "age"])
mtl_features = feature_store.get(player, features=["xba", "xslg", "sprint_speed"])
```

Benefits: deduplicates data fetching, ensures feature consistency across models, and makes it easy to add new features that all models can access.

#### 5. Per-Stat Model Selection

**Current state**: A single pipeline produces all stats for all players. The `marcel_gb` pipeline applies the same GB model to every player.

**Improvement**: Allow the pipeline to select the best model per stat and per player:

```python
# Based on backtesting, use different models for different stats
stat_model_map = {
    "hr": "marcel_gb",       # GB excels at HR
    "era": "marcel_mtl",     # MTL excels at ERA
    "sb": "marcel_gb",       # GB excels at SB
    "obp": "marcel_full",    # Pure Marcel is best for OBP
}
```

This could extend to per-player selection: use MLE for rookies, contextual for players with rich pitch data, and Marcel for everyone else. A routing layer would select which model's output to use based on data availability and backtested accuracy.

#### 6. Structured Adjuster Dependencies

**Current state**: Adjuster ordering is managed by the `PipelineBuilder`, which constructs adjusters in a hard-coded sequence. Adding a new adjuster requires understanding where it should go in the chain.

**Improvement**: Have adjusters declare their dependencies explicitly:

```python
class GBResidualAdjuster(RateAdjuster):
    depends_on = {"park_factors", "statcast"}   # must run after these
    provides = {"gb_residual"}                  # other adjusters can depend on this
```

The builder could then topologically sort adjusters, allowing new adjusters to be inserted without manual ordering. Invalid orderings (circular dependencies, missing requirements) would be caught at build time.

#### 7. Separate Rate vs. Counting Stat Models

**Current state**: Rate models and counting stat models are conflated. The pipeline computes rates, then multiplies by playing time at the end. But some phenomena are better modeled as counting stats (e.g., SB depends on opportunities, not just speed).

**Improvement**: Allow models to contribute at different levels:

- **Rate models**: Predict per-PA rates (HR/PA, K/PA). Current approach.
- **Counting models**: Predict season totals directly (SB, saves). Some stats have dependencies on opportunity (lineup position, closer role) that rate models miss.
- **Context models**: Predict adjustments conditional on team/league context (RBI driven by lineup position, runs by team quality).

The finalizer could merge rate-based and count-based predictions, using each where it's strongest.

#### 8. Online Learning for In-Season Updates

**Current state**: Models are trained offline on historical data. In-season updates use `ROSProjector` with Bayesian blending, but the models themselves don't update.

**Improvement**: Support incremental model updates as the season progresses:

- Retrain the GB residual model mid-season with partial-year actuals.
- Update playing time projections based on actual usage patterns.
- Adjust aging curves when a player's trajectory deviates from expectation.

This would require checkpointed model state and efficient incremental training rather than full retrains.

#### 9. Model Versioning & Experiment Tracking

**Current state**: Models are saved to `~/.fantasy_baseball/models/` by name. No versioning, no run metadata, no experiment comparison beyond the evaluation CLI.

**Improvement**: Track model experiments with:
- Version hashes tied to training data + hyperparameters.
- Stored evaluation metrics alongside model artifacts.
- Comparison tooling: "how does v3 of the GB model compare to v2 on 2024 data?"
- Integration with the evaluation harness for automated regression testing.

#### 10. Pluggable Data Source Adapters

**Current state**: Each data source (pybaseball, FanGraphs API, MLB Stats API, CSV) has its own implementation class. Adding a new source (e.g., Statcast from a Parquet lake, or a third-party projection system) requires implementing the full `DataSource[T]` protocol.

**Improvement**: Create a thin adapter layer that maps raw data into the canonical types:

```python
class StatcastParquetAdapter(DataSourceAdapter[StatcastRecord]):
    """Reads from a Parquet file and maps columns to StatcastRecord."""
    column_map = {"launch_speed": "exit_velocity", ...}
```

Combined with a source registry, this would allow swapping data backends (API vs. local file vs. database) without changing consumer code.
