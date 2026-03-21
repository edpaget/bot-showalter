# Architecture Overview

## Data Flow

```
External Sources (FanGraphs, Lahman, MLB API, Statcast, FantasyPros, Yahoo)
  -> Ingest (fetch, map, load)
    -> SQLite (fbm.db, statcast.db)
      -> Feature Assembler (declare, materialize, version)
        -> Models (prepare, train, predict, evaluate)
          -> Projections (per-player stat forecasts)
            -> Valuations (auction dollar values)
              -> Draft Board / Reports / Keeper Analysis / Yahoo sync
```

## Layers

### Domain (`domain/`)

Immutable value objects (`@dataclass(frozen=True)`) representing the core
vocabulary: `Player`, `PlayerBio`, `PlayerProfile`, `BattingStats`,
`PitchingStats`, `Projection`, `ProjectionAccuracy`, `ProjectionConfidence`,
`StatDistribution`, `Valuation`, `ModelRun`, `LeagueEnvironment`,
`LeagueSettings`, `ADP`, `ADPMovers`, `ADPAccuracy`, `DraftBoard`, `Tier`,
`DraftRecommendation`, `DraftReport`, `YahooLeague`, `YahooTeam`,
`YahooDraftPick`, `YahooPlayer`, `KeeperCost`, `KeeperDecision`,
`KeeperOptimization`, `Experiment`, `Checkpoint`, `PickValue`, `MockDraft`,
`ColumnProfile`, `CorrelationResult`, `TemporalStability`,
`ResidualAnalysis`, `ResidualPersistence`, `TalentQuality`,
`PerformanceDelta`, `CategoryTracker`, `Result`, etc. No I/O or
side effects.

### Repositories (`repos/`)

Protocol-based data access. Each repo is a `@runtime_checkable` Protocol
with a `Sqlite*` implementation using raw SQL. Constructor injection of
`sqlite3.Connection` makes testing straightforward — tests pass fakes
that satisfy the Protocol without touching a database.

Implementations: `SqlitePlayerRepo`, `SqliteBattingStatsRepo`,
`SqlitePitchingStatsRepo`, `SqliteProjectionRepo`, `SqliteValuationRepo`,
`SqliteADPRepo`, `SqliteModelRunRepo`, `SqliteStatcastPitchRepo`,
`SqliteSprintSpeedRepo`, `SqliteILStintRepo`, `SqliteLeagueEnvironmentRepo`,
`SqliteMinorLeagueBattingStatsRepo`, `SqlitePositionAppearanceRepo`,
`SqliteRosterStintRepo`, `SqliteLoadLogRepo`, `SqliteLevelFactorRepo`,
`SqliteCheckpointRepo`, `SqliteExperimentRepo`, `SqliteKeeperRepo`,
`SqliteYahooLeagueRepo`, `SqliteYahooTeamRepo`, `SqliteYahooDraftRepo`,
`SqliteYahooPlayerMapRepo`, `SqliteYahooRosterRepo`.

### Ingest (`ingest/`)

Data acquisition from external sources. The `DataSource` Protocol returns
a list of row dicts; row mappers convert to domain objects. `Loader`
orchestrates fetch-map-upsert within a transaction, logging results to
`LoadLogRepo`.

Sources: Chadwick (player register), Lahman (bios, appearances, roster
stints, teams), FanGraphs (batting/pitching stats), MLB Stats API
(transactions, minor leagues), Statcast Savant (pitch-level data, sprint
speed), FantasyPros (ADP via CSV or live fetch), CSV (third-party
projections).

### Database (`db/`)

SQLite with WAL mode and foreign keys. Two databases: `fbm.db` (main)
and `statcast.db` (pitch-level data and sprint speed, lazily attached for
cross-DB joins). Schema evolution via numbered SQL migration files applied
on connection open. Connection pooling via `pool.py`.

### Features (`features/`)

Declarative feature engineering. Features are described as data
(`Feature`, `TransformFeature`, `DerivedTransformFeature`, `DeltaFeature`)
using a fluent builder API (`.lag()`, `.rolling_mean()`, `.per()`,
`.percentile()`). A `FeatureSet` bundles features with seasons and
filters; its version is a SHA-256 content hash for reproducibility and
deduplication.

`SqliteDatasetAssembler` materializes a `FeatureSet` into a versioned
SQLite table, returning a `DatasetHandle` for downstream reading and
splitting.

Transform modules in `features/transforms/` compute derived Statcast
metrics: age interactions, batted ball profiles, batted ball interactions,
batter trends, pitcher command, expected stats (xwOBA etc.), league
averages, pitch mix, plate discipline, playing time, spin profiles,
sprint speed, and weighted rates.

### Models (`models/`)

Pluggable projection systems. Each model class is decorated with
`@register("name")` to auto-register in a global registry. The CLI
resolves models by name via `get("name")`.

Models implement subsets of the operation Protocols: `Preparable`,
`Trainable`, `Predictable`, `Evaluable`, `Tunable`, `Ablatable`,
`FineTunable`, `Sweepable`, `FeatureIntrospectable`. The CLI dispatcher
checks `isinstance` before invoking.

Implementations: Marcel (weighted averages + regression + aging), MLE
(minor-league translations), Statcast GBM (gradient-boosted on pitch
data), Statcast GBM Preseason (lagged features), Composite (feature-group
linear combination), Ensemble (weighted blend of systems), Playing-Time
(OLS regression for PA/IP), ZAR (Z-Score Above Replacement valuation).

### Services (`services/`)

Stateless business logic consuming repos and assemblers via constructor
DI. Services include:

- **Evaluation:** `ProjectionEvaluator` (RMSE, bias vs actuals),
  `ValuationEvaluator` (MAE, rank correlation), `TrueTalentEvaluator`
  (cross-season talent estimation), `ResidualPersistenceDiagnostic`,
  `ResidualAnalysisDiagnostic`, `RegressionGate` (multi-season validation).
- **Reporting:** `PerformanceReportService` (over/underperformer cohort
  analysis), `ADPReportService` (value-over-ADP), `ADPAccuracyEvaluator`,
  `ADPMoversService` (ADP ranking changes), `DraftReportService`
  (post-draft analysis).
- **Lookup:** `ProjectionLookupService`, `ValuationLookupService`,
  `ProjectionConfidenceService` (cross-system agreement),
  `PlayerBiographyService`, `PlayerProfileService`, `PlayerResolverService`.
- **Draft:** `DraftBoardService` (draft board assembly with ADP overlay),
  `TierGeneratorService` (tier breaks from valuations), `DraftSessionService`
  (interactive draft REPL), `DraftRecommenderService` (category need
  recommendations), `MockDraftService` (simulated drafts with bot
  strategies), `PickValueService` (draft pick value curve),
  `DraftTranslationService` (Yahoo draft integration).
- **Keeper:** `KeeperService` (surplus value, decisions, adjusted rankings),
  `KeeperOptimizerService` (optimal keeper set solver).
- **Experiment:** `ExperimentSummaryService` (exploration status),
  `CheckpointResolverService` (feature set checkpoint save/restore).
- **Data:** `DatasetCatalogService` (feature set materialization cache),
  `LeagueEnvironmentService` (league environment computation),
  `DataProfilerService` (column distributions, correlations, temporal
  stability), `QuickEvalService` (single-target fast evaluation),
  `PlayerUniverseService` (player filtering and universe selection).

### Agent (`agent/`)

LangGraph-based LLM agent using Claude for conversational access to
projections, valuations, and analysis. `graph.py` defines the agent
graph with tool-calling nodes. `chat.py` provides the interactive chat
loop. `prompt.py` contains the system prompt. `stream.py` handles
streaming responses.

### Tools (`tools/`)

Agent tool definitions that wrap service calls for the LLM agent.
Each tool module provides structured tool functions: `adp_tools`
(ADP lookups and trends), `performance_tools` (over/underperformer
analysis), `player_tools` (player search and biography),
`projection_tools` (projection lookups and comparisons),
`valuation_tools` (valuation rankings and lookups).
`_formatting.py` provides shared output formatting.

### Discord Bot (`discord_bot/`)

Discord integration for conversational access. `bot.py` sets up the
Discord client. `agent_handler.py` routes messages to the LLM agent
and streams responses back to Discord channels.

### Yahoo (`yahoo/`)

Yahoo Fantasy API integration. `YahooAuth` handles OAuth 2.0
authentication and token management. `YahooFantasyClient` wraps API
calls (game keys, league metadata, team rosters, draft results).
`YahooLeagueSource` fetches league and team data for syncing to the
local database. Supports live draft tracking with auto-pick ingestion.

### CLI (`cli/`)

Typer-based commands organized into subcommand groups: core model
operations (`prepare`, `train`, `predict`, `evaluate`, `tune`, `sweep`,
`ablate`, `finetune`, `gate`, `quick-eval`, `marginal-value`,
`compare-features`), `ingest` (data acquisition), `compute` (derived
data), `projections` (lookup, systems), `valuations` (lookup, rankings,
evaluate), `draft` (board, export, live server, interactive session,
tiers, needs, pick values, trade picks), `report` (14 report types),
`runs` (model run management), `datasets` (feature set cache),
`compare` (multi-system comparison), `keeper` (surplus value, decisions,
optimization, trade evaluation), `experiment` (journal, search, summary,
checkpoints), `profile` (column distributions, correlations, temporal
stability), `yahoo` (auth, sync, rosters, draft), `chat` (LLM agent),
and `discord` (bot).

`factory.py` is the composition root — context managers
(`build_model_context`, `build_ingest_container`, etc.) open a DB
connection, wire repos/assemblers/services, yield a typed context
dataclass, and close the connection on exit. `_dispatcher.py` routes
operations to the correct Protocol method.

## Key Design Decisions

**SQLite-first.** Single-file portable database, no ORM. Raw SQL for
clarity and performance. WAL mode enables concurrent reads. Separate
Statcast DB keeps the main database lean.

**Protocol-based DI.** All inter-layer boundaries use
`@runtime_checkable` Protocols. Constructor injection flows from the
composition root in `factory.py`. Tests inject fakes/stubs without
`monkeypatch`, and the type checker verifies doubles match interfaces.

**Immutable domain objects.** All domain types are frozen dataclasses.
Value semantics enable safe sharing and content-based equality.

**Content-addressed feature versioning.** A `FeatureSet`'s version is
the SHA-256 hash of its canonical representation. This detects feature
drift, enables caching, and ensures reproducibility.

**Model registry.** The `@register` decorator auto-registers model
classes on import. The CLI dispatches by name without hard-coding model
types, making new models a single-file addition.

**Composition root.** All wiring lives in `cli/factory.py`. Model
classes, services, and repos are unaware of each other's concrete types —
they depend only on Protocols.

**Interaction mode generality.** Business logic lives in services, not
in CLI commands or agent tools. The CLI, LLM agent, Discord bot, and
live draft server are thin adapters over shared services, so any
capability exposed in one mode is available in all others.

## Configuration

`fbm.toml` at the project root defines data/artifact paths, per-model
parameters, league settings, and Yahoo Fantasy API credentials.
`config.py` merges CLI overrides onto TOML values and returns a
`ModelConfig` dataclass. `config_league.py` parses league settings
(teams, budget, roster positions, scoring categories). `config_yahoo.py`
parses Yahoo OAuth credentials and league mappings. Roadmaps and phase
plans are managed via the rdm MCP server (project: fbm).
