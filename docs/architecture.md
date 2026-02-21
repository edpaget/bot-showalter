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
              -> Draft Board / Reports / Yahoo sync
```

## Layers

### Domain (`domain/`)

Immutable value objects (`@dataclass(frozen=True)`) representing the core
vocabulary: `Player`, `PlayerBio`, `BattingStats`, `PitchingStats`,
`Projection`, `StatDistribution`, `Valuation`, `ModelRun`,
`LeagueEnvironment`, `LeagueSettings`, `ADP`, `DraftBoard`, `Tier`,
`ProjectionConfidence`, `YahooLeague`, `YahooTeam`, etc. No I/O or
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
`SqliteYahooLeagueRepo`, `SqliteYahooTeamRepo`.

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
metrics: batted ball profiles, batted ball interactions, pitcher command,
expected stats (xwOBA etc.), league averages, pitch mix, plate discipline,
playing time, spin profiles, sprint speed, and weighted rates.

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
  (cross-season talent estimation), `ResidualPersistenceDiagnostic`.
- **Reporting:** `PerformanceReportService` (over/underperformer cohort
  analysis), `ADPReportService` (value-over-ADP), `ADPAccuracyEvaluator`,
  `ADPMoversService` (ADP ranking changes).
- **Lookup:** `ProjectionLookupService`, `ValuationLookupService`,
  `ProjectionConfidenceService` (cross-system agreement),
  `PlayerBiographyService`.
- **Draft:** `DraftBoardService` (draft board assembly with ADP overlay),
  `TierGeneratorService` (tier breaks from valuations).
- **Data:** `DatasetCatalogService` (feature set materialization cache),
  `LeagueEnvironmentService` (minor-league environment computation).

### Yahoo (`yahoo/`)

Yahoo Fantasy API integration. `YahooAuth` handles OAuth 2.0
authentication and token management. `YahooFantasyClient` wraps API
calls (game keys, league metadata, team rosters). `YahooLeagueSource`
fetches league and team data for syncing to the local database.

### CLI (`cli/`)

Typer-based commands organized into subcommand groups: core model
operations (`prepare`, `train`, `predict`, `evaluate`, `tune`, `sweep`,
`ablate`, `finetune`), `ingest` (data acquisition), `compute` (derived
data), `projections` (lookup, systems), `valuations` (lookup, rankings,
evaluate), `draft` (board, export, live server), `report` (12 report
types), `runs` (model run management), `datasets` (feature set cache),
`compare` (multi-system comparison), and `yahoo` (auth, sync).

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

## Configuration

`fbm.toml` at the project root defines data/artifact paths, per-model
parameters, league settings, and Yahoo Fantasy API credentials.
`config.py` merges CLI overrides onto TOML values and returns a
`ModelConfig` dataclass. `config_league.py` parses league settings
(teams, budget, roster positions, scoring categories). `config_yahoo.py`
parses Yahoo OAuth credentials and league mappings. Roadmaps and phase
plans live in `docs/plans/`.
