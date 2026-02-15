# Feature DSL — Roadmap

## Problem

Models need training and prediction datasets assembled from multiple source tables
(batting\_stats, pitching\_stats, player, projection) with various transformations
(lags, rolling aggregates, derived calculations). Today there is no standard way to
declare what a model needs — each model would have to hand-write SQL or data-wrangling
code.

We want a system where a model declares its features as Python objects, and the
framework automatically assembles (or reuses) a materialized SQLite table containing
exactly those columns.

## Design overview

### Core concept: a Feature is a column spec

A `Feature` is a declarative description of one column in the output dataset. It
captures:

- **Source** — which table the raw data lives in (`batting_stats`, `pitching_stats`,
  `player`, `projection`)
- **Column** — which column from that table
- **Temporal offset** — lag in seasons relative to the target season (0 = same season,
  1 = prior season, etc.)
- **Aggregation** — optional reduction across a window of seasons (`mean`, `sum`,
  `min`, `max`)
- **Output name** — the column alias in the materialized dataset

Features don't hold data. They're instructions that the assembler reads to build SQL.

### Entity spine

Every dataset is anchored to a *spine* — the set of `(player_id, season)` pairs the
dataset covers. The spine is the cartesian product of qualifying players and the
requested target seasons, optionally filtered (e.g. minimum PA threshold). All feature
lookups join against this spine.

### Builder API

A fluent builder makes feature lists readable and composable:

```python
from fantasy_baseball_manager.features import batting, pitching, player

marcel_batting_features = [
    # Weighted-average inputs: 3 years of counting stats
    batting.col("pa").lag(1).alias("pa_1"),
    batting.col("pa").lag(2).alias("pa_2"),
    batting.col("pa").lag(3).alias("pa_3"),
    batting.col("hr").lag(1).alias("hr_1"),
    batting.col("hr").lag(2).alias("hr_2"),
    batting.col("hr").lag(3).alias("hr_3"),
    batting.col("bb").lag(1).alias("bb_1"),

    # Rolling aggregates
    batting.col("hr").per("pa").rolling_mean(3).alias("hr_rate_3yr"),

    # Player metadata
    player.age(),          # season − birth_year, computed
    player.col("bats"),
    player.col("position"),

    # Target column (what we're predicting)
    batting.col("hr").lag(0).alias("hr_next"),
]
```

Each of `batting`, `pitching`, `player` is a `SourceRef` — a lightweight handle bound
to one source table. Its methods return `Feature` instances (frozen dataclasses),
never mutate state.

### FeatureSet

A named, versioned bundle of features plus assembly parameters:

```python
feature_set = FeatureSet(
    name="marcel_batting",
    features=marcel_batting_features,
    seasons=[2022, 2023, 2024],       # target seasons to include
    source_filter="fangraphs",        # which source rows to use for stats
    spine_filter=SpineFilter(min_pa=50, player_type="batter"),
)
```

The **version** is derived automatically by content-hashing the feature definitions
and parameters. Identical declarations produce the same version, enabling reuse.

### DatasetAssembler

The assembler turns a `FeatureSet` into a materialized SQLite table:

1. **Analyze features** — collect which sources and season offsets are needed.
2. **Build the spine** — query distinct `(player_id, season)` pairs matching filters.
3. **Generate SQL** — one `SELECT` with `LEFT JOIN` per distinct (source, lag) pair.
   Derived features (age, rate stats) become SQL expressions.
4. **Materialize** — `CREATE TABLE ds_{feature_set_id} AS SELECT ...`
5. **Record metadata** — insert into `feature_set` and `dataset` tables. Store the
   `source_query` (the generated SQL) for auditability.
6. **Return a `DatasetHandle`** — a reference the model uses to read the table.

### Caching and reuse

- The content hash of a `FeatureSet` determines its version.
- Before materializing, check if a `feature_set` row with that `(name, version)`
  already exists *and* its `ds_*` table is still present.
- If so, return the existing `DatasetHandle` — skip SQL generation and materialization.
- If source data has been reloaded (detectable via `load_log` timestamps vs. dataset
  `created_at`), the assembler can optionally invalidate stale datasets.

### Split management

`DatasetAssembler.split()` partitions a materialized dataset by season:

```python
handle = assembler.materialize(feature_set)
splits = assembler.split(
    handle,
    train=range(2015, 2022),
    validation=[2022, 2023],
    holdout=[2024],
)
# splits.train, splits.validation, splits.holdout are DatasetHandles
```

Each split becomes its own row in the `dataset` table (sharing the same
`feature_set_id`), with `split` set to `"train"`, `"validation"`, or `"holdout"`.
Split tables are views or filtered copies — implementation detail deferred to Phase 3.

---

## Type design

```
Source (Enum)
  BATTING, PITCHING, PLAYER, PROJECTION

Feature (frozen dataclass)
  name: str
  source: Source
  column: str
  lag: int = 0
  window: int = 1
  aggregate: str | None = None      # "mean", "sum", "min", "max"
  denominator: str | None = None    # for rate stats (.per("pa"))
  computed: str | None = None       # sentinel for derived features like age()

SourceRef
  source: Source
  col(column) -> FeatureBuilder
  age() -> Feature                  # player-only convenience

FeatureBuilder
  lag(n) -> FeatureBuilder
  rolling_mean(n) -> FeatureBuilder
  rolling_sum(n) -> FeatureBuilder
  per(denominator) -> FeatureBuilder
  alias(name) -> Feature            # terminal: returns frozen Feature

SpineFilter (frozen dataclass)
  min_pa: int | None = None
  min_ip: float | None = None
  player_type: str | None = None    # "batter" | "pitcher"

FeatureSet (frozen dataclass)
  name: str
  features: tuple[Feature, ...]
  seasons: tuple[int, ...]
  source_filter: str | None = None  # e.g. "fangraphs"
  spine_filter: SpineFilter = SpineFilter()
  version: str                      # computed content hash

DatasetHandle (frozen dataclass)
  dataset_id: int
  feature_set_id: int
  table_name: str                   # "ds_42"
  row_count: int
  seasons: tuple[int, ...]

DatasetSplits (frozen dataclass)
  train: DatasetHandle
  validation: DatasetHandle | None
  holdout: DatasetHandle | None
```

### Protocol

```python
class DatasetAssembler(Protocol):
    def materialize(self, feature_set: FeatureSet) -> DatasetHandle: ...
    def split(self, handle: DatasetHandle, ...) -> DatasetSplits: ...
    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle: ...
    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]: ...
```

---

## SQL generation strategy

Given features, the assembler builds a single query. Example for three lag-1 batting
features and player age:

```sql
CREATE TABLE ds_42 AS
SELECT
    spine.player_id,
    spine.season,
    b1.pa   AS pa_1,
    b1.hr   AS hr_1,
    b1.bb   AS bb_1,
    spine.season - CAST(SUBSTR(p.birth_date, 1, 4) AS INTEGER) AS age
FROM spine
LEFT JOIN batting_stats b1
    ON b1.player_id = spine.player_id
    AND b1.season = spine.season - 1
    AND b1.source = ?
LEFT JOIN player p
    ON p.id = spine.player_id;
```

Key rules:
- Each unique `(source, lag, source_filter)` tuple gets one `LEFT JOIN` with a
  table alias like `b1`, `b2`, `p0`, `pi1`.
- Features sharing the same join are grouped into that join's SELECT columns.
- Rolling aggregates over N seasons use a correlated subquery:
  `(SELECT AVG(hr) FROM batting_stats WHERE player_id = spine.player_id
   AND season BETWEEN spine.season - 3 AND spine.season - 1 AND source = ?)`
- Rate features (`.per("pa")`) become `CAST(col AS REAL) / NULLIF(denom, 0)`.
- The spine is a CTE or temp table built from the relevant stat table filtered by
  the `SpineFilter`.

---

## Integration with models

Models declare their features as a class attribute or method:

```python
class MarcelModel:
    name = "marcel"
    features = marcel_batting_features  # the list from above

    def prepare(self, config: ModelConfig, assembler: DatasetAssembler) -> PrepareResult:
        feature_set = FeatureSet(
            name=f"{self.name}_batting",
            features=tuple(self.features),
            seasons=tuple(config.seasons),
            source_filter="fangraphs",
        )
        handle = assembler.get_or_materialize(feature_set)
        return PrepareResult(
            model_name=self.name,
            rows_processed=handle.row_count,
            artifacts_path=config.artifacts_dir,
        )
```

The `DatasetAssembler` is injected into the model (via the dispatcher or `Preparable`
signature) — the model never touches SQL or raw connections.

---

## Phases

### Phase 1 — Core feature types and builder API

- `features/__init__.py` — public API: `batting`, `pitching`, `player` `SourceRef`
  instances
- `features/types.py` — `Source`, `Feature`, `FeatureBuilder`, `SpineFilter`,
  `FeatureSet`, `DatasetHandle`, `DatasetSplits`
- `features/protocols.py` — `DatasetAssembler` protocol
- Content-hash function for `FeatureSet` version derivation
- Tests: builder produces correct `Feature` dataclasses; content hash is stable and
  deterministic; identical feature lists yield the same hash

### Phase 2 — SQL generation

- `features/sql.py` — analyze a `FeatureSet`, produce a SQL string
- Join planning: group features by `(source, lag)`, assign aliases, emit JOINs
- Spine CTE generation from `SpineFilter`
- Support for: direct columns, lag, rolling aggregates, rate stats, age
- Tests: assert generated SQL against expected strings for known feature sets; run
  generated SQL against an in-memory SQLite with seeded test data and verify output
  columns/values

### Phase 3 — Dataset materialization and caching

- `features/assembler.py` — `SqliteDatasetAssembler` implementing the protocol
- `materialize()`: execute generated SQL as `CREATE TABLE`, write `feature_set` and
  `dataset` rows
- `get_or_materialize()`: check for existing dataset by content hash, reuse or rebuild
- `split()`: partition by season, create split entries in `dataset` table
- `read()`: return rows from materialized table as dicts
- Consider whether to add a `dataset_version` or `source_hash` column to `feature_set`
  for staleness detection (may require a schema migration)
- Tests: round-trip — define features, materialize, read back, verify values; reuse
  detection; split correctness

### Phase 4 — Model pipeline integration

- Update `ModelConfig` or `Preparable` signature so models receive a
  `DatasetAssembler`
- Wire assembler construction in CLI dispatcher (inject connection/pool)
- Update Marcel's `prepare()` to use the feature DSL
- Tests: end-to-end from feature declaration through materialized dataset to model
  prepare call

### Phase 5 — Ergonomics and extensions

- `features/library.py` — reusable feature bundles (e.g. `STANDARD_BATTING_COUNTING`,
  `STANDARD_BATTING_RATES`, `PLAYER_METADATA`) that models can import and combine
- Projection features: `projection.col("hr").system("steamer").alias("steamer_hr")`
- Cross-source features: delta between two sources or two projection systems
- Feature documentation / introspection: list all features in a set with descriptions
- CLI command: `fbm features <model>` to print a model's declared feature set

### Phase 6 — Python transforms (embeddings, complex aggregations)

Phases 1–5 produce features entirely via SQL generation. This phase adds a second path:
the assembler loads raw joined rows from the data store, hands them to a user-defined
Python callable, and writes the output back into the materialized dataset. This unlocks
features that SQL cannot express — embeddings, distribution encodings, clustering,
percentile ranks across pitch types, etc.

#### Core concept: `TransformFeature`

A `TransformFeature` declares:

- **Source query** — which table and columns to load (unaggregated, joined to the spine)
- **Group key** — the spine columns the raw rows are grouped by before being passed to
  the transform (typically `(player_id, season)`)
- **Transform function** — a Python callable that receives one group of raw rows and
  returns a flat dict of output columns
- **Output names** — the column names the transform produces (declared up front so the
  assembler can allocate the dataset schema before running the transform)

```python
from fantasy_baseball_manager.features import statcast, TransformFeature

def pitch_mix_profile(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute pitch-type usage rates and average metrics per pitch type."""
    total = len(rows)
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["pitch_type"], []).append(r)
    result = {}
    for pt in ("FF", "SL", "CH", "CU", "SI", "FC"):
        pitches = by_type.get(pt, [])
        result[f"{pt.lower()}_pct"] = len(pitches) / total if total else 0.0
        result[f"{pt.lower()}_velo"] = (
            sum(p["release_speed"] for p in pitches) / len(pitches)
            if pitches else 0.0
        )
    return result

pitch_mix = TransformFeature(
    source=statcast,
    columns=["pitch_type", "release_speed", "release_spin_rate",
             "pfx_x", "pfx_z"],
    group_by=("player_id", "season"),
    transform=pitch_mix_profile,
    outputs=["ff_pct", "ff_velo", "sl_pct", "sl_velo", "ch_pct", "ch_velo",
             "cu_pct", "cu_velo", "si_pct", "si_velo", "fc_pct", "fc_velo"],
)
```

`TransformFeature` instances can be mixed with regular `Feature` instances in a
`FeatureSet`. The assembler handles them in a second pass after the SQL-based features
are materialized.

#### How the assembler handles transforms

Materialization becomes a two-pass process:

1. **SQL pass** (existing) — build spine, generate SQL for all regular `Feature` items,
   `CREATE TABLE ds_{id}` with those columns plus placeholder columns for transform
   outputs.
2. **Transform pass** — for each `TransformFeature` (or group of transforms sharing the
   same source query):
   a. Query raw rows from the source table joined to the spine (using `ATTACH` for
      cross-database joins like `statcast.db` → `stats.db`).
   b. Group rows by the group key.
   c. Call the transform function for each group.
   d. `UPDATE ds_{id} SET col1=?, col2=?, ... WHERE player_id=? AND season=?` to fill
      in the transform output columns.

This keeps the SQL path untouched for pure-SQL feature sets — the transform pass only
runs when `TransformFeature` items are present.

#### Caching

The content hash for a `FeatureSet` already covers its feature definitions. For
`TransformFeature`, the hash includes:

- Source table and columns requested
- Group key
- Transform function identity (qualified name + source hash via `inspect.getsource`,
  or a user-supplied version string as a fallback for closures/lambdas)
- Output column names

If the transform function's source code changes, the content hash changes, and the
assembler rebuilds the dataset.

#### `Source.STATCAST`

Add `STATCAST` to the `Source` enum. Because statcast data lives in a separate database
file (`statcast.db`), the assembler must `ATTACH` it when loading raw rows for a
transform. The `ATTACH` helper already exists in `db/connection.py` (data-layer Phase 1).

Regular SQL-path features could also reference `Source.STATCAST` for pre-aggregated
statcast columns if an aggregation table exists, but the primary use case for statcast
in the DSL is via `TransformFeature`.

#### Transform Protocol

```python
class RowTransform(Protocol):
    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]: ...
```

Transform functions must be pure: same input rows → same output dict. Side effects
(network calls, file I/O) are not supported. This keeps transforms testable and
cache-safe.

#### Type additions

```
TransformFeature (frozen dataclass)
  source: Source
  columns: tuple[str, ...]         # columns to load from the source table
  group_by: tuple[str, ...]        # spine columns to group on
  transform: RowTransform           # the callable
  outputs: tuple[str, ...]         # output column names
  version: str | None = None       # optional manual version override

Source (Enum) — add:
  STATCAST
```

#### Deliverables

- `features/types.py` — `TransformFeature` dataclass, `RowTransform` protocol,
  `Source.STATCAST`
- `features/assembler.py` — two-pass materialization: SQL pass, then transform pass
  with grouped raw-row loading and batch `UPDATE`
- `features/transforms/` — package for reusable transform functions
  - `features/transforms/pitch_mix.py` — pitch-type usage and velocity profiles
  - `features/transforms/batted_ball.py` — exit velocity / launch angle distribution
    encoding
- Content-hash extension to include transform identity
- Tests:
  - Unit: transform functions in isolation (given rows → expected dict)
  - Integration: define a `FeatureSet` mixing regular features and
    `TransformFeature`, materialize against seeded in-memory SQLite (with attached
    statcast tables), verify output columns and values
  - Caching: changing transform source code invalidates the cached dataset

#### Dependencies

- **Data-layer Phase 4 (Statcast ingest)** — needed for real statcast data, but
  transforms can be built and tested with seeded in-memory tables before ingest is done.
- **Feature DSL Phases 1–3** — the assembler, SQL generation, and type system must exist
  before adding the transform pass.

---

## Relationship to the data-layer plan

This plan provides the detailed design for
[data-layer Phase 6 ("Feature store")](data-layer.md#phase-6--feature-store). The
data-layer roadmap defines the schema tables (`feature_set`, `dataset`, `model_run`)
and the `features/` package location but leaves the declaration API, SQL generation,
and caching strategy unspecified. This plan fills that in.

Mapping between the two:

| Data-layer Phase 6 item | Covered here |
|---|---|
| `features/protocols.py`: FeatureStore Protocol | `DatasetAssembler` protocol (Phase 1) |
| `features/feature_store.py`: materialize datasets as `ds_{id}` tables | `SqliteDatasetAssembler` (Phase 3) — plus the feature declaration types (Phase 1) and SQL generation (Phase 2) that the data-layer plan didn't detail |
| manage train/validation/holdout splits | `DatasetAssembler.split()` + `DatasetSplits` (Phase 3) |
| `features/model_run_repo.py`: track datasets + metrics | Deferred to the [model-registry plan](model-registry.md), which owns `model_run` provenance |
| Schema migration for `feature_set`, `dataset`, `model_run` | Already in `001_initial.sql`; may add `content_hash` column (Phase 3) |

### Dependencies on earlier data-layer phases

- **Phase 1 (Database foundation)** — complete. Provides the connection factory, pool,
  migration runner, and repo layer this plan builds on.
- **Phases 2–3 (Player seeding, stat ingest)** — needed for real data, but the feature
  DSL can be built and tested with seeded in-memory SQLite before ingest is done.
- **Phase 5 (Projections)** — needed before projection-source features work, but all
  other feature types are independent of it.

### What this plan does NOT cover from Phase 6

`model_run_repo.py` (recording which datasets a model run used, storing evaluation
metrics, comparing metrics across versions) belongs to the
[model-registry roadmap](model-registry.md), which depends on both this plan and the
data-layer plan. It is not duplicated here.

## Non-goals (for now)

- Real-time / streaming feature computation — SQLite is batch-only, which is fine for
  season-level projections.
- Feature versioning with backwards compatibility — a changed feature list produces a
  new version (new hash), old datasets remain until explicitly cleaned up.
- pandas/polars integration — the SQL-path assembler works at the SQL level and returns
  dicts. Python transforms (Phase 6) receive `list[dict]`, not DataFrames. Models can
  convert to DataFrames themselves if needed.
