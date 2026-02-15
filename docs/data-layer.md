# Data Layer

The data layer stores player statistics from multiple sources, projection data, and engineered feature datasets in SQLite. It is designed around dependency injection, protocol-typed repositories, and idempotent upsert-based imports.

## Two-database strategy

| Database | Contents | Notes |
|---|---|---|
| `stats.db` | Players, teams, season-level batting/pitching stats, projections, load log, feature sets, datasets, model runs | Core data. Moderate size, frequently joined. |
| `statcast.db` | Pitch-level Statcast data (700k+ rows per season) | Large and ephemeral. Can be dropped and rebuilt from source without affecting curated data. |

Both live under a configurable data directory (default `~/.fbm/data/`). Cross-database queries use SQLite's `ATTACH DATABASE`.

## Package overview

```
src/fantasy_baseball_manager/
    db/                  # Connection management, migrations, pooling
    domain/              # Frozen dataclasses — pure value objects
    repos/               # Protocol-first repository layer
    ingest/              # Data sources, loaders, column mappers
    features/            # Feature DSL, SQL generation, dataset assembly
```

---

## `db/` — Connections and migrations

### Creating connections

```python
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection

# stats.db (applies migrations from db/migrations/)
conn = create_connection("~/.fbm/data/stats.db")

# statcast.db (applies migrations from db/statcast_migrations/)
sc_conn = create_statcast_connection("~/.fbm/data/statcast.db")

# In-memory for tests
conn = create_connection(":memory:")
```

`create_connection` enables WAL mode (for on-disk databases), turns on foreign keys, and runs any pending migrations. The `migrations_dir` parameter allows custom migration paths — `create_statcast_connection` uses this to give `statcast.db` its own schema.

### Attaching databases

```python
from fantasy_baseball_manager.db.connection import attach_database

attach_database(conn, "~/.fbm/data/statcast.db", "statcast")

# Now query across both:
conn.execute("""
    SELECT p.name_first, p.name_last, sc.pitch_type
    FROM player p
    JOIN statcast.statcast_pitch sc ON p.mlbam_id = sc.batter_id
    WHERE sc.game_pk = 718001
""")
```

### Connection pooling

```python
from fantasy_baseball_manager.db.pool import ConnectionPool

pool = ConnectionPool("~/.fbm/data/stats.db", size=4)
with pool.connection() as conn:
    # use conn
    pass
```

Thread-safe, `queue.Queue`-backed. Connections are pre-opened with `check_same_thread=False` and WAL mode.

### Migration strategy

Numbered SQL files in `db/migrations/` (for `stats.db`) or `db/statcast_migrations/` (for `statcast.db`). A `schema_version` table tracks which have run. The connection factory applies pending migrations automatically on open.

---

## `domain/` — Data models

All domain types are frozen dataclasses. Required fields come first, optional fields default to `None`.

| Type | Key fields | Identity |
|---|---|---|
| `Player` | `name_first`, `name_last` | `mlbam_id`, `fangraphs_id`, `bbref_id` |
| `Team` | `abbreviation`, `name`, `league`, `division` | `abbreviation` |
| `BattingStats` | `player_id`, `season`, `source` | `(player_id, season, source)` |
| `PitchingStats` | `player_id`, `season`, `source` | `(player_id, season, source)` |
| `Projection` | `player_id`, `season`, `system`, `version`, `player_type`, `stat_json` | `(player_id, season, system, version, player_type)` |
| `StatcastPitch` | `game_pk`, `game_date`, `batter_id`, `pitcher_id`, `at_bat_number`, `pitch_number` | `(game_pk, at_bat_number, pitch_number)` |
| `ModelRunRecord` | `system`, `version`, `train_dataset_id` | `(system, version)` |
| `LoadLog` | `source_type`, `source_detail`, `target_table` | auto-increment |

### Projection stat storage

Projections use a `stat_json: dict[str, Any]` field rather than individual stat columns. This lets different projection systems (Steamer, ZiPS, custom models) carry different stat sets without schema changes. Query specific stats with SQLite's `json_extract()`.

### Projection accuracy

```python
from fantasy_baseball_manager.domain.projection_accuracy import (
    compare_to_batting_actuals,
    compare_to_pitching_actuals,
)

comparisons = compare_to_batting_actuals(projection, actual_batting_stats)
for c in comparisons:
    print(f"{c.stat_name}: projected={c.projected}, actual={c.actual}, error={c.error}")
```

Returns a `ProjectionComparison` for each stat present in both the projection's `stat_json` and the actual stats object.

---

## `repos/` — Repository layer

Every repository is defined as a `typing.Protocol` in `repos/protocols.py` and has a concrete `Sqlite*Repo` implementation. Repos accept a `sqlite3.Connection` via constructor injection.

### Available protocols

| Protocol | Key methods |
|---|---|
| `PlayerRepo` | `upsert`, `get_by_id`, `get_by_mlbam_id`, `get_by_bbref_id`, `search_by_name`, `all` |
| `TeamRepo` | `upsert`, `get_by_abbreviation`, `all` |
| `BattingStatsRepo` | `upsert`, `get_by_player_season`, `get_by_season` |
| `PitchingStatsRepo` | `upsert`, `get_by_player_season`, `get_by_season` |
| `ProjectionRepo` | `upsert`, `get_by_player_season`, `get_by_season`, `get_by_system_version` |
| `StatcastPitchRepo` | `upsert`, `get_by_pitcher_date`, `get_by_batter_date`, `get_by_game`, `count` |
| `ModelRunRepo` | `upsert`, `get`, `list`, `delete` |
| `LoadLogRepo` | `insert`, `get_recent`, `get_by_target_table` |

### Usage pattern

```python
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.domain.player import Player

conn = create_connection(":memory:")
repo = SqlitePlayerRepo(conn)

player_id = repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
player = repo.get_by_mlbam_id(545361)
```

All `upsert` methods use `INSERT ... ON CONFLICT ... DO UPDATE`, making re-imports idempotent. They commit immediately and return the row ID.

### Testing with protocols

Since repos are typed with `Protocol`, tests inject fakes via constructors rather than patching:

```python
class FakePlayerRepo:
    def upsert(self, player: Player) -> int: ...
    def get_by_id(self, player_id: int) -> Player | None: ...
    # ... satisfies PlayerRepo protocol
```

---

## `ingest/` — Data ingestion

### Data sources

All sources implement the `DataSource` protocol (`source_type`, `source_detail` properties and `fetch(**params) -> DataFrame`).

| Source | Wraps | `source_detail` |
|---|---|---|
| `ChadwickSource` | `pybaseball.chadwick_register` | `chadwick_register` |
| `FgBattingSource` | `pybaseball.fg_batting_data` | `fg_batting_data` |
| `FgPitchingSource` | `pybaseball.fg_pitching_data` | `fg_pitching_data` |
| `BrefBattingSource` | `pybaseball.batting_stats_bref` | `batting_stats_bref` |
| `BrefPitchingSource` | `pybaseball.pitching_stats_bref` | `pitching_stats_bref` |
| `StatcastSource` | `pybaseball.statcast` | `statcast` |
| `CsvSource` | `pandas.read_csv` | file path |

### Loaders

`StatsLoader` is the generic orchestrator. It fetches a DataFrame from a source, maps each row through a callable, upserts via a repo, and writes a `LoadLog` entry.

```python
from fantasy_baseball_manager.ingest.loader import StatsLoader

loader = StatsLoader(
    source=FgBattingSource(),
    repo=SqliteBattingStatsRepo(conn),
    load_log_repo=SqliteLoadLogRepo(conn),
    row_mapper=make_fg_batting_mapper(players),
    target_table="batting_stats",
)
log = loader.load(season=2024)
# log.status == "success", log.rows_loaded == N
```

`PlayerLoader` is a specialized variant for the player table.

### Column mappers

Mappers translate source-specific column names to domain objects. Factory functions close over a player lookup table to resolve external IDs to internal `player_id` values.

| Mapper | Input format | Output |
|---|---|---|
| `make_fg_batting_mapper(players)` | FanGraphs batting CSV | `BattingStats` |
| `make_fg_pitching_mapper(players)` | FanGraphs pitching CSV | `PitchingStats` |
| `make_bref_batting_mapper(players, season=)` | BBRef batting | `BattingStats` |
| `make_bref_pitching_mapper(players, season=)` | BBRef pitching | `PitchingStats` |
| `make_fg_projection_batting_mapper(players, season=, system=, version=)` | FG projection CSV | `Projection` |
| `make_fg_projection_pitching_mapper(players, season=, system=, version=)` | FG projection CSV | `Projection` |
| `chadwick_row_to_player(row)` | Chadwick register | `Player` |
| `statcast_pitch_mapper(row)` | Statcast DataFrame | `StatcastPitch` |

Rows with missing required fields (NaN player IDs, etc.) return `None` and are silently skipped by the loader.

### Chunked date loading (Statcast)

```python
from fantasy_baseball_manager.ingest.date_utils import chunk_date_range

chunks = chunk_date_range("2024-04-01", "2024-06-30", chunk_days=7)
for start, end in chunks:
    loader.load(start_dt=start, end_dt=end)
```

### End-to-end ingest recipe

```python
conn = create_connection("stats.db")
player_repo = SqlitePlayerRepo(conn)

# 1. Seed players from Chadwick register
player_loader = PlayerLoader(
    ChadwickSource(), player_repo, SqliteLoadLogRepo(conn), chadwick_row_to_player
)
player_loader.load()

# 2. Load batting stats
players = player_repo.all()
batting_loader = StatsLoader(
    FgBattingSource(),
    SqliteBattingStatsRepo(conn),
    SqliteLoadLogRepo(conn),
    make_fg_batting_mapper(players),
    "batting_stats",
)
batting_loader.load(season=2024)

# 3. Load projections from CSV
projection_loader = StatsLoader(
    CsvSource("steamer_batting_2025.csv"),
    SqliteProjectionRepo(conn),
    SqliteLoadLogRepo(conn),
    make_fg_projection_batting_mapper(players, season=2025, system="steamer", version="2025.1"),
    "projection",
)
projection_loader.load()

# 4. Load Statcast into separate database
sc_conn = create_statcast_connection("statcast.db")
statcast_loader = StatsLoader(
    StatcastSource(),
    SqliteStatcastPitchRepo(sc_conn),
    SqliteLoadLogRepo(conn),  # load log stays in stats.db
    statcast_pitch_mapper,
    "statcast_pitch",
)
for start, end in chunk_date_range("2024-06-01", "2024-06-30"):
    statcast_loader.load(start_dt=start, end_dt=end)
```

---

## `features/` — Feature DSL and dataset assembly

The feature system lets models declare the data they need as Python objects. The framework assembles (or reuses) a materialized SQLite table containing exactly those columns.

### Package structure

```
features/
    __init__.py          # Public API: SourceRef instances, exports
    types.py             # Feature, TransformFeature, FeatureSet, etc.
    protocols.py         # DatasetAssembler protocol
    sql.py               # SQL generation from feature declarations
    assembler.py         # SqliteDatasetAssembler (two-pass materialization)
    library.py           # Reusable feature bundles
    transforms/          # Python-based transform functions
        pitch_mix.py     # Pitch-type usage and velocity profiles
        batted_ball.py   # Exit velocity and launch angle metrics
```

### Feature types

There are three kinds of features, all usable together in a single `FeatureSet`:

| Type | Purpose | Handled by |
|---|---|---|
| `Feature` | SQL-expressible column lookups, lags, rolling aggregates, rate stats, computed values (age) | SQL generation pass |
| `DeltaFeature` | Difference between two `Feature` values (e.g. actual minus projected) | SQL generation pass |
| `TransformFeature` | Python callable applied to grouped raw rows — for features SQL cannot express | Transform pass |

### Declaring SQL features

```python
from fantasy_baseball_manager.features import batting, player, projection, delta
from fantasy_baseball_manager.features.types import FeatureSet, SpineFilter

feature_set = FeatureSet(
    name="batter_model_v1",
    features=(
        batting.col("hr").lag(1).alias("hr_1"),           # Previous season HR
        batting.col("avg").rolling_mean(3).alias("avg_3"), # 3-year rolling AVG
        batting.col("hr").per("pa").alias("hr_rate"),      # HR rate (HR / PA)
        player.age(),                                      # Age from birth_date
        projection.col("hr").system("steamer").alias("steamer_hr"),
        delta("hr_error",
              batting.col("hr").alias("actual_hr"),
              projection.col("hr").system("steamer").alias("proj_hr")),
    ),
    seasons=(2022, 2023),
    source_filter="fangraphs",
    spine_filter=SpineFilter(min_pa=200, player_type="batter"),
)
```

The builder API (`SourceRef.col()` → `FeatureBuilder` → `.lag()`, `.rolling_mean()`, `.per()`, `.system()`, `.alias()`) produces immutable `Feature` dataclasses. `SourceRef` instances (`batting`, `pitching`, `player`, `projection`, `statcast`) are pre-built in `features/__init__.py`.

### Declaring Python transforms

For features SQL cannot express — pitch-type distributions, batted-ball profiles, embeddings — use `TransformFeature`:

```python
from fantasy_baseball_manager.features.transforms import PITCH_MIX, BATTED_BALL

feature_set = FeatureSet(
    name="statcast_features",
    features=(
        batting.col("hr").lag(1).alias("hr_1"),
        PITCH_MIX,      # 12 outputs: ff_pct, ff_velo, sl_pct, sl_velo, ...
        BATTED_BALL,     # 5 outputs: avg_exit_velo, max_exit_velo, ...
    ),
    seasons=(2022, 2023),
    source_filter="fangraphs",
    spine_filter=SpineFilter(player_type="batter"),
)
```

A `TransformFeature` declares what data to load (`source`, `columns`), how to group it (`group_by`), a Python callable to run on each group (`transform`), and what columns it produces (`outputs`). The `RowTransform` protocol is simply:

```python
class RowTransform(Protocol):
    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]: ...
```

Custom transforms can be written by defining a function matching this signature and wrapping it in a `TransformFeature`.

### Built-in transforms

| Transform | Source | Outputs |
|---|---|---|
| `PITCH_MIX` | `statcast_pitch` | `ff_pct`, `ff_velo`, `sl_pct`, `sl_velo`, `ch_pct`, `ch_velo`, `cu_pct`, `cu_velo`, `si_pct`, `si_velo`, `fc_pct`, `fc_velo` |
| `BATTED_BALL` | `statcast_pitch` | `avg_exit_velo`, `max_exit_velo`, `avg_launch_angle`, `barrel_pct`, `hard_hit_pct` |

### Feature library

`features/library.py` provides pre-built feature bundles:

| Bundle | Contents |
|---|---|
| `STANDARD_BATTING_COUNTING` | 10 counting stats × 3 lags = 30 features |
| `STANDARD_BATTING_RATES` | avg, obp, slg, ops, woba, wrc_plus at lag 1 |
| `PLAYER_METADATA` | age |
| `STATCAST_PITCH_MIX` | `PITCH_MIX` transform |
| `STATCAST_BATTED_BALL` | `BATTED_BALL` transform |

### Content hashing and caching

`FeatureSet.version` is derived automatically by content-hashing all feature definitions, seasons, and filters. For `TransformFeature`, the hash includes the transform function's source code (via `inspect.getsource`) or an explicit `version` string if provided. Identical declarations produce the same version, enabling reuse.

### Two-pass materialization

The `SqliteDatasetAssembler` materializes a `FeatureSet` into a `ds_{id}` table:

**Pass 1 — SQL:** Generate SQL from `Feature` and `DeltaFeature` declarations (spine CTE, JOINs, SELECT expressions), execute as `CREATE TABLE ds_{id} AS SELECT ...`. `TransformFeature` items are skipped during SQL generation.

**Pass 2 — Transforms:** If any `TransformFeature` exists:
1. `ALTER TABLE` to add a `REAL` column for each transform output.
2. For each transform, query raw rows from the source table joined to the dataset spine.
3. For `Source.STATCAST`, lazily `ATTACH` the statcast DB and join through `player.mlbam_id` to map `statcast_pitch.batter_id` to the dataset's `player_id`.
4. Group rows by the transform's `group_by` key in Python.
5. Call the transform function for each group.
6. Batch `UPDATE` to write computed values back.

```python
from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler

assembler = SqliteDatasetAssembler(conn, statcast_path=Path("statcast.db"))

# Materialize (or reuse if already cached)
handle = assembler.get_or_materialize(feature_set)
# handle.table_name == "ds_42", handle.row_count == N

# Read the data
rows = assembler.read(handle)  # list[dict[str, Any]]

# Split into train/validation/holdout by season
splits = assembler.split(
    handle,
    train=[2020, 2021, 2022],
    validation=[2023],
    holdout=[2024],
)
# splits.train, splits.validation, splits.holdout are DatasetHandles
```

### SQL generation

Given features, `features/sql.py` builds a single parameterized query. Key rules:

- Each unique `(source, lag, system)` tuple gets one `LEFT JOIN` with a table alias like `b1`, `pi0`, `p`, `pr0`.
- Rolling aggregates use correlated subqueries: `(SELECT AVG(hr) FROM batting_stats WHERE player_id = spine.player_id AND season BETWEEN ...)`.
- Rate stats become `CAST(col AS REAL) / NULLIF(denom, 0)`.
- The spine is a CTE built from the relevant stat table, filtered by `SpineFilter`.
- Projection joins use `system` filters instead of `source` filters.

### DatasetAssembler protocol

```python
class DatasetAssembler(Protocol):
    def materialize(self, feature_set: FeatureSet) -> DatasetHandle: ...
    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle: ...
    def split(self, handle, train, validation=None, holdout=None) -> DatasetSplits: ...
    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]: ...
```

The `statcast_path` parameter is on the concrete `SqliteDatasetAssembler` only — the protocol stays unchanged.

---

## Schema

### `stats.db` tables

`player`, `team`, `batting_stats`, `pitching_stats`, `projection`, `feature_set`, `dataset`, `model_run`, `load_log`, `schema_version`

### `statcast.db` tables

`statcast_pitch`, `schema_version`

Indexes on `statcast_pitch`: `(pitcher_id, game_date)`, `(batter_id, game_date)`, `(game_pk)`.

---

## Key design decisions

- **No ORM.** `sqlite3` from the stdlib is used directly. SQLite's feature set (JSON functions, upsert, ATTACH, CTEs) covers all needs without ORM overhead.
- **Protocol-typed repos.** Every repo has a `Protocol` definition. Concrete implementations are injected, never imported directly by business logic. Tests use fakes that satisfy the same protocol.
- **Upsert everywhere.** All imports are idempotent. Re-running an import replaces existing data via `ON CONFLICT ... DO UPDATE`.
- **Frozen dataclasses.** Domain objects are immutable value types. No mutable state, no ORM base classes.
- **Mapper factories with closures.** Column mappers close over a player ID lookup table built at mapper creation time, avoiding repeated queries during row iteration.
- **Audit trail.** Every `StatsLoader.load()` call writes a `LoadLog` entry with timestamps, row count, and status (success/error).
- **Two-pass materialization.** SQL-expressible features run as a single `CREATE TABLE AS SELECT` for performance. Python transforms run as a second pass only when needed, keeping the common case fast.
- **Cross-database joins via ATTACH.** Statcast data lives in a separate DB file. The assembler lazily attaches it only when a `Source.STATCAST` transform is encountered, joining through `player.mlbam_id`.
