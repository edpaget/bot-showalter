# Data Layer Roadmap

## Goal

Build a SQLite-backed data layer for storing player statistics from multiple sources (pybaseball, CSV files, external APIs) and our own projections.

---

## SQLite File Strategy

Use **two databases**:

| File | Contents | Rationale |
|---|---|---|
| `stats.db` | Player master data, season-level batting/pitching stats, projections, data source metadata | Core data, moderate size, frequently queried together |
| `statcast.db` | Pitch-level and batted-ball-level Statcast data | Very large (millions of rows per season), queried separately, easy to rebuild from source |

Both live under a configurable data directory (default `~/.fbm/data/`). Keeping Statcast separate means we can blow it away and re-fetch without touching curated stats/projections, and it avoids bloating backups of the core data.

---

## Data Model

### `stats.db`

#### `player`

Canonical player record. One row per human being. All external IDs stored here so we can join across sources.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | Internal surrogate key |
| `name_first` | TEXT | |
| `name_last` | TEXT | |
| `mlbam_id` | INTEGER UNIQUE | MLB Advanced Media ID (used by Statcast, MLB API) |
| `fangraphs_id` | INTEGER UNIQUE | FanGraphs ID |
| `bbref_id` | TEXT UNIQUE | Baseball Reference ID (e.g. `troutmi01`) |
| `retro_id` | TEXT UNIQUE | Retrosheet ID |
| `bats` | TEXT | L/R/S |
| `throws` | TEXT | L/R |
| `birth_date` | TEXT | ISO 8601 date |
| `position` | TEXT | Primary position |

pybaseball ships a `playerid_lookup` / Chadwick register that maps all these IDs. We seed this table from that.

#### `team`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `abbreviation` | TEXT UNIQUE | e.g. `LAD`, `NYY` |
| `name` | TEXT | Full name |
| `league` | TEXT | `AL` / `NL` |
| `division` | TEXT | `E` / `C` / `W` |

#### `batting_stats`

One row per player per season per data source.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `player_id` | INTEGER FK → player | |
| `season` | INTEGER | e.g. 2025 |
| `team_id` | INTEGER FK → team | NULL if split-season |
| `source` | TEXT | `fangraphs`, `bbref`, `projection:steamer`, etc. |
| `pa` | INTEGER | Plate appearances |
| `ab` | INTEGER | At bats |
| `h` | INTEGER | Hits |
| `doubles` | INTEGER | 2B |
| `triples` | INTEGER | 3B |
| `hr` | INTEGER | Home runs |
| `rbi` | INTEGER | Runs batted in |
| `r` | INTEGER | Runs scored |
| `sb` | INTEGER | Stolen bases |
| `cs` | INTEGER | Caught stealing |
| `bb` | INTEGER | Walks |
| `so` | INTEGER | Strikeouts |
| `hbp` | INTEGER | Hit by pitch |
| `sf` | INTEGER | Sacrifice flies |
| `sh` | INTEGER | Sacrifice hits |
| `gdp` | INTEGER | Ground into double play |
| `ibb` | INTEGER | Intentional walks |
| `avg` | REAL | Batting average |
| `obp` | REAL | On-base percentage |
| `slg` | REAL | Slugging |
| `ops` | REAL | OPS |
| `woba` | REAL | Weighted on-base average |
| `wrc_plus` | REAL | wRC+ |
| `war` | REAL | WAR (version depends on source) |
| `loaded_at` | TEXT | ISO 8601 timestamp of when this row was imported |

UNIQUE constraint on `(player_id, season, source)` — re-importing replaces.

#### `pitching_stats`

Same shape, one row per player per season per source.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `player_id` | INTEGER FK → player | |
| `season` | INTEGER | |
| `team_id` | INTEGER FK → team | |
| `source` | TEXT | |
| `w` | INTEGER | Wins |
| `l` | INTEGER | Losses |
| `era` | REAL | |
| `g` | INTEGER | Games |
| `gs` | INTEGER | Games started |
| `sv` | INTEGER | Saves |
| `hld` | INTEGER | Holds |
| `ip` | REAL | Innings pitched |
| `h` | INTEGER | Hits allowed |
| `er` | INTEGER | Earned runs |
| `hr` | INTEGER | Home runs allowed |
| `bb` | INTEGER | Walks |
| `so` | INTEGER | Strikeouts |
| `whip` | REAL | |
| `k_per_9` | REAL | K/9 |
| `bb_per_9` | REAL | BB/9 |
| `fip` | REAL | Fielding Independent Pitching |
| `xfip` | REAL | Expected FIP |
| `war` | REAL | |
| `loaded_at` | TEXT | |

UNIQUE on `(player_id, season, source)`.

#### `projection`

Separating projections from actuals keeps things clean. One row per player per projection system per season.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `player_id` | INTEGER FK → player | |
| `season` | INTEGER | Season being projected |
| `system` | TEXT | `steamer`, `zips`, `atc`, `custom`, etc. |
| `version` | TEXT | Version identifier for the model/training run (e.g. `2025.1`, `v3-retrained`) |
| `player_type` | TEXT | `batter` or `pitcher` |
| `stat_json` | TEXT | JSON blob of projected stats (flexible schema per system) |
| `loaded_at` | TEXT | |

UNIQUE on `(player_id, season, system, version, player_type)`.

The `version` column lets us store multiple iterations of the same projection system for the same season, so we can compare how a model's accuracy changes across training runs. Using a JSON column for stats lets different projection systems carry different stat sets without needing to union incompatible column sets. Queries that need specific stats can use SQLite's `json_extract()`.

#### `feature_set`

Metadata about a named, versioned set of engineered features.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `name` | TEXT | e.g. `batter_rolling_v2`, `pitcher_stuff_plus` |
| `version` | TEXT | e.g. `1.0`, `2.1` |
| `description` | TEXT | What features are in this set, how they're derived |
| `source_query` | TEXT | SQL or description of the transformation that produced it |
| `created_at` | TEXT | |

UNIQUE on `(name, version)`.

#### `dataset`

A materialized snapshot: a specific feature set filtered/split for a purpose.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `feature_set_id` | INTEGER FK → feature_set | |
| `name` | TEXT | Human label, e.g. `batter_2020_2024_train` |
| `split` | TEXT | `train`, `validation`, `holdout`, `full`, etc. |
| `table_name` | TEXT | Actual SQLite table holding the rows, e.g. `ds_42` |
| `row_count` | INTEGER | |
| `seasons` | TEXT | JSON array of seasons included, e.g. `[2020,2021,2022,2023]` |
| `created_at` | TEXT | |
| `params_json` | TEXT | Any split parameters (random seed, split ratio, filter criteria) |

#### `ds_{id}` (dynamic tables)

One per dataset, created by the feature pipeline. Wide table with `player_id`, `season`, and all the engineered feature columns. Queryable with plain SQL. Dropped when no longer needed.

#### `model_run`

Links a projection version to the datasets it was trained/evaluated on.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `system` | TEXT | Matches `projection.system` |
| `version` | TEXT | Matches `projection.version` |
| `train_dataset_id` | INTEGER FK → dataset | |
| `validation_dataset_id` | INTEGER FK → dataset | NULL if not used |
| `holdout_dataset_id` | INTEGER FK → dataset | NULL if not used |
| `metrics_json` | TEXT | JSON blob of evaluation metrics (RMSE, MAE, etc.) |
| `created_at` | TEXT | |

This closes the loop: from a projection row, follow `system+version` → `model_run` → `dataset` → `ds_{id}` table to see the exact features and rows used.

#### `load_log`

Audit trail for every data import.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `source_type` | TEXT | `pybaseball`, `csv`, `api` |
| `source_detail` | TEXT | Function name, file path, or URL |
| `target_table` | TEXT | Which table was loaded |
| `rows_loaded` | INTEGER | |
| `started_at` | TEXT | |
| `finished_at` | TEXT | |
| `status` | TEXT | `success` / `error` |
| `error_message` | TEXT | NULL on success |

### `statcast.db`

#### `statcast_pitch`

One row per pitch. This table can hold 700k+ rows per season.

| Column | Type | Notes |
|---|---|---|
| `pitch_id` | INTEGER PK | Composite of game_pk + at_bat_number + pitch_number, or surrogate |
| `game_pk` | INTEGER | MLB game ID |
| `game_date` | TEXT | |
| `batter_id` | INTEGER | mlbam_id → joins to player table in stats.db via ATTACH |
| `pitcher_id` | INTEGER | mlbam_id |
| `pitch_type` | TEXT | FF, SL, CH, etc. |
| `release_speed` | REAL | mph |
| `release_spin_rate` | REAL | rpm |
| `pfx_x` | REAL | Horizontal movement |
| `pfx_z` | REAL | Vertical movement |
| `plate_x` | REAL | Horizontal plate location |
| `plate_z` | REAL | Vertical plate location |
| `zone` | INTEGER | Strike zone region |
| `events` | TEXT | NULL unless final pitch of PA (single, strikeout, etc.) |
| `description` | TEXT | ball, called_strike, swinging_strike, foul, hit_into_play, etc. |
| `launch_speed` | REAL | Exit velocity (NULL if not put in play) |
| `launch_angle` | REAL | (NULL if not put in play) |
| `hit_distance_sc` | REAL | |
| `barrel` | INTEGER | 0/1 |
| `estimated_ba_using_speedangle` | REAL | xBA |
| `estimated_woba_using_speedangle` | REAL | xwOBA |
| `loaded_at` | TEXT | |

Indexed on `(pitcher_id, game_date)`, `(batter_id, game_date)`, `(game_pk)`.

This is a subset of the ~90 columns Statcast provides. We store the columns we actually use and can always add more.

---

## Python Tooling

### Dependencies to add

| Package | Role |
|---|---|
| `pybaseball` | Data fetching: FanGraphs leaderboards, Baseball Reference, Statcast, Chadwick player ID register |
| `pandas` | Data transformation — pybaseball returns DataFrames, and pandas has `DataFrame.to_sql()` for bulk loading |

Both are runtime dependencies. No ORM — we use **`sqlite3` from the stdlib** directly. Reasons:

- Zero added dependency for DB access
- SQLite's feature set (JSON functions, `UPSERT`, `ATTACH`) is more than sufficient
- An ORM adds complexity without much value when the schema is stable and queries are straightforward
- Keeps the data layer easy to test with in-memory databases

### Architecture (module layout)

```
src/fantasy_baseball_manager/
    db/
        __init__.py
        connection.py      # Connection factory, ATTACH helper, migration runner
        pool.py             # Thread-safe connection pool
        schema.py           # SQL CREATE statements as strings, schema version
        migrations/         # Numbered .sql files for schema changes
    models/
        __init__.py
        player.py           # Player dataclass
        batting.py          # BattingStats dataclass
        pitching.py         # PitchingStats dataclass
        projection.py       # Projection dataclass
    repos/
        __init__.py
        protocols.py        # Protocol definitions for each repository
        player_repo.py      # PlayerRepo: CRUD for player table
        batting_repo.py     # BattingStatsRepo
        pitching_repo.py    # PitchingStatsRepo
        projection_repo.py  # ProjectionRepo
        load_log_repo.py    # LoadLogRepo
    features/
        __init__.py
        protocols.py        # FeatureStore Protocol
        feature_store.py    # Manages feature_set, dataset, ds_* tables
        model_run_repo.py   # ModelRunRepo: tracks which datasets a model used
    ingest/
        __init__.py
        protocols.py        # DataSource Protocol
        pybaseball_source.py  # Wraps pybaseball calls, returns DataFrames
        csv_source.py       # Reads CSVs into DataFrames
        api_source.py       # Fetches from HTTP APIs into DataFrames
        loader.py           # Orchestrator: DataSource → transform → repo.upsert, writes load_log
        column_maps.py      # Maps source column names → our schema column names
```

### Key design patterns

- **Dataclasses as models**: Plain `@dataclass` types for each entity. No ORM base class. The repos handle serialization to/from SQL.
- **Repository Protocol per table**: Each repo defines a Protocol (`PlayerRepo`, `BattingStatsRepo`, etc.) in `repos/protocols.py`. The concrete implementation uses `sqlite3`. Tests inject fakes that satisfy the same Protocol.
- **DataSource Protocol for ingest**: `ingest/protocols.py` defines a `DataSource` Protocol with a method like `fetch(params) -> pd.DataFrame`. Each source (pybaseball, CSV, API) implements this. The `Loader` orchestrator depends on the Protocol, not the concrete source.
- **Column mapping**: pybaseball, FanGraphs CSVs, and various APIs all use different column names for the same stats. `column_maps.py` centralizes the mapping from source columns to our canonical schema.
- **Upsert on import**: Use SQLite's `INSERT ... ON CONFLICT(...) DO UPDATE` so re-importing data is idempotent.
- **`ATTACH DATABASE`**: When querying Statcast data, attach `statcast.db` to a `stats.db` connection to join pitch-level data against the player table without duplicating player data.
- **Thread-safe connection pool**: `db/pool.py` provides a `ConnectionPool` backed by `queue.Queue`. Each thread checks out a connection, uses it, and returns it. The pool pre-opens connections with `check_same_thread=False` and `journal_mode=WAL` (write-ahead logging), which allows concurrent readers with a single writer — the sweet spot for SQLite multithreading. The pool exposes a `get()` / `release()` pair and a `connection()` context manager for automatic release. Repos accept a pool (or a single connection) via constructor injection, so single-threaded code and tests don't pay for pooling overhead.

### Schema migration strategy

Keep it simple: numbered SQL files (`001_initial.sql`, `002_add_barrel_column.sql`, etc.) in `db/migrations/`. A `schema_version` table tracks which migrations have run. The connection factory runs pending migrations on open. No heavy framework needed for a single-user local SQLite app.

---

## Phases

### Phase 1 — Database foundation
- `db/connection.py`: connection factory, migration runner, `ATTACH` helper
- `db/pool.py`: thread-safe connection pool with `queue.Queue`, WAL mode, context manager
- `db/schema.py`: initial schema SQL
- `db/migrations/001_initial.sql`
- `models/` dataclasses for player, batting, pitching, projection
- `repos/` with Protocol definitions and SQLite implementations for all tables
- Tests for all repos using in-memory SQLite

### Phase 2 — Player ID seeding
- `ingest/pybaseball_source.py`: wrap `playerid_reverse_lookup` / Chadwick register
- `ingest/loader.py`: orchestrator + load_log writing
- Seed `player` table from Chadwick register
- Tests with a fake DataSource

### Phase 3 — Batting & pitching stat ingest
- `ingest/pybaseball_source.py`: wrap `fg_batting_data`, `fg_pitching_data`, `batting_stats_bref`, `pitching_stats_bref`
- `ingest/csv_source.py`: generic CSV import
- `ingest/column_maps.py`: FanGraphs and BBRef column mappings
- Load historical season stats
- Tests

### Phase 4 — Statcast ingest
- Create `statcast.db` schema
- `ingest/pybaseball_source.py`: wrap `statcast` function (date-range based)
- Chunked loading (pybaseball recommends fetching ~1 week at a time)
- ATTACH-based queries joining Statcast to player
- Tests

### Phase 5 — Projections
- `ingest/csv_source.py` or `ingest/api_source.py` for importing projection CSVs (Steamer, ZiPS, etc.)
- `projection_repo.py` with JSON stat storage
- Queries that compare projections to actuals
- Tests

### Phase 6 — Feature store
- `features/protocols.py`: FeatureStore Protocol
- `features/feature_store.py`: create/register feature sets, materialize datasets as `ds_{id}` tables, manage train/validation/holdout splits
- `features/model_run_repo.py`: record which datasets a model run used, store evaluation metrics
- Schema migration for `feature_set`, `dataset`, `model_run` tables
- Queries to compare metrics across model versions
- Tests with in-memory SQLite

### Phase 7 — API source & extensibility
- `ingest/api_source.py`: generic HTTP JSON → DataFrame adapter
- Any additional stat sources (e.g. Savant leaderboards, Ottoneu prices)
- CLI or script entry points for running imports
