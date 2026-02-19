# Replace pybaseball & pylahman with Direct Sources

**Created:** 2026-02-18
**Status:** Active — Phase 6 complete
**Goal:** Incrementally replace `pybaseball` and `pylahman` with direct HTTP
fetchers backed by SQLite structured caching, then drop both dependencies.

## Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Decouple DataSource from pandas | ✅ Done |
| 2 | Replace Chadwick register source | ✅ Done |
| 3 | Replace Statcast & Sprint Speed sources | ✅ Done |
| 4 | Replace pylahman sources | ✅ Done |
| 5 | Replace FanGraphs sources | ✅ Done |
| 6 | Replace Baseball Reference sources | ✅ Done |
| 7 | Remove pybaseball & pylahman, clean up pandas | Not started |

**Remaining pybaseball usage:** `pybaseball` package dependency only —
no source classes remain in the ingest pipeline (Phase 7 removes the dep).

## Motivation

- `pybaseball` scrapes HTML (FanGraphs, Baseball Reference) and downloads large
  files (Chadwick ZIP, Statcast CSVs). These break when upstream HTML changes and
  cache data in opaque file formats (~/.pybaseball/).
- `pylahman` bundles static CSVs inside the package — data freshness depends on
  package releases.
- Both force a `pandas` dependency for the transport layer.
- We already have a working pattern for direct HTTP ingest (`MLBMinorLeagueBattingSource`,
  `MLBTransactionsSource`) using `httpx` + `tenacity`.
- SQLite tables are the natural cache: if data for a date/season already exists,
  skip the fetch.

## Architecture

### Current flow
```
pybaseball function → (HTTP + file cache) → pd.DataFrame
  → Loader.iterrows() → row_mapper → repo.upsert() → SQLite
```

### Target flow
```
Direct HTTP fetcher → raw response (CSV/JSON)
  → parse to list[dict] or domain objects
  → repo.upsert() → SQLite (serves as both store and cache)
```

Key changes:
- **DataSource protocol evolves:** `fetch()` returns `list[dict]` instead of
  `pd.DataFrame`. Loaders iterate dicts, not Series. Mappers accept `dict`.
- **Cache-aware fetching:** Sources accept a repo or connection to check what's
  already cached before making HTTP calls (e.g., statcast checks which dates
  are already loaded).
- **No pandas in the ingest pipeline.** `pd.read_csv` replaced by `csv.DictReader`
  or direct JSON parsing. pandas remains available for notebooks and analytics.

### Interaction with dataset management

The `datasets drop` / `datasets rebuild` commands only touch materialized
feature-set tables (`ds_N`). They never drop or modify the base ingest tables
(`batting_stats`, `statcast_pitch`, etc.). A `rebuild` re-runs `prepare`,
which queries the base tables — no re-fetch from upstream.

Today, if you delete the SQLite database and re-run `ingest`, pybaseball may
serve from its file cache (`~/.pybaseball/`), making re-ingestion fast. After
the migration, a full re-ingest with no database would require a full network
fetch. This is an acceptable tradeoff because:

1. The normal workflow (`datasets rebuild`) never re-fetches — it reads from
   the base tables that are already populated.
2. Cache-aware sources (Phase 3) make incremental `ingest` cheap — only
   missing dates/seasons are fetched.
3. The behavior is more predictable: one source of truth (SQLite) instead of
   two (SQLite + opaque file cache) that can get out of sync.

## Phases

### Phase 1 — Decouple DataSource from pandas ✅

Migrate the `DataSource` protocol and all loaders/mappers from `pd.DataFrame` /
`pd.Series` to `list[dict]` / `dict`. This is a pure refactor with no behavior
change — every existing source converts its DataFrame to `list[dict]` at the
boundary.

1. Change `DataSource.fetch()` return type from `pd.DataFrame` to `list[dict[str, Any]]`.
2. Update all loaders to iterate `list[dict]` instead of `df.iterrows()`.
3. Update all mapper signatures from `Callable[[pd.Series], T | None]` to
   `Callable[[dict[str, Any]], T | None]`.
4. Replace `pd.isna()` / `pd.notna()` in column_maps helpers with plain
   `value is None or value != value` (NaN check).
5. Each existing source's `fetch()` converts its DataFrame to
   `df.to_dict("records")` as a temporary bridge.
6. Update all tests.

After this phase, sources still use pybaseball/pylahman internally but the
rest of the pipeline is pandas-free.

### Phase 2 — Replace Chadwick register source ✅

The simplest replacement. pybaseball downloads a ZIP from GitHub containing
CSVs. Replace with a direct HTTP fetch.

- **Upstream:** `https://github.com/chadwickbureau/register/archive/refs/heads/master.zip`
- **Format:** ZIP containing `people-*.csv` files
- **Approach:** `httpx` GET → `zipfile` → `csv.DictReader` → `list[dict]`.
  Filter to rows with a non-null `key_mlbam`. Cache: if the `player` table
  already has entries with `mlbam_id`, skip re-fetch unless forced.
- Remove `ChadwickSource` and `from pybaseball import chadwick_register`.

### Phase 3 — Replace Statcast & Sprint Speed sources ✅

These are CSV endpoints on Baseball Savant — no HTML scraping needed.

**Statcast pitch data:**
- **Upstream:** `https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&game_date_gt={date}&game_date_lt={date}`
- **Approach:** Fetch day-by-day (pybaseball already does this). Use SQLite to
  track which dates are loaded (`load_log` or a dedicated tracking table).
  Only fetch missing dates. Parse CSV with `csv.DictReader`.
- **Cache:** Query `load_log` or `statcast_pitch` table for existing date
  coverage. On incremental ingest, only fetch new dates.

**Sprint speed:**
- **Upstream:** `https://baseballsavant.mlb.com/leaderboard/sprint_speed?year={year}&position=&team=&min={min_opp}&csv=true`
- **Approach:** Single CSV download per season. `httpx` GET → `csv.DictReader`
  → `list[dict]`.

Remove `StatcastSource`, `StatcastSprintSpeedSource`, and their pybaseball
imports.

### Phase 4 — Replace pylahman sources ✅

`pylahman` bundles static CSVs. Replace with the Lahman database CSV files
hosted on GitHub, or with equivalent MLB Stats API calls.

**People (biographical data):**
- Option A: Download Lahman `People.csv` from the Lahman GitHub repo directly.
- Option B: Use MLB Stats API `https://statsapi.mlb.com/api/v1/people/{id}`
  for individual lookups, or the roster endpoint for bulk.
- **Recommendation:** Option A for bulk historical, Option B for incremental
  current-season updates.

**Appearances (games by position):**
- Download Lahman `Appearances.csv` from GitHub, or use MLB Stats API
  `fielding` stat group.
- The existing `LahmanAppearancesSource` already melts wide columns into
  long format — keep that transform logic.

**Teams:**
- Download Lahman `Teams.csv` from GitHub, or use MLB Stats API
  `https://statsapi.mlb.com/api/v1/teams?sportId=1&season={year}`.
- **Recommendation:** MLB Stats API — it's always current and we already
  use `statsapi.mlb.com` for other sources.

Remove `pylahman` dependency entirely after this phase.

### Phase 5 — Replace FanGraphs sources ✅

Replaced `FgBattingSource` and `FgPitchingSource` with direct calls to the
FanGraphs JSON API (`/api/leaders/major-league/data`). Used Option B — the
JSON API returns structured data with native types, eliminating HTML scraping
and the `lxml` dependency.

Implementation details:
- New `fangraphs_source.py` with both source classes using `httpx` + `tenacity`
  constructor-injected retry (same pattern as `lahman_source.py`).
- `playerid` → `IDfg` remapping at the source boundary keeps mappers unchanged.
- Handles both bare JSON array and `{"data": [...]}` response formats.
- Source type/detail changed from `pybaseball`/`fg_*_data` to
  `fangraphs`/`batting`|`pitching`.
- Removed `FgBattingSource`, `FgPitchingSource`, `_translate_fg_params`, and
  `fg_batting_data`/`fg_pitching_data` imports from `pybaseball_source.py`.
- pybaseball now only used for Baseball Reference sources (`BrefBattingSource`,
  `BrefPitchingSource`).

### Phase 6 — Replace Baseball Reference sources ✅

Chose **Option B — drop bref as a stats source entirely.** Analysis showed
bref data was fully redundant: all projection models hardcode
`source_filter="fangraphs"`, so bbref rows were ingested but never consumed.
FanGraphs provides a strict superset of stats (adds woba, wrc_plus, war,
fip, xfip, bb_per_9, hld). The mlbID cross-referencing value was already
covered by the Chadwick register (Phase 2).

Implementation:
- Deleted `pybaseball_source.py` (contained `BrefBattingSource`,
  `BrefPitchingSource`) and its tests.
- Removed `_resolve_mlbam_id`, `make_bref_batting_mapper`,
  `make_bref_pitching_mapper` from `column_maps.py`.
- Simplified `IngestContainer.batting_source()` / `pitching_source()` —
  removed `name` parameter, always returns FanGraphs source.
- Removed `--source` CLI option from `ingest batting` / `ingest pitching`.
- Removed `"pybaseball"` from third-party logger suppression list.

### Phase 7 — Remove pybaseball & pylahman, clean up pandas

1. Remove `pybaseball` and `pylahman` from `pyproject.toml` dependencies.
2. Remove `pandas` from main dependencies if no longer used in `src/`
   (keep in dev dependencies for notebooks).
3. Remove `pyarrow` (pandas' Arrow backend) from locked deps.
4. Clean up any remaining `import pandas` in source files.
5. Verify all ingest commands still work end-to-end.

## Migration Strategy

- Each phase is independently shippable. After each phase, all existing CLI
  commands work identically — only the underlying data source changes.
- Existing tests use constructor-injected fakes, so replacing a real source
  doesn't require changing loader/mapper tests — only the source-level tests
  change.
- Add integration tests for new HTTP sources using recorded responses (or
  a thin mock HTTP layer) to avoid hitting real endpoints in CI.
