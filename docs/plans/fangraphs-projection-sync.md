# FanGraphs Projection Sync — Roadmap

## Goal

Fetch projections from FanGraphs' JSON API on demand, starting with Depth Charts (the industry-standard playing-time authority) and extending to Steamer and ZiPS for in-season updates. This replaces manual CSV downloads with a single CLI command that fetches, maps, and upserts projections into the existing `projection` table.

## Background

FanGraphs exposes an undocumented JSON API at:

```
https://www.fangraphs.com/api/projections?type={system}&stats={bat|pit}&pos=all&team=0&players=0
```

**Available `type` values:**

| Type | Description | Use Case |
|------|-------------|----------|
| `fangraphsdc` | Depth Charts | Playing-time authority (Steamer+ZiPS rates, RosterResource PT) |
| `steamer` | Steamer (preseason) | Rate projection baseline |
| `steamerr` | Steamer (rest-of-season) | In-season rate updates |
| `steameru` | Steamer (update) | In-season rate updates |
| `zips` | ZiPS (preseason) | Rate projection baseline |
| `rzips` | ZiPS (rest-of-season) | In-season rate updates |
| `rfangraphsdc` | Depth Charts (rest-of-season) | In-season PT + rate updates |

The JSON response is an array of objects with fields matching the existing CSV column map closely but not exactly. Key differences from CSV exports:

- Player ID fields: `playerid` (FG ID as string), `xMLBAMID` (MLBAM ID as int) — note `playerid` lowercase, unlike CSV's `PlayerId`
- Player name: `PlayerName` (not `Name`)
- Some fields use underscores vs slashes differently (e.g., `FPTS_G` vs `FPTS/G`)
- Batting responses include `~500` players; pitching `~450` players
- No `NameASCII` field

The existing CSV import path (`fbm import`) already handles FanGraphs projection data via `make_fg_projection_batting_mapper` / `make_fg_projection_pitching_mapper`. The `_resolve_fg_projection_id` function already reads `playerid` and `MLBAMID` — the API returns the same fields as `playerid` and `xMLBAMID`.

## Status

| Phase | Status |
|-------|--------|
| 1 — FanGraphs projection source | done (2026-03-07) |
| 2 — Sync CLI command | done (2026-03-07) |
| 3 — In-season refresh support | not started |

## Phase 1 — FanGraphs Projection Source

**Goal:** Create an `FgProjectionSource` that fetches projection data from the FanGraphs API, compatible with the existing `Loader` pipeline.

### Steps

1. **`FgProjectionSource`** — New class in `ingest/fangraphs_projection_source.py` implementing the `DataSource` protocol:
   - Constructor accepts `projection_type` (e.g., `"fangraphsdc"`, `"steamer"`, `"zips"`) and `stat_type` (`"bat"` or `"pit"`).
   - `fetch()` hits `https://www.fangraphs.com/api/projections` with query params `type`, `stats`, `pos=all`, `team=0`, `players=0`.
   - Uses `httpx.Client` with the same timeout/retry pattern as `FgStatsSource`.
   - Remaps `xMLBAMID` → `MLBAMID` and `playerid` → `PlayerId` so downstream mappers work unchanged.
   - `source_type` returns `"fangraphs"`, `source_detail` returns e.g. `"projections/fangraphsdc/bat"`.

2. **System name mapping** — Map API `type` values to the system names stored in the DB:
   - `fangraphsdc` → `"fangraphs-dc"`
   - `steamer` / `steamerr` / `steameru` → `"steamer"`
   - `zips` / `rzips` → `"zips"`

3. **Version convention** — Use date-based versions for synced projections: `"2026-03-07"` (sync date). This distinguishes multiple syncs over the season and avoids collisions with the year-based versions used for CSV imports.

4. **Tests:**
   - Unit test `FgProjectionSource` with a fake `httpx.Client` returning canned JSON.
   - Verify field remapping (`xMLBAMID` → `MLBAMID`, `playerid` stays as-is for the resolver).
   - Verify the source integrates with `Loader` + existing projection mappers end-to-end (using canned data).
   - Test error handling (HTTP errors, empty responses, malformed JSON).

### Acceptance Criteria

- `FgProjectionSource` implements `DataSource` and returns rows compatible with existing FG projection mappers.
- Canned integration test: source → mapper → `Loader` → repo produces correct `Projection` objects with `source_type="third_party"`.
- No network calls in tests (all HTTP interactions mocked).

## Phase 2 — Sync CLI Command

**Goal:** Add `fbm projections sync` command that fetches and imports projections from FanGraphs with a single invocation.

### Steps

1. **CLI command** — `fbm projections sync`:
   ```
   fbm projections sync <system> --season <year> [--version <version>]
   ```
   - `system` accepts: `fangraphs-dc`, `steamer`, `zips`
   - Maps system name back to API `type` value (e.g., `fangraphs-dc` → `fangraphsdc`)
   - `--version` defaults to today's date (`YYYY-MM-DD`)
   - Fetches both batting and pitching in one invocation
   - Uses existing `build_import_context` for DB access
   - Uses `FgProjectionSource` + existing mappers + `Loader` for each player type
   - Reports rows loaded per player type

2. **Multi-system shorthand** — Support `--all` flag to sync all three systems (fangraphs-dc, steamer, zips) in one command:
   ```
   fbm projections sync --all --season 2026
   ```

3. **Tests:**
   - CLI integration test with mocked HTTP verifying correct systems/versions stored.
   - Test `--all` flag dispatches to all three systems.
   - Test default version is today's date.

### Acceptance Criteria

- `fbm projections sync fangraphs-dc --season 2026` fetches and stores Depth Charts projections.
- `fbm projections sync --all --season 2026` fetches all three systems.
- `fbm projections lookup "Aaron Judge" --season 2026 --system fangraphs-dc` shows synced data.
- `fbm compare fangraphs-dc/<version> steamer/<version> --season <year> --stat pa --stat ip` works against synced data.

## Phase 3 — In-Season Refresh Support

**Goal:** Support rest-of-season projection variants and make it easy to re-sync periodically during the season.

### Steps

1. **Rest-of-season variants** — When syncing during the season (detected by date or explicit `--ros` flag), use rest-of-season API types:
   - `fangraphs-dc` → `rfangraphsdc`
   - `steamer` → `steamerr`
   - `zips` → `rzips`

   Store with the same system name but a version indicating ROS: `"2026-07-15-ros"`.

2. **`--ros` flag** — Explicit flag on `fbm projections sync`:
   ```
   fbm projections sync steamer --season 2026 --ros
   ```

3. **Refresh command** — `fbm projections refresh` as a convenience wrapper:
   ```
   fbm projections refresh --season 2026
   ```
   Equivalent to `fbm projections sync --all --season 2026 --ros` with today's date as version.

4. **Skill update** — Update the `fbm` CLI skill documentation to include the new `sync` and `refresh` commands.

5. **Tests:**
   - Verify `--ros` flag selects rest-of-season API types.
   - Verify `refresh` command works end-to-end.
   - Verify version string includes `-ros` suffix.

### Acceptance Criteria

- `fbm projections sync steamer --season 2026 --ros` fetches rest-of-season Steamer.
- `fbm projections refresh --season 2026` syncs all three systems with ROS variants.
- Repeated syncs with different dates create separate versions, preserving history.

## Out of Scope

- **Authentication.** The FanGraphs projections API appears to be unauthenticated. If this changes, authentication would be a separate concern.
- **Rate limiting / caching.** Each sync fetches ~6 requests (3 systems x 2 player types). This is well within reasonable usage. No caching layer needed.
- **Automated scheduling.** The CLI command is designed to be run manually or via cron/launchd. No built-in scheduler.
- **Historical backfill.** This roadmap covers current-season syncs. Importing historical CSVs continues to use `fbm import`.
- **Projection blending.** How synced Depth Charts data feeds into the ensemble or playing-time model is handled by existing infrastructure (the data lands in the `projection` table where other models already read from it).
