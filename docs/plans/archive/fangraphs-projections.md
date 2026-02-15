# FanGraphs Projection Integration Roadmap

## Goal

Import FanGraphs ZiPS and Steamer projection CSVs so they are queryable and evaluable on equal footing with first-party models. After this work, a user can import a CSV, look up any player's projected stats, evaluate a system's accuracy against actuals, and compare ZiPS vs Steamer vs Marcel (or any other system) — all through the same interfaces.

This roadmap assumes the model registry is complete (phases 1-5 of `model-registry.md`), so `fbm import`, `fbm eval`, `fbm compare`, `ProjectionEvaluator`, and `source_type` on the `projection` table are all available infrastructure.

---

## Context: What Already Exists

| Component | Status | Location |
|---|---|---|
| FG projection column mappers (batting + pitching) | **Needs update** — uses `playerid` column, actual CSVs use `PlayerId` | `ingest/column_maps.py` |
| `CsvSource` | Done | `ingest/csv_source.py` |
| `StatsLoader` pipeline | Done | `ingest/loader.py` |
| `Projection` domain object + `ProjectionRepo` | Done | `domain/projection.py`, `repos/projection_repo.py` |
| Per-player accuracy comparison | Done | `domain/projection_accuracy.py` |
| `projection.source_type` column | Done (model registry Phase 1) | Migration |
| `fbm import` CLI command | Done (model registry Phase 3) | CLI |
| `ProjectionEvaluator` (aggregate RMSE/MAE/correlation) | Done (model registry Phase 4) | Service |
| `fbm eval` / `fbm compare` CLI | Done (model registry Phase 4) | CLI |
| Projection lookup by player/stat | **Not done** | — |
| FanGraphs-specific import validation | **Not done** | — |
| End-to-end integration tests with ZiPS/Steamer data | **Not done** | — |

---

## CSV Format Reference

Source directory: `../fantasy-baseball/data/projections/`

### File naming convention

```
{system}_{year}_{batting|pitching}.csv
```

Examples: `zips_2026_batting.csv`, `steamer_2025_pitching.csv`

Season is encoded in the filename, not in a column.

### Available data

| System | Years | Batting files | Pitching files |
|---|---|---|---|
| ZiPS | 2014-2026 | 13 | 12 (none for 2014) |
| Steamer | 2015-2026 | 12 | 12 |

### Encoding

All files are UTF-8 with BOM (`\xef\xbb\xbf`). Must read with `encoding='utf-8-sig'` or equivalent.

### Batting CSV columns

All present in both ZiPS and Steamer, in this order:

```
Name, Team, G, PA, AB, H, 1B, 2B, 3B, HR, R, RBI, BB, IBB, SO, HBP, SF, SH,
GDP, SB, CS, AVG, BB%, K%, BB/K, OBP, SLG, wOBA, OPS, ISO, Spd, BABIP, UBR,
wSB, wRC, wRAA, wRC+, BsR, Fld, Off, Def, WAR, ADP, InterSD, InterSK,
IntraSD, Vol, Skew, Dim, FPTS, FPTS/G, SPTS, SPTS/G, P10, P20, P30, P40,
P50, P60, P70, P80, P90, TT10, TT20, TT30, TT40, TT50, TT60, TT70, TT80,
TT90, NameASCII, PlayerId, MLBAMID
```

**Key observations:**
- `PlayerId` is the FanGraphs ID — **not** `playerid`. The existing mapper uses `row.get("playerid")` which will not match. This is the primary compatibility issue.
- `MLBAMID` is the MLB Advanced Media ID — provides an alternative resolution path.
- `1B` (singles) is present — not in the current column map but derivable from `H - 2B - 3B - HR`.
- `GDP` is empty in Steamer, populated in ZiPS.
- `UBR` is empty in ZiPS.
- Percentile columns (`P10`–`P90`, `TT10`–`TT90`) are wOBA percentile distributions. Steamer populates these; ZiPS does not.
- Fantasy columns (`FPTS`, `SPTS`, `ADP`, etc.) and uncertainty columns (`InterSD`, `Vol`, `Skew`, `Dim`) are mostly empty in ZiPS, populated in Steamer.
- Steamer values are continuous (e.g., `141.36` G, `145.316` H). ZiPS values are integers for counting stats (e.g., `141` G, `144` H).

### Pitching CSV columns

All present in both ZiPS and Steamer, in this order:

```
Name, Team, W, L, QS, ERA, G, GS, SV, HLD, BS, IP, TBF, H, R, ER, HR, BB,
IBB, HBP, SO, K/9, BB/9, K/BB, HR/9, K%, BB%, K-BB%, AVG, WHIP, BABIP, LOB%,
GB%, HR/FB, FIP, WAR, RA9-WAR, ADP, InterSD, InterSK, IntraSD, Vol, Skew,
Dim, FPTS, FPTS/IP, SPTS, SPTS/IP, P10, P20, P30, P40, P50, P60, P70, P80,
P90, TT10, TT20, TT30, TT40, TT50, TT60, TT70, TT80, TT90, NameASCII,
PlayerId, MLBAMID
```

**Key observations:**
- No `xFIP` column — the existing pitching mapper has `"xFIP": "xfip"` but `_collect_stats` uses `row.get()` so this is handled gracefully (skipped when absent).
- `QS` (quality starts), `BS` (blown saves), `TBF` (total batters faced), `R` (runs), `HR/FB`, `GB%`, `LOB%`, `RA9-WAR` are present but not in the current column map.
- Same percentile/fantasy column pattern as batting.

### Player ID format

The `PlayerId` column contains two formats:

| Format | Example | Meaning | Resolution |
|---|---|---|---|
| Numeric string | `"15640"` | Established MLB player FanGraphs ID | Match against `player.fangraphs_id` |
| `sa`-prefixed string | `"sa3025154"` | Minor league / international player FanGraphs ID | Match against `player.fangraphs_id` (if stored) or fall back to `MLBAMID` → `player.mlbam_id` |

**Volume breakdown (2026 batting):**

| System | Total rows | Numeric `PlayerId` | `sa`-prefix `PlayerId` |
|---|---|---|---|
| ZiPS | 1,902 | 780 (41%) | 1,122 (59%) |
| Steamer | 4,192 | 979 (23%) | 3,213 (77%) |

Nearly all `sa`-prefix players also have a valid `MLBAMID` (3,213 of 3,214 in Steamer 2026 batting). The `MLBAMID` column is the reliable fallback for resolving minor league players.

### Sample rows

**ZiPS batting (integer counting stats):**
```
"Aaron Judge","NYY",141,624,500,144,76,25,1,42,107,115,113,...,"Aaron Judge","15640",592450
```

**Steamer batting (continuous counting stats):**
```
"Aaron Judge","NYY",141.36,634.656,509.805,145.316,77.8009,23.5922,1.2052,42.7178,...,"Aaron Judge","15640",592450
```

---

## Design Decisions

### System naming convention

Use lowercase system names matching FanGraphs nomenclature:
- `"zips"` for ZiPS projections
- `"steamer"` for Steamer projections

### Version convention

Use the projection season as the version string (e.g., `"2026"` for 2026 projections). The season is extracted from the filename. If FanGraphs publishes mid-season updates, use `"2026.2"` etc.

### Player resolution strategy

Two-step resolution, trying in order:

1. **`PlayerId` → `player.fangraphs_id`** — works for numeric FG IDs (established MLB players) and `sa`-prefix IDs if stored in our player table.
2. **`MLBAMID` → `player.mlbam_id`** — fallback for any row where step 1 fails. Covers nearly all minor league players.
3. **Skip** — if neither resolves, the row is skipped. Gaps are visible in `load_log` row counts.

This requires updating `_resolve_fg_projection_id` (or writing a new resolver) to:
- Read `PlayerId` instead of `playerid`
- Attempt FG ID lookup first (handling both numeric and `sa`-prefix)
- Fall back to `MLBAMID` lookup via `mlbam_id`

### Column mapping — what to store in `stat_json`

The existing mapper stores core counting stats and rate stats. We extend to include additional columns that are useful for analysis, organized into tiers:

**Tier 1 — Core stats (already mapped):**
`PA`, `AB`, `H`, `2B`, `3B`, `HR`, `RBI`, `R`, `SB`, `CS`, `BB`, `SO`, `HBP`, `SF`, `SH`, `GDP`, `IBB`, `AVG`, `OBP`, `SLG`, `OPS`, `wOBA`, `wRC+`, `WAR` (batting)
`W`, `L`, `G`, `GS`, `SV`, `HLD`, `H`, `ER`, `HR`, `BB`, `SO`, `ERA`, `IP`, `WHIP`, `K/9`, `BB/9`, `FIP`, `WAR` (pitching)

**Tier 2 — Additional stats to add:**
- Batting: `1B`, `G`, `ISO`, `BABIP`, `BB%`, `K%`, `wRC`, `wRAA`, `BsR`, `Off`, `Def`, `Fld`, `Spd`
- Pitching: `QS`, `BS`, `TBF`, `R`, `HBP`, `IBB`, `K/BB`, `HR/9`, `K%`, `BB%`, `BABIP`, `LOB%`, `GB%`, `HR/FB`, `RA9-WAR`, `AVG` (opponent)

**Not stored:** Fantasy points (`FPTS`, `SPTS`, `ADP`), percentile distributions (`P10`–`P90`, `TT10`–`TT90`), uncertainty metrics (`InterSD`, `Vol`, `Skew`, `Dim`), `NameASCII`. These are FanGraphs UI artifacts or fantasy-site-specific and don't contribute to projection evaluation.

### Encoding handling

Pass `encoding='utf-8-sig'` to `CsvSource` (via `**params` on `fetch()`). This strips the BOM transparently.

### Import is idempotent

Re-importing the same CSV for the same system/version upserts — existing rows are updated, not duplicated. This is handled by the `projection` table's unique constraint on `(player_id, season, system, version, player_type)`.

---

## Phases

### Phase 1 — Fix column mappers and player resolution

**Goal:** Update the FanGraphs projection mappers to match the actual CSV column names and add two-step player ID resolution.

- Update `_resolve_fg_projection_id` to read `PlayerId` (capital P, capital I) instead of `playerid`.
- Add MLBAMID fallback: accept an `mlbam_lookup` (built from `player.mlbam_id`) alongside the existing `fg_lookup`. Try FG ID first; if it fails (not found or non-numeric and not in table), try `MLBAMID`.
- Extend `_FG_BATTING_PROJECTION_COLUMNS` with Tier 2 batting columns: `{"1B": "singles", "G": "g", "ISO": "iso", "BABIP": "babip", "BB%": "bb_pct", "K%": "k_pct", "wRC": "wrc", "wRAA": "wraa", "BsR": "bsr", "Off": "off", "Def": "def_", "Fld": "fld", "Spd": "spd"}`.
- Extend `_FG_PITCHING_PROJECTION_COLUMNS` with Tier 2 pitching columns: `{"QS": "qs", "BS": "bs", "TBF": "tbf", "R": "r", "HBP": "hbp", "IBB": "ibb", "K/BB": "k_per_bb", "HR/9": "hr_per_9", "K%": "k_pct", "BB%": "bb_pct", "BABIP": "babip", "LOB%": "lob_pct", "GB%": "gb_pct", "HR/FB": "hr_per_fb", "RA9-WAR": "ra9_war", "AVG": "avg"}`.
- Update `make_fg_projection_batting_mapper` and `make_fg_projection_pitching_mapper` signatures to accept `players: list[Player]` (unchanged) but build both `fg_lookup` and `mlbam_lookup` internally.
- Tests: unit tests for the updated resolver with numeric IDs, `sa`-prefix IDs, MLBAMID fallback, and unresolvable rows.

### Phase 2 — Import validation and test fixtures

**Goal:** Verify the updated pipeline works end-to-end with realistic FanGraphs CSV data.

- Create sample CSV fixtures in `tests/fixtures/` using the exact column headers from the real files (all 71 batting columns, all 67 pitching columns). Use 5-10 rows of synthetic data per fixture, including:
  - Players with numeric `PlayerId` (MLB-established)
  - Players with `sa`-prefix `PlayerId` and valid `MLBAMID` (minor leaguers)
  - One player with `sa`-prefix `PlayerId` and no `MLBAMID` (should be skipped)
  - One player whose ID matches no one in the player table (should be skipped)
- Read CSVs with `encoding='utf-8-sig'` (include BOM in fixtures to match real files).
- Write integration tests exercising: `CsvSource` → FG projection mapper → `StatsLoader` → `ProjectionRepo` → query back. Cover:
  - ZiPS batting, ZiPS pitching, Steamer batting, Steamer pitching
  - `source_type = 'third_party'` set correctly
  - `load_log` records source file path, row count, status
  - Idempotent re-import (import same CSV twice, confirm no duplicates, stats updated)
  - Correct `system` and `version` values on stored projections
  - Tier 2 stats present in `stat_json` (e.g., `iso`, `babip`, `qs`)
  - Steamer continuous values stored as-is (no rounding)

### Phase 3 — Projection lookup service and CLI

**Goal:** Provide a way to query any player's projections across systems, so imported ZiPS/Steamer projections are as accessible as first-party model output.

- `ProjectionLookupService` — a thin service over `ProjectionRepo` and `PlayerRepo` that returns human-readable results:
  - `lookup(player_name: str, season: int, system: str | None = None) -> list[PlayerProjection]` — fuzzy-match player name, return projections optionally filtered by system.
  - `list_systems(season: int) -> list[SystemSummary]` — return all systems with projections for a season, with player counts and source type.
  - `PlayerProjection` dataclass: player name, system, version, source type, player type, and stat dict.
  - `SystemSummary` dataclass: system name, version, source type, player count (batters + pitchers).
- Player name resolution: look up by `name_last` (exact, case-insensitive), then disambiguate by `name_first` if multiple matches. Accept `"Last, First"` format for unambiguous lookup.
- `fbm projections lookup <player_name> --season 2026 [--system zips]` CLI command — prints a table of projected stats.
- `fbm projections systems --season 2026` CLI command — lists all available projection systems for a season.
- Tests with injected fakes for repos.

### Phase 4 — Evaluation and cross-system comparison

**Goal:** Verify the full evaluation and comparison workflow with third-party projections.

- Write integration tests that:
  1. Import ZiPS and Steamer projection fixtures for the same season.
  2. Load actual batting/pitching stats for the same players and season.
  3. Call `ProjectionEvaluator.evaluate()` for each system.
  4. Call `ProjectionEvaluator.compare()` across ZiPS, Steamer, and a first-party system (Marcel).
  5. Assert per-stat metrics (RMSE, MAE, correlation) are computed correctly.
  6. Assert `ComparisonResult` contains all three systems with correct `source_type`.
- Test edge cases:
  - Player has projection but no actuals — excluded from metrics.
  - Player has actuals but no projection — excluded.
  - Stat present in projection but absent in actuals source — stat skipped.
- Verify CLI output for `fbm eval` and `fbm compare` is identical in format regardless of whether the system is first-party or third-party.
- Support `--stats` filter to narrow comparison to specific stats (e.g., `--stats hr,avg,war`).

---

## Out of Scope

- **Automated FanGraphs scraping or API access.** This roadmap assumes the user has already downloaded CSVs. Automated download is a separate concern.
- **ATC, THE BAT, or other projection systems.** The infrastructure supports any FanGraphs-format CSV. Adding others is trivial — just import with `--system atc`.
- **Projection blending or ensemble models.** Combining multiple projection systems into a weighted average builds on this work but is a separate concern.
- **Percentile distributions and uncertainty.** The `P10`–`P90` and `TT10`–`TT90` columns in Steamer CSVs capture projection uncertainty. Storing and exposing these is valuable but deferred — the current `stat_json` model could accommodate them but the evaluation and lookup services would need uncertainty-aware features.
- **Visualization or dashboards.** Output is tabular CLI text. Charting or web UI is a separate concern.
