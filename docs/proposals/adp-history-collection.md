# ADP History Collection Plan

## Goal

Assemble a multi-year dataset of preseason consensus projections paired with NFBC ADP to train the ML valuation model described in `ml-valuation-function.md`. Target: ~1000 players/year across 3+ seasons.

## Current State

### What we have

- **Projection CSVs** in `data/projections/` for Steamer (2015-2026) and ZiPS (2014-2026), both batting and pitching
- **ADP column** exists in every CSV but is **only populated for 2026** (the current year)
- **FanGraphs API** at `/api/projections` returns an `ADP` field for the current year (NFBC ADP). The `season=` parameter is ignored — it always returns current-year data regardless of what you pass.
- **Existing ADP scrapers** for Yahoo and ESPN (Playwright-based), but these only provide current-season data with no historical access

### The gap

FanGraphs does not serve historical ADP. The NFBC ADP column on projection pages reflects the current preseason only. When you download a historical projection CSV with a FanGraphs+ membership, the ADP column is present but empty. There is no API parameter to request a past season's ADP.

This means historical ADP must be collected prospectively (captured each year going forward) or sourced from an external provider for backfill.

## Collection Strategy

### Phase 1: Capture 2026 (immediate)

The 2026 data is available right now. Two sources, both already working:

1. **FanGraphs API** — the `/api/projections` endpoint returns `ADP` alongside all projection stats. This is the cleanest source: projections and ADP in a single response, keyed by `playerid` (FanGraphs ID) and `xMLBAMID`.

2. **Existing CSV files** — `steamer_2026_batting.csv` and `steamer_2026_pitching.csv` already have populated ADP values.

**Action:** Build an ADP extraction path from the existing CSV files and/or the API response. Store as `data/adp/nfbc_{year}.csv` with columns: `playerid`, `mlbam_id`, `name`, `adp`, `player_type` (batter/pitcher).

### Phase 2: Automate annual capture (before 2027 draft season)

Each January-March, NFBC ADP stabilizes as drafts ramp up. We need to snapshot projections + ADP during this window before the regular season starts.

**Approach:** Add a CLI command or script that:

1. Fetches Steamer projections via the FanGraphs API (already implemented in `fangraphs.py`)
2. Extracts the `ADP` field from each player record (currently discarded by `_parse_batting` / `_parse_pitching`)
3. Writes a paired dataset: projection stats + ADP for each player, tagged with year and projection system
4. Stores the snapshot in `data/adp/nfbc_{year}.csv`

This runs once per preseason. After 2-3 years (2026-2028) we'll have enough data to train the valuation model.

### Phase 3: Historical backfill (2023-2025)

To train sooner, we need ADP for at least 2-3 prior years. The FanGraphs API and CSV exports don't provide this, so we need an external source.

#### Option A: FantasyData CSV downloads (recommended first attempt)

FantasyData (`fantasydata.com/mlb/fantasy-baseball-adp-rankings`) has a year filter showing 2023-2026 with CSV/XLS download buttons. If the downloads are accessible (free with registration), this gives us 3 years of NFBC-style ADP to pair with our existing Steamer/ZiPS projection CSVs.

**Join strategy:** Match players by name (normalized) between FantasyData ADP and FanGraphs projection CSVs. Use the Smart Fantasy Baseball Player ID Map (`smartfantasybaseball.com/tools/`) as a crosswalk for ambiguous matches. The ID map provides FanGraphs ID, MLBAM ID, ESPN ID, Yahoo ID, and more in a single spreadsheet.

#### Option B: FantasyPros historical ADP

FantasyPros aggregates ADP from Yahoo, ESPN, CBS, NFBC, and others with a year dropdown (2022-2026). Past years may require FantasyPros Premium (~$10-30/mo during draft season). If accessible, this is a higher-quality composite ADP since it averages across platforms.

**Join strategy:** Same name-based matching with ID map crosswalk.

#### Option C: Wayback Machine

Check `web.archive.org` for cached FanGraphs projections pages from February-March of 2023, 2024, 2025. If the API responses were cached, we get structured JSON with ADP for free. If HTML pages were cached, we can parse the projection tables.

**URLs to check:**
- `https://web.archive.org/web/2025*/https://www.fangraphs.com/api/projections?type=steamer*`
- `https://web.archive.org/web/2024*/https://www.fangraphs.com/api/projections?type=steamer*`
- `https://web.archive.org/web/2023*/https://www.fangraphs.com/api/projections?type=steamer*`

#### Option D: Re-download from FanGraphs member area

Log into FanGraphs with your membership and manually check whether the historical projections pages (e.g., "2025 Steamer Projections") display NFBC ADP from that year when accessed directly through the member-only historical section. The ADP values on historical pages may differ from what the API returns. If they do show historical ADP, re-export the CSVs for 2023-2025.

**This is worth checking first** since it would be the simplest path and the data would already be joined.

## Data Schema

### `data/adp/nfbc_{year}.csv`

| Column | Type | Source |
|--------|------|--------|
| `playerid` | str | FanGraphs player ID |
| `mlbam_id` | str | MLB Advanced Media ID |
| `name` | str | Player name |
| `team` | str | Team abbreviation |
| `adp` | float | NFBC Average Draft Position |
| `player_type` | str | `batter` or `pitcher` |
| `position` | str | Primary position |

### `data/adp/training/{year}_{system}.csv` (joined dataset for ML)

Projections + ADP in a single file, ready for model training:

| Column | Type | Notes |
|--------|------|-------|
| `playerid` | str | FanGraphs ID (join key) |
| `name` | str | Player name |
| `player_type` | str | `batter` or `pitcher` |
| `position` | str | Primary position |
| `adp` | float | Training target |
| `pa` or `ip` | int/float | Playing time |
| `hr`, `r`, `rbi`, `sb` | int | Batting counting stats |
| `obp`, `slg` | float | Batting rate stats |
| `w`, `sv`, `so` | int | Pitching counting stats |
| `era`, `whip` | float | Pitching rate stats |
| `year` | int | Season year |
| `system` | str | Projection system (steamer, zips) |

## Implementation Sketch

### Step 1: ADP extraction from existing data

Add an `adp` field to the CSV parsing path. The `CSVProjectionSource` in `projections/csv_source.py` currently ignores the ADP column. Either:
- Add an optional `adp` field to `BattingProjection` / `PitchingProjection`, or
- Create a separate ADP reader that parses just the `playerid` + `ADP` columns from projection CSVs

The second approach is cleaner since it keeps projection models focused on projections.

### Step 2: API ADP capture

Extend `FanGraphsProjectionSource` (or add a parallel fetcher) to capture the `ADP` field from API responses. Currently `_parse_batting` and `_parse_pitching` skip this field. The ADP data should be written alongside projections during the annual snapshot.

### Step 3: Build the joined training dataset

A script/command that:
1. Reads projection CSVs from `data/projections/{system}_{year}_{bat|pit}.csv`
2. Reads ADP from `data/adp/nfbc_{year}.csv`
3. Joins on `playerid`
4. Filters to players with both projections and ADP (removes undrafted players and unrojected players)
5. Outputs `data/adp/training/{year}_{system}.csv`

### Step 4: Validation

For 2026 (where we have both API ADP and CSV ADP), verify that the values match and the join is clean. This establishes confidence in the pipeline before applying it to backfilled historical data.

## Recommended Action Order

1. **Now:** Manually check FanGraphs member area for historical ADP (Option D) — this determines whether backfill is trivial or requires external sources
2. **Now:** Try FantasyData CSV downloads for 2023-2025 (Option A) — quick check if freely accessible
3. **Now:** Check Wayback Machine for cached API responses (Option C) — free, no account needed
4. **Soon:** Build the ADP extraction + joining pipeline (Steps 1-4)
5. **Jan 2027:** Run the automated capture for 2027 preseason data

## Expected Dataset Size

| Year | Batters | Pitchers | Total | Source |
|------|---------|----------|-------|--------|
| 2023 | ~500 | ~300 | ~800 | Backfill (external) |
| 2024 | ~500 | ~300 | ~800 | Backfill (external) |
| 2025 | ~500 | ~300 | ~800 | Backfill (external) |
| 2026 | ~500 | ~300 | ~800 | FanGraphs API/CSV |
| **Total** | | | **~3200** | |

Filtering to players with ADP < 500 (actually drafted) will reduce per-year counts but improve signal quality. The ~3200 sample target is close to the ~4500 in the ML proposal; adding ZiPS as a second projection system for the same years roughly doubles the samples.
