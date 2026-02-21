# ADP Integration & Value-Over-ADP Report Roadmap

Import Average Draft Position data and compare it against internal ZAR valuations. The value-over-ADP report is the single most actionable draft-day artifact — it identifies players the market is overvaluing or undervaluing relative to our projections, giving a concrete edge in every draft.

Historical FantasyPros ADP CSVs already exist in `../fantasy-baseball/data/adp/` for 2015-2023 and 2025-2026 (11 files, ~300-900 players each). The format is a composite multi-provider file with per-platform ADP columns and a composite average. The existing `Loader` + row mapper pattern and the `fbm import` command provide the template for ingest.

## Data shape

All files share a consistent core structure with provider columns varying by year:

```
"Rank","Player","Team","Positions","<provider1>","<provider2>",...,"AVG"
"1","Shohei Ohtani","LAD","SP,DH","","2","1","1","1","1","1.0"
```

- **Rank**: FantasyPros composite rank (integer, 1-indexed).
- **Player**: Full name, occasionally with suffix like `"(Batter)"` for two-way players.
- **Team**: MLB team abbreviation (may be empty for free agents).
- **Positions**: Comma-separated positions (e.g., `"CF,RF,DH"`, `"SP"`, `"1B,2B,SS"`).
- **Provider columns** (variable by year): Any subset of ESPN, Yahoo, CBS, NFBC, RTS, FT. Values are integer ADP picks or empty string when the provider doesn't list the player.
- **AVG**: Composite average ADP across reporting providers (float).

Key observations:
- Column order varies year to year, but the first 4 columns (Rank, Player, Team, Positions) and last column (AVG) are stable.
- Provider columns are snake-draft ADP (pick number), not auction dollar values.
- Player count per file: 400-900+ (2026 has 596 so far, mid-February).
- Empty-string provider values are common for players only listed on some platforms.
- Dual entries for two-way players (e.g., Ohtani as SP and Ohtani (Batter) as DH).

## Phase 1: ADP domain, storage, and FantasyPros ingest

Define the ADP domain model, persistence layer, and build a mapper tailored to the FantasyPros CSV format. Combines what would have been separate domain and ingest phases since the data shape is already known.

### Context

There is no ADP infrastructure in the codebase today. The FantasyPros CSV is the primary data source and its shape is well-understood from 11 years of historical files. The mapper can be purpose-built for this format rather than designed generically.

### Steps

1. Create `ADP` frozen dataclass in `src/fantasy_baseball_manager/domain/adp.py`:
   - `player_id: int` — internal player ID (resolved from name + team).
   - `season: int`
   - `provider: str` — `"fantasypros"` for the composite, or individual provider names (`"espn"`, `"yahoo"`, `"nfbc"`, `"cbs"`, `"rts"`, `"ft"`).
   - `overall_pick: float` — ADP pick number (float to support fractional composite averages).
   - `rank: int` — ordinal rank within this provider's list.
   - `positions: str` — comma-separated position string from the CSV (e.g., `"CF,RF,DH"`).
   - `as_of: str | None` — snapshot date (ISO format), for trend tracking. Defaults to None for historical data.
   - `id: int | None`, `loaded_at: str | None` — standard persistence fields.
2. Create `adp_repo.py` in `repos/` with:
   - `upsert(adp: ADP) -> int` — insert or update, returns row ID.
   - `upsert_batch(adps: list[ADP]) -> int` — batch upsert, returns count.
   - `find_by_season(season, provider?, as_of?) -> list[ADP]` — query with optional filters.
   - `find_by_player(player_id, season) -> list[ADP]` — all providers for one player.
3. Add SQL migration creating the `adp` table with UNIQUE constraint on `(player_id, season, provider, as_of)`. Use `as_of IS NULL` as the default for historical data without snapshot dates.
4. Create `adp_mapper.py` in `ingest/` with `make_fantasypros_adp_mapper(players, season, as_of?)`:
   - Resolve player names to internal IDs via name + team lookup (reuse existing player lookup pattern from `column_maps.py`).
   - Handle the `"(Batter)"` suffix for two-way players — strip it before name lookup, store the batter-only position list.
   - Parse the variable provider columns dynamically: any column that isn't Rank, Player, Team, Positions, or AVG is a provider column.
   - For each row, produce one `ADP` record per provider that has a non-empty value, plus one for the `"fantasypros"` composite (from the AVG column).
   - Handle empty-string provider values by skipping that provider for that player.
   - Return `list[ADP]` (multiple records per CSV row — one per provider plus composite).
5. Add `fbm ingest adp <csv-path> --season <year> [--as-of <date>]` CLI command. The provider is always `"fantasypros"` for the composite plus individual providers extracted from column headers.
6. Add `fbm ingest adp-bulk <directory> [--pattern "fantasypros_*.csv"]` command that ingests all matching files, extracting the season from the filename.
7. Write tests:
   - Repo: insert, upsert idempotency, query by season/provider.
   - Mapper: happy path, empty provider values, two-way player handling, unknown player logging.
   - CLI: end-to-end with fixture CSV.

### Acceptance criteria

- `ADP` dataclass is frozen and stored in the `adp` table.
- Single-file and bulk ingest both work.
- Each CSV row produces N+1 ADP records (one per non-empty provider column + one composite).
- Two-way players (e.g., "Shohei Ohtani (Batter)") are resolved correctly.
- Unmatched player names are logged with a count summary.
- Re-running the same import is idempotent.
- All 11 historical files can be loaded via bulk ingest.

## Phase 2: Value-over-ADP report

Build a report that joins ZAR valuations against ADP data to surface the biggest discrepancies — top buy targets and avoid-list players.

### Context

The goal is a single table: player name, position, our ZAR rank, ADP rank, rank delta, our dollar value. Sorted by rank delta, this tells you who the market is sleeping on and who it's overrating. The composite `fantasypros` ADP is the primary comparison target since it aggregates across platforms.

### Steps

1. Create `ValueOverADP` frozen dataclass: `player_id`, `player_name`, `position`, `zar_rank`, `zar_value`, `adp_rank`, `adp_pick`, `rank_delta` (adp_rank - zar_rank; positive = market undervalues), `provider`.
2. Create a service in `services/adp_report.py`:
   - `compute_value_over_adp(valuations, adp_records, top_n?)` — joins by player_id, computes deltas, returns sorted list.
   - Handle players in valuations but not in ADP (unranked by market — flag as potential sleepers if highly valued).
   - Handle players in ADP but not in valuations (market values them but we don't — flag as potential avoids).
3. Add `fbm report value-over-adp --season <year> --system <valuation-system> [--provider fantasypros] [--player-type batter|pitcher] [--position <pos>] [--top N]` CLI command.
4. Output columns: rank delta, player name, position, ZAR rank, ZAR $, ADP rank, ADP pick, delta.
5. Show two sections: "Buy targets" (positive delta, sorted descending) and "Avoid list" (negative delta, sorted ascending).
6. Write tests with synthetic valuation and ADP data.

### Acceptance criteria

- Report shows rank delta per player, split into buy/avoid sections.
- Players in valuations but missing from ADP are flagged as sleepers.
- Players in ADP but missing from valuations are flagged as avoids.
- Filters by player type, position, and top-N work correctly.
- Output matches the tabular format used by existing reports.

## Phase 3: Historical ADP evaluation

Evaluate how well ADP predicts actual season outcomes, and compare ADP predictive power against our projection systems.

### Context

With 11 years of historical ADP data, we can measure ADP's accuracy as a projection system — correlating pre-season ADP rank with end-of-season fantasy value. This tells us (a) how efficient the market is, and (b) where our projections add value above the market consensus.

### Steps

1. Implement `evaluate_adp_accuracy(adp_records, actuals, league_settings, season)` that:
   - Converts ADP rank to an expected dollar value (using the rank-to-value curve from ZAR).
   - Compares against actual end-of-season values.
   - Computes RMSE, rank correlation (Spearman rho), and top-N precision.
2. Add `fbm report adp-accuracy --season <year> [--compare-system <system>]` that shows ADP accuracy alongside a projection system's accuracy for direct comparison.
3. Add `fbm report adp-accuracy --season 2015:2023` for multi-year aggregate accuracy.
4. Write tests with known ADP and actuals data.

### Acceptance criteria

- ADP accuracy metrics (RMSE, rank correlation) are computed correctly.
- Side-by-side comparison with a projection system shows where each is stronger.
- Multi-year aggregate mode works across all available historical seasons.

## Phase 4: Automated ADP fetching

Automate periodic fetching of fresh FantasyPros ADP data during draft season via headless browser scraping.

### Context

FantasyPros ADP data is rendered via JavaScript on `https://www.fantasypros.com/mlb/adp/overall.php`. There is no public API or CSV export endpoint. The data table must be extracted via a headless browser. During draft season (Feb-March), ADP shifts meaningfully week to week — automated fetching with snapshot dates enables the trend tracking report.

### Steps

1. Add `playwright` as an optional dependency (dev/draft extra).
2. Create `src/fantasy_baseball_manager/ingest/fantasypros_adp_source.py` implementing the `DataSource` protocol:
   - Launch a headless Chromium browser via Playwright.
   - Navigate to `https://www.fantasypros.com/mlb/adp/overall.php`.
   - Wait for the ADP data table to render.
   - Extract the table HTML and parse it into row dicts matching the CSV column structure.
   - Return `list[dict[str, Any]]` compatible with the existing FantasyPros ADP mapper.
3. Add `fbm ingest adp-fetch --season <year> [--as-of <date>]` CLI command that fetches live data and ingests it in one step. Defaults `--as-of` to today.
4. Add `fbm ingest adp-fetch --save-csv <path>` option to also write the fetched data to a CSV file (for archival, matching the existing file format in `../fantasy-baseball/data/adp/`).
5. Write tests using a saved HTML fixture to verify table extraction without hitting the live site.

### Acceptance criteria

- `fbm ingest adp-fetch` downloads and ingests current ADP data.
- Fetched data matches the column structure of existing FantasyPros CSVs.
- `--save-csv` produces a file loadable by `fbm ingest adp`.
- Snapshot date (`as_of`) is stored, enabling trend tracking.
- Fails gracefully with a clear error if FantasyPros changes their page structure.

## Phase 5: ADP trend tracking

Track how ADP shifts over the offseason and surface the biggest movers.

### Context

ADP moves significantly between January and draft day. Players recovering from injury, changing teams, or having strong spring trainings see large ADP swings. Tracking these trends helps identify players whose stock is rising (draft them before the market catches up) or falling (wait for better value). This requires multiple snapshots stored via the `as_of` field from Phase 1.

### Steps

1. Implement `compute_adp_movers(season, provider, current_as_of, previous_as_of)` that:
   - Fetches two ADP snapshots for the same provider/season.
   - Computes rank delta (previous rank - current rank; positive = rising).
   - Returns sorted list of `ADPMover` frozen dataclass: `player_name`, `position`, `current_rank`, `previous_rank`, `rank_delta`, `direction`.
2. Add `fbm report adp-movers --season <year> [--window <days>] [--provider fantasypros]` command.
   - `--window` selects the comparison snapshot closest to N days before the latest snapshot.
   - Default window: 14 days.
3. Show risers (top 20) and fallers (bottom 20) in separate sections.
4. Write tests with two synthetic snapshots verifying mover detection.

### Acceptance criteria

- Movers report correctly identifies risers and fallers.
- Window selection finds the closest available snapshot.
- Players appearing in only one snapshot are noted as "new" or "dropped."

## Ordering

Phases 1 → 2 are the critical path — get ADP into the database and produce the value-over-ADP report. Phase 3 (historical evaluation) is independently valuable and can be done anytime after phase 1 since we have 11 years of historical CSVs. Phase 4 (auto-fetch) is needed before phase 5 (trend tracking) since trend tracking requires multiple snapshots. Phase 4 can also proceed in parallel with phases 2-3.

Suggested priority: **1 → 2 → 3 → 4 → 5**. Phases 1-2 are the pre-draft essentials. Phase 3 gives confidence in the value-over-ADP report. Phases 4-5 are draft-season enhancements.
