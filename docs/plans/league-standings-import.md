# League Standings Import Roadmap

Import team-level season category totals from Yahoo Fantasy leagues. This is infrastructure for SGP (Standings Gain Points) denominator computation, which in turn enables independent valuation targets that break the circular ZAR$-vs-ZAR$ evaluation problem diagnosed in `notebooks/pitcher_valuation_diagnosis.ipynb`.

The redraft league (ID 91300 in `fbm.local.toml`) has 4 seasons (2021–2025) of H2H category data with the same stat categories as the current league config. Walking the Yahoo league renewal chain discovers prior-season league keys, and the standings endpoint provides per-team category totals for each season.

## Status

| Phase | Status |
|-------|--------|
| 1 — Standings API pipeline | done (2025-03-09) |
| 2 — Historical league discovery and bulk import | done (2025-03-09) |

## Phase 1: Standings API pipeline

Build the end-to-end pipeline for fetching, parsing, storing, and displaying standings data for a single league-season.

### Context

The Yahoo Fantasy client already has a `get_standings(league_key)` method (added to `yahoo/client.py`), but there is no domain model, source class, migration, repo, or CLI command to parse, store, or use the response. The existing Yahoo integration follows a consistent pattern — source class parses API responses into domain objects, repo persists them to SQLite, CLI command orchestrates the flow — and this phase follows that same pattern.

The standings endpoint returns team metadata, win/loss records, and per-team stat totals for the season. We need the **per-team category totals** (HR, R, RBI, OBP, SB, ERA, WHIP, SO, W, SV+HLD) — these are the raw numbers needed for SGP computation.

### Steps

1. **Probe the API response.** Call `get_standings` on a known league key (e.g., the 2025 redraft) and capture the raw JSON structure. Document the path to per-team category totals and final standings rank.
2. **Create domain model.** Add a `TeamSeasonStats` dataclass to `domain/` with fields: `team_key`, `league_key`, `season`, `team_name`, `final_rank`, `stat_values` (dict mapping category key → float, e.g., `{"hr": 250.0, "era": 3.45}`). Keep it simple — one row per team per season with all category totals in a JSON column.
3. **Create migration 028.** Add a `yahoo_team_season_stats` table with columns: `id`, `team_key`, `league_key`, `season`, `team_name`, `final_rank`, `stat_values_json` (TEXT), with a unique constraint on `(team_key, league_key, season)`.
4. **Create repo.** `SqliteYahooTeamStatsRepo` with `upsert(stats)`, `get_by_league_season(league_key, season)`, and `get_all_seasons(league_key)`.
5. **Create source class.** `YahooStandingsSource` in `yahoo/standings_source.py` — takes a `YahooFantasyClient`, calls `get_standings`, parses the response into `list[TeamSeasonStats]`. Follow the same parsing patterns as `YahooLeagueSource`.
6. **Wire into YahooContext.** Add `yahoo_team_stats_repo` to the `YahooContext` dataclass and `build_yahoo_context()`.
7. **Add CLI command.** `yahoo standings --league NAME --season YEAR` — fetches standings, persists to DB, displays a table of team × category totals with final rank. Follow the same error handling pattern as `yahoo sync`.
8. **Add tests.** Unit tests for source parsing (mock API response), repo round-trip, and CLI smoke test.

### Acceptance criteria

- `yahoo standings --league redraft --season 2025` fetches and displays team category totals for the 2025 redraft league.
- Data is persisted to `yahoo_team_season_stats` and retrievable via the repo.
- Source class correctly maps Yahoo's stat IDs to category keys matching `fbm.toml` batting/pitching category keys.

---

## Phase 2: Historical league discovery and bulk import

Walk the Yahoo league renewal chain to discover all prior-season league keys for a league, then bulk-import standings for every discovered season.

### Context

Yahoo Fantasy leagues renew each season, creating a new league key. The `renew` field on `yahoo_league` points to the prior season's league in `"game_key_league_id"` format. The existing `_resolve_prior_league_key()` in the Yahoo CLI walks one step back, but we need to walk the full chain (2025→2024→2023→2022→2021) to discover all historical league keys.

The current renewal chain resolution requires each league to already be synced to the DB (to read its `renew` field). To walk multiple steps, we need to sync each league as we discover it, then read its `renew` to find the next one.

### Steps

1. **Build a chain walker.** Create a function `walk_renewal_chain(client, league_source, league_repo, team_repo, league_key) → list[(league_key, season)]` that:
   - Starts from the given league key
   - Syncs its metadata if not already in the DB
   - Reads its `renew` field to find the prior season
   - Recurses until `renew` is None or the prior-season API call fails (league too old)
   - Returns the full list of `(league_key, season)` pairs, newest first
2. **Add `yahoo standings --league NAME --all-seasons`.** Walk the renewal chain from the current season's league, then import standings for every discovered season. Display a summary table showing seasons imported and team counts.
3. **Handle edge cases.** The renewal chain may break if the league changed formats or the Yahoo API doesn't have old data. Log warnings and continue with whatever seasons are available.
4. **Add a `--since YEAR` flag** to limit how far back the chain walker goes (default: no limit).
5. **Add tests.** Test the chain walker with mocked API responses (3-season chain), test the `--all-seasons` flag.

### Acceptance criteria

- `yahoo standings --league redraft --all-seasons` discovers and imports standings for all available seasons of the redraft league (expected: 2021–2025).
- Each season's standings are persisted with correct league keys and team names.
- The renewal chain is walked automatically — the user only needs to provide the current league name.
- `yahoo standings --league redraft --all-seasons --since 2023` limits import to 2023–2025.

## Ordering

- **Phase 1** has no dependencies. The `get_standings` client method already exists.
- **Phase 2** depends on phase 1 (uses the standings source and repo). It also requires the league to be synced at least once (`yahoo sync --league redraft`) so the starting league key is known.
- **Downstream consumers**: The [Evaluation Framework](evaluation-framework.md) phase 2 (independent targets) and [Valuation Reform](valuation-reform.md) phase 1 (SGP denominators) depend on this data being available.
