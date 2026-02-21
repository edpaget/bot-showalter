# NPB/KBO Ingest Module Roadmap

Ingest historical batting and pitching stats from FanGraphs' international leaderboards (NPB and KBO) using a custom HTTP client, following the same patterns as the existing `FgStatsSource`. This data feeds two downstream use cases: (1) scouting international free agents by viewing their league-native stats, and (2) translating NPB/KBO performance into MLB-equivalent projections via the existing MLE engine — the same way we translate MiLB stats today.

FanGraphs serves international data via the same JSON API used for MLB stats. The endpoint `https://www.fangraphs.com/api/leaders/international/{league}/data` accepts query parameters for league, season, stat type (bat/pit), and position. It returns player-level season stats including advanced metrics (wRC+, wOBA, FIP, etc.). Historical coverage: NPB 2019–2025 (~7 seasons), KBO 2002–2025 (~23 seasons). Player IDs use the `sa#######` format (FanGraphs international ID) plus a `minormasterid` field.

## Phase 1: FanGraphs international source

Build the HTTP source that fetches raw JSON from the FanGraphs international API for both NPB and KBO.

### Context

The existing `FgStatsSource` hits `/api/leaders/major-league/data`. The international endpoint follows the exact same pattern at `/api/leaders/international/{league}/data` with identical query parameter semantics (`stats`, `season`, `season1`, `qual`, `pos`, `ind`, `type`, `team`, `pageitems`, `pagenum`). The response is a JSON array of player stat dicts with the same key structure (but using `playerids` instead of `playerid`, and including `JName`/`KName` for native-script names). A single new source class can handle both leagues by parameterising the league slug.

### Steps

1. Create `FgInternationalSource` in `src/fantasy_baseball_manager/ingest/fg_international_source.py`:
   - Constructor takes `league: str` (`"npb"` or `"kbo"`), `stat_type: str` (`"bat"` or `"pit"`), plus the standard optional `client` and `retry` parameters.
   - Base URL: `https://www.fangraphs.com/api/leaders/international/{league}/data`.
   - `fetch(season=..., qual=...)` builds query params matching the API's expectations (`stats`, `season`, `season1`, `ind`, `qual`, `type`, `pageitems`, `lg`, `pos`, `team`).
   - `source_type` returns `"fangraphs_international"`, `source_detail` returns `"{league}_{batting|pitching}"`.
   - Remap `playerids` → `fg_intl_id` in the response (analogous to existing `playerid` → `IDfg` remap).
2. Write tests in `tests/ingest/test_fg_international_source.py`:
   - Use `FakeTransport` returning canned JSON for NPB batting, NPB pitching, KBO batting, KBO pitching.
   - Test retry behaviour with `FailNTransport`.
   - Verify the source satisfies the `DataSource` protocol.
   - Parametrize over league × stat_type.

### Acceptance criteria

- `FgInternationalSource("npb", "bat").fetch(season=2024)` returns parsed JSON rows.
- `FgInternationalSource("kbo", "pit").fetch(season=2024)` returns parsed JSON rows.
- Invalid league or stat_type raises `ValueError`.
- Retry and error handling match existing source behaviour.
- All tests pass without network access.

## Phase 2: International stats domain and persistence

Define domain models and repos for storing NPB/KBO stats, plus column mappers to convert API rows into domain objects.

### Context

The FanGraphs international API returns stats that are a superset of our existing `BattingStats`/`PitchingStats` fields, but the player ID system is different (FanGraphs international `sa#######` IDs vs. our internal numeric IDs). We need a way to register international players in our player table and map their stats. The API provides `Name`, `Team`, `playerids` (FanGraphs intl ID), and `minormasterid` — these become the primary identifiers.

A new `InternationalBattingStats` domain model (parallel to `MinorLeagueBattingStats`) captures the league/team context needed for MLE translation. Pitching stats follow the same pattern.

### Steps

1. Create `InternationalBattingStats` frozen dataclass in `src/fantasy_baseball_manager/domain/international_batting_stats.py`:
   - Fields: `player_id`, `season`, `league` (`"npb"` / `"kbo"`), `team`, `g`, `pa`, `ab`, `h`, `doubles`, `triples`, `hr`, `r`, `rbi`, `bb`, `so`, `sb`, `cs`, `avg`, `obp`, `slg`, `age`, plus optional `hbp`, `sf`, `sh`, `ibb`, `gdp`.
   - Include FanGraphs advanced metrics: `woba`, `wrc_plus`, `iso`, `babip`, `spd`.
   - Standard `id`, `loaded_at` optional fields.
2. Create `InternationalPitchingStats` frozen dataclass in `src/fantasy_baseball_manager/domain/international_pitching_stats.py`:
   - Fields: `player_id`, `season`, `league`, `team`, `w`, `l`, `era`, `g`, `gs`, `sv`, `hld`, `ip`, `h`, `er`, `hr`, `bb`, `so`, `whip`, `k_per_9`, `bb_per_9`, `fip`, plus optional `cg`, `sho`, `bs`, `tbf`, `ibb`, `hbp`, `wp`, `bk`, `k_pct`, `bb_pct`, `babip`, `lob_pct`.
   - Standard `id`, `loaded_at` optional fields.
3. Create repos in `src/fantasy_baseball_manager/repos/`:
   - `InternationalBattingStatsRepo` protocol and `SqliteInternationalBattingStatsRepo` with upsert/query methods (unique on `player_id + season + league + team`).
   - `InternationalPitchingStatsRepo` protocol and `SqliteInternationalPitchingStatsRepo`.
4. Add SQL migrations for `international_batting_stats` and `international_pitching_stats` tables.
5. Create column mappers in `src/fantasy_baseball_manager/ingest/column_maps.py`:
   - `make_fg_international_batting_mapper(players, league)` — resolves player by FanGraphs international ID, maps API row → `InternationalBattingStats`.
   - `make_fg_international_pitching_mapper(players, league)` — same for pitching.
   - Player resolution: look up by `fg_intl_id` (the `sa#######` string) via a new lookup on the `Player` model. Falls back to name matching if needed.
6. Write tests for domain models, repos, and mappers.

### Acceptance criteria

- Domain objects are frozen dataclasses with all NPB/KBO-relevant fields.
- Repos support upsert (idempotent re-import) and query by season/league.
- Mappers convert raw API dicts to domain objects, returning `None` for unresolvable players.
- Migrations create correct table schemas.
- All tests pass.

## Phase 3: Player registration for international players

Handle the chicken-and-egg problem: to map stats we need `player_id`, but international players may not exist in the player table yet.

### Context

Our existing player table is keyed on MLB identifiers (MLBAM, FanGraphs numeric, BBRef, Retro). International-only players have none of these — they have a FanGraphs international ID (`sa#######`) and a `minormasterid`. Players who later move to MLB will have entries in both systems. We need to either (a) add international ID fields to the `Player` model or (b) maintain a separate mapping table. Option (a) is simpler and enables future cross-referencing when an NPB/KBO player signs with an MLB team.

### Steps

1. Add `fg_intl_id: str | None = None` field to the `Player` dataclass.
2. Add the column to the `players` table via migration.
3. Update `SqlitePlayerRepo` to persist and query by `fg_intl_id`.
4. Create an international player registration function that:
   - Takes a raw API row with `playerids`, `Name`, `Team`, `Age`.
   - Checks if a player with that `fg_intl_id` already exists.
   - If not, creates a new `Player` with `name_first`/`name_last` parsed from `Name`, `fg_intl_id` set, and other MLB IDs as `None`.
   - If yes, returns the existing player.
   - For players with a `minormasterid` that maps to an existing MLB player, links them by updating the existing player's `fg_intl_id`.
5. Integrate into the ingest pipeline: before mapping stats, run player registration for all rows in the API response.
6. Write tests for registration, linking, and idempotency.

### Acceptance criteria

- International-only players are created in the player table with `fg_intl_id`.
- Players who later appear in MLB are linked (not duplicated) when `minormasterid` matches.
- Re-importing the same season does not create duplicate players.
- All existing tests still pass (adding an optional field should be backward-compatible).

## Phase 4: CLI commands and ingest pipeline

Wire the source, mappers, and repos together via CLI commands for batch importing.

### Context

With the source (phase 1), domain/persistence (phase 2), and player registration (phase 3) in place, this phase wires them together using the existing `Loader` pattern and adds CLI commands to trigger imports.

### Steps

1. Add `fbm ingest international --league npb|kbo --stat-type bat|pit --season <year> [--qual 0]` CLI command.
2. Add `fbm ingest international-bulk --league npb|kbo --seasons 2019:2025` for batch import across multiple seasons.
3. The pipeline: fetch → register players → map → upsert stats.
4. Log summary: players registered, stats rows imported, unmatched rows.
5. Write integration tests with canned API responses.

### Acceptance criteria

- `fbm ingest international --league npb --stat-type bat --season 2024` fetches and stores NPB batting stats.
- Bulk import across all available seasons works for both leagues.
- Re-running is idempotent.
- Summary logging shows counts.

## Phase 5: NPB/KBO level factors for MLE translation

Define league-level translation factors so the existing MLE engine can translate NPB/KBO stats to MLB equivalents, just as it does for MiLB levels.

### Context

The MLE engine (`models/mle/engine.py`) translates minor league batting lines to MLB equivalents using `LevelFactor` (competition factor, K/BB/ISO/BABIP factors) and `LeagueEnvironment` (runs per game, league-average rates). The same framework applies to NPB and KBO — we just need level factors and league environments for those leagues. Research suggests NPB is roughly AAA+ quality and KBO is roughly AA-AAA quality, but the exact factors should be derived from cross-league player performance data (players who moved from NPB/KBO to MLB).

### Steps

1. Seed `LevelFactor` rows for `"npb"` and `"kbo"` levels with initial estimates:
   - NPB: factor ~0.80–0.85 (between AAA and MLB), k_factor ~1.10, bb_factor ~0.90, iso_factor ~0.85.
   - KBO: factor ~0.70–0.75 (roughly AAA), k_factor ~1.15, bb_factor ~0.85, iso_factor ~0.80.
   - These are starting points based on published research; they should be tunable.
2. Compute `LeagueEnvironment` rows for NPB and KBO from the ingested stats (aggregate league-wide rates per season).
3. Create a service `compute_international_league_environment(stats, season, league)` that aggregates batting stats into a `LeagueEnvironment`.
4. Adapt the MLE engine's `translate_batting_line` to accept `InternationalBattingStats` in addition to `MinorLeagueBattingStats` (they share the same essential fields: PA, AB, H, 2B, 3B, HR, BB, SO, etc.). This may mean extracting a common protocol or making `translate_batting_line` accept a structural type.
5. Write tests verifying NPB/KBO translation produces reasonable MLB-equivalent lines.

### Acceptance criteria

- `LevelFactor` and `LeagueEnvironment` exist for NPB and KBO leagues.
- The MLE engine translates NPB/KBO batting lines to MLB equivalents.
- Translated lines for known NPB→MLB movers (e.g., Yoshinobu Yamamoto's batting-against stats, Seiya Suzuki's hitting) produce values in the right ballpark compared to their actual MLB performance.
- Level factors are configurable, not hardcoded.

## Phase 6: Cross-league player matching

Match international players to their MLB counterparts when they sign with an MLB team, enabling validation of MLE translations.

### Context

The ultimate test of our NPB/KBO MLE is: for players who moved to MLB, how well did the translation predict their actual MLB performance? This requires linking NPB/KBO player records to MLB player records. FanGraphs' `minormasterid` field helps, but manual matching may be needed for some players.

### Steps

1. Build a cross-reference service that matches international players to MLB players via:
   - `minormasterid` from the FanGraphs international API (primary).
   - Name + age fuzzy matching as a fallback.
2. Store the link by setting `fg_intl_id` on the MLB player record (or vice versa, setting MLB IDs on the international player record if they're the same row).
3. Create a validation dataset: for each matched player, pair their last NPB/KBO season's stats with their first MLB season's stats.
4. Write a report/service that runs MLE translation on the international stats and compares predicted vs. actual MLB performance (RMSE on key rates: AVG, OBP, SLG, K%, BB%).
5. Use validation results to iteratively tune the level factors from Phase 5.

### Acceptance criteria

- Cross-league matches are identified for all available NPB/KBO → MLB movers.
- Validation report shows predicted vs. actual MLB performance.
- RMSE on key rates is reported per league.
- At least 20 player-seasons of validation data exist across NPB and KBO combined.

## Ordering

**Phase 1 → 2 → 3 → 4** is the critical ingest path — each phase depends on the prior. Phase 1 is the HTTP source, phase 2 adds domain/persistence, phase 3 solves player identity, phase 4 wires it all together with CLI.

**Phase 5** (level factors + MLE) can begin after phase 2, since it only needs the domain model and stored stats. It doesn't require CLI integration.

**Phase 6** (cross-league matching) requires phases 3 and 5. It is the validation step that closes the loop.

Suggested priority: **1 → 2 → 3 → 4 → 5 → 6**. Phases 1–4 are the ingest foundation. Phase 5 is where the real value emerges (MLE translation). Phase 6 validates accuracy and refines the model.
