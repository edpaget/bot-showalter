# Player Eligibility Roadmap

Provide a generic player eligibility service that extrapolates position data from prior seasons when the current season has no data yet. This solves the immediate problem of 2026 valuations having no position assignments (every batter falls back to "util"), and extends to pitcher SP/RP classification with separate roster slots, and player bio/profile lookups. The service sits between the data repos and the valuation models, so any model (not just ZAR) can use it.

## Status

| Phase | Status |
|-------|--------|
| 1 — Batter position eligibility with season fallback | done (2026-02-21) |
| 2 — Pitcher SP/RP classification and roster slots | done (2026-02-22) |
| 3 — Player profile service | done (2026-02-23) |

## Phase 1: Batter position eligibility with season fallback

Build a generic `PlayerEligibilityService` that provides batter position data, falling back to the prior season when the target season has no data. Integrate it into the ZAR model to fix the 2026 valuation problem.

### Context

When running `fbm predict zar --season 2026`, the model calls `position_repo.get_by_season(2026)` which returns nothing — position appearance data only exists for past seasons (sourced from Lahman). All batters then fall back to `["util"]`, destroying positional scarcity in the valuations. The fix needs to be generic (not hardcoded into ZAR) so that any future valuation model can reuse it.

### Steps

1. Create `src/fantasy_baseball_manager/services/player_eligibility.py` with a `PlayerEligibilityService` class. Constructor takes `PositionAppearanceRepo`. Primary method: `get_batter_positions(season, league, *, min_games=10) -> dict[int, list[str]]`.
2. Implement season fallback: if `get_by_season(season)` returns empty, try `season - 1`. Apply the `min_games` threshold when filtering.
3. Add `min_games` parameter to `build_position_map()` in `models/zar/positions.py` (default 1 for backward compatibility). Filter out appearances where `games < min_games` before building the map.
4. The service calls `build_position_map(appearances, league, min_games=min_games)` internally, keeping the existing position logic reusable.
5. Update `ZarModel.__init__` to accept an optional `PlayerEligibilityService` (or build one from the existing `position_repo`). In `predict()`, use the service instead of calling `position_repo.get_by_season()` directly.
6. Update `build_model_context()` in `cli/factory.py` to create and pass the service.
7. Write tests: service with current-season data (no fallback), service with missing season (fallback), min_games filtering, build_position_map with min_games.

### Acceptance criteria

- When position data exists for the target season, it is used directly (no fallback).
- When position data is missing for the target season, the service falls back to `season - 1`.
- Positions with fewer than `min_games` (default 10) games are excluded.
- `build_position_map()` accepts and respects `min_games` parameter.
- ZAR model produces correct positional assignments for 2026 using 2025 data.
- All existing tests continue to pass.

## Phase 2: Pitcher SP/RP classification and roster slots

Derive pitcher position eligibility (SP/RP) from prior-season pitching stats and add `pitcher_positions` to the league config to support separate SP/RP/P roster slots.

### Context

Currently all pitchers are hardcoded to a single `"p"` position with `roster_pitchers` slots. The user's league has SP=2, RP=2, and P=4 (flex) slots — pitchers who started games should be SP-eligible, pitchers who appeared in relief should be RP-eligible, and the P slot works like "util" for pitchers. This requires: (a) league config changes to express pitcher positions, (b) a way to derive SP/RP from pitching stats, and (c) integration into the valuation pipeline.

### Steps

1. Add `pitcher_positions: dict[str, int]` field to `LeagueSettings` (default empty dict). When empty, models fall back to `{"p": roster_pitchers}`.
2. Update `parse_league()` in `config_league.py` to read `pitcher_positions` from TOML.
3. Update `fbm.toml` with `[leagues.h2h.pitcher_positions]`: `sp = 2`, `rp = 2`, `p = 4`.
4. Add `get_pitcher_positions(season, league) -> dict[int, list[str]]` to `PlayerEligibilityService`. Constructor gains an optional `PitchingStatsRepo`. Rules: `gs > 0` → "sp"-eligible, `(g - gs) > 0` → "rp"-eligible. All pitchers with at least one designation also get "p" (flex). Uses same season-fallback logic as batters.
5. Only include positions present in `league.pitcher_positions`. If `pitcher_positions` is empty, return `{pid: ["p"]}` for all (backward compat).
6. Update `ZarModel.predict()` to use the service for pitcher positions and pass `league.pitcher_positions` (or `{"p": roster_pitchers}` if empty) as `pitcher_roster_spots`.
7. Wire `PitchingStatsRepo` into the eligibility service via `build_model_context()`.
8. Write tests: SP-only pitcher, RP-only pitcher, dual-eligible pitcher, flex "p" assignment, backward compat with empty pitcher_positions.

### Acceptance criteria

- `pitcher_positions` config is parsed from TOML and stored in `LeagueSettings`.
- Pitchers are classified as SP, RP, or both based on prior-season `gs` and `g` fields.
- Replacement levels are computed separately for sp, rp, and p positions.
- A pitcher eligible for both SP and RP can fill any of the three slot types.
- When `pitcher_positions` is empty, behavior matches the current single-pool approach.
- All existing tests continue to pass.

## Phase 3: Player profile service

Extend the eligibility service (or create a sibling service) to provide player bio/profile data (age, bats/throws) through the same generic lookup pattern. This enables models and reports to enrich output with player context without coupling to repo details.

### Context

Various parts of the system need player bio data (age for aging curves, bats/throws for platoon splits, etc.) but currently reach directly into `PlayerRepo`. A `PlayerProfileService` would provide a clean, consistent API with the same patterns as the eligibility service — caching, fallback logic, and a unified interface for enrichment.

### Steps

1. Create `PlayerProfileService` (or extend `PlayerEligibilityService` into a broader `PlayerDataService`) with methods like `get_age(player_id, season)`, `get_bats_throws(player_id)`, `get_profile(player_id, season)` returning a `PlayerProfile` dataclass.
2. `PlayerProfile` includes: `player_id`, `name`, `age` (computed from birth_date and season), `bats`, `throws`, `positions` (from eligibility), `pitcher_type` (SP/RP/both).
3. Add a convenience method `enrich_valuations(valuations, season)` that annotates a list of valuations with profile data for display.
4. Wire into draft board and CLI output so that player age and handedness appear alongside valuations.
5. Write tests for age calculation, profile assembly, and enrichment.

### Acceptance criteria

- `PlayerProfile` provides age, bats/throws, and position data in one call.
- Age is correctly computed from birth_date and season year.
- Draft board output includes player age and handedness.
- Service is injectable and testable with fakes.

## Ordering

Phase 1 → 2 → 3, sequential. Phase 1 is the most urgent — it fixes the broken 2026 valuations. Phase 2 adds pitcher granularity and is closely related. Phase 3 is a convenience enhancement that generalizes the pattern. No hard dependencies on other roadmaps, though this roadmap's output enhances the positional scarcity roadmap (which depends on correct position assignments).
