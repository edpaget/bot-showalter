# Position Normalization Roadmap

Player positions flow through three inconsistent representations: uppercase from data sources (DB, Yahoo, Lahman), lowercase from internal services (`POSITION_ALIASES` maps `"C"` → `"c"`), and uppercase again in display/draft layers. This causes silent mismatches — most critically, `build_position_map` compares lowercase aliases against uppercase `league.positions` keys, so no batter ever matches a real position and all fall to "util". The fix is to define a canonical `Position` enum, normalize at system boundaries, and use the enum everywhere internally.

## Status

| Phase | Status |
|-------|--------|
| 1 — Position enum and core fix | done (2026-03-06) |
| 2 — Propagate enum to services and domain | done (2026-03-06) |
| 3 — Normalize at ingest/Yahoo boundaries | not started |

## Phase 1: Position enum and core fix

Introduce a `Position` enum, fix the case mismatch in `build_position_map` / `build_roster_spots`, and ensure ZAR produces correctly-cased valuations.

### Context

Commit `c995479` uppercased league config keys (`league.positions` → `{"C": 1, "OF": 3, ...}`), but `POSITION_ALIASES` still maps to lowercase (`"C"` → `"c"`). Line 35 of `models/zar/positions.py` checks `league_pos in valid_positions` — lowercase `"c"` is never in `{"C", "OF", ...}`, so every batter gets no position match and falls to "util". This means ZAR produces only ~12 positive-value batters (the util slot count) instead of ~120.

### Steps

1. Create a `Position` StrEnum in `src/fantasy_baseball_manager/domain/position.py` with members for all known positions: `C`, `1B`, `2B`, `3B`, `SS`, `LF`, `CF`, `RF`, `OF`, `DH`, `UTIL`, `SP`, `RP`, `P`. Use uppercase values matching the display convention. Include a `from_raw(s: str) -> Position` classmethod that handles case-insensitive lookup plus common aliases (e.g., `"dh"` → `DH`).
2. Add an `OF_POSITIONS` frozenset (`{LF, CF, RF, OF}`) and a `consolidate()` method or mapping that converts `LF`/`CF`/`RF` → `OF` for league-slot matching.
3. Replace `POSITION_ALIASES` dict in `models/zar/positions.py` with logic that uses `Position.from_raw()` plus the OF consolidation. The output of `build_position_map` should use `Position` enum values (or their string `.value`, which is uppercase).
4. Fix `build_roster_spots` to return keys consistent with `build_position_map` output. Since `league.positions` keys are already uppercase (matching `Position` values), this should now work naturally.
5. Fix `best_position` and its fallback to return `Position.UTIL.value` (or `Position.UTIL`) instead of the string `"util"`.
6. Update the `_classify` method in `PlayerEligibilityService` to use `Position` values for pitcher positions (`"SP"`, `"RP"`, `"P"`) instead of lowercase strings. Update the `league.pitcher_positions` key checks to be case-consistent.
7. Update existing tests in `tests/models/zar/test_positions.py` and `tests/services/test_player_eligibility.py` to use the new enum values. Add a test that verifies `build_position_map` output keys match `league.positions` keys exactly.
8. Run ZAR for 2025 and verify positive-value batter count is ~120 (not 12).

### Acceptance criteria

- A `Position` enum exists with `from_raw()` for case-insensitive parsing.
- `build_position_map` output keys match `league.positions` keys (both uppercase).
- `build_roster_spots` output keys match `build_position_map` output keys.
- ZAR valuations for 2025 produce ~120 positive-value batters (matching 12 teams × 10 batter slots).
- All existing tests pass (with updated position string expectations).
- Pitcher position keys (`"SP"`, `"RP"`, `"P"`) are consistent between eligibility service and league config.

## Phase 2: Propagate enum to services and domain

Replace bare `str` position fields in domain models and service interfaces with `Position` (or at least ensure all string values match `Position` values).

### Context

After phase 1 fixes the core mismatch, position strings still live as bare `str` in `Valuation.position`, `DraftBoardRow.position`, `DraftPick.position`, `PositionAppearance.position`, and various service return types. This makes future regressions easy — any new code that introduces a lowercase position string will silently break comparisons.

### Steps

1. Update `Valuation.position`, `DraftBoardRow.position`, and `DraftPick.position` fields to use `str` values that are always uppercase `Position` values. (Using the enum directly in frozen dataclasses stored to SQLite may be complex; a pragmatic approach is to keep `str` but ensure all producers use `Position.value`.)
2. Fix `_BATTER_POSITION_ORDER` and `_PITCHER_POSITION_ORDER` in `services/draft_board.py` — these are already uppercase and should now match valuation positions.
3. Fix `_position_sort_key` — currently fails silently when positions are lowercase, falling back to `len(order)`. Verify it works after phase 1.
4. Update `build_draft_roster_slots` in `services/draft_state.py` — it already uppercases, verify consistency with enum values.
5. Update `keeper_optimizer.py` — currently does `.lower()` normalization at lines 349/354. Remove this and use positions as-is (since they'll now be consistently uppercase).
6. Add a test in `test_draft_board.py` that verifies position grouping/sorting works end-to-end with real ZAR output positions.

### Acceptance criteria

- Draft board HTML export groups batters by position correctly (C, 1B, 2B, SS, 3B, OF — not all lumped together).
- Keeper optimizer groups by position without ad-hoc `.lower()` calls.
- Draft state roster slot validation works without case mismatches.
- No `.lower()` or `.upper()` normalization hacks remain in service layer code — positions are already in canonical form by the time they reach services.

## Phase 3: Normalize at ingest/Yahoo boundaries

Ensure all external data sources produce `Position` values at the boundary, eliminating case/alias issues at the source.

### Context

Position strings enter from multiple external sources: Lahman CSV ingestion (`_POSITION_COLUMNS` mapping), Yahoo API responses (roster positions, draft picks, eligibility lists), and the TOML league config. Each has its own conventions. Normalizing at these boundaries means internal code never encounters raw position strings.

### Steps

1. Update `LahmanAppearancesSource` in `ingest/lahman_source.py` — the `_POSITION_COLUMNS` mapping already produces uppercase strings matching `Position` values. Verify and add `Position.from_raw()` call if needed.
2. Update Yahoo player parsing (`yahoo/player_parsing.py`) — normalize `eligible_positions` through `Position.from_raw()`, filtering out non-position entries (`"BN"`, `"IL"`, `"IL+"`, `"NA"`, `"DL"`).
3. Update Yahoo draft source (`yahoo/draft_source.py`) — `_primary_position()` should return a `Position` value.
4. Update Yahoo roster source (`yahoo/roster_source.py`) — normalize `selected_position` to `Position` value, keeping bench/IL as separate non-Position strings or a distinct type.
5. Update TOML league config parsing (`config_league.py`) — validate that position keys in `[leagues.*.positions]` and `[leagues.*.pitcher_positions]` are valid `Position` values after uppercasing. Raise `LeagueConfigError` for unknown positions.
6. Add boundary tests: unknown position strings from Yahoo/Lahman are handled gracefully (logged and skipped, not silently lost).

### Acceptance criteria

- `PositionAppearance.position` values always match a `Position` member.
- Yahoo-derived positions (roster entries, draft picks) use `Position` values.
- League config rejects unknown position keys at parse time.
- No raw position strings cross the boundary into domain/service code without normalization.

## Ordering

**Phase 1 is critical and urgent** — it fixes the active bug where all batters get "util" position, making ZAR valuations wrong. Phases 2 and 3 are hardening and can be done in either order, though phase 2 (propagate to services) is higher value since it fixes draft board display and keeper optimizer grouping.

Dependencies: Phase 1 → Phase 2 (services consume enum values produced by phase 1). Phase 3 is independent of phase 2 but benefits from phase 1's enum definition.
