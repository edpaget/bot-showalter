# Two-Way Player Identity

Ensure all draft, keeper, and valuation code identifies players by `(player_id, player_type)` instead of `player_id` alone. Two-way players (e.g. Ohtani) exist as separate batter and pitcher entries — using `player_id` alone causes dict collisions, incorrect pool filtering, and wrong valuation lookups.

## Status

| Phase | Status |
|-------|--------|
| 1 — Core valuation pipeline | not started |
| 2 — Keeper service lookups | not started |
| 3 — DraftPick & undo identity | not started |
| 4 — Peripheral services & CLI | not started |

## Phase 1 — Core valuation pipeline

The `ValuationAdjuster` protocol and `compute_adjusted_valuations` are the root cause: they accept `set[int]` and filter projections by `player_id` alone, which removes both types when only one is kept.

### Steps

1. Change `compute_adjusted_valuations` signature from `kept_player_ids: set[int]` to `kept_keys: set[tuple[int, str | None]]`.
2. Update projection filter (line 476) from `p.player_id not in kept_player_ids` to a helper that checks `(player_id, player_type)` pairs, with `None` player_type matching all types.
3. Change `orig_lookup` (line 503) from `dict[int, float]` to `dict[tuple[int, str], float]` keyed by `(player_id, player_type)`. Update the result-building loop (line 517) to look up by `(player_id, player_type)`.
4. Update `ValuationAdjuster` protocol in `session_manager.py` from `kept_ids: set[int]` to `kept_keys: set[tuple[int, str | None]]`.
5. Update `_make_valuation_adjuster` in `cli/commands/web.py`:
   - Pass typed keys through to `compute_adjusted_valuations`.
   - Change `orig_lookup` (line 67) from `dict[int, Valuation]` to `dict[tuple[int, str], Valuation]`.
6. Simplify `_build_player_pool` in `session_manager.py`: remove `_fully_kept_ids` workaround and the post-adjuster restoration loop. Pass `keeper_player_ids` directly to the adjuster (it now handles typed keys natively).
7. Remove `_keeper_player_ids_only` if no longer used anywhere.

### Acceptance criteria

- Keeping batter-Ohtani excludes only batter projections from re-valuation; pitcher-Ohtani projections remain and get re-valued.
- The adjusted value for pitcher-Ohtani reflects the correct original value (pitcher), not the batter's.
- `_fully_kept_ids` workaround is deleted.
- All existing tests pass; new test covers partial two-way keep scenario through `compute_adjusted_valuations`.

## Phase 2 — Keeper service lookups

Multiple functions in `keeper_service.py` build `dict[int, Valuation]` or `dict[int, float]` lookups keyed by `player_id` alone. For two-way players, only the higher-value type survives, which can assign the wrong valuation.

### Steps

1. `compute_surplus` (line 92): change `val_lookup` to `dict[tuple[int, str], Valuation]`, keyed by `(player_id, player_type)`. Update surplus calculation to match keeper costs by player type where available.
2. `estimate_other_keepers` (line 146): change `val_lookup` to `dict[tuple[int, str], float]`. Ensure roster-based keeper estimation considers that two-way players may appear twice.
3. `build_league_keeper_overview` (line 177): change `val_lookup` to `dict[tuple[int, str], Valuation]`. Trade target and projection displays should show the correct type's value.
4. `evaluate_trade` (line 384): change `val_lookup` to `dict[tuple[int, str], Valuation]`. Trade evaluation must compare correct types.

### Acceptance criteria

- `compute_surplus` returns correct surplus for a two-way player kept as pitcher (not the batter valuation).
- `estimate_other_keepers` can handle a roster containing both types of the same player.
- Trade evaluation uses the correct type's valuation for two-way players.
- All existing keeper service tests pass.

## Phase 3 — DraftPick & undo identity

`DraftPick` does not carry `player_type`, so the undo path searches `_removed_rows` by `player_id` alone. If both types of a two-way player were ever drafted (by different teams), undo would restore the wrong one.

### Steps

1. Add `player_type: str = ""` field to `DraftPick` dataclass.
2. Populate `player_type` in `DraftEngine.pick()` from the pool entry being consumed.
3. Update `DraftEngine.undo()` to search `_removed_rows` by `(player_id, player_type)` instead of `player_id` alone.
4. Add `player_type: str` to `DraftPickType` GraphQL type and `from_domain`.
5. Update frontend `PickFields` fragment to include `playerType`.
6. Run codegen. Update frontend types/tests as needed.
7. Update pick persistence in `session_manager.py` to include `player_type` in stored pick data.
8. Verify undo round-trips correctly for two-way players.

### Acceptance criteria

- `DraftPick` carries the drafted `player_type`.
- Undo after drafting pitcher-Ohtani restores the correct pitcher pool entry (not batter).
- Pick log displays show which type was drafted.
- Session restore correctly rehydrates `player_type` on picks.

## Phase 4 — Peripheral services & CLI

Remaining `dict[int, ...]` lookups in supporting code.

### Steps

1. `pick_value.py` `compute_pick_value_curve` (line 97): change `val_by_id` to `dict[tuple[int, str], float]`. Two-way players contribute two entries to the value curve, not one.
2. `keeper_planner.py` `_compute_scenario` (line 139): change `val_lookup` to `dict[tuple[int, str], Valuation]`.
3. `cli/commands/keeper.py` trade evaluation lookup (line 480): change to typed key.
4. Audit any remaining `dict[int, ...]` valuation lookups across the codebase.

### Acceptance criteria

- Pick value curve treats two-way player types as separate entries.
- Keeper planner scenarios show correct per-type valuations.
- No remaining `dict[int, Valuation]` or `dict[int, float]` valuation lookups that could collide.
