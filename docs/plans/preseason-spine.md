# Preseason Spine Roadmap

Enable projections for future seasons where no stats data exists yet. The feature pipeline's spine CTE queries `batting_stats`/`pitching_stats` to determine which players to project, so `predict --season 2026` returns zero rows because no 2026 data exists. The spine is the only blocker — all feature columns use lagged joins (`season - 1`) that reference prior-season data which does exist. The fix adds an explicit player ID list to bypass the stats table, a `PlayerUniverseProvider` service to determine the player list from prior-season stats, and wires it into the model's predict path as a fallback.

## Status

| Phase | Status |
|-------|--------|
| 1 — Extend SpineFilter with explicit player IDs | in progress |
| 2 — Create PlayerUniverseProvider | not started |
| 3 — Wire into model predict() | not started |

## Phase 1: Extend SpineFilter with explicit player IDs

Add an explicit player ID list to `SpineFilter` so the spine CTE can bypass the stats table entirely, generating rows from the ID list x seasons using SQLite's `json_each()`.

### Context

`SpineFilter` (`features/types.py`) has `min_pa`, `min_ip`, `player_type` but no way to specify an explicit player list. When `player_ids` is provided, the spine should bypass the stats table entirely and generate rows from the ID list x seasons using SQLite's `json_each()`.

### Steps

1. Add `player_ids: tuple[int, ...] | None = None` to `SpineFilter`.
2. Update `_spine_filter_to_dict()` — include a hash of sorted IDs (not the full list) for compact version hashing; `None` when unset for backward compat.
3. Modify `_spine_cte()` in `features/sql.py` — when `player_ids` is set, emit `SELECT CAST(value AS INTEGER) AS player_id, ? AS season FROM json_each(?)` per season, unioned together. Skip stats table, `min_pa`/`min_ip`, and `source_filter`.
4. Add unit tests in `tests/features/test_sql.py` for the new branch.
5. Add round-trip integration test: explicit player IDs for a future season, verify lag features materialize correctly.

### Acceptance criteria

- `SpineFilter(player_ids=(1, 2, 3))` is valid and frozen.
- `_spine_cte()` uses `json_each()` when `player_ids` is set; does not reference stats table.
- Version hash is order-independent (`(1,2)` and `(2,1)` produce same hash).
- Version hash for `SpineFilter(player_ids=None)` is unchanged (no cache invalidation).
- Round-trip: feature set with explicit IDs for future season returns correct rows with lag features.
- All existing `tests/features/test_sql.py` pass unchanged.

## Phase 2: Create PlayerUniverseProvider

Create a service-layer protocol and implementation that determines which players to include in a future-season projection using prior-season stats.

### Context

To project a future season, we need to know which players to include. The simplest heuristic: "players who appeared in the prior season's stats." `BattingStatsRepo.get_by_season()` and `PitchingStatsRepo.get_by_season()` already exist. This is a service-layer concern (the season-1 fallback is a policy decision per Principle 8), implemented as a single-method protocol per Principle 11.

### Steps

1. Create `PlayerUniverseProvider` protocol and `StatsBasedPlayerUniverse` in `services/player_universe.py`.
   - Single method: `get_player_ids(season, player_type, *, source=None, min_pa=None, min_ip=None) -> set[int]`.
   - Uses `batting_repo.get_by_season(season - 1, source)` for batters, `pitching_repo.get_by_season(season - 1, source)` for pitchers.
   - Filters by min thresholds.
2. Add to `AnalysisContainer` as a `cached_property`.
3. Write tests in `tests/services/test_player_universe.py` — normal case, empty prior season, filtering.
4. Wire through `build_model_context` in `cli/factory.py`.

### Acceptance criteria

- `StatsBasedPlayerUniverse` implements `PlayerUniverseProvider` protocol.
- `get_player_ids(2026, "batter", source="fangraphs")` returns player IDs from 2025 fangraphs batting stats.
- Uses `season - 1` lookback, not `season` itself.
- `min_pa`/`min_ip` filtering works.
- Empty prior season returns empty set (no error).
- Constructor injection: repos passed as constructor params.
- Registered in `AnalysisContainer` and wired through `build_model_context`.

## Phase 3: Wire into model predict()

Integrate the `PlayerUniverseProvider` into the model's predict path so it falls back to the provider-based spine when the normal spine returns zero rows.

### Context

`_StatcastGBMBase.predict()` builds feature sets and materializes them. If the spine returns zero rows, prediction fails. With Phases 1-2 in place, predict can fall back to the `PlayerUniverseProvider` when the normal spine is empty.

### Steps

1. Accept optional `PlayerUniverseProvider` in `_StatcastGBMBase.__init__()` (default `None`).
2. Add `_build_feature_set_with_universe()` helper — calls standard builder, gets player IDs from provider, uses `dataclasses.replace()` to swap in a `SpineFilter` with `player_ids` (recomputes version hash automatically via `__post_init__`).
3. In `predict()`: after materialization returns zero rows, retry with provider-based spine if available.
4. Update `StatcastGBMPreseasonModel` to accept and forward the provider.
5. Update `build_model_context` to pass `StatsBasedPlayerUniverse` to the model.
6. Write integration tests: future season returns predictions via fallback; existing season unchanged.
7. Smoke test: `fbm predict statcast-gbm-preseason --season 2026` produces non-empty output.

### Acceptance criteria

- `fbm predict statcast-gbm-preseason --season 2026` produces predictions.
- Predictions use 2025 player universe, each with `season=2026`.
- Predictions for seasons with actuals (e.g., `--season 2025`) unchanged.
- Fallback only activates when normal spine returns zero rows.
- `PlayerUniverseProvider` injected via constructor (Principle 1).
- All existing model tests pass unchanged.

## Key Files

| File | Change |
|------|--------|
| `src/fantasy_baseball_manager/features/types.py` | Add `player_ids` to `SpineFilter`, update `_spine_filter_to_dict` |
| `src/fantasy_baseball_manager/features/sql.py` | Branch `_spine_cte()` for `json_each()` path |
| `src/fantasy_baseball_manager/services/player_universe.py` | New: `PlayerUniverseProvider` protocol + `StatsBasedPlayerUniverse` |
| `src/fantasy_baseball_manager/models/statcast_gbm/model.py` | Accept provider, add fallback in `predict()` |
| `src/fantasy_baseball_manager/analysis_container.py` | Register `StatsBasedPlayerUniverse` |
| `src/fantasy_baseball_manager/cli/factory.py` | Wire provider into model context |
| `tests/features/test_sql.py` | Spine CTE tests for `player_ids` branch |
| `tests/services/test_player_universe.py` | New: provider unit tests |

## Ordering

Phases are sequential: Phase 1 (feature pipeline) -> Phase 2 (service layer) -> Phase 3 (model integration). Each is independently testable and mergeable. No hard dependencies on other roadmaps.
