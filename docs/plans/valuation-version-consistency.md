# Valuation Version Consistency Roadmap

The `--system` and `--version` options should be configurable everywhere valuations are consumed. Today, `yahoo keeper-decisions` is the only keeper-related command that accepts `--version` (defaulting to `"production"`). All other keeper commands (`decisions`, `adjusted-rankings`, `trade-eval`, `optimize`, `scenario`, `trade-impact`) and yahoo commands (`keeper-league`, `draft-needs`) fetch all versions for the given system and never filter, which can silently mix production and experimental valuations. The draft commands already handle this correctly — they all accept `--version` and filter after fetching. This roadmap brings the remaining commands and services into alignment.

## Status

| Phase | Status |
|-------|--------|
| 1 — Add `--version` to keeper and yahoo CLI commands | not started |
| 2 — Push version filtering into the repo layer | not started |

## Phase 1: Add `--version` to keeper and yahoo CLI commands

Add a `--version` option (defaulting to `"production"`) to every command and service call site that fetches valuations without version filtering.

### Context

The draft commands (in `cli/commands/draft.py`) already follow a consistent pattern: accept `--version` with default `"production"`, fetch via `get_by_season`, then filter with `[v for v in valuations if v.version == version]`. The keeper commands and several yahoo commands skip this step entirely, meaning they return valuations from all versions (production, experimental, injury-adjusted, etc.) mixed together. When multiple versions exist for the same player, `compute_surplus` picks the highest value — which may come from an unintended version.

### Steps

1. Add `version: Annotated[str, typer.Option("--version", help="Valuation version")] = "production"` to all six `keeper_app` commands that take `--system`: `decisions`, `adjusted-rankings`, `trade-eval`, `optimize`, `scenario`, `trade-impact`.
2. After each `ctx.valuation_repo.get_by_season(season, system)` call in those commands, add `valuations = [v for v in valuations if v.version == version]`.
3. Add the same `--version` parameter to `yahoo keeper-league` and `yahoo draft-needs` commands, with the same default and filtering pattern.
4. Update `_build_cost_translator` in `keeper.py` to accept a `version` parameter and filter valuations before passing them to `compute_pick_value_curve`. Thread the parameter through from callers (`import` and `set` commands).
5. Add `version` parameter to `ValuationLookupService.rankings()` and filter valuations after fetching. Update CLI callers of `rankings()` to pass version through.
6. Add or update tests for each modified command to verify that version filtering is applied (i.e., valuations with a non-matching version are excluded from results).

### Acceptance criteria

- Every CLI command that accepts `--system` also accepts `--version` (default `"production"`).
- When multiple valuation versions exist for a player, only the specified version is used in keeper decisions, trade evaluations, optimization, and rankings.
- `ValuationLookupService.rankings()` filters by version when provided.
- `_build_cost_translator` filters by version.
- Existing tests continue to pass; new tests cover the version filtering behavior.

## Phase 2: Push version filtering into the repo layer

Eliminate the repeated `[v for v in valuations if v.version == version]` pattern by adding an optional `version` parameter to `ValuationRepo.get_by_season`.

### Context

After phase 1, every call site will follow the same two-line pattern: fetch all valuations for a system, then filter by version in Python. This is wasteful (fetches rows from SQLite only to discard them) and error-prone (easy to forget the filter line when adding new call sites). The draft commands alone have ~15 instances of this pattern. Moving the filter into the repo query is more efficient and makes the correct behavior the default.

### Steps

1. Add `version: str | None = None` parameter to `ValuationRepo` Protocol's `get_by_season` method.
2. Update `SqliteValuationRepo.get_by_season` to add `AND version = ?` to the SQL query when version is provided.
3. Update all call sites across the codebase to pass `version=` directly to `get_by_season` and remove the subsequent Python-side filter line. This affects:
   - `cli/commands/keeper.py` (6 commands + `_build_cost_translator`)
   - `cli/commands/yahoo.py` (`keeper-decisions`, `keeper-league`, `draft-needs`)
   - `cli/commands/draft.py` (~15 commands)
   - `cli/commands/report.py`
   - `services/valuation_lookup.py`
   - `services/valuation_evaluator.py`
   - `services/adp_report.py`
   - `services/adp_accuracy.py`
   - `services/yahoo_draft_setup.py`
   - `services/injury_valuation.py`
   - `web/schema.py`
   - `web/session_manager.py`
4. Update test fakes/stubs that implement `ValuationRepo` to accept the new parameter.
5. Verify no call site relies on receiving multiple versions from a single `get_by_season` call (the `lookup` method uses `get_by_player_season`, which intentionally returns all versions — that stays unchanged).

### Acceptance criteria

- `ValuationRepo.get_by_season` accepts an optional `version` parameter.
- SQLite implementation filters at the query level when version is provided.
- No call site uses the old two-line fetch-then-filter pattern.
- All existing tests pass; repo tests cover the new version filtering path.
- `get_by_player_season` (used by player lookup) still returns all versions so users can compare them.

## Ordering

Phase 1 is independent and delivers the user-facing fix — every command respects `--version`. Phase 2 is a pure refactor that cleans up the implementation. Phase 1 should land first; phase 2 can follow immediately or be deferred.
