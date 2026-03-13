# Yahoo Draft Init Fixes Roadmap

Phase 3 of the Yahoo Draft Init roadmap introduced auto-derivation of keeper costs at draft start, but the implementation has a data-flow bug: the deriver writes to `keeper_cost` (via `KeeperCostRepo`) but `SessionManager.start_session` re-reads from `league_keeper` (via `LeagueKeeperRepo`). These are different tables — the re-read finds nothing, so keepers silently fail to load. Additionally, the draft UI only shows numeric team IDs ("Team 1", "Team 2") even when Yahoo team names are available from the prefill query.

This roadmap fixes the keeper auto-import pipeline and surfaces team names throughout the draft UI.

## Status

| Phase | Status |
|-------|--------|
| 1 — Fix keeper auto-import repo mismatch | done (2026-03-12) |
| 2 — Surface team names in draft sessions | done (2026-03-12) |

## Phase 1: Fix keeper auto-import repo mismatch

Fix the data pipeline so that auto-derived keeper costs are actually loaded into the draft session.

### Context

`_derive_keeper_costs_from_yahoo` (in `schema.py`) calls `derive_and_store_keeper_costs` / `derive_best_n_keeper_costs`, which write `KeeperCost` rows to the `keeper_cost` table via `KeeperCostRepo`. However, `SessionManager.start_session` auto-loads keepers from `LeagueKeeperRepo` (the `league_keeper` table), which is a separate table that stores league-wide keeper assignments (team_name, player_id). The deriver never populates `league_keeper`, so the retry read at lines 110-113 of `session_manager.py` always comes back empty.

The cleanest fix: after calling the deriver, have `SessionManager` read from `KeeperCostRepo` as a fallback. This matches the existing `yahooDraftSetup` query, which already reads keeper player IDs from `keeper_cost_repo`. Injecting `KeeperCostRepo` into `SessionManager` (via a Protocol or directly) keeps the layers clean — it's a repo, not a service.

### Steps

1. Add an optional `keeper_cost_repo: KeeperCostRepo | None = None` parameter to `SessionManager.__init__` (using a Protocol for the lookup method to keep service/repo layer boundaries clean, or accepting the concrete repo since repos are already imported via TYPE_CHECKING).

2. In `start_session`, after calling `self._keeper_cost_deriver(season, league_key)`, read from `keeper_cost_repo.find_by_season_league(season, self._league.name)` instead of re-reading from `league_keeper_repo`. Extract player IDs from the `KeeperCost` objects.

3. Wire `keeper_cost_repo` into `SessionManager` in `web.py` where it is constructed — `container.keeper_cost_repo` is already available there.

4. Update the existing `TestKeeperAutoDerivation` tests: the fake deriver should write to a `KeeperCostRepo` (or fake equivalent) instead of `LeagueKeeperRepo`, since that matches what the real deriver does. Add a test verifying that keeper player IDs from `keeper_cost_repo` are loaded after derivation.

5. Verify backward compatibility: when no `keeper_cost_repo` is provided and no `league_key` is passed, behavior is unchanged.

### Acceptance criteria

- Starting a keeper draft with `league_key` and no pre-existing keeper data triggers the deriver and successfully loads keeper player IDs from the `keeper_cost` table.
- Pre-existing `league_keeper` rows are still loaded (existing auto-load path at lines 101-105 is unchanged).
- Pre-existing `keeper_cost` rows without derivation are NOT auto-loaded (avoid double-loading — the `keeper_cost` fallback only runs after derivation).
- Backend tests verify the fix end-to-end with a fake deriver that writes to `keeper_cost_repo`.

## Phase 2: Surface team names in draft sessions

Thread Yahoo team names from the prefill query into the draft session so the UI shows names instead of numbers.

### Context

`yahooDraftSetup` returns `teamNames` (a `dict[int, str]` mapping team ID → team name), and `SessionControls` already fetches this data during prefill. However, the names are discarded — they never flow into `DraftSessionRecord`, `DraftStateType`, or the frontend draft components. The pick log shows "Team 1", "Team 2"; the trade dialog uses "Team 1", "Team 2"; and resume/restore has no way to recover names.

The fix requires changes at three layers: persist team names in the session record, return them in `DraftStateType`, and use them in frontend components.

### Steps

1. Add `team_names: dict[str, str] | None = None` (JSON-serialized `{team_id: name}`) to `DraftSessionRecord`. Add a `team_names TEXT` column to `draft_session` via a new migration. Update `SqliteDraftSessionRepo` to read/write it as JSON (same pattern as `roster_slots`).

2. Add `team_names: list[TeamNameType] | None` to `DraftStateType` (where `TeamNameType` is a simple strawberry type with `team_id: int` and `name: str`, or use `strawberry.scalars.JSON`). Populate it in `from_state` by accepting an optional `team_names` parameter and in the `start_session` / `session` resolvers.

3. Add `team_names: dict[int, str] | None = None` parameter to `SessionManager.start_session`. Store it in the `DraftSessionRecord`. Thread it from the `start_session` GraphQL mutation (accept `teamNames: JSON` and pass through).

4. Update the frontend `START_SESSION` mutation to accept `$teamNames: JSON` and pass it. In `SessionControls`, include `teamNames` from the prefill response in the `onStart` config. In `DraftDashboard`, pass it through to `startSession` variables.

5. Store team names in `DraftSessionContext` so they survive across component re-renders. Expose a `teamNames` map (or `getTeamName(id)` helper) from the context.

6. Update `PickLogPanel` to display team names: replace `Team {pick.team}` with the name (falling back to `Team {n}` when no names are available). Similarly update `TradeDialog` team selects and trade display.

7. On session resume, load team names from the persisted `DraftSessionRecord` via the `session` query (which returns `DraftStateType` including `teamNames`).

### Acceptance criteria

- When starting a draft with Yahoo prefill, team names appear in the pick log, trade dialog, and trade history instead of numeric IDs.
- Team names persist across page refresh (stored in DB, returned on resume).
- Drafts started without Yahoo (no team names) fall back to "Team N" display — no regressions.
- Frontend and backend tests cover both named and unnamed team paths.

## Ordering

Phase 1 is a bug fix and should land first — without it, keeper auto-import is broken. Phase 2 is an independent enhancement that can land after Phase 1. Neither phase has external roadmap dependencies.
