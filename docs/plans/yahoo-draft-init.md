# Yahoo Draft Initialization Roadmap

When the web UI's draft dashboard is associated with a Yahoo league (via `webConfig.yahooLeague`), starting a new draft session should offer to auto-populate draft settings from Yahoo — specifically the number of teams, draft format, user team position, keeper players, and (for snake drafts) the draft order. Today the user manually enters teams/format/userTeam and keepers are only auto-loaded from the local `league_keeper` table. This roadmap adds a "prefill from Yahoo" flow that queries the Yahoo API at session-start time and wires the results into the existing `startSession` mutation.

## Status

| Phase | Status |
|-------|--------|
| 1 — Yahoo draft-setup query | done (2026-03-12) |
| 2 — Frontend prefill UX | not started |
| 3 — Keeper auto-import at draft init | not started |

## Phase 1: Yahoo draft-setup query

Add a new GraphQL query that returns everything the frontend needs to prefill the draft-start form from a Yahoo league.

### Context

The Yahoo API exposes league settings (teams, draft type) via `get_league_settings`, team list with `is_owned_by_user` via `get_teams`, and draft results (which reveal pick order for completed/in-progress drafts) via `get_draft_results`. The web backend already has `YahooLeagueSource`, `YahooTeamRepo`, and `YahooLeagueRepo` that store synced data. However, there is no single query that returns "here's what you need to start a draft" — the frontend would have to call multiple queries and do the mapping itself.

The Yahoo API also exposes a `draft_order_list` field inside league settings for leagues where the draft order has been set (either manually by the commissioner or randomized). This needs to be parsed and surfaced.

### Steps

1. Add a `get_league_draft_order(league_key)` method to `YahooFantasyClient` that calls the `league/{key}/settings` endpoint (already exists as `get_league_settings`) and extracts draft order from the `draft_order_list` field in the settings response. If the draft order hasn't been set yet, return an empty list.
2. Add a `YahooDraftSetupInfo` domain dataclass containing: `num_teams: int`, `draft_format: str` (snake/auction/live), `user_team_id: int`, `team_names: dict[int, str]`, `draft_order: list[int]` (team IDs in pick order for round 1 — empty if unset or auction), `is_keeper: bool`, `max_keepers: int | None`, `keeper_player_ids: list[int]` (from the local keeper cost table, empty if none).
3. Add a `yahoo_draft_setup(league_key, season)` resolver on the `Query` type that:
   - Reads the stored Yahoo league metadata and teams from repos.
   - Identifies the user team via `is_owned_by_user`.
   - Looks up local keeper costs for the season/league if the league is a keeper league.
   - Returns a `YahooDraftSetupInfoType` with all fields populated.
4. Add corresponding `YahooDraftSetupInfoType` strawberry type in `web/types.py`.
5. Add the query to the frontend GraphQL documents and run codegen.
6. Write backend tests for the new resolver covering: non-keeper league, keeper league with costs, auction format, league with no user team (error).

### Acceptance criteria

- `yahooDraftSetup(leagueKey, season)` query returns correct teams, format, userTeam, and keeper IDs for a synced league.
- Draft order list is populated when available in stored league data.
- Query raises a clear error when the league hasn't been synced.
- Frontend codegen succeeds with the new query types.

## Phase 2: Frontend prefill UX

Wire the Yahoo draft-setup query into `SessionControls` so that when a Yahoo league is configured, the form auto-fills from Yahoo data with a single click.

### Context

Currently `SessionControls` has manual inputs for season, teams, format, userTeam, and budget. When `webConfig.yahooLeague` is present, we should show a "Use Yahoo settings" button (or auto-fill on mount) that populates these fields from the `yahooDraftSetup` query. The user can still override any value before clicking "Start Draft".

### Steps

1. In `SessionControls`, accept `yahooLeague` (from `webConfig`) as an optional prop.
2. When `yahooLeague` is present, run the `yahooDraftSetup` query on mount (or lazily on a button press).
3. When the query returns, populate the form state: `teams`, `format`, `userTeam`, `season` from the response. If draft format is auction, also set `budget` to the league default (260 or from settings).
4. Show the Yahoo league name and a visual indicator that settings were prefilled. Allow the user to override any field.
5. Pass `keeperPlayerIds` from the Yahoo setup response through to the `startSession` mutation call.
6. If the Yahoo setup query fails (e.g., not synced), show the manual form as a fallback with an info message.
7. Write a frontend test that mocks the `yahooDraftSetup` query and verifies the form is prefilled.

### Acceptance criteria

- When a Yahoo league is configured, the draft-start form auto-fills teams, format, userTeam, and season from Yahoo data.
- Keeper player IDs from Yahoo setup are passed to `startSession`.
- Manual override of any prefilled field works correctly.
- Graceful fallback when Yahoo setup query fails.
- Frontend test covers the prefill flow.

## Phase 3: Keeper auto-import at draft init

Allow the `startSession` mutation to derive and import keeper costs on-the-fly if they don't already exist, so the user doesn't need to run a separate CLI command before starting a keeper draft.

### Context

Today, keeper costs must be derived ahead of time via `fbm yahoo keepers derive` (CLI) or `deriveKeeperCosts` (GraphQL mutation). If the user forgets, the draft starts with no keepers even though the Yahoo league is a keeper league. This phase makes the draft-start flow self-contained: if the league is a keeper league and no keeper costs exist for the season, the backend automatically derives them.

### Steps

1. In `SessionManager.start_session`, add an optional `league_key: str | None` parameter.
2. When `league_key` is provided, the league is a keeper league, and no keeper costs exist for the season/league, call the existing `derive_and_store_keeper_costs` (or `derive_best_n_keeper_costs`) logic to populate them before building the keeper snapshot.
3. This requires `SessionManager` to accept a keeper-cost derivation callable (Protocol) via constructor injection — it must not import the Yahoo service directly.
4. Update the `start_session` GraphQL mutation to accept an optional `leagueKey` parameter and thread it through.
5. Update the frontend `START_SESSION` mutation to pass `leagueKey` when Yahoo is configured.
6. Write backend tests covering: auto-derivation triggers when costs are missing, skips when costs already exist, works for both round-based and best-N keeper types.

### Acceptance criteria

- Starting a keeper draft with a Yahoo league key auto-derives keeper costs if none exist for the season.
- Existing keeper costs are not overwritten.
- The new `leagueKey` parameter is optional and the mutation works without it (backward compatible).
- Backend tests verify auto-derivation triggers correctly.

## Ordering

Phases are sequential:

1. **Phase 1** must land first — it provides the backend query that Phase 2 consumes.
2. **Phase 2** depends on Phase 1 — it wires the query into the frontend.
3. **Phase 3** is independently implementable after Phase 1 (doesn't need Phase 2), but logically should come last since it's an enhancement to the flow established by Phases 1-2.

No external roadmap dependencies — all Yahoo infrastructure (client, repos, sync) and draft session machinery already exist.
