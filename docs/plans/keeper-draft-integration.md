# Keeper Draft Integration Roadmap

Connect keeper league data to the live draft tool so the web UI accounts for kept players, adjusted valuations, and keeper-aware recommendations. Today the draft tool operates as a pure redraft simulator — it builds the player pool from raw valuations and ADP without considering which players are kept and how that changes positional scarcity. The keeper system already has the building blocks (`compute_adjusted_valuations`, `estimate_other_keepers`, `build_keeper_draft_needs`) but none of them feed into the draft session.

Today's keeper data model (`KeeperCost`) only stores the user's own keeper costs for surplus computation. There is no persisted league-wide keeper list — the `--league-keepers` flag on `keeper optimize` takes an ephemeral CSV. Phase 1 adds a persisted league keeper list with CLI commands for entry, which downstream phases read at draft time.

This roadmap depends on: keeper-optimization-solver (done), keeper-surplus-value (done), web-ui-foundation phases 1-5 (done), adp-arbitrage-alerts (done), draft-pick-keeper-support (done).

## Status

| Phase | Status |
|-------|--------|
| 1 — League keeper list | not started |
| 2 — Keeper-adjusted draft pool | not started |
| 3 — Keeper state display | not started |
| 4 — Keeper-aware recommendations | not started |
| 5 — Pre-draft keeper planner | not started |

## Phase 1: League keeper list

Add a persisted league-wide keeper list and CLI commands to populate it. This is the data foundation — every downstream phase reads from this list to know which players are kept, by which team, and at what cost.

### Context

Today the only keeper data stored is the user's own `KeeperCost` records (player_id, cost, source). When running `keeper optimize --league-keepers`, the user passes an ad-hoc CSV of other teams' keepers that is never saved. Before the draft, leagues typically announce all keepers — the user needs a way to enter this list once and have it available for both the optimizer and the draft tool.

### Steps

1. Define a `LeagueKeeper` frozen dataclass in `domain/keeper.py`: `player_id`, `season`, `league`, `team_name`, `cost` (optional), `source` (optional). This represents "Team X is keeping Player Y" — distinct from `KeeperCost` which represents the user's own cost basis.
2. Add a `league_keepers` table to the SQLite schema and a `LeagueKeeperRepo` with `upsert_batch()`, `find_by_season_league()`, `delete_by_season_league()`, and `find_by_team()`.
3. Add `fbm keeper league-set` CLI command: `fbm keeper league-set "Mike Trout" --team "Team A" --season 2026 --league dynasty`. Sets a single player as kept by a specific team. Resolves player name via `resolve_players()`.
4. Add `fbm keeper league-import` CLI command: imports a CSV with columns `player_name,team_name` (and optional `cost`). Batch-resolves names and upserts. Reports unmatched names.
5. Add `fbm keeper league-list` CLI command: shows the current league keeper list for a season/league, grouped by team.
6. Add `fbm keeper league-clear` CLI command: removes all league keepers for a season/league (for re-entry if the list changes).
7. Wire `LeagueKeeperRepo` into `build_keeper_context()` so it's available alongside the existing repos.
8. Write tests: CRUD operations on the repo, CLI commands for set/import/list/clear, name resolution with fuzzy matching, CSV import with unmatched names reported.

### Acceptance criteria

- `fbm keeper league-set` persists a single keeper assignment and is idempotent (re-running updates rather than duplicates).
- `fbm keeper league-import` loads a CSV of keepers and reports unmatched names.
- `fbm keeper league-list` displays the persisted list grouped by team.
- `fbm keeper league-clear` removes all entries for a season/league.
- The `LeagueKeeperRepo` can be queried to get all kept player IDs for a season/league.

## Phase 2: Keeper-adjusted draft pool

Remove kept players from the draft pool and re-value the remaining players using the existing `compute_adjusted_valuations()` pipeline so the draft board reflects true post-keeper scarcity.

### Context

When a league has 12 teams keeping 5 players each, 60 players are removed before the draft starts. This changes replacement level at every position — if 8 of the top 12 shortstops are kept, SS scarcity spikes and the remaining SS should be valued higher. The `compute_adjusted_valuations()` function already does this re-valuation, but the draft session doesn't call it. Phase 1's `LeagueKeeperRepo` provides the persisted keeper list to drive this.

### Steps

1. Add optional `keeper_player_ids: set[int] | None` parameter to `SessionManager.start_session()` and thread it through `_build_player_pool()`. When provided, filter these player IDs out of the available pool before building the draft board, and pass the adjusted valuations to `build_draft_board()` instead of raw valuations.
2. Wire up the keeper-adjusted valuation pipeline: when `keeper_player_ids` is provided, call `compute_adjusted_valuations()` to re-value the remaining pool, then use those adjusted values in the draft board.
3. Add `keeper_player_ids` field to `DraftSessionRecord` so keeper-adjusted sessions can be restored from the DB.
4. Extend the `start_session` GraphQL mutation with an optional `keeperPlayerIds: [Int!]` argument that flows through to `SessionManager.start_session()`.
5. Inject `LeagueKeeperRepo` and the projection data needed by the adjusted-valuation pipeline into `SessionManager` via the composition root (`app.py` / factory).
6. Add a convenience mode: when `keeper_player_ids` is not explicitly provided but a `league` parameter is given, auto-load keepers from `LeagueKeeperRepo.find_by_season_league()` and use their player IDs. This is the primary flow — the user runs `keeper league-import` before the draft, then `start_session` picks them up automatically.
7. Write tests: starting a session with keeper IDs removes those players from the board; adjusted valuations differ from raw valuations (SS scarcity example); session restore preserves keeper context; auto-load from `LeagueKeeperRepo` produces the correct keeper set.

### Acceptance criteria

- A draft session started with keeper IDs excludes those players from `available()`.
- The remaining players' values reflect post-keeper replacement levels (e.g., a mid-tier SS is worth more when top SS are kept).
- `DraftBoardRow.value` uses the adjusted value, not the raw value.
- Sessions can be restored from the DB with the same keeper context.
- When no keeper IDs are provided and no league keepers are persisted, behavior is identical to today (backward compatible).

## Phase 3: Keeper state display

Show which players are kept, by which teams, and at what cost in the draft UI so the user has full context during the draft.

### Steps

1. Add a `KeeperInfoType` Strawberry type with fields: `playerId`, `playerName`, `position`, `teamName`, `cost`, `source`.
2. Add a `keepers(sessionId) -> [KeeperInfoType!]!` query to the GraphQL schema. When a session was started with keeper IDs, return the keeper details by joining against `LeagueKeeperRepo` and `PlayerRepo`.
3. Store the league keeper snapshot in the session (extend `DraftSessionRecord` or add a related table) so keeper display doesn't depend on the repo state changing after the draft starts.
4. Build a `KeeperPanel` React component for the draft dashboard sidebar:
   - Table showing all kept players grouped by team, with name, position, cost, and projected value.
   - Highlight the user's keepers distinctly.
   - Show summary stats: total keepers, total value kept, average cost.
   - Collapsible team sections to avoid overwhelming the display.
5. Show a "Kept" badge on `DraftBoardTable` rows for any player in the keeper set (they won't appear in the available pool, but may show in other views like the full board query).
6. Include keeper count and total kept value in `DraftStateType` so the frontend knows the session is keeper-aware.
7. Write tests: keeper query returns correct data, panel renders grouped by team, kept badge appears on correct players.

### Acceptance criteria

- `keepers` query returns all kept players with team assignment, cost, and value.
- Dashboard shows a keeper panel with per-team breakdown.
- User's own keepers are visually distinct from opponent keepers.
- Non-keeper sessions return an empty keeper list (no errors).

## Phase 4: Keeper-aware recommendations

Integrate keeper draft needs analysis into the recommendation engine so pick suggestions account for what positions are scarce after keepers and what category gaps the user's keeper set creates.

### Context

The existing `build_keeper_draft_needs()` function computes category gaps after keepers are locked in. The draft tool's recommendation engine ranks available players by value and positional need. Combining these means recommendations can say "you need SB from the draft because your keepers are SB-light" or "prioritize SP because 6 aces were kept league-wide."

### Steps

1. Define a `KeeperDraftContext` dataclass holding the user's keeper roster analysis (`RosterAnalysis`) and category needs (`list[CategoryNeed]`), computed once at session start via `build_keeper_draft_needs()`.
2. Cache `KeeperDraftContext` in the `DraftEngine` or `SessionManager` alongside the engine. Persist enough to reconstruct on session restore.
3. Extend the recommendation scoring to factor in category needs: boost the score of players who fill the user's weakest categories (as identified by keeper draft needs). Use a configurable weight so keeper-need signal doesn't overwhelm raw value.
4. Add a `categoryNeeds` field to `RecommendationType` or a new `KeeperNeedsType` in the GraphQL schema, exposing which categories are weak and how each recommended player addresses them.
5. Update `RecommendationPanel` to show category-need indicators next to recommendations (e.g., a tag showing "fills SB need" or a color-coded bar for category gap closure).
6. Update arbitrage alerts to be keeper-aware: a falling player who also fills a keeper-identified category gap should score higher in arbitrage ranking. Extend `detect_falling_players()` to accept optional category-need weights.
7. Write tests: recommendations change when keeper context is present (SB-needy keeper set boosts SB contributors); category needs are correctly computed; arbitrage scoring respects category weights.

### Acceptance criteria

- Recommendations in a keeper session factor in category gaps from the user's keeper set.
- A player who fills a weak category ranks higher than one who doesn't, all else being equal.
- `RecommendationPanel` shows which category need each recommendation addresses.
- Arbitrage alerts boost players who fill keeper-identified gaps.
- Non-keeper sessions produce identical recommendations to today.

## Phase 5: Pre-draft keeper planner

Add a pre-draft planning view that lets the user explore keeper scenarios and see how each affects the draft pool and strategy before committing.

### Context

Keeper decisions and draft strategy are tightly coupled — keeping an extra player means one fewer draft pick, but it also changes which positions you need to draft. The keeper optimizer already solves for optimal keeper sets, but the user can't see how each scenario changes the draft board. This phase connects the optimizer output to the draft tool for "what if" analysis.

### Steps

1. Add a `plan_keeper_draft` GraphQL query that accepts a list of keeper scenarios (each a set of player IDs) and returns, for each scenario: the adjusted draft board (top N players with adjusted values), the user's category needs, and a summary of positional scarcity shifts.
2. Implement the backend by calling `compute_adjusted_valuations()` for each scenario, building a draft board from each, and computing `build_keeper_draft_needs()` for the user's keeper set in each scenario.
3. Build a `KeeperPlannerView` React page (or modal) accessible before starting a draft session:
   - Show the user's keeper optimizer results (optimal set + alternatives from `solve_keepers()`).
   - For each scenario, show a mini draft board preview: top available players, scarcity shifts, and category needs.
   - Let the user toggle players in/out of their keeper set and see the draft board update.
   - "Start draft with this keeper set" button that calls `start_session` with the selected keeper IDs.
4. Add a `KeeperScenarioComparisonType` to the GraphQL schema with fields for each scenario's board preview, scarcity data, and category needs.
5. Cache scenario computations server-side to avoid redundant re-valuation (the adjusted valuation pipeline is CPU-intensive with full ZAR recalculation).
6. Write tests: multiple scenarios return different adjusted boards; toggling a keeper in/out changes scarcity; "start draft" flows through to a keeper-adjusted session.

### Acceptance criteria

- User can compare 2+ keeper scenarios side-by-side with adjusted draft boards.
- Each scenario shows how positional scarcity and category needs change.
- Toggling a player in/out of the keeper set updates the preview.
- "Start draft" creates a session pre-loaded with the chosen keeper set.
- Scenario computation is cached to avoid redundant work.

## Ordering

Phase 1 is the data foundation — it persists the league keeper list via CLI. Phase 2 depends on phase 1 for the keeper IDs to feed into the draft session. Phase 3 depends on phase 2 for the keeper-to-team mapping in the session. Phase 4 depends on phases 2-3 for the keeper context and adjusted pool. Phase 5 depends on phase 2 for adjusted valuations and can proceed in parallel with phases 3-4 for the planning UI, though the "start draft" flow requires phase 2.

Phases 1-2 are the minimum viable integration. Phase 3 adds visibility. Phase 4 adds intelligence. Phase 5 is the capstone that connects pre-draft planning to draft execution.
