# Draft Pick Trades Roadmap

Allow pick-for-pick trades in live draft sessions. The approach adds a pick-ownership override map to `DraftState` so that `_snake_team()` respects traded picks. The existing `PickTrade` domain model, `evaluate_pick_trade()`, and cascade analysis from the pick-trade evaluator roadmap provide the analytical foundation — this roadmap wires that into live sessions with persistence, GraphQL mutations, and a frontend trade dialog.

## Status

| Phase | Status |
|-------|--------|
| 1 — Domain & engine | done (2026-03-11) |
| 2 — Persistence & session integration | done (2026-03-11) |
| 3 — GraphQL API | done (2026-03-11) |
| 4 — Frontend trade dialog | not started |

## Phase 1: Domain & engine

Add pick-ownership overrides to `DraftState` and wire `DraftEngine` to respect them during snake-order validation and `team_on_clock()`.

### Context

`DraftEngine._snake_team()` computes team ownership purely from the snake formula. There is no mechanism to override this, so traded picks cannot be represented in a live session. The `PickTrade` domain type and `evaluate_pick_trade()` already exist but are disconnected from live draft state.

### Steps

1. Add a `pick_overrides: dict[int, int]` field to `DraftState` (default empty dict). This maps `pick_number → team` for any picks whose ownership differs from the snake formula.
2. Add a `trade_picks(gives: list[int], receives: list[int], partner_team: int)` method to `DraftEngine` that:
   - Validates all `gives` picks belong to `user_team` (checking overrides first, then snake formula).
   - Validates all `receives` picks belong to `partner_team`.
   - Validates none of the traded picks have already been used (pick_number < current_pick).
   - Updates `pick_overrides` for all affected picks (gives → partner_team, receives → user_team).
   - Returns a `DraftTrade` domain object (new frozen dataclass capturing session_id, teams, pick lists, timestamp).
3. Update `_snake_team()` (or introduce a `team_for_pick()` instance method) to consult `pick_overrides` before falling back to the formula.
4. Update `team_on_clock()` and the snake validation branch in `pick()` to use the new resolution path.
5. Add an `undo_trade()` method that reverses the most recent trade (removes its overrides from `pick_overrides`).
6. Add a `DraftTrade` frozen dataclass to the domain layer capturing: trade_id, session_id, team_a, team_b, team_a_gives, team_b_gives, executed_at.
7. Unit-test: snake draft with traded picks — verify `team_on_clock()` returns the new owner, `pick()` validates against the override, and `undo_trade()` restores original ownership.

### Acceptance criteria

- `team_on_clock()` returns the traded-to team for overridden picks.
- `pick()` enforces the overridden team, not the snake formula team.
- `trade_picks()` rejects trades involving already-used picks.
- `trade_picks()` rejects trades where gives/receives don't belong to the claimed teams.
- `undo_trade()` restores original snake ownership.
- All existing draft engine tests continue to pass (no regressions for sessions without trades).

## Phase 2: Persistence & session integration

Persist pick trades alongside draft sessions so traded pick ownership survives session resume.

### Context

`SqliteDraftSessionRepo` persists picks and session metadata but has no concept of trades. Without persistence, resumed sessions would lose trade information and revert to default snake order.

### Steps

1. Add a `draft_session_trade` migration table: `id, session_id, team_a, team_b, team_a_gives (JSON list), team_b_gives (JSON list), executed_at`. Foreign key to `draft_session(id)`.
2. Add `save_trade()`, `load_trades()`, and `delete_trade()` methods to `SqliteDraftSessionRepo`.
3. Update `DraftSession._persist_pick()` flow: after `trade_picks()`, call `save_trade()`.
4. Update session resume logic (web + CLI) to call `load_trades()` and rebuild `pick_overrides` from the trade ledger.
5. Wire `undo_trade()` to `delete_trade()` for persistence.
6. Test round-trip: create session → trade picks → resume session → verify overrides restored.

### Acceptance criteria

- Trades survive session close and resume.
- `load_trades()` correctly rebuilds `pick_overrides`.
- Deleting a trade via `undo_trade()` removes the DB row and restores snake ownership.
- Resuming a session with multiple trades reconstructs the correct override map.

## Phase 3: GraphQL API

Expose pick trades via GraphQL mutations and subscription events so the web UI can execute and observe trades.

### Context

The web layer has `pick`, `undo`, `startSession`, `endSession` mutations and `DraftEventType` subscription events. Trades need analogous mutation + event support, plus a query for trade evaluation (reusing the existing `evaluate_pick_trade` service).

### Steps

1. Add `DraftTradeType` Strawberry type mapping from the `DraftTrade` domain object.
2. Add `PickTradeEvaluationType` Strawberry type mapping from `PickTradeEvaluation`.
3. Add `tradePicks(sessionId, gives, receives, partnerTeam)` mutation that:
   - Calls `DraftEngine.trade_picks()`.
   - Persists via repo.
   - Publishes a `TradeEvent` to subscriptions.
   - Returns the updated `DraftStateType` plus the `DraftTradeType`.
4. Add `undoTrade(sessionId)` mutation that reverses the last trade.
5. Add `evaluateTrade(sessionId, gives, receives)` query that returns `PickTradeEvaluationType` using the existing `evaluate_pick_trade()` — provides a preview before committing.
6. Add `TradeEvent` to the `DraftEventType` union so subscribed clients see trades in real time.
7. Update `DraftStateType` to include a `trades: list[DraftTradeType]` field.
8. Run `bun run schema:export && bun run codegen` to update frontend types.

### Acceptance criteria

- `tradePicks` mutation executes a trade and returns updated state with correct `team_on_clock`.
- `undoTrade` mutation reverses the last trade.
- `evaluateTrade` query returns value analysis without modifying state.
- `TradeEvent` is broadcast to subscribed clients.
- Schema export and codegen produce updated frontend types with no diff errors.

## Phase 4: Frontend trade dialog

Add a trade UI to the draft dashboard that lets users evaluate and execute pick trades.

### Context

The draft dashboard has no trade controls. Users need to select picks to give/receive, preview the value analysis, and execute or cancel.

### Steps

1. Add a "Trade Picks" button to `DraftDashboard` (visible only for snake format, disabled when no picks remain).
2. Build a `TradeDialog` modal component:
   - Two columns: "You Give" and "You Receive".
   - Each column shows remaining picks for the respective team, selectable via checkboxes.
   - Partner team selector (dropdown of other teams).
   - "Evaluate" button calls the `evaluateTrade` query and displays the `PickTradeEvaluation` result (gives value, receives value, net value, recommendation).
   - "Execute Trade" button calls `tradePicks` mutation, closes dialog, updates state via `applyPickResult`-style handler.
   - "Cancel" button closes without action.
3. Update `DraftSessionContext` to handle `TradeEvent` from subscriptions — apply pick overrides to local state.
4. Update `PickLogPanel` to show trade events in the pick history timeline.
5. Visual indicator on the draft board for picks whose ownership has been traded (e.g., team badge or color change on the pick-order row).
6. Add Vitest tests for `TradeDialog` with mocked GraphQL responses.

### Acceptance criteria

- User can open trade dialog, select picks, and see value evaluation before committing.
- Executing a trade updates `team_on_clock` display and pick-order indicators immediately.
- Trade events received via subscription update all connected clients.
- Pick log shows trade entries alongside regular picks.
- Dialog is only available during snake-format drafts with future picks remaining.

## Ordering

Phases are strictly sequential:

1. **Phase 1** (domain & engine) has no dependencies beyond existing code.
2. **Phase 2** (persistence) depends on phase 1's `DraftTrade` type and `trade_picks()` method.
3. **Phase 3** (GraphQL) depends on phase 2 for persistence wiring.
4. **Phase 4** (frontend) depends on phase 3 for the GraphQL schema and generated types.

The existing `PickTrade` evaluation infrastructure (from the completed draft-pick-trade-evaluator roadmap) is a foundation — phase 3 reuses `evaluate_pick_trade()` directly.
