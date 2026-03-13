# Draft Order Roadmap

Custom draft order support so snake drafts respect the actual pick order rather than assuming team 1 picks first, team 2 picks second, etc. Today the snake draft math in `_snake_team()` maps team numbers directly to pick positions. The Yahoo API doesn't expose an explicit draft order endpoint, so the order must be set manually (via CLI or web UI). The plumbing already exists — `pick_overrides` in `DraftState` can remap any pick to any team — but there's no way to seed it at session start.

## Status

| Phase | Status |
|-------|--------|
| 1 — Domain & engine support | done (2026-03-12) |
| 2 — CLI + web wiring | not started |

## Phase 1: Domain & engine support

Add `draft_order` as a first-class concept that flows from session creation through to pick resolution, persisted alongside the session record.

### Context

`DraftState.pick_overrides` already supports remapping pick → team, but it's only populated by trades. There's no way to set an initial draft order. `DraftConfig` has no field for it, `DraftSessionRecord` doesn't persist it, and `load_draft_from_db` doesn't restore it. The `YahooDraftSetupInfo` domain type has a `draft_order: list[int]` field that's always empty.

### Steps

1. Add `draft_order: list[int] | None` to `DraftConfig` — a list where index 0 is the team that picks first, index 1 picks second, etc. (e.g., `[12, 11, 10, ...]` for reverse-standings order). `None` means default snake math.
2. Update `DraftEngine.start()` to seed `pick_overrides` from `draft_order` when provided. For a 12-team snake draft with order `[12, 11, ..., 1]`, picks 1–12 map to teams 12, 11, ..., 1; picks 13–24 reverse; and so on. This replaces `_snake_team()` for the initial layout while still allowing trades to layer on top.
3. Add `draft_order: list[int] | None` to `DraftSessionRecord`.
4. Add a DB migration adding a `draft_order` JSON column to `draft_session`.
5. Update `SqliteDraftSessionRepo` to persist and load `draft_order`.
6. Update `load_draft_from_db` to pass `draft_order` into `DraftConfig` so resumed sessions restore the custom order before replaying trades.
7. Update `SessionManager.start_session()` to accept and pass through `draft_order`.

### Acceptance criteria

- `DraftEngine` with `draft_order=[12, 11, ..., 1]` assigns pick 1 to team 12, pick 2 to team 11, etc., and reverses correctly in round 2.
- Trades layer on top of custom draft order (a traded pick overrides the draft-order assignment).
- A session with custom draft order can be persisted, reloaded, and the pick assignments are identical after reload.
- `draft_order=None` preserves existing behavior (team N picks Nth).

## Phase 2: CLI + web wiring

Expose draft order through the CLI and web UI so users can set it when starting a draft.

### Context

After phase 1 the engine and persistence support draft order, but there's no user-facing way to set it. The CLI `draft start` command and the GraphQL `startSession` mutation need a new parameter. The web UI's `SessionControls` component should accept draft order from the Yahoo prefill (when available) or allow manual entry.

### Steps

1. Add `--draft-order` option to the CLI `draft start` command, accepting a comma-separated list of team IDs in pick order (e.g., `--draft-order 12,11,10,9,8,7,6,5,4,3,2,1`).
2. Add `draftOrder: [Int!]` parameter to the `startSession` GraphQL mutation. Parse and pass to `SessionManager.start_session()`.
3. Populate `YahooDraftSetupInfo.draft_order` from `previous_season_team_rank` on Yahoo teams — derive reverse-standings order as a reasonable default for keeper leagues. This replaces the hardcoded `[]` in the `yahoo_draft_setup` query.
4. Update `SessionControls.tsx` to use `draftOrder` from the Yahoo prefill when available: set it in state and include it in the `onStart` payload.
5. Add a draft-order input to `SessionControls` (a text field or reorderable list) so users can manually adjust the order before starting.
6. Include `draftOrder` in the `START_SESSION` mutation variables and the `DraftStateType` response so the frontend knows the active order.

### Acceptance criteria

- `fbm draft start --draft-order 12,11,10,...,1` starts a session where team 12 picks first.
- The web UI prefills draft order from Yahoo previous-season standings (reverse order) for keeper leagues.
- Users can manually edit draft order in the web UI before starting.
- The `startSession` mutation accepts and persists `draftOrder`.
- Omitting `--draft-order` / `draftOrder` preserves existing default behavior.

## Ordering

Phase 1 must complete before phase 2. No external dependencies — all prerequisite infrastructure (pick_overrides, session persistence, Yahoo team data) already exists.
