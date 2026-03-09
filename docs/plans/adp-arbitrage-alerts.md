# ADP Arbitrage Alerts Roadmap

Surface real-time "falling player" alerts during a live draft: players whose ADP says they should have been picked already but are still available, representing buy-low opportunities. Conversely, flag players being drafted well above their ADP as market inefficiencies to avoid. This is a lightweight overlay on the existing draft state and ADP data that adds a value-hunting lens to draft decisions.

This roadmap depends on: draft state engine (done), ADP integration (done), draft board service (done). Phase 3 additionally depends on web-ui-foundation phases 2-3 (session mutations and subscriptions).

## Status

| Phase | Status |
|-------|--------|
| 1 — Falling player detection | done (2026-03-09) |
| 2 — Draft REPL integration | done (2026-03-09) |
| 3 — Web UI integration | not started |

## Phase 1: Falling player detection

Build the core detection engine that identifies players who have fallen past their ADP and scores the magnitude of the value opportunity.

### Context

ADP represents market consensus on when a player "should" be drafted. When a player's ADP is 45 but pick 60 has passed and they're still available, that's a signal — either the market is wrong (opportunity) or other drafters know something (trap). The detection engine quantifies the opportunity by combining ADP slip with the player's valuation.

### Steps

1. Define domain types in `src/fantasy_baseball_manager/domain/adp_arbitrage.py`:
   - `FallingPlayer` frozen dataclass: `player_id`, `player_name`, `position`, `adp`, `current_pick`, `picks_past_adp` (current_pick - adp), `value`, `value_rank` (rank among available players), `arbitrage_score` (composite of slip magnitude and value).
   - `ArbitrageReport` frozen dataclass: `current_pick`, `falling: list[FallingPlayer]`, `reaches: list[ReachPick]`.
   - `ReachPick` frozen dataclass: `player_id`, `player_name`, `position`, `adp`, `pick_number`, `picks_ahead_of_adp` (adp - pick_number), `drafter_team`.
2. Implement `detect_falling_players()` in `src/fantasy_baseball_manager/services/adp_arbitrage.py`:
   - Takes current pick number, available player pool with ADP and valuations.
   - Identifies players where `current_pick > adp + threshold` (configurable, default 10 picks).
   - Scores each by `arbitrage_score = value * log(1 + picks_past_adp)` — higher value and longer falls score higher.
   - Returns top N falling players sorted by arbitrage score.
3. Implement `detect_reaches()`:
   - Scans completed picks for players drafted `N+ picks` ahead of their ADP.
   - Useful for post-draft analysis and mid-draft awareness of opponent tendencies.
4. Write tests with a mock draft state at various pick numbers, verifying correct detection thresholds and scoring.

### Acceptance criteria

- A player with ADP 40 still available at pick 55 is detected as falling (with default threshold of 10).
- A player with ADP 40 still available at pick 45 is NOT detected (within threshold).
- Higher-value falling players score higher than low-value ones at the same slip magnitude.
- Reaches are correctly identified from completed picks.
- Players without ADP data are excluded (no false positives from missing data).

## Phase 2: Draft REPL integration

Wire falling player detection into the draft session so alerts appear automatically and are available on demand.

### Steps

1. Add `falls` command to the draft REPL:
   - Displays the current falling player list with ADP, current pick, slip, value, and arbitrage score.
   - Accepts optional `--position` filter.
   - Accepts optional `--threshold` to adjust sensitivity (default from config).
2. Add automatic alerts after each pick:
   - When a player crosses a significant threshold (e.g., 20+ picks past ADP and top-50 in value), print a brief alert line after the normal pick confirmation.
   - Keep alerts concise — one line per player, no more than 3 per pick to avoid noise.
3. Add `reaches` command to the REPL:
   - Shows recent reach picks across all teams.
   - Useful for gauging opponent irrationality / different rankings.
4. Add `fbm draft arbitrage` standalone command for pre-draft ADP analysis:
   - Given a draft position and pick number, show which players are likely to be available based on ADP and which represent value if they slip.
5. Write tests for the REPL command parsing and alert triggering logic.

### Acceptance criteria

- `falls` command shows falling players sorted by arbitrage score.
- Automatic alerts fire for significant fallers without overwhelming the output.
- `reaches` command shows actual reach picks from the draft log.
- Alerts respect the `--threshold` setting.
- The standalone `arbitrage` command works for pre-draft scenario analysis.

## Phase 3: Web UI integration

Expose arbitrage detection through GraphQL and add a dashboard panel so falling player alerts appear in real time during a web-based draft.

### Context

The detection engine (phase 1) is a pure service function that takes draft state and returns scored results. Wiring it into the web UI means adding a resolver, a subscription event type, and a React panel. This phase depends on web-ui-foundation phases 2-3 (session mutations and subscriptions) for the session context and real-time event delivery.

### Steps

1. Define Strawberry types: `FallingPlayerType`, `ReachPickType`, `ArbitrageReportType` mirroring the domain dataclasses from phase 1.
2. Add a `arbitrage(sessionId, threshold, position, limit) -> ArbitrageReportType` query that retrieves the active `DraftEngine` from the session manager, extracts the current pick and available pool, and delegates to `detect_falling_players()` and `detect_reaches()`.
3. Add an `ArbitrageAlertEvent` variant to the `DraftEventType` subscription union. After each `pick` mutation, run `detect_falling_players()` and publish alerts for any player crossing the significant-faller threshold (e.g., 20+ picks past ADP and top-50 in value). Limit to 3 alerts per pick to avoid noise.
4. Include the `ArbitrageReportType` in `PickResultType` so the frontend can refresh the arbitrage panel from the same mutation response used for recommendations and roster — no extra round trip.
5. Build an `ArbitragePanel` React component for the draft dashboard sidebar:
   - Shows top falling players sorted by arbitrage score with ADP, current pick, slip, value, and score columns.
   - Position filter dropdown.
   - Threshold slider or input (defaults to config value).
   - Toast-style alerts when significant fallers appear via subscription.
   - Reaches tab showing opponent reach picks from the draft log.
6. Wire the panel into the phase 5 dashboard layout alongside recommendations, roster, needs, and balance.
7. Write tests: resolver returns correct arbitrage data for a given session state, subscription emits alert events after picks, panel renders and filters correctly with mocked Apollo responses.

### Acceptance criteria

- `arbitrage` query returns falling players and reaches scoped to the active session's draft state.
- `PickResultType` includes arbitrage data so all panels update in one mutation response.
- Subscription delivers `ArbitrageAlertEvent` for significant fallers after each pick.
- Dashboard panel displays falling players with sorting, position filtering, and threshold control.
- Reaches tab shows actual reach picks from the session's draft log.
- Alerts are concise — no more than 3 per pick event.

## Ordering

Phase 1 is independent and can start immediately. Phase 2 depends on phase 1 for the detection engine. Phase 3 depends on phase 1 for the detection engine and on web-ui-foundation phases 2-3 (session mutations and subscriptions) for the GraphQL session infrastructure; it also slots into the web-ui-foundation phase 5 dashboard layout. Phases 1-2 are lightweight CLI work; phase 3 is the web integration layer.
