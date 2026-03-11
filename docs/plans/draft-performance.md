# Draft Performance Roadmap

The draft UI is sluggish both when starting a session and when making picks. Profiling reveals several compounding issues: duplicate expensive computations per pick (category balance scores computed 3×, arbitrage computed 2×), O(n²) `_tier_urgency` calls in the recommendation reason builder, a redundant `refetchQueries` round-trip on the frontend, and missing database indexes on the valuation and ADP tables that make session startup slower than necessary. This roadmap addresses each bottleneck in order of impact.

## Status

| Phase | Status |
|-------|--------|
| 1 — Eliminate duplicate computation per pick | done (2026-03-10) |
| 2 — Cache tier urgency within recommend() | done (2026-03-10) |
| 3 — Remove redundant frontend refetchQueries | done (2026-03-11) |
| 4 — Add database indexes for session startup | not started |

## Phase 1: Eliminate duplicate computation per pick

The `pick` mutation in `schema.py` computes expensive work twice: once inline (lines 529-534) for arbitrage alerts, and again inside `_build_pick_result()` (lines 96-109) for the mutation response. The category balance function (`cat_bal_fn`) — which iterates all available players against the roster — is called 3 times: once in the pick mutation for arbitrage cat scores, once in `_build_pick_result` for arbitrage cat scores, and once inside `recommend()`.

### Context

Each pick triggers `_compute_arbitrage_cat_scores()` twice and `recommend()` once, all of which invoke the category balance function. The category balance function iterates all available players and simulates roster additions, making it the most expensive per-call operation. `detect_falling_players` is also called twice — once for alerts and once inside `build_arbitrage_report`.

### Steps

1. Refactor the `pick` mutation to compute category balance scores (`_compute_arbitrage_cat_scores`) once and pass the result to both the alert-publishing code and `_build_pick_result`.
2. Change `_build_pick_result` to accept pre-computed `arb_cat_scores` as an optional parameter instead of recomputing them internally.
3. Pass the same `arb_cat_scores` into `recommend()` via the `category_balance_fn` parameter — or restructure so `recommend()` receives the pre-computed `cat_scores` dict directly instead of calling the function again.
4. Verify that the arbitrage alert subscription still fires correctly and that `PickResultType` still contains correct arbitrage data.

### Acceptance criteria

- `_compute_arbitrage_cat_scores` is called at most once per `pick` mutation execution.
- `category_balance_fn` (the underlying scoring function) is invoked at most once per pick, not once per consumer.
- All existing draft tests pass unchanged.
- Manual test: start a draft session, make 5 picks, verify recommendations and arbitrage alerts still appear correctly.

## Phase 2: Cache tier urgency within recommend()

`_tier_urgency` sorts all players at a position on every call. It's called once per player in `_score_player`, then again for each of the top-10 players in `_build_reason` — resulting in redundant O(n log n) sorts for each recommendation.

### Context

`_tier_urgency(player, pool)` in `draft_recommender.py` builds a sorted list of same-position players every time it's called. With ~200 available players and ~10 positions, each call sorts ~20 players. Across scoring + reason-building, this happens ~210+ times per pick. Pre-computing tier urgency once per position eliminates all but ~10 sorts.

### Steps

1. Add a `_compute_tier_urgency_map` function that pre-computes tier urgency for all players in the pool, returning a `dict[int, float]` keyed by `player_id`.
2. Call it once at the top of `recommend()` and pass the map to both `_score_player` and `_build_reason` instead of the raw `pool` list.
3. Update `_score_player` and `_build_reason` signatures to accept the pre-computed map instead of calling `_tier_urgency` per player.
4. Update all existing tests for `_score_player`, `_build_reason`, and `recommend` to pass the new parameter.

### Acceptance criteria

- `_tier_urgency` (or its replacement) performs at most one sort per position per `recommend()` call, not one per player.
- All existing `draft_recommender` tests pass.
- Recommendation output is identical before and after (same scores, same reasons for the same inputs).

## Phase 3: Remove redundant frontend refetchQueries

Both `pickMutation` and `undoMutation` in `DraftDashboard.tsx` specify `refetchQueries: [BALANCE_QUERY]`, which triggers a separate GraphQL query to the backend after each mutation. The `PickResultType` response already contains all the data the UI needs — the `BALANCE_QUERY` refetch is redundant and adds a full backend round-trip (including recomputing category balance from scratch).

### Context

The `BALANCE_QUERY` is also used on initial load (`useQuery` at line 45) to populate balance data when a session is active. After a pick, though, `ctx.applyPickResult` already processes the mutation response which includes updated state. The refetch re-fetches the same data at the cost of another backend computation cycle.

### Steps

1. Remove the `refetchQueries` option from both `pickMutation` and `undoMutation` in `DraftDashboard.tsx`.
2. Verify that `applyPickResult` already updates the balance state from the mutation response. If it doesn't, extend it to do so — the `PickResultType` includes the necessary data.
3. If `BALANCE_QUERY` is still needed for initial session load or resume, keep the `useQuery` call but remove the mutation-triggered refetch.
4. Update the `DraftDashboard` test mocks to remove the `BALANCE_QUERY` refetch mock from pick/undo test scenarios.

### Acceptance criteria

- No `refetchQueries` on `pickMutation` or `undoMutation`.
- Category balance data still updates in the UI after each pick and undo.
- Network tab shows one GraphQL request per pick (the mutation), not two.
- All frontend tests pass.

## Phase 4: Add database indexes for session startup

The `valuation` and `adp` tables have no indexes beyond their UNIQUE constraints. `_build_player_pool` queries both tables by `(season, system, version)` and `(season, provider)` respectively. On session start, these are full table scans over potentially thousands of rows.

### Context

SQLite's UNIQUE constraints do create implicit indexes, but only on the full composite key. The valuation table's UNIQUE is on `(player_id, season, system, version, player_type)` — a query filtering on `(season, system, version)` can't efficiently use this index because `player_id` is the leading column. Similarly, the ADP table's UNIQUE on `(player_id, season, provider, positions, as_of)` doesn't help queries filtering on `(season, provider)`.

### Steps

1. Create migration `032_draft_query_indexes.sql` with covering indexes:
   - `CREATE INDEX idx_valuation_season_system_version ON valuation(season, system, version);`
   - `CREATE INDEX idx_adp_season_provider ON adp(season, provider);`
2. Run the migration and verify it applies cleanly.
3. Verify session start time improvement with a quick before/after timing test (e.g., `time` the `start_session` mutation or the `_build_player_pool` call).

### Acceptance criteria

- Migration `032_draft_query_indexes.sql` exists and applies without error.
- `EXPLAIN QUERY PLAN` for the valuation and ADP `get_by_season` queries shows index usage instead of full table scans.
- All existing tests pass (migration is picked up automatically by the test DB setup).

## Ordering

Phases are independent and can be implemented in any order. Suggested priority by impact:

1. **Phase 1** (duplicate computation) — largest impact; eliminates 2× redundant category balance scoring per pick.
2. **Phase 2** (tier urgency cache) — reduces O(n²) to O(n) in the recommendation hot path.
3. **Phase 3** (refetchQueries) — eliminates a full redundant backend round-trip per pick.
4. **Phase 4** (indexes) — improves session startup; less impact on per-pick latency since data is in-memory after start.
