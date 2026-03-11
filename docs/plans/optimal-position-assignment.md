# Optimal Position Assignment Roadmap

The current ZAR and SGP valuation pipelines share a greedy position-assignment flaw: each player independently picks whichever eligible position has the lowest replacement level (`compute_var` in `zar/engine.py`). With flex slots (P for pitchers, UTIL for batters), nearly every player claims the flex replacement level, which was computed assuming only a fraction of the pool would use it. The result: fewer players end up with positive VAR than there are roster slots (e.g., 63 pitchers valued instead of 96), and the dollar curve is top-heavy.

A secondary issue is the budget split: `compute_budget_split` divides money proportionally to category count (50/50 with 5 batting / 5 pitching categories), ignoring that batters have more roster slots (9 per team) than pitchers (8 per team). This over-funds pitchers relative to draft demand.

This roadmap replaces the greedy assignment with an optimal solver (using `scipy.optimize.linear_sum_assignment`, already a dependency), fixes the budget split, and validates the changes against holdout seasons using independent targets.

## Status

| Phase | Status |
|-------|--------|
| 1 — Optimal position assignment solver | done (2025-07-08) |
| 2 — Budget split by roster spots | in progress |
| 3 — Integrate into ZAR and SGP models | not started |
| 4 — Holdout validation and production adoption | not started |

## Phase 1: Optimal position assignment solver

Build a standalone solver that optimally assigns players to position slots, respecting slot capacities and position eligibility, maximizing total composite score.

### Context

The current pipeline has three functions that interact to produce the bug:

1. `compute_replacement_level` — for each position, sorts eligible players by composite z and picks the Nth-best (N = spots x teams). This is correct in isolation.
2. `compute_var` — for each player, VAR = composite z minus `min(replacement[pos] for pos in eligible_positions)`. This is the greedy step: every SP-eligible pitcher also sees the P replacement level and picks whichever is lower.
3. `var_to_dollars` — caps the draftable pool at `roster_spots_total` **and** requires positive VAR. When the greedy step makes many players negative-VAR, the pool shrinks below roster slots.

The fix is to replace steps 2 and 3 with a single optimal assignment that:
- Assigns each of the top N players to exactly one slot, where N = total roster slots
- Respects slot capacity (e.g., 24 SP slots, 24 RP slots, 48 P slots)
- Maximizes total composite score across all assignments
- Derives replacement levels from the assignment (worst player per position = replacement)
- Guarantees exactly N players have positive VAR

### Steps

1. **Create `src/fantasy_baseball_manager/models/zar/assignment.py`** with a function:
   ```python
   def assign_positions(
       composite_scores: list[float],
       player_positions: list[list[str]],
       roster_spots: dict[str, int],
       num_teams: int,
   ) -> AssignmentResult
   ```
   The `AssignmentResult` should contain: player-to-slot assignments, per-position replacement levels, and per-player VAR values.

2. **Expand position slots into individual slot entries.** Convert `{"SP": 2, "RP": 2, "P": 4}` with 12 teams into a flat list of 96 slot entries (24 SP, 24 RP, 48 P). Each slot is a column in the cost matrix.

3. **Build the cost matrix.** Select candidates — take approximately 1.5x the total slot count, sorted by composite score descending, as rows. For each candidate-slot pair, set cost = `-composite_score` if the player is eligible for that slot type, else `+inf`. This is a rectangular matrix (more candidates than slots); `linear_sum_assignment` handles this.

4. **Run `scipy.optimize.linear_sum_assignment`** to get the optimal assignment. Extract: which players are assigned, which slot each got, and which players are unassigned (below replacement).

5. **Derive replacement levels from the assignment.** For each position, the replacement level = the composite score of the lowest-assigned player at that position. This replaces `compute_replacement_level`.

6. **Compute VAR from the assignment.** For each assigned player: `VAR = composite_score - replacement[assigned_position]`. Unassigned players get VAR = 0. This replaces `compute_var`.

7. **Add comprehensive tests.** Test cases should include:
   - Simple case: players with single position eligibility (should match current behavior)
   - Multi-position eligibility with capacity constraints (the bug case)
   - Edge case: fewer eligible players than slots for a position
   - Edge case: a player eligible for only a flex slot
   - Verify exactly `roster_spots_total` players get positive VAR
   - Verify the assignment is optimal (total composite score is maximized)

### Acceptance criteria

- `assign_positions` returns an optimal assignment that respects all slot capacities.
- Exactly `sum(spots * num_teams)` players are assigned (or all players if fewer than slots).
- Per-position replacement levels are derived from the worst assigned player at each position.
- All assigned players have VAR >= 0; the player at replacement level has VAR = 0.
- The solver handles the h2h league configuration: 7 batter positions + UTIL on the batter side, SP + RP + P on the pitcher side.
- Performance is acceptable: the solver runs in under 1 second for ~1000 candidates and ~100 slots.

---

## Phase 2: Budget split by roster spots

Change the budget split from category-count-proportional to roster-spot-proportional.

### Context

`compute_budget_split` currently divides the total league budget ($260 x 12 = $3,120) by category count: 5 batting / 5 pitching = 50/50. But the league drafts 9 batters and 8 pitchers per team (108 vs 96 slots). This over-funds pitchers by ~$78 relative to how many are actually drafted.

The roster-spot-proportional split better reflects draft economics: a team spending $260 should allocate roughly in proportion to how many players they need at each pool.

### Steps

1. **Add a `budget_split` option to `LeagueSettings`.** Support two modes: `"categories"` (current default, backward compatible) and `"roster_spots"`. Default to `"roster_spots"` for new behavior.

2. **Update `compute_budget_split`** to accept the split mode. When `"roster_spots"`:
   ```
   batter_fraction = roster_batters / (roster_batters + roster_pitchers)
   ```
   where `roster_batters` = sum of batter position spots + util, and `roster_pitchers` = sum of pitcher position spots. For the h2h league: 9/17 = 52.9% batters ($1,651), 8/17 = 47.1% pitchers ($1,469).

3. **Update `fbm.toml` parsing** to read the new option from the league config section.

4. **Add tests.** Verify both split modes produce correct budget allocations. Verify backward compatibility: omitting the option uses `"roster_spots"` (the new default).

### Acceptance criteria

- Budget split is proportional to roster spots by default.
- The `"categories"` mode is available for leagues that prefer the old behavior.
- Dollar totals still sum to the full league budget.
- The h2h league produces approximately $1,651 batter / $1,469 pitcher split.

---

## Phase 3: Integrate into ZAR and SGP models

Replace the greedy replacement/VAR/dollar pipeline in both `ZarModel` and `SgpModel` with the optimal assignment solver.

### Context

Both models follow the same pattern in their `_value_pool` methods:
1. Compute per-player scores (z-scores for ZAR, SGP scores for SGP)
2. Call `run_zar_pipeline` or `run_sgp_pipeline`, which internally calls `compute_replacement_level` → `compute_var` → `var_to_dollars`
3. Call `best_position` to label each player's position for display

The solver from phase 1 replaces the middle three functions. Both pipelines already produce composite scores before the replacement/VAR step, so the solver slots in cleanly. The position label comes directly from the assignment rather than the `best_position` heuristic.

### Steps

1. **Create a shared pipeline function** (e.g., `run_optimal_pipeline`) that takes composite scores, player positions, roster spots, num_teams, and budget — runs the solver and then `var_to_dollars`. This function is scoring-method-agnostic (works with z-scores or SGP scores).

2. **Update `run_zar_pipeline`** to call the optimal assignment solver instead of the greedy `compute_replacement_level` → `compute_var` chain. Keep the old functions available (they're tested and may be useful for comparison), but the pipeline default should use the solver.

3. **Update `run_sgp_pipeline`** identically — it already delegates to the ZAR replacement/VAR functions, so the change is the same.

4. **Update `ZarModel._value_pool` and `SgpModel._value_pool`** to use the assigned position from the solver result instead of calling `best_position`. The `Valuation.position` field should reflect the solver's assignment.

5. **Wire in the new budget split** from phase 2 — both models call `compute_budget_split` and should now get roster-spot-proportional budgets by default.

6. **Add integration tests.** For both ZAR and SGP:
   - Verify the valued player count equals `roster_spots_total` (not fewer)
   - Verify dollar totals match the budget
   - Verify position assignments respect slot capacities (no more than N players assigned to a position with N slots)

7. **Preserve backward compatibility.** The old greedy behavior should be accessible via a flag or model variant for comparison purposes.

### Acceptance criteria

- `fbm predict zar` and `fbm predict sgp` use optimal assignment by default.
- Both models value exactly `roster_spots_total` players (108 batters + 96 pitchers = 204 for h2h).
- Position assignments respect slot capacities.
- Dollar totals sum to the league budget.
- Position labels on valuations reflect the solver assignment, not the `best_position` heuristic.
- A flag or variant exists to run the old greedy pipeline for comparison.

### Dependencies

- Phase 1 (solver) and phase 2 (budget split) must be complete.

---

## Phase 4: Holdout validation and production adoption

Validate the optimal-assignment pipeline against holdout seasons using independent targets, compare to the current `zar-reformed` production system, and adopt if improved.

### Context

The [Valuation Reform](valuation-reform.md) roadmap established the validation protocol: evaluate on 2024 and 2025 holdout seasons using WAR correlation (overall, batters, pitchers), top-N hit rates (25, 50, 100), and pitcher-focused mispricing analysis. The current production system (`zar-reformed`) achieved pitcher WAR rho of 0.218/0.248 on those seasons.

The optimal assignment should produce more realistic dollar values (less top-heavy, more players valued) and better position-adjusted rankings, which should improve hit rates and WAR correlation — especially for pitchers, where the flex-slot problem was most severe.

### Steps

1. **Generate holdout valuations** for the new system:
   ```bash
   fbm predict zar --season 2024 --param league=h2h --param projection_system=steamer --version holdout-optimal
   fbm predict zar --season 2025 --param league=h2h --param projection_system=steamer --version holdout-optimal
   ```
   Also generate SGP variants for comparison:
   ```bash
   fbm predict sgp --season 2024 --param league=h2h --param projection_system=steamer --version holdout-optimal
   fbm predict sgp --season 2025 --param league=h2h --param projection_system=steamer --version holdout-optimal
   ```

2. **Run the evaluation framework comparison** against the current production baseline:
   ```bash
   fbm valuations compare zar-reformed/production zar/holdout-optimal --season 2024 --league h2h --check
   fbm valuations compare zar-reformed/production zar/holdout-optimal --season 2025 --league h2h --check
   ```

3. **Build a comparison matrix** covering: system x season x metric (WAR rho all/batter/pitcher, hit-25/50/100, valued player count, dollar distribution shape).

4. **Pitcher-focused analysis.** For each system, examine the top-20 pitcher valuations. Verify that the optimal assignment:
   - Values the expected number of pitchers (96)
   - Distributes dollars more evenly (top pitcher < $80, not $118)
   - Correctly distinguishes SP scarcity from RP scarcity

5. **SGP reassessment.** With the flex-slot problem fixed, re-evaluate whether SGP becomes competitive. The previous rejection was partly driven by reliever inflation, which the optimal assignment should mitigate. If SGP improves significantly, note this for future work.

6. **Adopt the winner.** If the optimal-assignment ZAR improves on holdout metrics without degrading batter accuracy, adopt as the new production default and regenerate 2026 valuations.

7. **Document results** in this roadmap's status table with the comparison matrix and go/no-go decision.

### Acceptance criteria

- Comparison matrix covering optimal-assignment ZAR (and optionally SGP) vs current zar-reformed on both holdout seasons.
- Valued player count is exactly 204 (108 batters + 96 pitchers) — up from current ~153.
- Pitcher WAR rho does not regress from baseline (>= 0.218 on 2024, >= 0.248 on 2025).
- Top-N hit rates do not regress on any N value.
- Dollar distribution is less top-heavy: top pitcher value < $80 (down from $118).
- If adopted, 2026 production valuations are regenerated.

### Gate: go/no-go

**Go** if optimal-assignment ZAR matches or improves all independent targets (WAR rho, hit rates) on both holdout seasons and produces a more realistic dollar distribution. **No-go** if WAR rho regresses on either season, indicating the greedy assignment was accidentally helping by concentrating value on high-WAR players. If no-go, document findings — the solver is still architecturally correct and the regression would point to a separate scoring problem.

---

## Ordering

- **Phase 1** (solver) has no dependencies and is the foundation for everything else.
- **Phase 2** (budget split) is independent of phase 1 and can be implemented in parallel.
- **Phase 3** (integration) depends on both phases 1 and 2.
- **Phase 4** (validation) depends on phase 3 and requires holdout projection data (already available from the valuation-reform roadmap).
- The [Valuation Reform](valuation-reform.md) roadmap's `zar-reformed` settings (SV+HLD weight = 0, min_ip = 60) remain in effect — this roadmap changes the assignment/budget mechanics, not the category signal.
