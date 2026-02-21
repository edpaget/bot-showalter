# Positional Scarcity Curves Roadmap

Analyze and report the value dropoff curve per position to guide draft timing decisions. Positional scarcity answers the question: "how much value do I lose by waiting to draft this position?" A position with a steep dropoff (e.g., catcher) demands early attention, while a position with a flat curve (e.g., outfield) can be deferred safely.

The ZAR valuation system already computes per-player dollar values with positional assignments. This roadmap adds analysis on top of that data — computing dropoff metrics, generating scarcity reports, and optionally visualizing the curves.

## Status

| Phase | Status |
|-------|--------|
| 1 — Scarcity metrics and report | not started |
| 2 — Per-position value curves | not started |
| 3 — Scarcity-adjusted rankings | not started |

## Phase 1: Scarcity metrics and report

[Phase plan](positional-scarcity/phase-1.md)

Compute positional scarcity metrics and expose them through a CLI report.

### Context

The key insight for draft strategy is not just "who is the best available?" but "where is value disappearing fastest?" A position where the gap between rank 5 and rank 15 is $20 is far more urgent to address than one where the same gap is $3. Currently, the `fbm valuations rankings` command shows a flat list — there's no positional dropoff analysis.

### Steps

1. Create `src/fantasy_baseball_manager/services/positional_scarcity.py` with a `PositionScarcity` frozen dataclass: `position`, `tier_1_value` (avg of top N, where N = league roster slots for that position), `replacement_value`, `total_surplus` (sum of value above replacement), `dropoff_slope` (linear fit slope of value vs. rank), `steep_rank` (rank at which the dropoff accelerates — elbow detection).
2. Implement `compute_scarcity(valuations, league_settings)` that:
   - Groups valuations by position.
   - Computes the above metrics using league roster slot counts to set "starter" thresholds (e.g., 12-team league with 1 C slot → top 12 catchers are starters).
   - Ranks positions by scarcity (steepest dropoff first).
3. Add `fbm draft scarcity --season <year> --system <system>` CLI command.
4. Output a table: position, top-tier avg value, replacement value, surplus, dropoff slope, recommended draft round window.
5. Write tests with synthetic valuation data where some positions are scarce and others are deep.

### Acceptance criteria

- Scarcity report ranks positions by dropoff severity.
- Metrics are computed relative to league settings (number of teams × roster slots per position).
- Positions with steep dropoffs rank higher (more scarce).
- Report includes all batting and pitching positions.

## Phase 2: Per-position value curves

Add a detailed per-position view showing the full value curve from rank 1 through replacement level.

### Context

The summary metrics from phase 1 give the headline, but sometimes you want the full picture — especially to identify "cliffs" at specific ranks. For example, knowing that SS value holds steady through rank 8 then drops sharply at rank 9 tells you exactly when to target SS.

### Steps

1. Add `PositionValueCurve` dataclass: `position`, `values` (list of `(rank, player_name, value)` tuples from rank 1 through N × teams).
2. Implement `compute_value_curves(valuations, league_settings)` returning curves for all positions.
3. Add `fbm draft scarcity --position <pos> --detail` flag that shows the full value curve for one position.
4. Format output to visually indicate the "cliff" point (e.g., marker or separator at the elbow).
5. Write tests verifying curve generation and cliff detection.

### Acceptance criteria

- Detailed view shows every player at the position with rank and value.
- Cliff point is identified and marked in the output.
- Works for all positions including multi-slot positions (OF, UTIL).

## Phase 3: Scarcity-adjusted rankings

Produce an alternative ranking that adjusts player values by their position's scarcity — boosting players at scarce positions and discounting those at deep positions.

### Context

Standard ZAR already accounts for replacement level, but it uses a fixed replacement level per position. Scarcity-adjusted rankings go further by incorporating the *shape* of the dropoff curve. A player at a scarce position who is 2 tiers above the cliff is worth more than a player at a deep position with the same raw dollar value, because the opportunity cost of missing the scarce-position player is higher.

### Steps

1. Implement `scarcity_adjusted_value(value, position_scarcity)` that applies a multiplier based on positional scarcity (e.g., value × (1 + normalized_scarcity_score)).
2. Add `fbm draft scarcity-rankings --season <year> --system <system>` command showing the re-ranked player list.
3. Show both original value and adjusted value for transparency.
4. Write tests verifying that players at scarce positions get boosted.

### Acceptance criteria

- Scarcity-adjusted rankings differ from raw ZAR rankings.
- Players at scarce positions rank higher than they would in raw rankings.
- Both original and adjusted values are displayed.
- Adjustment is proportional — huge scarcity = big boost, mild scarcity = small boost.

## Ordering

Phase 1 → 2 → 3, sequential. Phase 1 is the most important — the scarcity summary directly informs draft strategy. Phase 2 adds granularity. Phase 3 is experimental and should be validated against historical draft outcomes before relying on it. No dependencies on other roadmaps, though the results complement the tier generator and draft board export.
