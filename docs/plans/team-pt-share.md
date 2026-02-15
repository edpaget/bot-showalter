# Playing Time v2 — Phase 1d: Team PT Share

## Status

Deferred from Phase 1 feature engineering. This feature requires cross-table aggregation that the current feature DSL does not support.

## Goal

Add a team-level playing-time share feature (e.g., a player's PA as a fraction of their team's total PA at their position) to the playing-time model. Team PT share captures roster competition dynamics — a player on a crowded roster may see fewer PA even with strong individual metrics.

## Approach Options

### Option A: New Source enum + cross-table TransformFeature

Extend the `Source` enum with a `TEAM_BATTING` / `TEAM_PITCHING` variant. The assembler would join team-level aggregates alongside player rows, and a `DerivedTransformFeature` would compute the ratio. This keeps the feature DSL consistent but requires assembler changes to handle team-level queries.

### Option B: Pre-computed team totals table

Materialize team-level PT totals (e.g., team PA by position-season) into a separate table or view during data ingestion. The feature would then reference this table via a standard `Feature` with a new source. Simpler assembler changes but adds a data pipeline step.

### Option C: Inline aggregation in transform

Pass the full team roster data into the transform via a multi-row group. The transform would compute the share internally. This avoids schema changes but breaks the current single-row-per-player transform contract.

## Recommendation

Option B (pre-computed team totals) is likely the best balance of simplicity and correctness. It avoids complicating the assembler's join logic and keeps transforms stateless. The materialized table can also serve other features (e.g., team strength, positional depth).

## Dependencies

- Requires team/roster data to be available in the data layer
- May depend on position assignment logic (which position does a multi-position player count toward?)
