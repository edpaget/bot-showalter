# Tier Generator Roadmap

Cluster players into draft tiers by position so that during a draft you can quickly see when a meaningful talent dropoff occurs and decide whether to reach for a player or wait. Tiers collapse the granularity of per-player dollar values into actionable bands — "these 5 shortstops are roughly interchangeable, but a big gap separates them from the next group."

The existing ZAR valuation system provides the per-player dollar values and positional assignments that serve as input. The tier generator adds a clustering layer on top, producing labeled tier assignments that can be consumed by the draft board export, the live draft tracker, and the category balance tracker.

## Status

| Phase | Status |
|-------|--------|
| 1 — Tier clustering engine | not started |
| 2 — CLI command and formatted output | not started |
| 3 — Cross-position tier map | not started |

## Phase 1: Tier clustering engine

[Phase plan](tier-generator/phase-1.md)

Build the core tier assignment algorithm as a standalone service, operating on valuation data.

### Context

ZAR rankings produce a continuous dollar value per player per position. For draft purposes, these need to be discretized into tiers. Natural breaks (Jenks) or simple gap detection on the sorted value curve are more appropriate than k-means here, since tier boundaries should correspond to visible dropoffs in value rather than arbitrary cluster centers.

### Steps

1. Create `src/fantasy_baseball_manager/services/tier_generator.py` with a `TierAssignment` frozen dataclass (`player_id`, `player_name`, `position`, `tier`, `value`, `rank`).
2. Implement a `generate_tiers(valuations, method, max_tiers)` function that:
   - Groups valuations by position.
   - Sorts by value descending within each position.
   - Applies the selected method to find tier boundaries (gap-based or Jenks natural breaks).
   - Returns `list[TierAssignment]`.
3. Implement **gap-based** method: compute pairwise value differences in the sorted list, mark a new tier wherever the gap exceeds a threshold (e.g., 1.5× the median gap for that position).
4. Implement **Jenks natural breaks** method using the `jenkspy` library (or a lightweight pure-Python implementation to avoid the dependency).
5. Accept a `max_tiers` parameter that caps the number of tiers per position.
6. Write tests with synthetic valuation data that has obvious clusters, verifying tier boundaries land in the gaps.

### Acceptance criteria

- `generate_tiers` returns tier-labeled players grouped by position.
- Gap-based and Jenks methods both produce reasonable tiers on test data.
- `max_tiers` correctly limits the number of tiers.
- Players with equal value are placed in the same tier.

## Phase 2: CLI command and formatted output

Expose the tier generator through the CLI with clear, position-grouped tabular output.

### Context

Draft-day usage requires a quick-reference view: for each position, show players grouped by tier with the tier boundary visually marked. This follows the pattern of `fbm valuations rankings` but adds tier grouping.

### Steps

1. Add `fbm draft tiers --season <year> --system <valuation-system> --method <gap|jenks>` CLI command under a new `draft` command group.
2. Format output as position-grouped tables with tier separators (blank line or horizontal rule between tiers).
3. Include columns: tier number, rank (within position), player name, value, and (if ADP data exists) ADP rank.
4. Support `--position <pos>` filter to show a single position.
5. Support `--max-tiers <n>` to control granularity.
6. Write integration tests verifying CLI output format.

### Acceptance criteria

- `fbm draft tiers` produces position-grouped tier output.
- Tier boundaries are visually clear in the output.
- Position and max-tier filters work correctly.

## Phase 3: Cross-position tier map

Add a unified cross-position tier view that helps answer "which positions have the most remaining value in the current tier?"

### Context

During a draft, you often want to compare across positions: "there are 3 tier-1 outfielders left but only 1 tier-1 shortstop — grab the shortstop." A cross-position summary aggregates tier counts and remaining value.

### Steps

1. Add a `tier_summary(tiers)` function that computes per-position, per-tier: player count, total value, average value, best available player name.
2. Add `fbm draft tier-summary --season <year> --system <system>` command showing a matrix: positions as rows, tiers as columns, with player counts in each cell.
3. Write tests covering the summary aggregation.

### Acceptance criteria

- Summary matrix correctly counts players per position per tier.
- Total value and best-available are accurate.
- Output is compact enough to scan at a glance.

## Ordering

Phase 1 → 2 → 3, strictly sequential. Phase 1 is the core logic, phase 2 makes it usable, phase 3 adds draft-day polish. No external dependencies — this roadmap can proceed independently of ADP integration (though phase 2 optionally displays ADP data if available).
