# Improved Age Curves

## What

Replace the single linear age multiplier with component-specific and position-specific aging curves derived from the delta method (comparing matched-pairs of player seasons year over year).

## Why

The current Marcel aging curve applies a uniform +0.6% per year (young) / -0.3% per year (old) to all stats equally. In reality, power peaks later than speed, and catchers age differently from outfielders. Component-specific curves correct systematic over/under-estimation at age extremes, which directly impacts projection accuracy for the youngest and oldest cohorts.

## Pipeline Fit

Replace `MarcelAgingAdjuster` with a new `ComponentAgingAdjuster` that applies per-stat, per-position aging multipliers. Slot it into the same position in the adjuster chain.

## Data Requirements

- 10+ years of player-season pairs from pybaseball (already available)
- Position data per player-season (available via FanGraphs fielding data)

## Key References

- Lichtman, M. "How Do Baseball Players Age?" (The Hardball Times)
- Tango, T. et al. "The Book: Playing the Percentages in Baseball" (Chapter 10)
- JC Bradbury. "Peak Athletic Performance and Ageing" (Journal of Sports Sciences, 2009)

## Implementation Sketch

1. Build aging delta dataset: for each player with consecutive seasons, compute the year-over-year change in each rate stat
2. Group deltas by age and stat (optionally by position group)
3. Fit smoothed curves (LOESS or polynomial) to the deltas
4. Store curves as lookup tables in a new `aging_curves.py` module
5. `ComponentAgingAdjuster` reads position from metadata and applies per-stat multipliers
