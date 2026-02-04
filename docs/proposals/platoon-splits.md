# Platoon Splits Projections

## What

Project performance separately by batter-pitcher handedness matchup (vs. LHP and vs. RHP), then weight by expected matchup frequency to produce a blended full-season projection.

## Why

Platoon effects are large and persistent: the average LHB hits ~30 points of wOBA worse against LHP than RHP. The current system projects a single rate for each player, which undervalues platoon-advantaged hitters and overvalues platoon-disadvantaged ones. For fantasy purposes, this matters when evaluating part-time players who face favorable matchups disproportionately.

## Pipeline Fit

New `PlatoonRateComputer` wrapping the base `RateComputer`. It computes separate rate sets for vs-LHP and vs-RHP splits, then blends them using expected matchup distribution (typically ~72% vs. RHP, ~28% vs. LHP for right-handed batters).

## Data Requirements

- Split stats by pitcher handedness via pybaseball's `batting_stats()` with split parameter
- League-wide matchup frequency distribution (available from Retrosheet or FanGraphs)

## Key References

- Tango, T. et al. "The Book" (Chapter 6: Platoon Effects)
- Lichtman, M. "Platoon Splits and Sample Size" (The Book Blog)
- Dolphin, A. "Predicting Platoon Splits" (Beyond the Box Score)

## Implementation Sketch

1. Create split data source: `SplitStatsDataSource` providing vs-LHP and vs-RHP season stats
2. Run base `RateComputer` twice: once with vs-LHP data, once with vs-RHP
3. Apply higher regression to splits (smaller samples) — approximately 2x the full-season regression
4. Blend: `rate = pct_vs_rhp * rate_vs_rhp + pct_vs_lhp * rate_vs_lhp`
5. Store split rates in metadata for downstream consumers
