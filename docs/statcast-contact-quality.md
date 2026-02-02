# Statcast Contact Quality Integration

## What

Incorporate Statcast metrics (barrel rate, exit velocity, expected wOBA) into HR and BABIP-adjacent projections as a secondary signal alongside traditional counting stats.

## Why

Traditional Marcel uses only counting stats (HR, H, 2B, etc.), which conflate batted-ball quality with luck. A hitter whose barrel rate increased but whose HR count stayed flat (due to bad luck or park) is undervalued. Statcast data provides the most direct measure of a hitter's true batted-ball authority, reducing projection noise for power and hit quality.

## Pipeline Fit

New `StatcastRateAdjuster` inserted after the base `RateComputer` but before rebaseline. It blends xwOBA-derived expected rates with the Marcel rate estimates using a configurable blending weight.

## Data Requirements

- Statcast data via pybaseball's `statcast_batter_expected_stats()` (available from 2015+)
- Barrel rate, hard-hit rate, xwOBA, xBA, xSLG per player-season

## Key References

- Baseball Savant methodology documentation
- Tango, T. "Expected Weighted On-Base Average" (Statcast glossary)
- Arthur, R. "How Well Does xwOBA Predict Future Performance?" (FiveThirtyEight, 2017)

## Implementation Sketch

1. Create `StatcastDataSource` protocol fetching Statcast leaderboards per year
2. Derive expected HR rate from barrel rate and league HR/barrel conversion
3. Derive expected singles/doubles/triples from xBA, xSLG decomposition
4. `StatcastRateAdjuster` blends: `final_rate = w * statcast_rate + (1-w) * marcel_rate`
5. Default blending weight of 0.3-0.4 for stats with 1+ year of Statcast data
