# Rest-of-Season Projection Updater

## What

Blend pre-season MARCEL projections with in-season actual performance to produce rest-of-season (ROS) projections that improve as sample size grows throughout the year.

## Why

Pre-season projections are the best estimate on Opening Day but become stale as the season progresses. A player who's changed his swing, added a pitch, or is playing through an undisclosed injury will diverge from his projection. In-season actuals carry real signal, but raw stats overweight small samples early in the year. A Bayesian blending approach gives in-season data appropriate influence based on how much we've observed, producing the most accurate ROS estimates at any point in the season.

ROS projections are the foundation for every other in-season tool: start/sit, waiver pickups, and trade evaluation all depend on current expected future production.

## Pipeline Fit

New `RestOfSeasonProjector` that wraps the existing `ProjectionPipeline`. It takes a full-season pre-season projection and observed in-season stats, then produces a blended ROS projection with reduced PA/IP for the remaining schedule.

## Data Requirements

- Pre-season full-season projections (from existing pipeline)
- In-season stats to date: batting and pitching lines via pybaseball or Yahoo API
- MLB schedule: remaining games per team to estimate remaining PA/IP opportunity
- Current date or scoring period to determine where we are in the season

## Key References

- Tango, T. "Marcel After the Season Starts" (The Book Blog)
- Lichtman, M. "Combining Projections with In-Season Performance" (The Book Blog)
- Sievert, C. "Updating Player Projections In-Season" (FanGraphs Community Research)

## Implementation Sketch

1. Load pre-season projection as the prior: extract projected rate stats (HR/PA, K/IP, BB/IP, etc.)
2. Fetch in-season actuals to date: compute observed rates and sample size (PA or IP)
3. Compute blended rate for each stat using a reliability weighting:
   - `blended_rate = (prior_rate * regression_pa + actual_rate * actual_pa) / (regression_pa + actual_pa)`
   - Regression PA/IP constants are stat-specific (same values used in `StatSpecificRegressionRateComputer`)
4. Estimate remaining PA/IP:
   - Remaining team games from MLB schedule
   - Player's share based on recent playing time patterns (e.g., rolling 14-day PA/G)
5. Apply blended rates to remaining PA/IP to produce ROS counting stat projections
6. Expose as a new CLI command: `players ros-project --date 2025-06-15`

## Open Questions

- Should the blending use the same regression constants as the pre-season pipeline, or are separate in-season constants warranted?
- How to handle players with no pre-season projection (call-ups, rookies with limited history)? Fall back to league-average rates with high regression.
- Should recent performance be weighted more heavily than early-season performance (e.g., recency-weighted actuals)?
