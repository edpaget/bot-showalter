# Two-Start Pitcher Identifier

## What

Each scoring period, identify starting pitchers with two scheduled starts and score them by matchup quality, projection, and park factors to support streaming decisions.

## Why

In H2H leagues with weekly scoring periods, a two-start pitcher provides roughly double the counting stats (K, W, QS) of a one-start pitcher. Streaming two-start pitchers from the waiver wire is one of the most reliable ways to gain an edge in pitching categories. However, not all two-start pitchers are worth starting — a bad pitcher with two tough matchups in hitter-friendly parks can sink your ratios. Scoring two-start pitchers by expected value helps identify the streamers worth adding and the rostered arms that might be worth benching.

## Pipeline Fit

New `players two-start` command. Can also be integrated into the waiver ranker as a filter/flag. Uses existing park factor infrastructure and ROS projections.

## Data Requirements

- Probable pitcher schedule for the scoring period: MLB Stats API or community-maintained probable pitcher data
- ROS pitcher projections from existing pipeline
- Park factors for each start's venue: from existing `ParkFactorProvider`
- Opponent team quality: aggregate wRC+ or runs scored
- Pitcher handedness and opponent lineup handedness splits (if platoon data available)

## Key References

- Cistulli, C. "Two-Start Pitcher Rankings Methodology" (FanGraphs)
- Staff, "Weekly Streaming Pitcher Analysis" (PitcherList)

## Implementation Sketch

1. Fetch probable pitcher schedule for the target scoring period
2. Identify all pitchers with two scheduled starts
3. For each two-start pitcher, compute a matchup score per start:
   - Base: pitcher's ROS projected per-start stats (IP, K, ER, W probability)
   - Adjust for opponent: scale by opponent wRC+ relative to league average
   - Adjust for park: apply park factor from existing infrastructure
4. Combine both starts into a projected weekly line: total IP, K, ER, W, and derived ERA/WHIP
5. Score and rank by composite value (weighted by league category needs if available)
6. Display:
   - Ranked list of two-start pitchers with both matchups shown
   - Projected weekly line (IP, K, ERA, WHIP, W)
   - Matchup grades (A/B/C/D/F) based on opponent + park
   - Roster status: rostered (by whom) or free agent
7. Flag pitchers who are two-start but risky (bad matchups, recent poor performance)

## Open Questions

- How reliable are probable pitcher schedules more than a few days out? Should the tool indicate confidence level?
- Should it also flag one-start pitchers with elite matchups as potential starts over mediocre two-start arms?
- How to handle doubleheaders and schedule changes mid-week?
