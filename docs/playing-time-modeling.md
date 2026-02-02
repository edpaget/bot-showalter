# Playing Time Modeling

## What

Replace the simple Marcel playing time formula with a model incorporating injury history, depth chart position, prospect ETA timelines, and team context to produce more accurate PA/IP projections.

## Why

The current formula (`0.5 * PA(y-1) + 0.1 * PA(y-2) + 200`) treats all players identically regardless of role security, injury risk, or organizational context. A player coming off a major injury gets projected for substantial playing time. A top prospect blocked by a veteran gets no playing time. For fantasy baseball, playing time accuracy is often more impactful than rate accuracy because it directly scales all counting stats.

## Pipeline Fit

New `EnhancedPlayingTimeProjector` implementing the `PlayingTimeProjector` protocol. Drop-in replacement for `MarcelPlayingTime` in any pipeline preset.

## Data Requirements

- Injury history: days on IL per player per season (available from Fangraphs transactions or Spotrac)
- Depth charts: team roster information and positional depth (MLB.com, FanGraphs)
- Prospect rankings and ETA estimates (community prospect lists)
- Service time and option status (affects roster flexibility)

## Key References

- PECOTA playing time methodology (Baseball Prospectus)
- Steamer playing time projections methodology (FanGraphs)
- Szymborski, D. "ZiPS Playing Time Framework" (FanGraphs)

## Implementation Sketch

1. Base PA/IP from Marcel formula as starting point
2. Injury adjustment: `pa_adj = base_pa * (1 - injury_risk)` where injury_risk is derived from:
   - Days on IL in last 3 years (exponentially weighted)
   - Age-based injury probability curves
3. Role adjustment:
   - Starters: cap at ~700 PA / 220 IP
   - Platoon/bench: reduce to 300-450 PA range
   - Closers: 60-70 IP cap
4. Prospect adjustment: for players not yet on MLB roster, project call-up date and prorate PA/IP
5. Team context: adjust based on team competitiveness (contenders play veterans, rebuilders play youth)
