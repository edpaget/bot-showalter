# Category Pace Tracker

## What

Track your team's season-long performance in each scoring category relative to the rest of the league, project end-of-season standings placement per category, and identify categories to target or punt in roster moves.

## Why

In H2H leagues, season-long category strength informs trade and waiver strategy even though matchups are weekly. If your team is bottom-3 in steals across the season, you either need to make a move to address it or consciously punt the category. If you're middle-of-the-pack in strikeouts, a single pitching pickup could move you up several spots. Understanding your category profile relative to the league drives strategic decisions about which players to target and which to trade away.

## Pipeline Fit

New `teams category-pace` command under the `teams` CLI group. Consumes league-wide roster data from Yahoo API and ROS projections to produce standings projections.

## Data Requirements

- All teams' rosters: from Yahoo API
- ROS projections for all rostered players
- Season-to-date actual stats per team (from Yahoo or computed from player actuals)
- League schedule: total weeks, weeks remaining
- Scoring categories from league settings

## Key References

- Hershey, S. "Category Management Strategy" (Baseball HQ)
- Kruse, P. "Standings Gain Points for H2H" (FanGraphs Community)

## Implementation Sketch

1. Fetch all league rosters via Yahoo API
2. For each team, compute:
   - Season-to-date category totals (from Yahoo matchup history or player stats)
   - ROS projected category totals (from ROS projections applied to current roster)
   - Full-season projected totals: `to_date + ros_projected`
3. Rank all teams in each category based on full-season projected totals
4. For your team, display:
   - Current rank in each category
   - Projected end-of-season rank
   - Gap to next rank up and down (e.g., "3 more SB to move from 8th to 7th")
   - Trend: improving or declining based on recent weekly performance vs. season average
5. Identify strategic recommendations:
   - Categories where a small improvement yields a rank gain ("buy SB")
   - Categories where you're so far behind it's not worth investing ("punt saves")
   - Categories where you're dominant and could trade from strength ("sell HR")
6. Output: ranked category table with current/projected standings, gaps, and recommendations

## Open Questions

- How to compute season-to-date team totals efficiently? Yahoo matchup results vs. aggregating player stats?
- Should the tool factor in schedule strength (remaining opponents' category profiles)?
- How to handle ratio categories (ERA, WHIP) in pace tracking — project based on IP volume and rate trends?
- Should it integrate with the trade evaluator to suggest specific trade targets that address weak categories?
