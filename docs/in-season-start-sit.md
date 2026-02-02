# Weekly Start/Sit Optimizer

## What

Recommend optimal lineups for a H2H scoring period by maximizing expected category wins against a specific opponent, given roster constraints and the week's schedule.

## Why

Setting lineups in H2H is not about maximizing raw stats — it's about winning categories relative to your opponent. The optimal lineup depends on which categories are competitive this week. If you're projected to lose ERA and WHIP regardless, you should start high-upside pitchers for volume stats (W, K, NSVH). If steals are a toss-up, you should start your speed guys over a marginal power bat. Manual lineup optimization is tedious and error-prone; an automated optimizer can evaluate all feasible lineup combinations against the matchup context.

## Pipeline Fit

New `lineup optimize` command under the `teams` CLI group. Depends on the matchup analyzer for category win probabilities and ROS projections for player-level estimates. Uses roster slot constraints from the existing `RosterConfig` model.

## Data Requirements

- Your roster and eligible positions: from Yahoo API
- Opponent's projected category totals: from matchup analyzer
- Player projections scaled to the week: games, probable starts, matchup quality
- Roster slot constraints: from league settings or config (C, 1B, 2B, SS, 3B, OF, UTIL, SP, RP, BN)
- Schedule: games per team this week, probable pitcher matchups

## Key References

- Tango, T. "Lineup Optimization and Category Games" (The Book Blog)
- Grey, R. "Start/Sit Framework for H2H" (Razzball)

## Implementation Sketch

1. Load your roster, eligible positions per player, and roster slot constraints
2. Load opponent's projected category totals from the matchup analyzer
3. For each of your players, project weekly stats:
   - Batters: scale ROS rates by team games this week, adjust for handedness matchups if platoon data available
   - Starting pitchers: project per-start stats, multiply by expected starts this week (0, 1, or 2)
   - Relief pitchers: scale by expected appearances (team games * appearance rate)
4. Enumerate feasible lineups respecting position eligibility and slot limits
   - For large rosters, use greedy optimization or branch-and-bound rather than brute force
5. Score each lineup by expected category wins against the opponent:
   - For each category, compute your team's projected total given the lineup
   - Compare against opponent's projected total using win probability model from matchup analyzer
   - Lineup score = sum of category win probabilities
6. Return the lineup that maximizes expected category wins
7. Display: recommended starters, bench players, and the marginal impact of each start/sit decision
   - "Starting X over Y in UTIL: +4% chance of winning SB, -1% chance of winning HR"

## Open Questions

- How to handle daily lineup leagues vs. weekly lock leagues? Daily leagues need rolling re-optimization.
- Should the optimizer consider opponent's likely lineup changes (e.g., streaming pitchers)?
- How granular should pitcher matchup quality be? Team-level wRC+ vs. individual batter projections?
- For leagues with daily moves and acquisition limits, should the optimizer account for remaining moves?
