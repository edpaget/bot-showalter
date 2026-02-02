# Weekly Matchup Analyzer

## What

Compare your team against your H2H opponent for the current (or upcoming) scoring period, projecting category-by-category win probabilities and identifying which categories are toss-ups worth optimizing around.

## Why

In H2H leagues, you don't need to maximize every category every week — you need to win more categories than your opponent. Knowing which categories are locked wins, locked losses, or toss-ups lets you make targeted lineup decisions: benching a high-K pitcher with a bad ERA if you're already losing strikeouts but competing in ratios, or starting a speed-only bench bat if SB is close. This is the most impactful weekly decision framework for H2H.

## Pipeline Fit

New `matchup` command under the `teams` CLI group. Consumes ROS projections (or pre-season projections as a fallback), roster data from Yahoo, and the weekly schedule to produce per-category projections for both teams.

## Data Requirements

- Both teams' rosters: from Yahoo Fantasy API (existing `YahooRosterSource`)
- Player projections: ROS projections preferred, pre-season as fallback
- Weekly schedule: number of games per MLB team in the scoring period
- Probable pitchers: to estimate which starters will pitch during the period
- League scoring categories: from config or CLI flags

## Key References

- Cockcroft, T. "Matchup Strategy for H2H Leagues" (ESPN)
- Karabell, E. "Category Management in Head-to-Head" (ESPN)

## Implementation Sketch

1. Fetch both teams' rosters via Yahoo API
2. For each rostered player, load ROS projections (falling back to pre-season)
3. Scale projections to the scoring period:
   - Batters: `weekly_stat = ros_rate * (team_games_this_week / remaining_team_games) * player_pa_share`
   - Pitchers: similar scaling by starts expected this week (probable pitcher data) or relief appearance rate
4. Aggregate each team's projected stats across all categories
5. For ratio categories (ERA, WHIP, OBP/AVG), compute the weighted ratio rather than summing
6. Estimate win probability per category:
   - Use historical projection error variance to model each category as a normal distribution
   - P(win) = P(my_projected > their_projected) given combined variance
7. Classify categories: "likely win" (>65%), "toss-up" (35-65%), "likely loss" (<35%)
8. Output: table showing projected values, difference, and win probability per category
9. Suggest optimization moves: which bench players could swing a toss-up category

## Open Questions

- How to source probable pitcher schedules reliably? Options: MLB Stats API, FanGraphs, manual entry.
- Should the tool account for players on IL or DTD status automatically via Yahoo roster status?
- How to handle uncertainty in playing time for part-time players within a single week?
