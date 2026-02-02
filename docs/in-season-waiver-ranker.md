# Waiver / Free Agent Pickup Ranker

## What

Rank available free agents by their marginal value to your specific team rather than by raw projection, accounting for your team's category needs, roster composition, and positional gaps.

## Why

Generic player rankings don't account for team context. A player who provides 15 HR and 20 SB is more valuable to a team that's weak in steals than one already leading the category. In H2H, marginal category wins are what matter — picking up a player who flips a loss to a win in one category is worth more than a player who adds volume to a category you're already dominating. Streaming pitchers (targeting two-start pitchers with favorable matchups) is a specific high-value application of this tool.

## Pipeline Fit

New `players waiver-rank` command. Consumes your team's roster projections, the pool of available free agents (from Yahoo API), and ROS projections to produce a team-context-aware ranking.

## Data Requirements

- Your roster and current projected category totals: from Yahoo API + ROS projections
- Available free agents: from Yahoo API free agent endpoint
- ROS projections for all players (including unrostered)
- League scoring categories
- Weekly schedule for streaming pitcher identification (games per team, probable pitchers)

## Key References

- Podhorzer, N. "Surplus Value and the Waiver Wire" (FanGraphs)
- Cistulli, C. "Streaming Pitchers Methodology" (FanGraphs)

## Implementation Sketch

1. Fetch your roster and compute current ROS projected totals per category
2. Fetch available free agents from Yahoo API
3. For each free agent with a ROS projection:
   a. Determine which roster slot they'd fill (weakest position or droppable player)
   b. Compute your team's new projected category totals with this player added
   c. Calculate marginal value: change in expected category wins across a typical matchup
      - Use league-average opponent as the baseline, or actual upcoming opponent
4. Rank free agents by marginal category win impact
5. For streaming pitchers specifically:
   a. Identify two-start pitchers for the upcoming week
   b. Score by: opponent quality (wRC+), park factor, pitcher projection, and which categories your team needs
   c. Flag streamers separately in output
6. Display: ranked free agent list with projected stats, marginal value, and which categories they help
7. Optionally suggest a drop candidate from your roster for each pickup

## Streaming Pitcher Sub-Feature

- Fetch probable pitcher schedule for the upcoming week
- Cross-reference with free agent pool
- Score each streamer: `value = projected_K * k_need + projected_W * w_need - projected_ER * era_penalty`
- Weight by matchup quality: opponent team wRC+ and park factor (existing park factor infrastructure)

## Open Questions

- How to handle FAAB budgeting suggestions? Should the tool recommend bid amounts based on marginal value?
- For waiver priority leagues, should it factor in waiver position (don't burn #1 priority on a marginal pickup)?
- How to model the "drop" side — should it compute net value (pickup minus drop) automatically?
- Should it distinguish between short-term streaming adds and long-term roster improvements?
