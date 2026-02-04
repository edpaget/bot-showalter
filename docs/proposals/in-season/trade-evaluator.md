# Trade Evaluator

## What

Evaluate proposed trades by comparing rest-of-season value for both sides, accounting for positional scarcity, category impact on your team, and roster fit.

## Why

Trades are the highest-leverage in-season decisions and the hardest to evaluate. Raw player value comparisons miss critical context: losing your only catcher-eligible player matters more than the value gap suggests, and gaining RBI when you're already leading the category is worth less than gaining SB where you're last. A good trade evaluator shows the full picture — raw value exchange, category impact, and positional consequences — so you can make informed decisions rather than relying on gut feel.

## Pipeline Fit

New `teams trade-eval` command. Consumes ROS projections, your roster, and the proposed trade to produce a multi-dimensional evaluation. Reuses the z-score valuation infrastructure and the matchup analyzer's category projection logic.

## Data Requirements

- Your current roster: from Yahoo API
- ROS projections for all involved players
- League scoring categories and settings
- Positional scarcity data: replacement-level values by position from existing draft ranking infrastructure
- League-wide category distributions: to assess where your team stands in each category

## Key References

- Tango, T. "Positional Adjustments and Trade Value" (The Book Blog)
- Karabell, E. "Trade Value Charts" (ESPN)
- Cockcroft, T. "Evaluating Fantasy Trades with Category Impact" (ESPN)

## Implementation Sketch

1. Parse trade: `--give "Player A, Player B" --receive "Player C, Player D"`
2. Load your current roster and compute ROS projected category totals
3. Compute post-trade roster and projected category totals
4. **Raw value comparison:**
   - Sum ROS z-score values for each side of the trade
   - Show the value gap: `received_value - given_value`
5. **Category impact analysis:**
   - For each scoring category, show projected total before and after the trade
   - Flag categories where the trade moves you from below-average to above-average (or vice versa)
   - Compute net expected category wins change against a league-average opponent
6. **Positional scarcity adjustment:**
   - If the trade leaves you without a viable starter at a position, penalize
   - If the trade upgrades a position where replacement level is low (C, SS), add value
   - Use replacement-level values from the draft ranking infrastructure
7. **Roster fit score:**
   - Account for position eligibility overlap (multi-position players have more roster flexibility)
   - Flag if the trade creates a roster construction problem (e.g., too many players at one position)
8. Output: summary table with raw value, category impact, positional adjustment, and overall recommendation
   - "Trade improves your team by +1.2 expected category wins per week"
   - Category breakdown: "+SB, +R, -HR, neutral K"

## Scoring Format Considerations

The scoring format significantly affects trade strategy and how the evaluator should frame its recommendations:

**H2H Each Category:** Every category counts. Trading away your best source of saves costs you a loss in saves most weeks, regardless of what you gain elsewhere. Punt strategies are expensive because you pay the cost in every matchup. The evaluator should weight all categories roughly equally and warn when a trade creates a new category weakness.

**H2H Most Categories (winner-take-all):** Punt strategies are viable and often optimal. You only need to win the majority of categories (e.g., 6 of 10), so deliberately sacrificing 2-3 categories to dominate the others is a valid approach. The evaluator should:

- **Assess punt viability:** If the trade punts a category, show the impact on `P(win matchup)` rather than just category win count. Losing saves but gaining HR, RBI, and SB might increase your matchup win rate even though you lose a category.
- **Model category concentration:** In winner-take-all, a team that wins 8 categories 60% of the time is worse than a team that wins 6 categories 80% of the time. The evaluator should favor trades that create reliable wins in a majority of categories over trades that spread value thinly.
- **Flag format-dependent recommendations:** A trade might be bad in each-category but good in most-categories (or vice versa). The evaluator should show the assessment under both formats when they disagree.

The implementation should accept a `--format each-category|most-categories` flag and adjust the "expected category wins" output to either a sum of win probabilities or a matchup win probability accordingly.

## Open Questions

- Should the evaluator suggest counter-offers (e.g., "if you swap Player B for Player E, the trade becomes more balanced")?
- How to handle keeper/dynasty value vs. single-season value? Should there be a flag for keeper league context?
- How to weight positional scarcity vs. raw value? A fixed penalty, or dynamic based on available free agents at the position?
- Should it pull in recent trade data from the league to calibrate what "fair" looks like in this specific league's trade market?
