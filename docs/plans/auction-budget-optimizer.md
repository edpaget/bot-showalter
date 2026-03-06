# Auction Budget Optimizer Roadmap

Real-time auction budget allocation that tells you how much to bid on each remaining position given your current roster, remaining budget, and the players still available. During an auction draft, the hardest question isn't "who is best?" (the draft board answers that) but "how much should I spend?" This tool maintains a spend plan that updates after every pick, balancing the need to land elite talent early against the risk of overspending and filling bench spots with $1 players.

This roadmap depends on: draft state engine (done), draft board service (done), valuations (done). It benefits from: category balance tracker (done), positional scarcity (done).

## Status

| Phase | Status |
|-------|--------|
| 1 — Budget allocation engine | not started |
| 2 — Bid recommendation and price tracking | not started |
| 3 — CLI integration | not started |

## Phase 1: Budget allocation engine

Build the core allocation model that distributes remaining budget across unfilled roster slots based on positional value curves.

### Context

The draft state engine already tracks auction budgets per team and the recommendation engine scores available players by value. What's missing is an allocation layer that answers: "I have $180 left and need 2B, OF, OF, SP, SP, RP, UT — how should I split the money?" The allocation should reflect the value distribution at each position in the remaining player pool, not just divide evenly.

### Steps

1. Define domain types in `src/fantasy_baseball_manager/domain/auction_budget.py`:
   - `PositionBudget` frozen dataclass: `position`, `slots_remaining`, `allocated_dollars`, `top_target` (player name), `target_value`.
   - `BudgetPlan` frozen dataclass: `total_remaining`, `reserve` (minimum $1 per unfilled slot), `spendable` (total - reserve), `allocations: list[PositionBudget]`, `inflation_factor` (ratio of league dollars to remaining player value).
2. Implement `compute_budget_plan()` in `src/fantasy_baseball_manager/services/auction_budget.py`:
   - Accepts roster state (filled/unfilled slots), remaining budget, available player pool (with valuations), and league settings.
   - Computes inflation factor: `total_league_dollars_remaining / total_remaining_player_value`. This adjusts for a shrinking pool where prices inflate as dollars chase fewer players.
   - For each unfilled position, estimates the optimal spend as `positional_value_share * spendable_budget`, where positional value share is the fraction of remaining value at that position relative to total remaining value.
   - Caps any single position allocation at the budget for the best available player at that position (no point budgeting $40 for catcher if the best catcher left is worth $25).
   - Redistributes excess to other positions proportionally.
3. Write tests with various roster states (empty roster full budget, nearly full roster small budget, one big hole) verifying allocations are sensible.

### Acceptance criteria

- `compute_budget_plan()` returns a plan where allocations sum to `spendable` (total minus $1 reserves).
- Inflation factor correctly reflects the ratio of remaining dollars to remaining value across all teams.
- Positions with deeper talent pools (e.g., OF) get proportionally more budget than shallow positions (e.g., C) when values are similar.
- A nearly-full roster with one unfilled premium position concentrates remaining budget on that position.
- Reserve is always maintained ($1 per unfilled slot minimum).

## Phase 2: Bid recommendation and price tracking

Add a bid advisor that recommends maximum bid prices for specific players and tracks market prices to detect inflation/deflation trends.

### Context

The budget plan says how much to allocate per position, but during an active nomination the user needs to know: "Player X is nominated — should I bid, and what's my max?" This phase bridges the allocation plan to specific bid decisions and tracks actual prices vs. expected prices to detect market trends.

### Steps

1. Add `BidRecommendation` frozen dataclass to the domain: `player_id`, `player_name`, `position`, `projected_value`, `inflation_adjusted_price`, `max_bid` (based on position budget), `bid_grade` ("must-buy" / "fair" / "overpay" / "skip"), `reason`.
2. Implement `recommend_bid()` in the auction budget service:
   - Given a nominated player, the current budget plan, and roster state, compute the inflation-adjusted fair price and the maximum the user should pay given their position budget.
   - Grade the bid: "must-buy" if the player is significantly underpriced relative to positional need, "skip" if the position is filled or the price exceeds budget.
   - Factor in category needs from the category balance tracker when available — boost max bid for players who fill weak categories.
3. Implement `PriceTracker` that records actual sale prices vs. projected prices:
   - `record_sale(player_id, price)` — logs the sale.
   - `market_trend()` — returns current inflation/deflation rate as a rolling ratio of actual prices to projected prices over the last N sales.
   - Update the inflation factor in real-time as sales happen.
4. Write tests for bid grading logic and price tracker trend detection.

### Acceptance criteria

- `recommend_bid()` returns a higher max bid for players at scarce positions the user still needs.
- Bid grade correctly distinguishes bargains from overpays based on inflation-adjusted price vs. allocation.
- `PriceTracker` detects inflation (actual > projected) and deflation accurately.
- Real-time inflation updates cause the budget plan to adjust (remaining allocations shift up/down).

## Phase 3: CLI integration

Wire the auction budget optimizer into the draft REPL and add standalone commands.

### Steps

1. Add `budget` command to the draft REPL (`DraftSession`):
   - Displays the current budget plan with per-position allocations, top targets, and inflation factor.
   - Updates automatically after each pick (same as recommendations).
2. Add `bid <player>` command to the draft REPL:
   - Shows the bid recommendation for a specific player including max bid and grade.
3. Add `fbm draft auction-plan` standalone command:
   - Accepts `--season`, `--system`, `--budget`, `--league`, `--roster` (current roster names).
   - Displays a pre-draft budget plan for planning purposes.
4. Ensure the budget plan renders cleanly in the existing Rich console output.
5. Write integration tests for the REPL commands.

### Acceptance criteria

- `budget` command in the REPL shows per-position dollar allocations that update after each pick.
- `bid <player>` shows inflation-adjusted price, max bid, and a clear buy/skip recommendation.
- `fbm draft auction-plan` works standalone for pre-draft planning without starting a REPL session.
- All commands degrade gracefully in snake drafts (show a message that auction tools are not applicable).

## Ordering

Phases are sequential: 1 -> 2 -> 3. Phase 1 (allocation engine) provides the math. Phase 2 (bid recommendations + price tracking) builds on the plan. Phase 3 (CLI) exposes everything to the user. Phase 1 can start immediately — all dependencies are satisfied.
