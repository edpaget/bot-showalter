# Roster Construction Optimizer Roadmap

Given league settings and a draft format (snake with a known pick position, or auction with a budget), compute an optimal roster construction strategy — how much to allocate to each position and which tier targets to aim for at each pick. This is the bridge between "I have player valuations" and "I have a draft plan."

The optimizer sits on top of the valuation, tier, and positional scarcity systems. It doesn't replace human judgment during the draft but provides a quantitative starting framework: a budget allocation (auction) or a positional target schedule (snake) that maximizes expected roster value.

## Status

| Phase | Status |
|-------|--------|
| 1 — Auction budget allocator | not started |
| 2 — Snake draft position targets | not started |
| 3 — Monte Carlo draft simulation | not started |

## Phase 1: Auction budget allocator

For auction drafts, compute the optimal budget split across positions.

### Context

In a standard auction draft (e.g., $260 budget, 23 roster spots), the key strategic decision is how to distribute spending across positions. Naively splitting the budget evenly is suboptimal — surplus value is concentrated at certain positions and spending patterns should reflect that. The ZAR system already computes dollar values per player, but there's no tool to say "spend $X on catchers, $Y on shortstops, etc."

### Steps

1. Create `src/fantasy_baseball_manager/services/roster_optimizer.py` with a `BudgetAllocation` frozen dataclass: `position`, `budget`, `target_tier`, `target_player_names` (top candidates at that budget level).
2. Implement `optimize_auction_budget(valuations, league_settings)` that:
   - Computes the value-maximizing budget split using a greedy or knapsack approach.
   - Respects roster constraints (number of slots per position).
   - Accounts for the $1 minimum bid floor per roster spot.
   - Returns `list[BudgetAllocation]` summing to the league budget.
3. Support a `strategy` parameter: `"balanced"` (spread value evenly) vs. `"stars_and_scrubs"` (concentrate budget on a few elite players, fill remaining at minimum bid).
4. Add `fbm draft budget --season <year> --system <system> --strategy <balanced|stars_and_scrubs>` CLI command.
5. Write tests with known valuation distributions verifying the allocation sums to the budget and respects constraints.

### Acceptance criteria

- Budget allocation sums to the league budget exactly.
- Every roster slot is accounted for (including bench/UTIL if configured).
- Stars-and-scrubs strategy allocates >50% of budget to top 5 players.
- Balanced strategy keeps position budgets within 2× of each other (excluding minimum-bid slots).

## Phase 2: Snake draft position targets

For snake drafts, compute recommended position targets for each round given a specific draft slot.

### Context

In a snake draft, the key strategic decision is which position to target in each round. With a 1st-overall pick you should target differently than with a 12th pick, because the player pool available at your next pick (24th vs. 13th) is very different. This phase maps the value curves and scarcity data to a specific draft slot.

### Steps

1. Create `SnakeDraftPlan` frozen dataclass: list of `RoundTarget` entries, each with `round`, `pick_number`, `recommended_position`, `target_tier`, `expected_value`, `alternative_positions`.
2. Implement `plan_snake_draft(valuations, tiers, scarcity, league_settings, draft_slot)` that:
   - Simulates a snake draft (picks go 1→N, N→1, repeat).
   - At each pick, estimates the available player pool based on ADP (if available) or value-based ranking.
   - Recommends the position that maximizes remaining roster value using a greedy look-ahead.
3. Show alternative positions at each pick for flexibility.
4. Add `fbm draft plan --season <year> --system <system> --slot <n> --teams <n>` CLI command.
5. Write tests with known valuations and a fixed draft slot verifying sensible position targeting.

### Acceptance criteria

- Draft plan covers all roster slots for the given draft position.
- Pick numbers are correct for the snake format.
- Early picks target scarce high-value positions; late picks fill deep positions.
- Plan adapts to draft slot (early vs. late first-round picks produce different strategies).

## Phase 3: Monte Carlo draft simulation

Simulate thousands of drafts to estimate expected roster value and identify robust strategies.

### Context

The greedy optimizer from phases 1-2 finds a locally optimal plan but doesn't account for opponent behavior or draft variance. A Monte Carlo simulation drafts against simulated opponents (using ADP-based picks), measuring the distribution of roster outcomes for different strategies.

### Steps

1. Implement a simple draft simulator that picks for opponents based on ADP rank (with noise).
2. Implement `simulate_drafts(valuations, adp, league_settings, strategy, n_simulations)` that runs N simulated drafts and records the resulting roster value for each.
3. Report percentile outcomes (p10, p25, p50, p75, p90) for total roster value.
4. Compare strategies (balanced vs. stars_and_scrubs, different position-targeting orders) by their outcome distributions.
5. Add `fbm draft simulate --season <year> --slot <n> --simulations <n>` CLI command.
6. Write tests verifying simulation mechanics (correct pick order, roster constraint enforcement, value tallying).

### Acceptance criteria

- Simulation runs to completion without constraint violations.
- Output shows percentile outcomes for total roster value.
- Different strategies produce measurably different outcome distributions.
- Opponent behavior roughly matches ADP patterns.

## Ordering

Phase 1 and phase 2 are independent and can be built in parallel. Phase 3 depends on both and also benefits from the ADP integration roadmap being complete (for realistic opponent modeling). Phase 1 is higher priority for auction leagues, phase 2 for snake leagues — choose based on the primary league format.
