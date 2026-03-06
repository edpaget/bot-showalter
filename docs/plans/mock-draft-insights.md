# Mock Draft Insights Roadmap

Bridge the gap between mock draft simulation results and live draft decisions. The mock draft simulator already runs hundreds of simulations and produces strategy comparisons, player draft frequencies, and roster quality distributions. This roadmap turns those raw outputs into actionable draft-day guidance: position-round targeting rules, player availability windows, and a draft plan that the live recommender can consume.

This roadmap depends on: mock draft simulator (done), draft state engine (done), recommendation engine (done), ADP integration (done).

## Status

| Phase | Status |
|-------|--------|
| 1 — Draft plan generation | not started |
| 2 — Availability windows | not started |
| 3 — Recommender integration | not started |

## Phase 1: Draft plan generation

Analyze batch mock draft results to produce a position-round targeting plan: which positions to prioritize in which rounds given the user's draft slot.

### Context

The batch simulator produces `PlayerDraftFrequency` (how often each player lands on the user's team) and `SimulationSummary` (roster quality stats). What's missing is distillation: "from slot 5 in a 12-team snake draft, the best strategy is SP in rounds 2-3, then 2B in round 5-6, then grab a catcher by round 10." This phase extracts that signal from simulation data.

### Steps

1. Define domain types in `src/fantasy_baseball_manager/domain/draft_plan.py`:
   - `RoundTarget` frozen dataclass: `round_range` (tuple of start/end rounds), `position`, `confidence` (% of winning simulations that drafted this position in this range), `example_players` (top 3 most-frequently-drafted players at this position in this range).
   - `DraftPlan` frozen dataclass: `slot`, `teams`, `strategy_name`, `targets: list[RoundTarget]`, `n_simulations`, `avg_roster_value`.
2. Implement `generate_draft_plan()` in `src/fantasy_baseball_manager/services/draft_plan.py`:
   - Takes batch simulation results (list of `DraftResult` from winning or above-median simulations).
   - For each round, counts position frequency across simulations to find the modal position drafted.
   - Groups adjacent rounds with the same modal position into ranges.
   - Computes confidence as the percentage of simulations that followed this pattern.
   - Lists the most-frequently-drafted players at each position-round intersection.
3. Write tests with synthetic simulation results verifying plan extraction.

### Acceptance criteria

- `generate_draft_plan()` produces a ordered list of round-position targets.
- Targets from winning simulations have higher confidence than targets from losing simulations.
- Adjacent rounds with the same position are merged into ranges (e.g., "SP in rounds 2-3" not "SP in round 2, SP in round 3").
- Example players are the most frequently seen names at that position-round.

## Phase 2: Availability windows

Compute per-player availability windows: the probability that a specific player is still available at each pick number, based on mock simulation data.

### Context

During a draft, the user often asks: "Can I wait on Player X until round 7?" The mock data can answer this empirically — if 80% of simulations show Player X gone by pick 70 but available at pick 60, the user knows the window. This phase computes and presents these windows.

### Steps

1. Add `AvailabilityWindow` frozen dataclass to the domain: `player_id`, `player_name`, `position`, `earliest_pick` (5th percentile of when drafted across sims), `median_pick`, `latest_pick` (95th percentile), `available_at_user_pick: float` (probability available at the user's next pick).
2. Implement `compute_availability_windows()` in the draft plan service:
   - Takes batch simulation results and the user's draft slot.
   - For each player, collects all pick numbers at which they were drafted across simulations.
   - Computes percentile distribution and probability of availability at each of the user's pick numbers.
3. Add `fbm mock availability` CLI command:
   - `--season`, `--slot`, `--teams`, `--simulations`, `--player` (optional filter to specific player).
   - Displays a table of players with their availability windows.
   - When `--player` is specified, shows a detailed round-by-round availability curve.
4. Write tests verifying percentile math and availability probabilities.

### Acceptance criteria

- Players with low ADP variance have tight windows (earliest ~ latest).
- Players with high variance (inconsistently drafted) have wide windows.
- `available_at_user_pick` correctly reflects the fraction of simulations where the player was still on the board at that pick.
- The CLI command is filterable by player name for quick lookups.

## Phase 3: Recommender integration

Feed draft plan targets and availability windows into the live draft recommendation engine so that simulation insights influence real-time pick suggestions.

### Context

The recommendation engine currently scores by value + scarcity + need. This phase adds a "mock-informed" signal: boost players at the mock-recommended position for this round, and penalize players who are "safe" to wait on (high availability at the user's next pick).

### Steps

1. Add optional `draft_plan: DraftPlan | None` and `availability: list[AvailabilityWindow] | None` parameters to the `recommend()` function.
2. When a draft plan is provided, add a scoring bonus for players whose position matches the current round's target position. Scale the bonus by the target's confidence.
3. When availability data is provided, add a "wait penalty" for players with >80% availability at the user's next pick (they'll likely still be there — don't rush). Conversely, add an urgency bonus for players with <30% availability at the next pick.
4. Add `--mock-plan` flag to `draft start` that pre-runs a batch simulation (or loads cached results) and feeds the plan into the recommender.
5. Write tests verifying that mock-informed recommendations differ from baseline recommendations in expected ways (mock-targeted positions get boosted, safe-to-wait players get deprioritized).

### Acceptance criteria

- With a mock plan loaded, recommendations shift toward the plan's target position for the current round.
- Players with high next-pick availability are deprioritized vs. players with low availability (all else equal).
- The `--mock-plan` flag works end-to-end: runs simulations, generates plan, feeds into recommender.
- When no mock plan is provided, recommendations behave identically to the current system (no regression).

## Ordering

Phase 1 and phase 2 are independent of each other (both consume raw simulation data) and can be done in parallel. Phase 3 depends on both. All phases can start immediately — the mock draft simulator is complete.
