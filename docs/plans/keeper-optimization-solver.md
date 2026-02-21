# Keeper Optimization Solver Roadmap

Solve the optimal keeper set given N keeper slots, balancing surplus value against draft pool impact. Simple surplus ranking ("keep the N highest surplus players") ignores that your keeper choices change the replacement-level baseline for the draft — keeping a $40 player at $10 is great, but if that position is deep in the draft pool, you might be better off keeping a $25 player at $5 at a scarce position. This solver finds the combination of keepers that maximizes total expected team value across keepers + draft.

This roadmap depends on: keeper/surplus value (planned — domain types and basic surplus computation), valuations (done), positional scarcity (planned), draft board service (done). Extends the keeper-surplus-value roadmap with optimization logic.

## Status

| Phase | Status |
|-------|--------|
| 1 — Combinatorial keeper selection | not started |
| 2 — Draft pool adjustment | not started |
| 3 — Scenario analysis | not started |
| 4 — CLI commands | not started |

## Phase 1: Combinatorial Keeper Selection

Enumerate and score all valid keeper combinations to find the set that maximizes total surplus, subject to roster constraints.

### Context

The keeper-surplus-value roadmap (planned) provides `KeeperDecision` records with per-player surplus values. But naive "keep the top N by surplus" doesn't account for roster constraints (can't keep 5 outfielders if you only have 3 OF slots) or the interaction between keeper choices (keeping Player A changes the value of keeping Player B if they're at the same position). This phase adds constraint-aware optimization.

### Steps

1. Define domain types in `src/fantasy_baseball_manager/domain/keeper_optimization.py`:
   - `KeeperConstraints` frozen dataclass: `max_keepers: int`, `max_per_position: dict[str, int] | None` (e.g., max 1 catcher keeper), `max_cost: float | None` (total keeper cost budget), `required_keepers: list[int] | None` (player IDs that must be kept).
   - `KeeperSet` frozen dataclass: `players: list[KeeperDecision]`, `total_surplus: float`, `total_cost: float`, `positions_filled: dict[str, int]`, `score: float` (total expected value = surplus + estimated draft value of remaining slots).
   - `KeeperSolution` frozen dataclass: `optimal: KeeperSet`, `alternatives: list[KeeperSet]` (top-5 next-best combinations), `sensitivity: list[tuple[str, float]]` (which player's inclusion is most marginal — name + surplus gap to next alternative).
2. Build `solve_keepers()` in `src/fantasy_baseball_manager/services/keeper_optimizer.py`:
   - Accepts `candidates: list[KeeperDecision]`, `constraints: KeeperConstraints`, `league: LeagueSettings`.
   - If `len(candidates) choose max_keepers` is small enough (< 100,000): enumerate all valid combinations and score each.
   - Score = sum of surplus values, with positional constraint violations filtered out.
   - Returns `KeeperSolution` with the top-scoring set and alternatives.
3. Add branch-and-bound pruning for larger candidate pools:
   - Sort candidates by surplus descending.
   - Prune branches where remaining candidates can't exceed the current best score.
4. Write tests with small candidate pools (8 candidates, keep 4) verifying constraint enforcement and optimal selection.

### Acceptance criteria

- `solve_keepers()` finds the combination with the highest total surplus among valid sets.
- Position constraints are enforced (e.g., max 1 catcher keeper).
- `required_keepers` are always included in the solution.
- `sensitivity` identifies the most marginal keeper (the one closest to being swapped out).
- Runs in under 5 seconds for realistic inputs (20 candidates, keep 6).

## Phase 2: Draft Pool Adjustment

Account for how keeper choices across all teams change the available draft pool, shifting replacement-level values and position scarcity.

### Context

In keeper leagues, the draft pool is depleted by every team's keepers. If the league keeps 10 elite shortstops, the remaining shortstops in the draft are weaker, making your own SS keeper more valuable. Phase 1 ignores this effect; phase 2 models it.

### Steps

1. Extend `domain/keeper_optimization.py`:
   - `LeagueKeeperState` frozen dataclass: `my_keepers: list[int]` (player IDs), `league_keepers: list[int]` (all other teams' kept player IDs), `draft_pool: list[Valuation]` (remaining players available to draft).
2. Build `compute_adjusted_draft_pool()` in `services/keeper_optimizer.py`:
   - Accepts all league keepers + full valuation list.
   - Removes kept players from the pool.
   - Recomputes replacement-level values per position from the reduced pool.
   - Returns adjusted `draft_pool` and new replacement-level baselines.
3. Build `solve_keepers_with_pool()`:
   - Wraps `solve_keepers()` but scores each keeper set by: `total_surplus + estimated_draft_value(remaining_slots, adjusted_pool)`.
   - `estimated_draft_value` uses the pick value curve concept (from draft-pick-trade-evaluator roadmap) or a simpler approximation: for each unfilled roster slot, assign the Nth-best available player at that position from the adjusted pool.
   - This makes keeper decisions context-aware: keeping a player at a deep position is less valuable because the draft pool is rich there.
4. Support importing other teams' keepers from a CSV file.
5. Write tests showing that pool adjustment changes the optimal keeper set (e.g., when the league keeps many SS, keeping your own SS becomes more valuable).

### Acceptance criteria

- Removing kept players from the pool changes replacement-level baselines.
- `solve_keepers_with_pool()` sometimes produces a different optimal set than `solve_keepers()` (demonstrating the pool effect).
- Importing league-wide keepers from CSV works correctly.
- Estimated draft value for remaining slots is reasonable (decreasing value per slot).

## Phase 3: Scenario Analysis

Let users explore "what if" keeper scenarios — swap players in/out and see how total team value changes, enabling informed conversations with trade partners.

### Context

The optimal solution from phases 1-2 is a starting point, but real keeper decisions involve uncertainty and negotiation. Users need to explore scenarios: "What if I trade for Player X — does he become a keeper?" or "What if I release Player Y — how much do I lose?" This phase provides interactive scenario comparison.

### Steps

1. Define types in `domain/keeper_optimization.py`:
   - `KeeperScenario` frozen dataclass: `name: str`, `keepers: list[int]`, `set: KeeperSet`, `delta_vs_optimal: float` (how much worse than the optimal set).
2. Build `compare_scenarios()` in `services/keeper_optimizer.py`:
   - Accepts `scenarios: list[tuple[str, list[int]]]` (name + player IDs for each scenario).
   - Scores each scenario using the same scoring function from phases 1-2.
   - Returns `list[KeeperScenario]` sorted by score descending.
3. Build `keeper_trade_impact()`:
   - Accepts `current_keepers: list[int]`, `acquire: list[int]`, `release: list[int]`, `constraints`.
   - Re-solves the optimization with the new candidate pool (add acquired, remove released).
   - Returns the new optimal set and the delta vs. the original optimal.
4. Write tests for scenario comparison with known inputs.

### Acceptance criteria

- `compare_scenarios()` correctly ranks multiple keeper configurations by total expected value.
- `keeper_trade_impact()` shows how acquiring/releasing a player changes the optimal keeper set.
- `delta_vs_optimal` accurately reflects the value gap between each scenario and the best one.

## Phase 4: CLI Commands

Expose keeper optimization through the CLI.

### Steps

1. Add `fbm keeper optimize --season <year> --system <system> --max-keepers <n>`:
   - Prints the optimal keeper set with total surplus and per-player breakdown.
   - Prints the top-3 alternative sets with their surplus gaps.
   - Prints sensitivity analysis (most marginal keeper).
   - Supports `--max-per-position`, `--max-cost`, `--required <player-names>`.
2. Add `fbm keeper optimize --league-keepers <csv-path>` for pool-adjusted optimization (phase 2).
3. Add `fbm keeper scenario --season <year> --scenarios <name1:players1,name2:players2,...>`:
   - Compares named keeper scenarios side-by-side.
4. Add `fbm keeper trade-impact --acquire <names> --release <names> --season <year>`:
   - Shows how a trade changes the optimal keeper set.
5. Register commands in `cli/app.py` under the `keeper` group.

### Acceptance criteria

- `fbm keeper optimize` prints the optimal keeper set with alternatives and sensitivity.
- Pool-adjusted mode changes results when league-wide keepers are provided.
- Scenario comparison prints a clear ranking of configurations.
- Trade impact shows before/after optimal sets and the value delta.

## Ordering

Phase 1 depends on the keeper-surplus-value roadmap (needs `KeeperDecision` and `KeeperCost` types) but is otherwise self-contained. Phase 2 depends on phase 1 and benefits from the positional scarcity roadmap for replacement-level computation. Phase 3 depends on phases 1-2. Phase 4 depends on all prior phases. The keeper-surplus-value roadmap should be implemented first to provide the foundational domain types and basic surplus calculation that this roadmap extends.
