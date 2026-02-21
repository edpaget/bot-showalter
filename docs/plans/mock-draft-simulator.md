# Mock Draft Simulator Roadmap

Simulate full drafts against configurable bot strategies before draft day. Run hundreds of mock drafts to discover optimal draft strategies — which rounds to target specific players, when position runs hurt you, and how different approaches (best-value vs. category-need vs. positional-scarcity) perform across many simulations. The simulator uses the existing `DraftBoard`, valuations, and ADP data as inputs and produces per-strategy win-rate and roster-quality metrics.

This roadmap depends on: draft board service (phase 1, done), ADP integration (done), tier generator (planned), category balance tracker (planned), positional scarcity (planned).

## Phase 1: Single-Draft Engine

Build the core simulation loop that runs a single snake draft with one human strategy and N-1 bot strategies, producing a completed roster for each team.

### Context

The draft board service can already build ranked player pools and the ADP system provides market consensus picks. What's missing is a turn-by-turn engine that simulates pick order, enforces roster constraints, and lets bots make decisions. This phase builds that minimal loop.

### Steps

1. Define domain types in `src/fantasy_baseball_manager/domain/mock_draft.py`:
   - `BotStrategy` enum: `ADP_BASED`, `BEST_VALUE`, `POSITIONAL_NEED`, `RANDOM`.
   - `DraftPick` frozen dataclass: `round`, `pick`, `team_idx`, `player_id`, `player_name`, `position`, `value`.
   - `DraftResult` frozen dataclass: `picks: list[DraftPick]`, `rosters: dict[int, list[DraftPick]]`, `snake: bool`.
2. Define `DraftBot` protocol in `src/fantasy_baseball_manager/domain/mock_draft.py` with method `pick(available: list[DraftBoardRow], roster: list[DraftPick], league: LeagueSettings) -> int` (returns `player_id`).
3. Implement bot strategies in `src/fantasy_baseball_manager/services/mock_draft_bots.py`:
   - `ADPBot`: picks the player with the lowest `adp_overall` from the available pool.
   - `BestValueBot`: picks the highest-value player from the available pool.
   - `PositionalNeedBot`: picks the highest-value player at a position the roster still needs (based on `LeagueSettings.positions`).
   - `RandomBot`: picks uniformly at random from the top 20 available players.
4. Build `run_mock_draft()` in `src/fantasy_baseball_manager/services/mock_draft.py`:
   - Accepts `board: DraftBoard`, `league: LeagueSettings`, `strategies: list[DraftBot]`, `snake: bool`.
   - Iterates rounds × teams in snake order, calling each team's bot.
   - Enforces positional limits from `LeagueSettings.positions` (skip positions already filled).
   - Returns `DraftResult`.
5. Write tests in `tests/services/test_mock_draft.py` covering snake ordering, positional enforcement, and each bot strategy picking legally.

### Acceptance criteria

- `run_mock_draft()` completes a full draft for a 12-team league with mixed bot strategies.
- Snake pick ordering is correct (odd rounds ascending, even rounds descending).
- No roster exceeds positional limits defined in `LeagueSettings.positions`.
- Each bot strategy produces deterministic results given a fixed random seed.

## Phase 2: Human Strategy Configuration

Allow the user to define a custom strategy for their team slot — a composable rule stack that approximates their intended draft approach.

### Context

Hard-coded bot strategies are useful for opponents, but the user needs a configurable strategy to test their own plan. Rather than requiring interactive input for hundreds of simulations, this phase lets users define a rule-priority stack (e.g., "take tier-1 value first, then fill catcher by round 10, then best available").

### Steps

1. Define `StrategyRule` protocol in `domain/mock_draft.py` with method `score(player: DraftBoardRow, roster: list[DraftPick], round: int, league: LeagueSettings) -> float | None` (None means rule doesn't apply).
2. Implement rule types in `services/mock_draft_bots.py`:
   - `TierValueRule`: boosts score for players in tier 1-2 at any position.
   - `PositionTargetRule`: boosts score for a specific position in a specific round range (e.g., "catcher in rounds 8-12").
   - `CategoryNeedRule`: boosts score for players who improve the roster's weakest category z-score (requires projections or `category_z_scores` from `DraftBoardRow`).
   - `FallbackBestValueRule`: uses raw `value` as score.
3. Build `CompositeBot` that accepts `list[StrategyRule]` with priority weights and picks the player with the highest weighted score.
4. Add `--strategy` option to CLI (phase 4) that loads a strategy from a TOML config section.
5. Write tests verifying `CompositeBot` respects rule priority and that `PositionTargetRule` fires in the correct round window.

### Acceptance criteria

- `CompositeBot` with a `PositionTargetRule(position="C", rounds=(8, 12))` drafts a catcher in rounds 8-12 when quality catchers are available.
- `CategoryNeedRule` shifts picks toward players improving the weakest category.
- Rules compose cleanly — adding/removing rules changes behavior without breaking the bot.

## Phase 3: Batch Simulation and Analytics

Run N simulations and aggregate results into actionable statistics: how often each player lands on the user's roster, roster quality distributions, and strategy win rates.

### Context

A single mock draft is noisy. Running hundreds of drafts with randomized bot behavior reveals which strategies consistently produce better rosters and which players are reliably available at certain picks.

### Steps

1. Define analytics types in `domain/mock_draft.py`:
   - `SimulationSummary`: `n_simulations`, `team_idx` (user's slot), `avg_roster_value`, `median_roster_value`, `p10_roster_value`, `p90_roster_value`.
   - `PlayerDraftFrequency`: `player_id`, `player_name`, `pct_drafted` (how often user drafted them), `avg_round_drafted`, `avg_pick_drafted`.
   - `StrategyComparison`: `strategy_name`, `avg_value`, `win_rate` (% of sims where this strategy had the highest total roster value).
2. Build `run_batch_simulation()` in `services/mock_draft.py`:
   - Accepts `n_simulations: int`, `board`, `league`, `user_strategy`, `opponent_strategies`, `draft_position: int | None` (fixed or random).
   - Runs N drafts, varying opponent randomness (inject noise into bot picks by adding jitter to ADP/value scores).
   - Returns `SimulationSummary`, `list[PlayerDraftFrequency]`, `list[StrategyComparison]`.
3. Add jitter parameter to `ADPBot` and `BestValueBot` (e.g., multiply value by `random.gauss(1.0, noise)`) to model opponent unpredictability.
4. Write tests for batch simulation with small N (e.g., 10) verifying aggregation math.

### Acceptance criteria

- `run_batch_simulation(n_simulations=100)` completes in under 30 seconds for a 12-team, 23-round draft.
- `PlayerDraftFrequency.pct_drafted` sums to roughly `roster_size / total_players` across all players.
- `StrategyComparison` correctly identifies the strategy with the highest average roster value.
- Results are reproducible given a fixed random seed.

## Phase 4: CLI Commands

Expose the simulator through the CLI so users can run mock drafts and review results from the terminal.

### Context

All the logic from phases 1-3 needs a user-facing interface. This phase adds CLI commands under a `fbm draft mock` subgroup.

### Steps

1. Add `mock_single` command:
   - `fbm draft mock single --season <year> --system <system> --teams <n> --position <pick> --strategy <name>`.
   - Prints the draft log (pick-by-pick) and final roster with total value.
2. Add `mock_batch` command:
   - `fbm draft mock batch --season <year> --system <system> --teams <n> --simulations <n> --strategy <name>`.
   - Prints summary statistics: avg/median/p10/p90 roster value.
   - Prints top-20 most-frequently-drafted players with avg round.
3. Add `mock_compare` command:
   - `fbm draft mock compare --season <year> --system <system> --strategies <name1,name2,...> --simulations <n>`.
   - Prints strategy comparison table with win rates.
4. Register commands in `cli/app.py` under a `draft` typer group.

### Acceptance criteria

- `fbm draft mock single` prints a complete draft log and roster summary.
- `fbm draft mock batch` prints aggregate statistics from multiple simulations.
- `fbm draft mock compare` ranks strategies by win rate and average value.
- All commands respect `--season` and `--system` filters.

## Ordering

Phase 1 is independent and can be implemented immediately. Phase 2 depends on phase 1 (builds on the bot protocol). Phase 3 depends on phase 1 (batch wraps single drafts). Phase 4 depends on all prior phases. The tier generator and category balance tracker roadmaps enhance phase 2 rules but are not hard blockers — rules can degrade gracefully when those systems aren't available.
