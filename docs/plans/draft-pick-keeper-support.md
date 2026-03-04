# Draft-Pick Keeper Support Roadmap

Add keeper league support for leagues where keeping a player costs a draft pick (e.g., "keeping Trout costs your 3rd-round pick") rather than auction dollars. The existing keeper infrastructure — surplus calculation, decisions, trade evaluation, and the planned optimization solver — all operate on dollar-denominated costs. This roadmap builds the translation layer that converts pick-slot costs into dollar equivalents so the full keeper pipeline works for both league formats.

The key insight is that a draft pick has an expected dollar value based on the pick value curve (ADP × valuation). A 3rd-round pick in a 12-team league (~pick 25-36) has an expected value around $15-20, depending on the draft class. Converting that to dollars lets surplus = projected_value - pick_dollar_equivalent, and everything downstream works unchanged.

This roadmap depends on: draft-pick-trade-evaluator phase 1 (pick value curves), keeper-surplus-value (done). It also touches keeper-optimization-solver to ensure draft-pick constraints are handled correctly.

## Status

| Phase | Status |
|-------|--------|
| 1 — Pick-to-dollar translation | in progress |
| 2 — Draft-pick keeper import | not started |
| 3 — Round-based constraints for keeper optimizer | not started |

## Phase 1: Pick-to-dollar translation

Build the function that converts a draft round + league size into an expected dollar value, using the pick value curve from the draft-pick-trade-evaluator roadmap.

### Context

The draft-pick-trade-evaluator roadmap (phase 1) produces a `PickValueCurve` that maps pick number → expected dollar value. For keeper purposes, we need to go from "round 3" → pick range (e.g., picks 25-36 in a 12-team league) → average expected dollar value for that range. This translation is the bridge between draft-pick keeper leagues and the dollar-based surplus pipeline.

### Steps

1. Add `round_to_dollar_cost()` to `src/fantasy_baseball_manager/services/pick_value.py`:
   - Accepts `round: int`, `league: LeagueSettings`, `curve: PickValueCurve`.
   - Computes the pick range for the round: `start = (round - 1) * league.teams + 1`, `end = round * league.teams`.
   - Returns the average expected value across that pick range from the curve.
   - For snake drafts, account for pick order reversal in even rounds (the mid-round pick value, not just the range average).
2. Add `picks_to_dollar_costs()` batch version that converts a list of `(player_id, round)` pairs into `KeeperCost` records with `source="draft_round"` and the dollar-equivalent cost.
3. Write tests verifying:
   - Round 1 produces a higher dollar cost than round 10.
   - The conversion is consistent with the pick value curve (round 1 in a 12-team league ≈ average of picks 1-12).
   - Missing curve data (rounds beyond the curve) returns a floor value of $1.

### Acceptance criteria

- `round_to_dollar_cost()` returns a dollar amount that decreases monotonically with later rounds.
- The dollar amount for round 1 is close to the average of the top N picks on the curve (where N = number of teams).
- Rounds beyond the curve's range return a minimum floor value rather than erroring.
- `picks_to_dollar_costs()` produces valid `KeeperCost` records usable by `compute_surplus()`.

## Phase 2: Draft-pick keeper import

Extend the keeper import and set commands to accept round-based costs, automatically converting them to dollar equivalents via the pick value curve.

### Context

The current CSV import expects a `Cost` column with a dollar amount. Draft-pick keeper leagues need a `Round` column instead. The CLI `set` command similarly accepts `--cost` as dollars. This phase adds a `--format draft-pick` mode that reads rounds and translates them using the phase 1 conversion.

### Steps

1. Extend `import_keeper_costs()` in `ingest/keeper_mapper.py`:
   - Accept an optional `cost_translator` callback: `(int) -> float` (round → dollar cost).
   - When a translator is provided, read the `Round` column instead of `Cost`, call the translator to get the dollar amount, and store with `source="draft_round"`.
   - The existing dollar path (`Cost` column, no translator) remains the default.
2. Add `--format` option to `fbm keeper import`: `auction` (default) or `draft-pick`.
   - When `draft-pick`, load the pick value curve for the given season/system/provider, build the translator via `round_to_dollar_cost()`, and pass it to the import function.
   - Require `--system` and `--provider` flags when using `draft-pick` format (needed to look up the curve).
3. Add `--round` option to `fbm keeper set` as an alternative to `--cost`:
   - Mutually exclusive with `--cost`. When `--round` is given, convert via the pick value curve and store with `source="draft_round"`.
4. Store the original round number in a new optional `KeeperCost.original_round` field so the display layer can show "Round 3 (~$18)" instead of just "$18".
5. Write tests for CSV import with round-based costs and the `set --round` command.

### Acceptance criteria

- `fbm keeper import keepers.csv --season 2026 --league dynasty --format draft-pick --system zar --provider nfbc` reads a `Round` column and stores dollar-equivalent costs.
- `fbm keeper set "Trout" --round 3 --season 2026 --league dynasty --system zar --provider nfbc` converts round 3 to dollars and stores the cost.
- The original round number is preserved and displayed alongside the dollar equivalent.
- All existing dollar-based import paths are unchanged (no regressions).
- `fbm keeper decisions` and `fbm keeper trade-eval` work correctly with draft-pick-imported costs (they just see dollar amounts).

## Phase 3: Round-based constraints for keeper optimizer

Extend the keeper-optimization-solver to handle draft-pick-specific constraints: round penalties, pick forfeiture rules, and round escalation for multi-year keepers.

### Context

Draft-pick keeper leagues often have rules the auction model doesn't capture: "keepers cost the round before where they were drafted" (escalation), "undrafted free agents cost a last-round pick", "you can't keep more than one player per round", or "first-round picks can't be used as keeper slots." These constraints affect which keeper combinations are valid and change the optimization problem.

### Steps

1. Extend `KeeperConstraints` in `domain/keeper_optimization.py` (from the keeper-optimization-solver roadmap):
   - Add `round_escalation: int` (how many rounds earlier the keeper costs next year, default 0). E.g., a player drafted in round 5 costs a round 4 pick next year if escalation = 1.
   - Add `max_per_round: int | None` (max keepers from the same round, typically 1).
   - Add `protected_rounds: set[int] | None` (rounds that can't be used for keepers, e.g., round 1).
   - Add `undrafted_round: int | None` (which round undrafted free agents cost as keepers).
2. Extend `solve_keepers()` to enforce round-based constraints when `source="draft_round"`:
   - Apply round escalation to compute the effective keeper round.
   - Validate max-per-round limits.
   - Exclude players whose escalated round falls in `protected_rounds`.
3. Extend the CLI `fbm keeper optimize` to accept `--round-escalation`, `--max-per-round`, `--protected-rounds`.
4. In the output, show the keeper's effective round (after escalation) alongside the dollar equivalent.
5. Write tests with round-based constraints verifying that the optimizer respects round limits and escalation changes the optimal set.

### Acceptance criteria

- Round escalation correctly bumps keeper costs (a round-5 pick with escalation=1 costs round 4 next year, round 3 the year after).
- `max_per_round` prevents keeping two players from the same round.
- `protected_rounds` excludes players whose escalated cost falls in a protected round.
- The optimizer produces different results with round constraints vs. without.
- Display shows both the effective round and dollar equivalent.

## Ordering

Phase 1 depends on draft-pick-trade-evaluator phase 1 (pick value curves) — that must be implemented first to provide the `PickValueCurve` and `compute_pick_value_curve()` function. Phase 2 depends on phase 1 of this roadmap. Phase 3 depends on phase 2 of this roadmap and at least phase 1 of the keeper-optimization-solver roadmap (needs `KeeperConstraints` and `solve_keepers()` to exist).

```
draft-pick-trade-evaluator phase 1 ──► this roadmap phase 1 ──► phase 2
keeper-optimization-solver phase 1 ──► this roadmap phase 3
```

Phase 1 is the critical path — without pick-to-dollar translation, nothing else works. Phase 2 is the user-facing payoff (draft-pick leagues can actually use the keeper tools). Phase 3 is an enhancement for the optimizer and can wait until the keeper-optimization-solver is underway.
