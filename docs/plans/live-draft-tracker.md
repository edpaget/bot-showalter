# Live Draft Tracker Roadmap

Build a stateful, interactive draft assistant that tracks picks as they happen and recommends the best available player at each selection. During a live draft, the tracker maintains the state of the player pool, your roster, and your category needs, synthesizing all pre-draft analysis (valuations, tiers, scarcity, ADP) into a single real-time recommendation.

This is the capstone draft-day tool. It depends on the valuation system (already built) and benefits from — but does not require — the tier generator, positional scarcity, category balance tracker, and ADP integration roadmaps.

## Status

| Phase | Status |
|-------|--------|
| 1 — Draft state engine | not started |
| 2 — Recommendation engine | not started |
| 3 — Interactive CLI session | not started |
| 4 — Draft log and post-draft analysis | not started |

## Phase 1: Draft state engine

Build the core stateful draft tracker — a state machine that tracks picks, maintains the available player pool, and records rosters.

### Context

During a live draft, the user needs to record each pick (theirs and opponents') and get updated recommendations. The state engine is the foundation: it enforces pick order, removes drafted players from the pool, assigns them to the correct team's roster, and validates roster constraints.

### Steps

1. Create `src/fantasy_baseball_manager/services/draft_state.py` with:
   - `DraftConfig` frozen dataclass: `teams`, `roster_slots` (from LeagueSettings), `format` ("snake" | "auction"), `user_team` (int), `season`.
   - `DraftPick` frozen dataclass: `pick_number`, `team`, `player_id`, `player_name`, `position`, `price` (for auction, None for snake).
   - `DraftState` mutable dataclass: `config`, `picks` (list), `available_pool` (set of player_ids), `team_rosters` (dict of team → list of picks), `current_pick` (int).
2. Implement `DraftEngine` class with methods:
   - `start(valuations, config)` → initializes state with full player pool.
   - `pick(player_name_or_id, team, position, price?)` → records a pick, removes from pool, validates roster constraints, advances pick counter.
   - `undo()` → reverses the last pick.
   - `available(position?)` → returns remaining players, optionally filtered by position.
   - `my_roster()` → returns the user's current roster.
   - `my_needs()` → returns unfilled roster slots.
3. Handle snake draft pick ordering (1→N, N→1 alternating).
4. Handle auction format (any order, budget tracking per team).
5. Write comprehensive tests: pick/undo, roster constraint enforcement, snake ordering, auction budget limits.

### Acceptance criteria

- Draft state correctly tracks all picks and rosters.
- `pick()` rejects players not in the available pool.
- `pick()` rejects picks that violate roster constraints (e.g., drafting a 2nd catcher when only 1 slot).
- `undo()` fully reverses the last pick (player returns to pool, roster updated).
- Snake pick ordering is correct for any number of teams.
- Auction budget tracking prevents overspending.

## Phase 2: Recommendation engine

Add a recommendation engine that suggests the best available player at each pick based on valuations, scarcity, and roster needs.

### Context

The recommendation engine is the brain of the draft tracker. At each pick, it needs to answer: "given what I already have, what I still need, and what's likely to be available at my next pick, who should I draft?" This combines value (ZAR), scarcity (positional dropoff), and need (unfilled roster slots).

### Steps

1. Implement `recommend(state, valuations, tiers?, scarcity?)` that returns a ranked list of recommendations with reasoning:
   - `Recommendation` frozen dataclass: `player_id`, `player_name`, `position`, `value`, `score` (composite recommendation score), `reason` (e.g., "best value", "positional scarcity", "fills need at C").
2. Scoring formula: weighted combination of raw value, positional scarcity multiplier, and need bonus (unfilled position slots get a boost).
3. If tier data is available, penalize picks within the same tier as the next-best alternative at that position (no urgency — you can wait).
4. If ADP data is available, factor in "picks until your next selection" to estimate who will still be available.
5. Show top 5-10 recommendations at each pick.
6. Write tests with a partially completed draft state, verifying that recommendations shift as roster fills.

### Acceptance criteria

- Recommendations change as the draft progresses and roster fills.
- Positional needs are reflected in the scoring (empty positions get boosted).
- Players at scarce positions are prioritized when the dropoff is imminent.
- Recommendation list is re-ranked each time a pick is recorded.

## Phase 3: Interactive CLI session

Build a REPL-style interactive CLI for live draft tracking.

### Context

During a draft, the user needs a fast, keyboard-driven interface. A REPL loop with short commands (`pick`, `undo`, `best`, `need`, `roster`) is the right UX — no subcommand overhead, minimal typing.

### Steps

1. Add `fbm draft live --season <year> --system <system> --teams <n> --slot <n> --format <snake|auction>` command that enters an interactive session.
2. Implement a REPL loop with commands:
   - `pick <player> [position] [price]` — record a pick (auto-detects team from pick order in snake, requires team in auction).
   - `undo` — reverse last pick.
   - `best [position]` — show top recommendations (optionally filtered by position).
   - `need` — show unfilled roster slots.
   - `roster [team]` — show a team's roster (default: user's team).
   - `pool [position]` — show remaining player pool.
   - `status` — show current pick number, round, team on the clock.
   - `quit` — exit the session.
3. Support fuzzy player name matching (the user won't type full names during a live draft).
4. Auto-display recommendations after each pick.
5. Support saving/loading draft state to a file for interruption recovery.
6. Write tests for the REPL command parsing and state persistence.

### Acceptance criteria

- Interactive session starts and accepts commands.
- Fuzzy player name matching resolves unambiguous inputs correctly.
- Ambiguous names prompt for clarification.
- Recommendations auto-display after each pick.
- Draft state can be saved and resumed.
- `quit` exits cleanly.

## Phase 4: Draft log and post-draft analysis

After the draft completes, generate a summary report analyzing the draft.

### Context

Once the draft is over, a post-draft report shows how your team looks: total value captured, category strengths/weaknesses, and a grade relative to the pre-draft optimal roster.

### Steps

1. Implement `draft_report(state, valuations, league_settings)` that produces:
   - Total roster value vs. budget or vs. optimal achievable value.
   - Per-category projected standings (where you rank across teams in each category).
   - Position-by-position grade (actual pick value vs. what was available).
   - Biggest steals and reaches (picks that were far above/below ADP).
2. Add `report` command to the REPL and a `fbm draft report <draft-state-file>` standalone command.
3. Write tests with a completed draft state.

### Acceptance criteria

- Report covers total value, category breakdown, and per-pick analysis.
- Steals/reaches are correctly identified relative to ADP or value.
- Report can be generated from a saved draft state file.

## Ordering

Phases are sequential: 1 → 2 → 3 → 4. Phase 1 (state engine) is a prerequisite for everything. Phase 2 (recommendations) is the core value-add. Phase 3 (interactive CLI) makes it usable during a live draft. Phase 4 (post-draft analysis) is a nice-to-have.

This roadmap benefits from the following being completed first: tier generator (for tier-aware recommendations), positional scarcity (for scarcity scoring), ADP integration (for opponent modeling and steal/reach detection), and category balance tracker (for need-based scoring). However, phase 1 can start immediately.
