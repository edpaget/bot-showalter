# Positional Upgrade Calculator Roadmap

Given a partially drafted roster, quantify the marginal fantasy value of adding each available player. Instead of simply ranking by raw value, this tool answers "how much does Player X improve my team?" — accounting for category balance, positional slots already filled, and diminishing returns. This is the analytical core behind smart mid-draft decisions.

This roadmap depends on: draft board service (done), valuations with `category_scores` (done), league settings with `positions` (done). Benefits from: category balance tracker (planned), tier generator (planned).

## Status

| Phase | Status |
|-------|--------|
| 1 — Marginal value engine | not started |
| 2 — Upgrade comparison view | not started |
| 3 — Opportunity cost scoring | not started |
| 4 — CLI commands | not started |

## Phase 1: Marginal Value Engine

Build the core calculation that scores each available player by how much they improve a partially filled roster, considering both total value and category balance.

### Context

The draft board ranks players by absolute value, which is correct at pick 1 but increasingly wrong as the draft progresses. After drafting three elite power hitters, the marginal value of another HR-heavy bat is lower than a speed-and-average player who fills a gap. This engine computes roster-relative marginal value.

### Steps

1. Define domain types in `src/fantasy_baseball_manager/domain/positional_upgrade.py`:
   - `RosterSlot` frozen dataclass: `position: str`, `player_id: int | None`, `player_name: str | None`, `value: float`, `category_z_scores: dict[str, float]`.
   - `RosterState` frozen dataclass: `slots: list[RosterSlot]`, `open_positions: list[str]`, `total_value: float`, `category_totals: dict[str, float]` (sum of z-scores per category across rostered players).
   - `MarginalValue` frozen dataclass: `player_id: int`, `player_name: str`, `position: str`, `raw_value: float`, `marginal_value: float`, `category_impacts: dict[str, float]` (per-category z-score improvement), `fills_need: bool` (True if player fills an open positional slot), `upgrade_over: str | None` (name of player this would replace at the position, if applicable).
2. Build `build_roster_state()` in `src/fantasy_baseball_manager/services/positional_upgrade.py`:
   - Accepts `drafted_player_ids: list[int]`, `board: DraftBoard`, `league: LeagueSettings`.
   - Maps drafted players to roster slots based on `league.positions` and player positions.
   - Handles multi-position eligibility (assign player to slot that maximizes open flexibility).
   - Identifies remaining open positions.
   - Computes category totals across rostered players using `DraftBoardRow.category_z_scores`.
   - Returns `RosterState`.
3. Build `compute_marginal_values()` in `services/positional_upgrade.py`:
   - Accepts `state: RosterState`, `available: list[DraftBoardRow]`, `league: LeagueSettings`.
   - For each available player:
     - If the player fills an open positional slot: marginal value = raw value + category-need bonus.
     - If the player would replace an existing player at a position: marginal value = value difference + category impact.
     - If the player doesn't fit any open slot and isn't an upgrade: marginal value = 0 (or UTIL slot value).
   - Category-need bonus: for each category where `category_totals[cat]` is below the league median (z < 0), add a bonus proportional to the player's z-score in that category.
   - Returns `list[MarginalValue]` sorted by `marginal_value` descending.
4. Write tests verifying:
   - A player at an open position scores higher than a same-value player at a filled position.
   - A player who improves a weak category scores higher than one who piles onto a strong category.
   - Multi-position players are assigned optimally.

### Acceptance criteria

- `compute_marginal_values()` re-ranks available players based on roster context.
- Players at open positions are boosted over those at filled positions.
- Category-need bonus shifts rankings toward players who improve weak categories.
- Multi-position eligibility is handled (a SS/2B-eligible player can fill either open slot).

## Phase 2: Upgrade Comparison View

For a specific position, show the current starter, the best available upgrade, and the net value gained — enabling quick "should I upgrade catcher now or wait?" decisions.

### Context

During a draft, the common question is "should I take the best available player, or should I fill position X now before the good options dry up?" This phase provides a per-position view comparing current starters against the best available upgrade at each position.

### Steps

1. Define types in `domain/positional_upgrade.py`:
   - `PositionUpgrade` frozen dataclass: `position: str`, `current_player: str | None`, `current_value: float`, `best_available: str`, `best_available_value: float`, `upgrade_value: float` (best - current), `next_best: str | None`, `dropoff_to_next: float` (best - next best at this position), `urgency: str` ("high" / "medium" / "low").
2. Build `compute_position_upgrades()` in `services/positional_upgrade.py`:
   - Accepts `state: RosterState`, `available: list[DraftBoardRow]`, `league: LeagueSettings`.
   - For each position in `league.positions`:
     - Find current starter (or None if unfilled).
     - Find top-2 available players at this position.
     - Compute upgrade value and dropoff.
     - Assign urgency: "high" if open slot + large dropoff to next, "medium" if open slot + moderate dropoff, "low" if filled + small upgrade available.
   - Returns `list[PositionUpgrade]` sorted by urgency then upgrade_value.
3. Write tests for urgency assignment logic and correct identification of upgrades.

### Acceptance criteria

- Open positions with steep dropoff are flagged as "high" urgency.
- Filled positions show accurate upgrade value (can be negative if current starter is better than best available).
- Dropoff metric correctly measures the gap between the #1 and #2 available player at each position.

## Phase 3: Opportunity Cost Scoring

Add opportunity cost awareness: factor in what you give up by taking a positional fill now vs. waiting a round, based on the rate at which players are being drafted.

### Context

"High urgency" from phase 2 doesn't account for what else you could draft instead. If the best catcher is available but so is a top-5 outfielder who won't last another round, the opportunity cost of taking the catcher may be too high. This phase estimates that cost.

### Steps

1. Add to `domain/positional_upgrade.py`:
   - `OpportunityCost` frozen dataclass: `position: str`, `recommended_player: str`, `marginal_value: float`, `opportunity_cost: float` (value of the best alternative you'd forgo), `net_value: float` (marginal - opportunity cost), `recommendation: str` ("draft now" / "wait" / "borderline").
2. Build `compute_opportunity_costs()` in `services/positional_upgrade.py`:
   - Accepts `marginal_values: list[MarginalValue]`, `state: RosterState`, `league: LeagueSettings`, `picks_until_next: int` (how many picks until user's next turn).
   - Estimates which players will be gone by next pick (top N from `marginal_values` where N ≈ `picks_until_next`).
   - For each position fill candidate: opportunity cost = value of the best non-position-fill player who would be gone by next pick.
   - Net value = marginal value - opportunity cost.
   - Returns `list[OpportunityCost]` sorted by net_value descending.
3. Write tests with scenarios where opportunity cost flips the recommendation (e.g., elite outfielder available makes drafting a catcher suboptimal despite high urgency).

### Acceptance criteria

- Opportunity cost correctly identifies when waiting is better than filling a position.
- `picks_until_next` parameter controls how many players are estimated to be drafted between user picks.
- "draft now" recommendation only appears when net value is positive.

## Phase 4: CLI Commands

Expose the positional upgrade calculator through the CLI for use during draft preparation and live drafts.

### Steps

1. Add `fbm draft upgrades --roster <player-names-or-ids> --season <year> --system <system>`:
   - Prints top-10 marginal value players given the current roster.
   - Columns: rank, player, position, raw value, marginal value, fills need, category impacts.
2. Add `fbm draft position-check --roster <player-names-or-ids> --season <year> --system <system>`:
   - Prints per-position upgrade comparison.
   - Columns: position, current player, best available, upgrade value, urgency.
3. Add `--opportunity-cost --picks-until-next <n>` flag to the `upgrades` command for opportunity cost analysis.
4. Support `--roster-file <path>` for loading roster from a text file (one player per line).
5. Register under the `draft` typer group in `cli/app.py`.

### Acceptance criteria

- `fbm draft upgrades` re-ranks players by marginal value given an input roster.
- `fbm draft position-check` shows a clean per-position comparison table.
- Opportunity cost flag changes recommendations when a superior non-fill player is available.
- Roster can be specified as a list of names or loaded from a file.

## Ordering

Phase 1 is independent and the most impactful — it delivers the core marginal value calculation. Phase 2 depends on phase 1 (needs `RosterState`). Phase 3 depends on phases 1-2 and benefits from mock draft simulator data for better "who will be gone" estimates but can use simple heuristics without it. Phase 4 depends on all prior phases. The category balance tracker roadmap is complementary — if implemented first, its `RosterAnalysis` output can feed directly into the category-need bonus in phase 1.
