# Keeper League Analysis Roadmap

Two new features for keeper league draft preparation: (1) a league-wide keeper overview that projects every team's keepers, compares total values, and surfaces trade targets, and (2) a post-keeper draft board that removes projected keepers from the player pool so the user can see who will actually be available in the draft.

Both features build on existing infrastructure: `estimate_other_keepers()` for projecting league keepers, `adjust_valuations_for_league_keepers()` for revaluing the post-keeper pool, Yahoo roster fetching, and the draft board renderer.

## Status

| Phase | Status |
|-------|--------|
| 1 — League-wide keeper overview | not started |
| 2 — Post-keeper draft board | not started |
| 3 — Category gap analysis | not started |

## Phase 1: League-wide keeper overview

Add a `yahoo keeper-league` command that projects every team's keepers, ranks teams by total keeper value, and identifies trade targets (other teams' Nth+ most valuable players that exceed the user's worst keeper).

### Context

The `yahoo keeper-decisions` command only analyzes the user's own team. During preseason trade discussions, the user needs to see what every team is likely keeping and where surplus talent exists on other rosters. Currently this requires ad-hoc scripts that combine roster data with valuations.

The existing `estimate_other_keepers()` service already estimates which players other teams will keep. The `fetch_league_rosters()` helper already fetches non-user team rosters. This phase wraps those into a proper CLI command with structured output.

### Steps

1. Create a `LeagueKeeperOverview` domain model holding per-team keeper projections (team name, is_user flag, projected keepers with values, total value, category score totals) and a list of trade targets (player name, value, position, owning team, rank on that team).
2. Create a `build_league_keeper_overview()` service function that takes rosters, valuations, max_keepers, and the user's team key. It projects each team's top-N keepers, computes category totals, ranks teams, and identifies trade targets (players ranked > max_keepers on other teams whose value exceeds the user's Nth keeper).
3. Add a `print_league_keeper_overview()` output function that renders: (a) a team ranking table with keeper names and total value, (b) a per-category comparison showing the user's rank out of all teams, and (c) a trade targets table sorted by value.
4. Wire up the `yahoo keeper-league` CLI command with parameters: `--league`, `--season`, `--system`, `--max-keepers` (defaults from league config), `--top-targets` (how many trade targets to show, default 15).
5. Add tests for `build_league_keeper_overview()` covering: correct keeper projection, category aggregation, trade target identification, and edge cases (unmapped players, ties at the keeper cutoff).

### Acceptance criteria

- `uv run fbm yahoo keeper-league --league keeper` shows all 12 teams' projected keepers ranked by total value.
- Each team row shows keeper names, values, and total.
- A category comparison section shows the user's rank in each category (e.g., "OBP: 2.6 (rank 10/12)").
- A trade targets section lists other teams' surplus players valued above the user's worst keeper.
- The command reuses existing roster data from the DB when available (does not re-fetch from Yahoo API unless rosters are missing).
- Service function is testable with injected fakes (no Yahoo API dependency in tests).

## Phase 2: Post-keeper draft board

Add a `--exclude-keepers` flag to the `draft board` command (and a standalone `yahoo draft-board` variant) that removes projected league keepers from the player pool and re-ranks the remaining players.

### Context

The existing `draft board` command shows all players ranked by valuation. In a keeper league, the top ~48 players (12 teams x 4 keepers) won't be available in the draft. The user needs to see the actual draft pool — who's left after keepers — to plan picks effectively. The existing `adjust_valuations_for_league_keepers()` already handles the revaluation math; this phase exposes it through the draft board UI.

### Steps

1. Add a `--exclude-keepers` flag to `draft board` that accepts a Yahoo league name. When set, the command fetches league rosters, estimates keepers, removes them from the valuation pool, and builds the board from the remaining players.
2. Alternatively (or additionally), add a `yahoo draft-board` command that defaults to excluding keepers and uses Yahoo league context for roster data. This avoids requiring the user to pass both `--league` (for league settings) and `--exclude-keepers` (for Yahoo league name).
3. Extend the draft board output to optionally show a "Kept by" annotation for players near the cutoff (e.g., a player who barely missed being kept, or who was kept by a specific team). This helps the user understand why certain players are/aren't available.
4. Support `--exclude-keepers` on related commands: `draft tiers`, `draft scarcity`, `draft scarcity-rankings`, and `draft upgrades`. These all start from a valuation list that can be filtered the same way.
5. Add tests verifying that the keeper-excluded board correctly removes kept players and that the remaining players are re-ranked properly.

### Acceptance criteria

- `uv run fbm draft board --season 2026 --league h2h --exclude-keepers keeper --top 36` shows the top 36 available (non-kept) players.
- Kept players do not appear in the board.
- Player rankings reflect the post-keeper pool (not just the original rankings with gaps).
- The `--exclude-keepers` flag works with `draft tiers`, `draft scarcity`, `draft scarcity-rankings`, and `draft upgrades`.
- When combined with `--top`, the count refers to available players (not total players before filtering).

## Phase 3: Category gap analysis

Add a `yahoo draft-needs` command that combines the user's projected keepers with the post-keeper draft board to recommend which categories to target and which available players best fill those gaps.

### Context

The existing `draft needs` command requires the user to manually specify their roster. With keeper league context, the system already knows the user's keepers and their category contributions. This phase automates the gap analysis: given your keepers' category strengths and weaknesses, which available players provide the most marginal value?

### Steps

1. Create a `build_keeper_draft_needs()` service that takes the user's projected keepers, their category scores, the post-keeper player pool, and league settings. It computes category deficits relative to league average (or a target rank), then scores each available player by how much they fill the weakest categories.
2. Add a `yahoo draft-needs` command that automatically uses the user's projected keepers as the starting roster and the post-keeper pool as the available players. Parameters: `--league`, `--season`, `--system`, `--top` (number of recommendations).
3. The output should show: (a) category strength/weakness summary from keepers, (b) top recommended players with a breakdown of which categories they help, and (c) a "draft priority" list (e.g., "1. OBP bat, 2. Workhorse SP, 3. Power bat").
4. Add tests for the needs scoring logic, verifying that players who fill weak categories rank higher than those who reinforce already-strong categories.

### Acceptance criteria

- `uv run fbm yahoo draft-needs --league keeper --season 2026` shows category gaps and player recommendations without manual roster input.
- Recommendations prioritize players who fill the user's weakest categories.
- The command uses post-keeper pool (not full player pool) for recommendations.
- Output includes both a category summary and specific player recommendations.

## Ordering

**Phase 1** has no dependencies and should be done first — it establishes the league-wide keeper projection logic that phases 2 and 3 build on.

**Phase 2** depends on phase 1's keeper estimation (to know which players to exclude). It could technically reuse `estimate_other_keepers()` directly, but the phase 1 service provides a cleaner interface.

**Phase 3** depends on both phase 1 (keeper category scores) and phase 2 (post-keeper pool). It should be done last.

All three phases depend on existing completed roadmaps: Yahoo Fantasy Integration, Keeper Surplus Value, and Keeper Optimization Solver.
