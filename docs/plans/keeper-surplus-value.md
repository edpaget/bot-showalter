# Keeper / Dynasty Surplus Value Roadmap

Calculate surplus value for keeper and dynasty leagues, where players are retained at a cost (auction price, draft pick, or contract salary) from the previous season. Surplus value — the difference between a player's projected value and their keeper cost — is the fundamental currency of keeper league strategy. It determines which players to keep, which to throw back, and how to value trades.

This is a new capability with no existing keeper/dynasty infrastructure in the codebase. It builds on the ZAR valuation system for projected dollar values and adds a keeper cost layer with contract tracking, surplus calculation, and trade analysis.

## Status

| Phase | Status |
|-------|--------|
| 1 — Keeper cost storage and import | not started |
| 2 — Surplus value calculation | not started |
| 3 — Keeper-adjusted draft pool | not started |
| 4 — Trade value calculator | not started |

## Phase 1: Keeper cost storage and import

[Phase plan](keeper-surplus-value/phase-1.md)

Define the keeper cost domain model and provide a way to input keeper costs for a league.

### Context

Keeper costs vary by league: some use last year's auction price + inflation, some use the draft round, some have multi-year contracts with escalating salaries. The data model needs to be flexible enough to represent all common formats while the initial import targets the most common case (auction price from last year).

### Steps

1. Create `src/fantasy_baseball_manager/domain/keeper.py` with:
   - `KeeperCost` frozen dataclass: `player_id`, `season`, `league` (str, matches league name in fbm.toml), `cost` (float, in auction dollars), `years_remaining` (int, default 1), `source` ("auction" | "draft_round" | "contract" | "free_agent").
   - `KeeperDecision` frozen dataclass: `player_id`, `player_name`, `position`, `cost`, `projected_value`, `surplus`, `years_remaining`, `recommendation` ("keep" | "release").
2. Create `keeper_repo.py` in `repos/` with `upsert_batch`, `find_by_season_league`, and `find_by_player` methods.
3. Add SQL migration for the `keeper_cost` table with UNIQUE constraint on `(player_id, season, league)`.
4. Add `fbm keeper import <csv-path> --season <year> --league <name>` CLI command to import keeper costs from CSV (columns: player name, cost, years remaining).
5. Support manual entry: `fbm keeper set <player-name> --cost <n> --season <year> --league <name>`.
6. Write tests for the repo and import command.

### Acceptance criteria

- Keeper costs are stored and queryable by season and league.
- CSV import resolves player names to internal IDs.
- Manual `set` command works for individual players.
- Re-importing is idempotent (upsert).

## Phase 2: Surplus value calculation

Compute surplus value for all keepable players and recommend keep/release decisions.

### Context

Surplus value = projected dollar value − keeper cost. Players with positive surplus should be kept; players with negative surplus should be released back into the draft pool. The threshold can be adjusted (e.g., only keep players with surplus > $3 to account for uncertainty).

### Steps

1. Implement `compute_surplus(keeper_costs, valuations, league_settings)` in a `keeper_service.py` that:
   - Joins keeper costs with ZAR valuations by player_id.
   - Computes surplus = value − cost.
   - Marks players as "keep" or "release" based on a configurable threshold.
   - For multi-year contracts, computes total expected surplus across remaining years (discounting future years by a decay factor for projection uncertainty).
2. Add `fbm keeper decisions --season <year> --league <name> --system <valuation-system>` CLI command showing all keeper-eligible players ranked by surplus.
3. Show: player name, position, cost, projected value, surplus, years remaining, recommendation.
4. Support `--threshold <n>` to set the minimum surplus for a keep recommendation.
5. Write tests with known costs and valuations.

### Acceptance criteria

- Surplus is correctly computed as value minus cost.
- Keep/release recommendations align with surplus sign and threshold.
- Multi-year contracts are valued using discounted future surplus.
- Output is sorted by surplus descending.

## Phase 3: Keeper-adjusted draft pool

After keeper decisions, compute the adjusted draft pool and re-run valuations on the remaining players.

### Context

Keeper selections remove players from the draft pool, which changes replacement level and therefore all remaining players' values. If 5 of the top 10 catchers are kept, catcher scarcity increases and the remaining catchers become more valuable. This phase re-runs the valuation engine on the post-keeper player pool.

### Steps

1. Implement `adjusted_pool(all_valuations, kept_players)` that removes kept players and identifies the new pool.
2. Re-run ZAR valuation on the reduced pool with adjusted replacement levels.
3. Add `fbm keeper adjusted-rankings --season <year> --league <name> --system <system>` command showing post-keeper valuations.
4. Highlight players whose value increased significantly due to keepers reducing supply at their position.
5. Write tests verifying that keeping top players at a position increases remaining players' values.

### Acceptance criteria

- Kept players are excluded from the draft pool.
- Replacement level recalculates based on the smaller pool.
- Players at positions with many keepers see a value increase.
- Adjusted rankings reflect the true draft-day market.

## Phase 4: Trade value calculator

Evaluate trades between teams using surplus value as the common currency.

### Context

In keeper leagues, trades involve players with different costs and contract lengths. A $5 player with $25 value (surplus $20) is worth more than a $20 player with $30 value (surplus $10), even though the latter has a higher raw projection. The trade calculator should compare surplus across traded assets.

### Steps

1. Implement `evaluate_trade(team_a_gives, team_b_gives, keeper_costs, valuations)` that computes net surplus exchanged by each side.
2. Create `TradeEvaluation` frozen dataclass: `team_a_surplus_delta`, `team_b_surplus_delta`, `winner`, `category_impact` (optional — how the trade affects each team's category balance).
3. Add `fbm keeper trade-eval --gives <players> --receives <players> --season <year> --league <name>` CLI command.
4. Show per-player surplus and net trade value for each side.
5. Write tests with known surplus values verifying the evaluation.

### Acceptance criteria

- Trade evaluation correctly computes surplus exchanged.
- Multi-player trades are supported.
- Winner/loser determination is based on net surplus delta.
- Multi-year contracts are valued with discounted future surplus.

## Ordering

Phases are sequential: 1 → 2 → 3 → 4. Phase 1 (storage) is a prerequisite. Phase 2 (surplus calculation) is the core deliverable. Phase 3 (adjusted pool) is important for draft preparation in keeper leagues. Phase 4 (trade eval) is useful year-round but lower priority for draft prep. No dependencies on other roadmaps, though the adjusted rankings from phase 3 feed naturally into the draft board export and live draft tracker.
