# Category Balance Tracker Roadmap

Track projected category strengths and weaknesses as a roster is assembled during a draft. In H2H categories and roto leagues, winning requires balance across all scoring categories — not just total value. A roster stacked with power hitters might lead the league in HR and RBI but finish last in SB and AVG. The category balance tracker projects where a partial roster stands in each category and identifies which categories need reinforcement.

This builds on the league settings (which define scoring categories) and the projection system (which provides per-player stat projections). It integrates with the live draft tracker but is also useful standalone for evaluating any roster.

## Status

| Phase | Status |
|-------|--------|
| 1 — Roster category projection | not started |
| 2 — Category need identification | not started |
| 3 — Draft integration and running balance | not started |

## Phase 1: Roster category projection

[Phase plan](category-balance-tracker/phase-1.md)

Given a set of players (a partial or complete roster), project the team's totals and rates in each scoring category.

### Context

The projection system produces per-player stat projections, and the league settings define which stats are scoring categories (counting vs. rate, direction). What's missing is a service that sums counting stats and computes weighted-average rate stats across a roster to produce team-level category projections.

### Steps

1. Create `src/fantasy_baseball_manager/services/category_tracker.py` with:
   - `TeamCategoryProjection` frozen dataclass: `category` (str), `projected_value` (float), `league_rank_estimate` (int, 1-based), `strength` ("strong" | "average" | "weak").
   - `RosterAnalysis` frozen dataclass: `projections` (list of TeamCategoryProjection), `strongest_categories` (list of str), `weakest_categories` (list of str).
2. Implement `analyze_roster(player_ids, projections, league_settings)`:
   - For counting categories: sum across roster.
   - For rate categories: compute weighted average using the configured denominator (e.g., AVG = total H / total AB).
   - Estimate league rank by comparing against typical league baselines (e.g., multiply per-player averages by roster size × teams to estimate league median).
3. Write tests with a known roster and projections, verifying category totals and rate calculations.

### Acceptance criteria

- Counting stats are summed correctly across the roster.
- Rate stats are weighted correctly (not simple averages — uses denominator weighting).
- League rank estimation produces reasonable values (1-N for N teams).
- Both batting and pitching categories are covered.

## Phase 2: Category need identification

Identify which categories a partial roster is weakest in and recommend player profiles that would address those gaps.

### Context

During a draft, after picking several players, you need to know: "what am I missing?" and "what kind of player should I target next?" This phase adds the gap analysis and targeting logic.

### Steps

1. Implement `identify_needs(analysis, available_players, projections)` that:
   - Ranks categories by weakness (lowest projected league rank).
   - For each weak category, finds the available players who contribute most to that category.
   - Returns `CategoryNeed` objects: `category`, `current_rank`, `target_rank`, `best_available` (top 5 players who improve this category the most).
2. Handle tradeoffs: a player who helps SB but hurts AVG should be flagged when both are needs.
3. Add `fbm draft needs --roster <player-names-or-file> --season <year> --system <system>` CLI command.
4. Write tests with a roster that's weak in a specific category, verifying the correct need is identified and recommendations make sense.

### Acceptance criteria

- Weakest categories are correctly identified.
- Recommended players genuinely improve the weak categories.
- Tradeoff warnings are shown when a player helps one category but hurts another.
- Works with partial rosters (not all slots filled).

## Phase 3: Draft integration and running balance

Integrate the category tracker into the live draft tracker so that needs update automatically after each pick.

### Context

During a live draft, the category balance should update in real time as picks are made. This phase hooks into the draft state engine so the tracker runs automatically rather than requiring manual roster input.

### Steps

1. Integrate `analyze_roster` into the draft state engine's `pick()` flow — after each user pick, recompute the category balance.
2. Add a `balance` command to the draft REPL that shows the current category projection.
3. Add a `needs` command to the draft REPL that shows category needs and player recommendations.
4. Modify the recommendation engine from the live draft tracker to incorporate category needs as a scoring factor — players who address weak categories get a boost.
5. Auto-display a compact category summary after each user pick (e.g., "Weak: SB, AVG | Strong: HR, RBI, K").
6. Write tests verifying that the category balance updates correctly as picks are recorded.

### Acceptance criteria

- Category balance updates automatically after each pick.
- `balance` and `needs` commands produce correct output.
- Recommendations factor in category needs.
- Compact summary is shown after each pick without being verbose.

## Ordering

Phases are sequential: 1 → 2 → 3. Phase 1 is useful standalone for pre-draft roster evaluation. Phase 2 adds the targeting intelligence. Phase 3 integrates with the live draft tracker (and therefore depends on that roadmap's phase 1 being complete). Phases 1-2 can proceed in parallel with the live draft tracker roadmap.
