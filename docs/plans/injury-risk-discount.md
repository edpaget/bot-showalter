# Injury Risk Discount Roadmap

Discount player projections and valuations by injury probability, using historical IL stint data already ingested into the system. A player projected for 600 PA but with a history of missing 30% of seasons to injury should be valued at closer to 420 PA. This adjustment is especially important for draft strategy — injury-prone stars are systematically overvalued by systems that assume a full healthy season.

The IL stint data is already in the database (via `fbm ingest il`), and the playing-time model projects PA/IP. This roadmap builds an injury risk model on top of that data and integrates it into the valuation pipeline.

## Status

| Phase | Status |
|-------|--------|
| 1 — Injury history profile | not started |
| 2 — Games-lost probability model | not started |
| 3 — Projection and valuation adjustment | not started |

## Phase 1: Injury history profile

[Phase plan](injury-risk-discount/phase-1.md)

Compute per-player injury history summaries from the existing IL stint data.

### Context

The `il_stint` table stores individual IL placements with player_id, season, il_type (10/15/60-day), days, and injury_location. There's no aggregated view of a player's injury track record. Phase 1 builds that profile — how many stints, how many days lost, recurrence patterns, and which body parts are affected.

### Steps

1. Create `src/fantasy_baseball_manager/domain/injury_profile.py` with `InjuryProfile` frozen dataclass: `player_id`, `seasons_tracked` (int), `total_stints` (int), `total_days_lost` (int), `avg_days_per_season` (float), `max_days_in_season` (int), `pct_seasons_with_il` (float), `injury_locations` (dict of location → count), `recent_stints` (last 2 seasons).
2. Create `src/fantasy_baseball_manager/services/injury_profiler.py` with `build_profiles(il_stints, seasons)` that aggregates IL stint data into profiles.
3. Add `fbm report injury-profile <player-name>` CLI command showing a player's injury history summary.
4. Add `fbm report injury-risks --season <year> --min-stints <n>` command listing the most injury-prone players in the projectable pool.
5. Write tests with synthetic IL stint data covering healthy players, chronically injured players, and single-incident players.

### Acceptance criteria

- Injury profiles are correctly computed from IL stint data.
- Days-lost calculation accounts for IL stint duration (using `days` field or `end_date - start_date`).
- Per-player and leaderboard views both work.
- Players with no IL history get a clean profile (0 stints, 0 days).

## Phase 2: Games-lost probability model

Build a simple model that estimates the probability of missing games next season based on the injury profile.

### Context

Not all injury histories are equal — a player with one freak 60-day IL stint is different from one with chronic 10-day IL placements every year. The model should weight recency, frequency, and severity to produce a single "expected games lost" estimate.

### Steps

1. Implement `estimate_games_lost(profile, projection_season)` that produces `ExpectedGamesLost` frozen dataclass: `player_id`, `expected_days_lost` (float), `p_full_season` (probability of playing 140+ games / 160+ IP), `confidence` (low/medium/high based on data volume).
2. Use a weighted approach: recent seasons weighted more heavily (e.g., last season 3×, two seasons ago 2×, three+ seasons ago 1×).
3. Apply a base rate — even players with no injury history have a ~10-15% chance of an IL stint in any given season. Regress toward the population base rate based on sample size.
4. For players with recurring injuries to the same location, apply a recurrence multiplier.
5. Write tests verifying that chronic injury histories produce higher expected days lost than clean histories, and that recency weighting works correctly.

### Acceptance criteria

- Expected days lost is higher for injury-prone players.
- Clean-history players still have a nonzero baseline risk.
- Recency weighting gives more weight to recent seasons.
- Recurring same-location injuries boost the estimate.

## Phase 3: Projection and valuation adjustment

Integrate the injury risk estimate into the projection and valuation pipelines.

### Context

The playing-time model projects PA/IP assuming a healthy season. The injury discount should scale down these projections, which in turn reduces counting stats and dollar values. This doesn't change the rate projections (a player's AVG or ERA is the same when healthy) — it changes the volume projections.

### Steps

1. Implement `apply_injury_discount(projection, expected_games_lost)` that adjusts PA/IP-based stats proportionally. If a player is projected for 600 PA but expected to lose 30 days (~18% of the season), discount PA to ~492 and scale counting stats accordingly.
2. Add an `--injury-adjusted` flag to `fbm predict` that applies the discount to projections.
3. Add an `--injury-adjusted` flag to `fbm valuations rankings` that re-computes values with discounted projections.
4. Add `fbm report injury-adjusted-values --season <year> --system <system>` showing the biggest value changes from injury adjustment — which players lose the most value, and which healthy players gain relative rank.
5. Write tests verifying that PA/IP reduction flows through to counting stats and dollar values.

### Acceptance criteria

- Injury-adjusted projections have lower PA/IP than raw projections.
- Counting stats scale proportionally with the PA/IP reduction.
- Rate stats are unchanged.
- Dollar values reflect the reduced counting stats.
- The biggest movers are players with extensive injury histories.

## Ordering

Phases are strictly sequential: 1 → 2 → 3. Phase 1 is a useful standalone deliverable (injury profiles and risk leaderboard). Phase 2 adds the probabilistic model. Phase 3 integrates it into the draft pipeline. No dependencies on other roadmaps, though the output feeds naturally into the draft board export and live draft tracker.
