# Feature Candidate Factory

Enable an autonomous agent to define and test new feature candidates without writing permanent Python transform code. Today, creating a new feature requires adding a `TransformFeature` with a Python function, registering it in a feature set, busting the dataset cache, and re-materializing — a heavyweight process that blocks rapid exploration. These tools collapse that into ad-hoc queries and lightweight evaluation.

## Status

| Phase | Status |
|-------|--------|
| 1 — Ad-hoc aggregation tool | not started |
| 2 — Interaction feature generator | not started |
| 3 — Binned feature constructor | not started |

## Phase 1: Ad-hoc aggregation tool

Given a SQL-like spec, aggregate raw statcast pitch data into a player-season vector without writing a permanent transform.

### Context

The fundamental bottleneck for feature exploration is that every new idea requires a full round-trip through the feature pipeline: write a Python `TransformFeature` → register it in a `FeatureSet` → invalidate the dataset cache → re-materialize. The ad-hoc aggregation tool bypasses all of this by running a parameterized query directly against `statcast_pitch` and returning a player-season Series that can be immediately correlated with targets or injected into a training run.

### Steps

1. Create `src/fantasy_baseball_manager/services/feature_factory.py` with an `aggregate_candidate(expression, seasons, player_type, min_pa, min_ip)` function. The `expression` parameter is a SQL aggregation fragment — e.g., `"AVG(launch_speed)"`, `"AVG(launch_speed) FILTER (WHERE barrel = 1)"`, `"COUNT(*) FILTER (WHERE description = 'swinging_strike') * 1.0 / COUNT(*)"`.
2. The function constructs a full query: `SELECT {player_id_col} AS player_id, SUBSTR(game_date, 1, 4) AS season, {expression} AS value FROM statcast_pitch WHERE ... GROUP BY player_id, season`. It returns a list of `(player_id, season, value)` tuples.
3. Validate the expression against a whitelist of allowed SQL functions (AVG, SUM, COUNT, MIN, MAX, FILTER, CASE) to prevent injection. Reject expressions containing disallowed keywords (DROP, INSERT, UPDATE, DELETE, etc.).
4. Add a `fbm feature candidate "<expression>" --seasons 2021 2022 2023 2024 --player-type batter [--correlate] [--min-pa 100]` CLI command. With `--correlate`, immediately run the target correlation scanner from the data-profiling-tools roadmap on the result.
5. Support naming candidates: `--name barrel_ev` so results can be referenced later.
6. Write tests with synthetic statcast data verifying correct aggregation, NULL handling, and expression validation.

### Acceptance criteria

- Arbitrary SQL aggregation expressions produce correct player-season vectors.
- Expression validation rejects dangerous SQL but allows legitimate aggregation patterns.
- The `--correlate` flag chains directly into target correlation scanning.
- NULL values are handled correctly (players with no qualifying data get NULL, not 0).
- Results can be named for later reference.

## Phase 2: Interaction feature generator

Given two existing features (or candidates from phase 1), compute their product, ratio, or difference as a new candidate.

### Context

The statcast GBM model already has some interaction features (batted ball interactions in the live model), but they were hand-picked. The agent needs to test arbitrary combinations to discover interactions that the tree model might not find on its own — especially ratio features (e.g., barrel rate / chase rate) that compress two dimensions into one meaningful signal.

### Steps

1. Add an `interact_candidates(feature_a, feature_b, operation, seasons, player_type)` function to the feature factory. `operation` is one of: `"product"`, `"ratio"`, `"difference"`, `"sum"`. `feature_a` and `feature_b` can be column names from a materialized dataset or named candidates from phase 1.
2. For ratios, handle division by zero gracefully (return NULL when denominator is 0 or NULL).
3. Add a `fbm feature interact <feature_a> <feature_b> --op ratio --seasons 2021 2022 2023 2024 --player-type batter [--correlate]` CLI command.
4. Support a `--scan` flag that tries all four operations and reports which produces the highest target correlation, saving the agent from running four separate commands.
5. Write tests verifying correct arithmetic, NULL propagation, and division-by-zero handling.

### Acceptance criteria

- All four operations produce correct results.
- Division by zero yields NULL, not an error.
- Features can come from materialized datasets or named candidates.
- The `--scan` flag tries all operations and ranks by correlation.

## Phase 3: Binned feature constructor

Convert continuous features into categorical bins, creating discrete archetypes (e.g., "high barrel + low launch angle").

### Context

While GBM trees can learn arbitrary splits, explicit bins sometimes help with small samples — a bin like "high-barrel, low-launch-angle ground-ball hitters" might have only 30 players per season, but the bin membership itself is a strong signal for BABIP. Bins also make it easier for the agent to reason about player archetypes.

### Steps

1. Add a `bin_candidate(feature, method, n_bins, seasons, player_type)` function to the feature factory. `method` is one of: `"quantile"` (equal-frequency bins), `"uniform"` (equal-width bins), `"custom"` (user-specified breakpoints).
2. Support multi-feature binning: `bin_candidates([feature_a, feature_b], ...)` creates a cross-product of bins, producing archetype labels (e.g., "high_barrel__low_launch_angle").
3. Return both the bin labels and the within-bin target means, so the agent can see if bins meaningfully separate outcomes.
4. Add a `fbm feature bin <feature> --method quantile --bins 4 --seasons 2021 2022 2023 2024 --player-type batter [--cross <feature_b>]` CLI command.
5. Write tests verifying correct bin assignment, cross-product labeling, and within-bin target means.

### Acceptance criteria

- Quantile and uniform binning produce correct bin boundaries.
- Cross-product binning creates meaningful archetype labels.
- Within-bin target means are reported for quick signal assessment.
- Bins are computed per-season (not pooled) to avoid lookahead.

## Ordering

Phases are sequential: 1 → 2 → 3. Phase 1 is the critical piece — it removes the biggest bottleneck. Phase 2 extends phase 1 with combinatorial exploration. Phase 3 adds a different lens (discrete archetypes). Phase 2's `--correlate` flag depends on the data-profiling-tools roadmap (phase 2).

## Dependencies

- **data-profiling-tools (phase 2)**: The `--correlate` flag in all three phases chains into the target correlation scanner from that roadmap. Without it, the feature factory still works but requires manual correlation checking.
