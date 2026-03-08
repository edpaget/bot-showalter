# Distributional ZAR Roadmap

The current ZAR pipeline uses point-estimate playing time projections — a single PA/IP value per player. But playing time is the hardest thing to project, and point estimates hide important risk asymmetry: an injury-prone star's downside (100 PA) is much worse than his upside is good (650 PA vs 600 PA). A player with high PT variance should be valued lower than a player with identical expected PT but low variance.

This roadmap creates a `zar-distributional` model that runs ZAR at multiple playing-time scenarios (using the existing residual percentile buckets from the playing-time model), then computes the expected dollar value across the distribution. This is a new registered model composing `ZarModel`, following principle 4.

## Status

| Phase | Status |
|-------|--------|
| 1 — Scenario projection generator | done (2026-03-08) |
| 2 — Expected-value ZAR engine | not started |
| 3 — `zar-distributional` model | not started |

## Phase 1: Scenario projection generator

Generate multiple playing-time scenario projections for each player, weighted by probability.

### Context

The playing-time model already produces residual percentiles (P10, P25, P50, P75, P90) bucketed by age and injury history. Each percentile represents a plausible PA/IP outcome. This phase converts those percentiles into weighted scenario projections — full stat lines scaled to each PT scenario.

### Steps

1. Create `src/fantasy_baseball_manager/domain/pt_scenario.py` with a `PlayingTimeScenario` frozen dataclass: `percentile` (int, e.g. 10/25/50/75/90), `pa_or_ip` (float), `weight` (float — probability mass assigned to this scenario).
2. Create `src/fantasy_baseball_manager/services/scenario_generator.py` with:
   - `DEFAULT_SCENARIO_WEIGHTS`: a mapping of percentile → weight that approximates a continuous distribution. E.g., P10=0.15, P25=0.20, P50=0.30, P75=0.20, P90=0.15 (symmetric, sums to 1.0).
   - `generate_scenarios(projection, residual_percentiles, scenario_weights=None) -> list[tuple[Projection, float]]`: For each percentile, scale the projection's counting stats proportionally to the scenario PA/IP vs. the point-estimate PA/IP. Rate stats are unchanged. Return (scaled_projection, weight) pairs.
   - `generate_pool_scenarios(projections, residual_buckets, features, scenario_weights=None) -> dict[int, list[tuple[Projection, float]]]`: Apply `generate_scenarios` across the full pool, looking up each player's bucket from their features (age, il_days_1).
3. Handle edge cases: players with no residual bucket get a single scenario at P50 with weight 1.0 (degrades to point estimate). Clamp PA/IP to valid ranges (0-750 / 0-250).
4. Write tests:
   - Scenario generation produces 5 scenarios with correct weights summing to 1.0.
   - Counting stats scale proportionally to PT scenario; rate stats are unchanged.
   - Fallback to single-scenario when no residual bucket is available.

### Acceptance criteria

- Scenarios are generated for each player with weights summing to 1.0.
- Counting stats scale correctly; rate stats are preserved.
- Players without residual data degrade gracefully to point-estimate behavior.

## Phase 2: Expected-value ZAR engine

Run ZAR independently at each scenario level and compute probability-weighted expected dollar values.

### Context

The key challenge: ZAR's dollar values depend on the **entire pool**, not just one player. When one player's PA drops in a downside scenario, it shifts replacement levels and the dollar conversion for everyone. The correct approach is to run the full ZAR pipeline independently for each scenario, then take the expected value of each player's dollar values across scenarios.

However, running 5 full ZAR passes is expensive and the cross-player interaction effects are second-order. A practical simplification: run ZAR once on the point-estimate projections to establish replacement levels and z-score means/stdevs, then compute per-player dollar values at each scenario using those fixed parameters. This makes the computation O(players x scenarios) rather than O(scenarios x players^2).

### Steps

1. Create `src/fantasy_baseball_manager/services/distributional_valuation.py` with:
   - `compute_expected_value(scenario_values: list[tuple[float, float]]) -> float`: Given a list of (dollar_value, weight) pairs, return the weighted mean.
   - `run_distributional_zar(stats_list, categories, positions, roster_spots, num_teams, budget, scenario_map, stdev_overrides=None) -> list[float]`:
     - Run `convert_rate_stats` and `compute_z_scores` once on the point-estimate stats to get pool-wide means, stdevs, and replacement levels.
     - For each player, compute z-scores and dollar values at each PT scenario using the fixed means/stdevs/replacement.
     - Return expected dollar values (probability-weighted mean across scenarios).
2. Expose the ZAR engine's intermediate values (means, stdevs, replacement levels) so they can be reused across scenarios. This may require a small refactor to `compute_z_scores` to optionally accept pre-computed means/stdevs rather than always recomputing.
3. Write tests:
   - A player with symmetric PT distribution gets approximately the same value as the point estimate.
   - A player with left-skewed distribution (injury risk = more downside) gets lower expected value than point estimate.
   - A player with no variance (single scenario) gets exactly the point-estimate value.

### Acceptance criteria

- Expected dollar values correctly weight scenario outcomes.
- Fixed-parameter approach produces values close to (but not identical to) a full multi-pass approach.
- Symmetric distributions produce values close to point estimates.
- Left-skewed (injury-heavy) distributions produce lower values.

## Phase 3: `zar-distributional` model

Register the composed model that wires scenario generation into the distributional ZAR engine.

### Context

This model composes `ZarModel` for the base ZAR computation and adds the distributional layer on top. It needs the playing-time model's residual buckets (loaded from the trained artifact) and the injury profiler's data (for bucket assignment features). The model reads projections, generates per-player PT scenarios, runs distributional ZAR, and persists results.

### Steps

1. Create `src/fantasy_baseball_manager/models/zar_distributional/model.py` with `ZarDistributionalModel`, registered as `"zar-distributional"`.
2. Constructor takes the same dependencies as `ZarModel` plus:
   - A way to load residual buckets (either a path to the playing-time artifacts or a `ResidualBuckets` protocol/callable).
   - Player feature data for bucket assignment (age, il_days_1) — sourced from the player repo and injury profiler.
3. `predict()` implementation:
   - Read projections from repo.
   - Load residual buckets from the playing-time model's trained artifacts.
   - Gather player features (age, recent IL days) for bucket assignment.
   - Generate PT scenarios for each player via `generate_pool_scenarios()`.
   - Run distributional ZAR via `run_distributional_zar()` for batters and pitchers separately.
   - Build `Valuation` objects with `system="zar-distributional"` and expected dollar values.
   - Persist and return.
4. Wire in `factory.py`: construct `ZarDistributionalModel` with standard ZAR deps plus playing-time artifact path and injury profiler.
5. Add `__init__.py` with re-exports and import in `models/__init__.py` for auto-registration.
6. Write tests:
   - End-to-end: synthetic projections + residual buckets → distributional valuations with `system="zar-distributional"`.
   - Injury-prone players (wide PT distributions) are valued lower than in point-estimate ZAR.
   - Stable-PT players get values close to standard ZAR.
   - Model is discoverable via `fbm models list`.
   - `fbm valuations lookup <player> --system zar-distributional` works.

### Acceptance criteria

- `fbm predict zar-distributional --param league=<name> --param projection_system=<sys> --season 2026` produces and persists valuations.
- Players with high PT variance are valued lower than in standard ZAR.
- Players with low PT variance get values approximately equal to standard ZAR.
- Residual buckets are loaded from the playing-time model's trained artifacts (no retraining required).
- All existing ZAR tests remain passing.
- Model is discoverable via `fbm models list`.

## Ordering

Phases are strictly sequential: 1 → 2 → 3.

### Dependencies

- **valuation-system-unification phase 1** (planned): Needed for `ZarModel` system-name configurability. Same dependency as `zar-replacement-padded`.
- **playing-time model** (done): Provides trained residual buckets (P10-P90 by age/injury bucket).
- **injury-risk-discount** (done): Provides injury profiler for bucket assignment features (il_days_1).

### Relationship to `zar-replacement-padded`

These two roadmaps are complementary, not competing. `zar-replacement-padded` addresses **what happens during missed time** (replacement production vs. zero). `zar-distributional` addresses **uncertainty in how much time is missed** (point estimate vs. distribution). A future model could combine both: generate PT scenarios, apply replacement padding at each scenario, then take the expected value. That composition is out of scope here but would be straightforward given both models exist.
