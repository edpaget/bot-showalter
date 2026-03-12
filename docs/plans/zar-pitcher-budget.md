# ZAR Pitcher Budget Roadmap

ZAR systematically overvalues pitchers relative to ADP and independent targets. Empirical analysis of the 2025 ensemble projections identified three compounding structural causes:

1. **Budget split is too pitcher-heavy.** The `ROSTER_SPOTS` mode allocates 47.1% of the auction budget to pitchers (9 batter slots, 8 pitcher slots). Industry consensus and empirical research (Smart Fantasy Baseball, FanGraphs) place the optimal pitcher allocation at 30–35%. This alone inflates every pitcher's dollar value by ~40% relative to market pricing.
2. **Pitcher VAR is top-concentrated.** The top 5 pitchers capture 25.6% of all positive pitcher VAR vs 17.2% for batters. This is driven by high inter-category correlation among SP-relevant categories (ERA↔WHIP r=0.85, SO↔W r=0.97) and the structural SP/RP bifurcation (SV+HLD correlates negatively with SO and W). The result: ace SPs absorb a disproportionate share of the pitcher budget, producing $75+ valuations that never materialize in real auctions.
3. **No cross-pool normalization.** Batter and pitcher composite z-scores are computed in independent pools with different variance structures (batter composite stdev=4.03, pitcher composite stdev=3.47, pitcher skew=1.57 vs batter skew=0.66). Dollar conversion is proportional to VAR within each pool, so the more skewed pitcher distribution concentrates dollars on fewer players.

This roadmap addresses the budget split and cross-pool normalization. It does not attempt SP/RP sub-pool separation, which is a larger structural change that should only be considered if these simpler fixes prove insufficient.

**Important caveat:** The [Valuation Accuracy](valuation-accuracy.md) roadmap concluded that projection accuracy — not the valuation formula — is the binding constraint on overall ranking quality. These fixes target a specific, measurable symptom (pitcher overvaluation relative to WAR and ADP) rather than promising large overall accuracy gains. Each phase has strict go/no-go gates.

## Status

| Phase | Status |
|-------|--------|
| 1 — Configurable budget split ratio | done (2026-03-11) |
| 2 — Pitcher composite normalization | done (2026-03-11) |
| 3 — Holdout validation and adoption | not started |

## Phase 1: Configurable budget split ratio

Add a `FIXED_RATIO` budget split mode that allows specifying an explicit hitter/pitcher percentage, replacing the mechanistic roster-slot derivation.

### Context

The current `ROSTER_SPOTS` mode derives the split from `roster_batters / (roster_batters + roster_pitchers)`. With 9 batter and 8 pitcher slots, this produces a 52.9/47.1 split. The alternative `CATEGORIES` mode would yield 50/50 (5 batting, 5 pitching categories) — even worse.

Neither mode reflects how real auction markets allocate money. Across formats, the empirical consensus is 65–70% hitting / 30–35% pitching ([Smart Fantasy Baseball](https://www.smartfantasybaseball.com/2014/02/what-is-the-ideal-spending-allocation-between-pitchers-and-hitters/)). The reason: pitchers are more volatile, more replaceable from the waiver wire, and carry higher injury risk — factors that ZAR's formula-driven split ignores.

A fixed ratio mode lets us empirically calibrate the split against holdout data rather than deriving it from roster geometry.

### Steps

1. **Add `FIXED_RATIO` to `BudgetSplitMode`** in `domain/league_settings.py`. Add a `budget_hitter_pct: float | None` field to `LeagueSettings` (default `None`; required when mode is `FIXED_RATIO`).
2. **Update `compute_budget_split()`** in `engine.py` to handle the new mode: `batter_budget = total * hitter_pct`, `pitcher_budget = total * (1 - hitter_pct)`.
3. **Update league config parsing** in `config_league.py` to read `budget_split = "fixed_ratio"` and `budget_hitter_pct = 0.67` from `fbm.toml`.
4. **Create a `zar-budget` model variant** (or parameterize via `model_params`) that uses `FIXED_RATIO` with a configurable percentage. This allows testing multiple splits without modifying the league config.
5. **Add tests.** Unit test for the new split mode. Verify backward compatibility — existing `ROSTER_SPOTS` and `CATEGORIES` modes produce identical output.

### Acceptance criteria

- `FIXED_RATIO` mode with `budget_hitter_pct = 0.67` produces a 67/33 split.
- Existing league configs with `ROSTER_SPOTS` or `CATEGORIES` are unaffected.
- The split can be overridden at predict time via model params (for grid-search testing in phase 3).
- Tests cover edge cases: `hitter_pct = 0.0`, `1.0`, `0.5`.

---

## Phase 2: Pitcher composite normalization

Normalize pitcher composite z-scores to match the batter pool's variance before dollar conversion, so that VAR concentration is comparable across pools.

### Context

The core insight: because dollar conversion is proportional to VAR within each pool, the *shape* of the composite z-score distribution determines how dollars are distributed. The pitcher pool's higher skew (1.57 vs 0.66) and top-concentration means ace SPs absorb outsized dollar shares.

The industry's standard fix — the "0.8 pitcher multiplier" — is a crude proxy for the 4/5 category participation ratio. Since SPs don't contribute to SV+HLD, they effectively participate in 4/5 pitching categories, making their raw composite z-score inflated relative to a batter who contributes across all 5 batting categories. The 0.8 multiplier addresses this, but a normalization approach (equalizing composite z variance across pools) is more principled and adapts to any category configuration.

A more principled approach: after computing composite z-scores in each pool, divide the pitcher composites by `pitcher_composite_stdev / batter_composite_stdev`. This rescales the pitcher distribution to have the same spread as the batter distribution, equalizing VAR concentration without requiring an arbitrary multiplier. The effect: dollars within the pitcher pool are distributed more evenly, reducing the $75+ ace valuations.

### Steps

1. **Add a `normalize_cross_pool` flag** to `run_zar_pipeline()` (default `False` for backward compatibility). When enabled, accept an external reference stdev and scale composite z-scores by `reference_stdev / pool_stdev` before replacement-level computation.
2. **Wire the two-pool normalization in `ZarModel.predict()`**. Run both pipelines once to compute composite z stdevs, then re-run the pitcher pipeline (or post-adjust composites) with the batter stdev as reference. Consider caching the converted stats to avoid redundant rate-stat conversion.
3. **Alternatively, implement as a post-hoc composite scaling.** After `compute_z_scores()` returns, scale each pitcher's `composite_z` by the ratio before passing to `run_optimal_pipeline()`. This avoids modifying the pipeline signature and keeps the change minimal.
4. **Make the normalization configurable** via `model_params["normalize_cross_pool"] = True` so it can be tested independently of the budget split change.
5. **Add tests.** Verify that normalization equalizes composite z stdev across pools. Verify that dollar values change but budget totals are preserved. Verify backward compatibility when flag is off.

### Acceptance criteria

- When normalization is enabled, pitcher and batter composite z-score stdevs are within 5% of each other.
- Pitcher pool dollar distribution is less top-concentrated: top-5 pitcher share of positive pitcher VAR decreases (target: from 25.6% to <20%).
- Total pitcher budget is unchanged (normalization affects within-pool distribution, not the split).
- Baseline ZAR without the flag produces identical output to current behavior.

---

## Phase 3: Holdout validation and adoption

Test all combinations of budget split × normalization against holdout seasons. Adopt the best configuration if it passes regression gates.

### Context

Phase 1 and 2 produce two independent levers. They may interact — a corrected budget split may reduce the need for normalization, or normalization may matter more at certain split ratios. This phase tests the full grid.

The evaluation must use independent targets (WAR ρ, top-N hit rate) per the established protocol. The `--check` flag on `fbm valuations compare` gates on WAR ρ regression (<0.01 drop) and hit-rate regression (<5pp drop).

### Steps

1. **Define the test grid.** Budget split ratios: 60%, 65%, 67%, 70%. Normalization: on/off. Total: 8 configurations. Baseline: current baseline ZAR (ROSTER_SPOTS split, no normalization, all categories active including SV+HLD).
2. **Generate holdout valuations** for each configuration on seasons 2024 and 2025 using ensemble projections.
3. **Evaluate each configuration** using `fbm valuations compare` against the baseline ZAR:
   - WAR ρ (overall, batters, pitchers separately)
   - Top-25, top-50, top-100 hit rates
   - Pitcher WAR ρ specifically (the metric most likely to improve)
4. **Build a comparison matrix.** Configuration × season × metric. Identify the Pareto-optimal configurations (no other config dominates on all metrics).
5. **Run regression gate checks** (`--check`) for the top 1-2 candidates against baseline on both seasons.
6. **Inspect pitcher-specific rankings.** For the best candidate: are the top 20 pitchers more plausible? Are ace SP valuations in the $25-40 range rather than $60-75? Are relievers appropriately discounted?
7. **Adopt or reject.** If a configuration passes `--check` on both seasons and shows meaningful pitcher WAR ρ improvement, adopt it as the new production default. Update ZAR params or create a new variant.
8. **Document results** in this roadmap's status table, including the comparison matrix and go/no-go rationale.

### Acceptance criteria

- Comparison matrix covers all 8 configurations on both holdout seasons with WAR ρ and hit-rate metrics.
- The adopted configuration (if any) passes `--check` against baseline ZAR on both 2024 and 2025.
- Pitcher WAR ρ improves by >0.02 on at least one holdout season without degrading batter metrics.
- Top-5 pitcher dollar values are in a more realistic range (under $50).
- If no configuration improves on the baseline, document findings and close the roadmap as no-go.

### Gate: go/no-go

**Go** if any configuration improves pitcher WAR ρ by >0.02 on at least one holdout season, does not degrade batter WAR ρ, and passes the `--check` regression gate on both seasons. **No-go** if all configurations either degrade overall metrics or show no meaningful improvement, confirming that the pitcher overvaluation is a projection problem (pitchers projected for more production than they deliver) rather than a formula problem.

---

## Ordering

- **Phase 1** has no dependencies. The `BudgetSplitMode` extension and config parsing are self-contained.
- **Phase 2** has no hard dependency on phase 1 but benefits from it — normalization can be tested with or without the budget split change.
- **Phase 3** depends on phases 1 and 2 (needs both levers available for grid search). Also depends on existing infrastructure: `fbm valuations compare --check` (done), ensemble projections for 2024-2025 (done), evaluation framework (done).
- Phases 1 and 2 can be implemented in parallel.
- If phase 3 results in no-go, consider the [Composite GBM Tuning](composite-gbm-tuning.md) roadmap as the alternative path — improving pitcher projection accuracy rather than valuation formula.
