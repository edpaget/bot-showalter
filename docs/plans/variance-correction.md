# Variance Correction Roadmap

Projection systems (ensemble, Steamer, Marcel, etc.) regress toward the mean by design, compressing the standard deviation of rate stats like ERA and WHIP to roughly half of what actual end-of-season outcomes produce. When the ZAR valuation engine computes z-scores from projections, it divides by this compressed stdev, inflating z-scores for top pitchers. The result: Skubal projects at $124 (vs $62 actual in 2025), Skenes at $99, Crochet at $95 — roughly 2x what elite pitchers are actually worth in-season.

This roadmap adds a variance correction layer to the ZAR engine that substitutes historical actual standard deviations for projection-derived stdevs when computing z-scores. The correction is opt-in, backward-compatible, and validated against holdout seasons before adoption.

## Status

| Phase | Status |
|-------|--------|
| 1 — Historical stdev computation | done (2026-03-07) |
| 2 — Engine integration and opt-in flag | not started |
| 3 — Holdout validation and production rollout | not started |

## Phase 1: Historical stdev computation

Build the infrastructure to compute and retrieve per-category standard deviations from historical actual stats, using the same marginal-contribution conversion that the ZAR engine already applies.

### Context

The ZAR engine's `convert_rate_stats` transforms ERA/WHIP into innings-weighted marginal contributions before z-scoring. The variance correction must operate at the same level — we need historical stdevs of the *converted* values, not raw ERA/WHIP. The actual valuations service (`actual_valuations.py`) already loads actual stats and runs them through the same pipeline, so the building blocks exist.

### Steps

1. Create a new module `src/fantasy_baseball_manager/services/historical_stdev.py` with a function that, given a list of seasons, a league config, and the batting/pitching stats repos:
   - Loads actual stats for each season via the repos.
   - Runs them through `convert_rate_stats` using the league's category configs.
   - Computes the population stdev of each category's converted values across all players in each season.
   - Returns the per-category stdev averaged across the requested seasons (e.g., 3-year rolling average).
2. Add a dataclass or TypedDict (e.g., `CategoryStdevs`) to hold the per-category stdev mapping (`dict[str, float]`), keyed by category key.
3. Write unit tests with synthetic data confirming:
   - Converted stdevs for rate categories differ from raw stdevs (validates that the marginal-contribution step matters).
   - Multi-season averaging produces the expected result.
   - Edge cases: single season, season with missing stats, zero-variance category.

### Acceptance criteria

- `compute_historical_stdevs(seasons, league, batting_repo, pitching_repo)` returns a `dict[str, float]` of per-category stdevs computed from actual stats after marginal-contribution conversion.
- Multi-season averaging is tested and correct.
- Unit tests pass with synthetic data — no dependency on the real database.

## Phase 2: Engine integration and opt-in flag

Thread the historical stdev overrides through `compute_z_scores` and the full ZAR pipeline, gated behind an opt-in parameter so existing behavior is preserved by default.

### Context

`compute_z_scores` currently computes `pstdev` from the player pool for each category. Adding an optional `stdev_overrides: dict[str, float] | None` parameter lets callers substitute historical values. The override must flow through `run_zar_pipeline` → `ZarModel._value_pool` and be triggerable from `ModelConfig.model_params` (e.g., `{"variance_correction_seasons": [2022, 2023, 2024, 2025]}`).

### Steps

1. Add an optional `stdev_overrides: dict[str, float] | None = None` parameter to `compute_z_scores`. When provided, use the override value for any category present in the dict; fall back to pool-computed stdev for categories not in the dict.
2. Thread `stdev_overrides` through `run_zar_pipeline` (new optional parameter) so it reaches `compute_z_scores`.
3. In `ZarModel.predict`, check `config.model_params` for a `"variance_correction_seasons"` key. If present, call `compute_historical_stdevs` for those seasons and pass the result as `stdev_overrides` to the pipeline. Compute stdevs separately for the batter and pitcher pools.
4. Update the CLI `valuations rankings` command (or the `compute` command that triggers model prediction) to accept an optional `--variance-correction` flag that populates the model param.
5. Write unit tests for:
   - `compute_z_scores` with overrides produces different z-scores than without.
   - `compute_z_scores` with partial overrides falls back to pool stdev for missing categories.
   - `compute_z_scores` without overrides is unchanged (backward compatibility).
   - Full pipeline integration test with overrides threaded end-to-end.

### Acceptance criteria

- `compute_z_scores` accepts optional `stdev_overrides` and uses them when provided.
- `run_zar_pipeline` passes overrides through to `compute_z_scores`.
- `ZarModel.predict` loads historical stdevs when `variance_correction_seasons` is set in model params.
- All existing ZAR engine tests pass without modification (backward compatibility).
- New tests verify override behavior at unit and integration levels.

## Phase 3: Holdout validation and production rollout

Validate the variance correction against holdout seasons using the before/after comparison protocol, then make it the production default if results improve.

### Context

The correction is only valuable if it improves valuation accuracy against actual outcomes. We need to run the standard `compare old new --season YEAR --top 300 --check` protocol on multiple holdout seasons (at minimum 2024 and 2025) comparing corrected vs uncorrected valuations.

### Steps

1. Generate variance-corrected valuations for holdout seasons (2024, 2025) using the `--variance-correction` flag with 3-4 prior seasons of actuals as the stdev source (e.g., for 2025 holdout, use 2021-2024 actuals).
2. Run the before/after comparison protocol:
   - `compare uncorrected corrected --season YEAR --top 300 --check` for both 2024 and 2025.
   - `compare uncorrected corrected --season YEAR` (full population) for both.
3. Analyze specifically:
   - Whether pitcher valuations at the top are less inflated.
   - Whether the overall value curve shape better matches actuals (from the distribution analysis notebook).
   - Whether batter valuations are unharmed (they should be largely unaffected since counting stats have less compression).
4. If validation passes (`--check` green on all tested seasons for both top-300 and full population):
   - Update the production valuation generation to include `variance_correction_seasons` by default.
   - Regenerate 2026 production valuations with the correction.
   - Re-run the distribution comparison notebook to confirm the pitcher inflation is resolved.
5. If validation fails, investigate:
   - Try different numbers of lookback seasons (2 vs 3 vs 5).
   - Try the scaling-factor variant (multiply z-scores by `actual_stdev / projected_stdev`) instead of full substitution.
   - Document findings and decide whether to iterate or pivot to SGP (out of scope for this roadmap).

### Acceptance criteria

- Before/after comparison run on at least two holdout seasons (2024, 2025).
- If adopted: `--check` passes on all tested seasons for both top-300 and full population.
- If adopted: 2026 production valuations regenerated with correction; top pitcher values no longer 2x actual.
- If rejected: findings documented with data showing why the correction didn't help, and next steps identified.

## Ordering

Phases are strictly sequential:

1. **Phase 1** (historical stdev computation) has no dependencies — it's a new module with new tests.
2. **Phase 2** (engine integration) depends on Phase 1 for the `compute_historical_stdevs` function.
3. **Phase 3** (validation) depends on Phase 2 for the ability to generate corrected valuations.

No external roadmap dependencies. The existing actual valuations infrastructure and ZAR engine are all that's needed.
