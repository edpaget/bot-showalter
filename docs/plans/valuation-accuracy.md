# Valuation Accuracy Roadmap

ZAR valuations systematically overvalue top pitchers (~2x actual) and miss breakout performers entirely. The variance-correction investigation (2026-03-07) established that the root cause is not z-score stdev compression but rather two structural gaps: (1) projections assume full health, inflating injury-prone pitchers, and (2) projections regress breakout candidates toward career norms, undervaluing high-upside players.

This roadmap attacks valuation accuracy through a series of independent corrections, each validated on holdout seasons (2024, 2025) with an explicit go/no-go gate before proceeding. Phases are ordered by expected impact and implementation cost: fix a known bug first, then layer on injury and breakout adjustments. If these adjustments prove insufficient, a deeper structural change (SGP or distributional ZAR) should be roadmapped separately — see the existing [Valuation System Unification](valuation-system-unification.md), [ZAR Replacement-Padded](zar-replacement-padded.md), and [ZAR Distributional](zar-distributional.md) roadmaps.

## Status

| Phase | Status |
|-------|--------|
| 1 — Fix injury discount PA/IP threshold floor | not started |
| 2 — Injury discount holdout validation | not started |
| 3 — Breakout/bust valuation integration | not started |
| 4 — Combined adjustment validation | not started |

## Phase 1: Fix injury discount PA/IP threshold floor

Fix a bug where the injury discount drops borderline players from the valuation pool entirely, making injury-adjusted evaluations incomparable to baseline.

### Context

The injury discount in `discount_projections` scales counting stats (including PA and IP) by `max(0, 1 - expected_days_lost / 183)`. The ZAR model then filters players below `min_pa=200` / `min_ip=30`. Players originally just above these thresholds get pushed below by the discount and are excluded entirely — they receive no valuation at all, not even $0. This drops ~45-53 players from the evaluation set, making apples-to-apples accuracy comparisons impossible.

### Steps

1. In `ZarModel.predict`, apply the PA/IP eligibility filter **before** injury discounting, so the player pool is determined by the original (undiscounted) projections.
2. Alternatively, floor discounted PA/IP at the eligibility threshold so players stay in the pool but with reduced counting stats. Either approach preserves the same player set as the uncorrected baseline.
3. Add a test confirming that injury-adjusted predictions produce the same player count as uncorrected predictions for the same season/league.
4. Regenerate `injury-adjusted` valuations for 2024 and 2025 holdout seasons and verify matched player counts now equal the uncorrected versions (883 and 769 respectively).

### Acceptance criteria

- Injury-adjusted ZAR predictions include the same players as uncorrected predictions (identical player count).
- Existing injury discount behavior is preserved for players well above the threshold — only the edge case changes.
- All existing ZAR and injury discount tests pass.

### Gate

**Automatic pass** — this is a bug fix, not a speculative improvement. Proceed to phase 2 after merging.

---

## Phase 2: Injury discount holdout validation

Re-evaluate the injury discount against holdout seasons now that the player-count bug is fixed, producing a true apples-to-apples comparison.

### Context

Our initial evaluation (2026-03-07) showed injury adjustment improving 2024 MAE by 5% (3.90 vs 4.12 on the filtered set) but flat on 2025 (4.63 vs 4.63). However, those results were distorted by the threshold bug — the injury-adjusted set excluded ~50 players. With phase 1 fixed, we can get a clean signal.

### Steps

1. Generate uncorrected and injury-adjusted ZAR valuations for 2024 and 2025 holdout seasons using Steamer projections.
2. Run `valuations evaluate` on all four runs and record MAE and Spearman ρ.
3. Inspect the top 20 mispricings in each: are the biggest pitcher overvaluations reduced? Are any batter valuations harmed?
4. Test with different `--seasons-back` values (3 vs 5) to see if the injury lookback window matters.

### Acceptance criteria

- Evaluation run on both holdout seasons with identical player counts.
- Results table with MAE, ρ, and top mispricings documented.

### Gate: go/no-go

**Go** if injury adjustment improves MAE on at least one holdout season without degrading ρ by more than 0.01 on either season. **No-go** if MAE worsens on both seasons or ρ drops significantly. If no-go, skip phase 4 (combined) and proceed directly to phase 5 (SGP).

---

## Phase 3: Breakout/bust valuation integration

Integrate breakout/bust classifier probabilities into ZAR valuations to discount bust-risk players and boost breakout candidates.

### Context

The breakout/bust classifier (completed 2026-03-05) outputs calibrated probabilities: `p_breakout`, `p_bust`, `p_neutral` per player. Currently these are used only for standalone reports (`fbm report breakout-candidates`, `fbm report bust-risks`). The second-largest bucket of valuation errors comes from breakout misses — players like Seth Lugo ($0 predicted, $38 actual) and Bryce Miller ($0 predicted, $36 actual) who vastly outperform projections.

Integrating P(bust) as a value discount and P(breakout) as a value boost could reduce these errors. The challenge is calibrating the magnitude: too aggressive and we introduce new errors; too conservative and we gain nothing.

### Steps

1. Add an optional `breakout_bust_adjustment` parameter to the ZAR prediction pipeline. When enabled, load breakout/bust predictions for the target season's player pool.
2. Design the adjustment formula. Starting point: `adjusted_value = value * (1 - bust_discount * p_bust) + breakout_bonus * p_breakout`, where `bust_discount` and `breakout_bonus` are tunable parameters.
3. Run a parameter sweep on a development season (e.g., 2023) to find reasonable values for `bust_discount` and `breakout_bonus`. Avoid overfitting — use coarse grid (e.g., 0.25, 0.5, 0.75, 1.0).
4. Add a `--breakout-bust` CLI flag to `fbm predict zar` that enables the adjustment.
5. Generate breakout/bust-adjusted valuations for holdout seasons (2024, 2025) and evaluate against actuals.
6. Inspect specifically: are previously-missed breakouts now valued higher? Are bust-risk pitchers reduced? Are false-positive breakout boosts introducing new errors?

### Acceptance criteria

- `ZarModel.predict` accepts an optional breakout/bust adjustment parameter.
- CLI flag `--breakout-bust` triggers the adjustment.
- Adjustment formula is tested with synthetic data (known P(breakout)/P(bust) values produce expected value shifts).
- Holdout evaluation results documented with MAE, ρ, and top mispricings.

### Gate: go/no-go

**Go** if breakout/bust adjustment improves MAE on at least one holdout season without degrading ρ by more than 0.01. **No-go** if it worsens both MAE and ρ, or if the parameter sweep shows high sensitivity (small changes in discount/bonus cause large accuracy swings, indicating overfitting). If no-go, document findings and proceed to phase 5.

---

## Phase 4: Combined adjustment validation

Test whether injury + breakout/bust adjustments together outperform either individually.

### Context

If phases 2 and 3 both pass their gates, the natural question is whether combining them yields additive gains. Injury discounts attack overvaluation of fragile pitchers; breakout/bust adjustments attack undervaluation of emerging players and overvaluation of decline candidates. These target different error buckets, so they may complement each other. But they could also interfere — e.g., an injury-discounted pitcher who is also flagged as a bust candidate gets double-penalized.

### Steps

1. Generate combined (injury + breakout/bust) valuations for 2024 and 2025 holdout seasons.
2. Compare four versions head-to-head on the same player set: uncorrected, injury-only, breakout/bust-only, combined.
3. Check for double-penalty effects: identify players where both adjustments fire and verify the combined discount is reasonable.
4. If combined outperforms, adopt as the new production default. Regenerate 2026 valuations.
5. If individual adjustments outperform combined, adopt whichever single adjustment performed best.

### Acceptance criteria

- Four-way comparison table (uncorrected, injury, breakout/bust, combined) on both holdout seasons.
- Production 2026 valuations regenerated with the winning configuration.
- Top pitcher overvaluations demonstrably reduced compared to uncorrected baseline.

### Gate: go/no-go

**Go** if combined improves over the best individual adjustment on at least one metric (MAE or ρ) without degrading the other. **No-go** if combined is worse than the best individual; in that case, adopt the better individual adjustment. If neither individual adjustment passed its gate (phases 2-3 both no-go), document findings and consider a deeper structural change via the [Valuation System Unification](valuation-system-unification.md), [ZAR Replacement-Padded](zar-replacement-padded.md), or [ZAR Distributional](zar-distributional.md) roadmaps.

## Ordering

- **Phase 1** has no dependencies — it is a standalone bug fix and should be done first.
- **Phases 2 and 3** depend on phase 1 (for clean evaluation baselines) but are independent of each other. They can be implemented in either order or in parallel.
- **Phase 4** depends on phases 2 and 3. If either phase's gate is no-go, phase 4 is skipped or simplified to adopt the passing phase's result.

The go/no-go gates create natural off-ramps: if injury adjustment alone solves the problem (phase 2 go, phase 3 no-go), there's no need to build breakout integration. If all phases here fail to move the needle, the next step would be a separate roadmap for SGP or distributional valuation — not more tweaks to the current ZAR framework.
