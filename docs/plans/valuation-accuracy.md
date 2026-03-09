# Valuation Accuracy Roadmap

ZAR valuations systematically overvalue top pitchers (~2x actual) and miss breakout performers entirely. The variance-correction investigation (2026-03-07) established that the root cause is not z-score stdev compression but rather two structural gaps: (1) projections assume full health, inflating injury-prone pitchers, and (2) projections regress breakout candidates toward career norms, undervaluing high-upside players.

This roadmap attacks valuation accuracy through a series of independent corrections, each validated on holdout seasons (2024, 2025) with an explicit go/no-go gate before proceeding. Phases are ordered by expected impact and implementation cost: fix a known bug first, then layer on injury and breakout adjustments. If these adjustments prove insufficient, a deeper structural change (SGP or distributional ZAR) should be roadmapped separately — see the existing [Valuation System Unification](valuation-system-unification.md), [ZAR Replacement-Padded](zar-replacement-padded.md), and [ZAR Distributional](zar-distributional.md) roadmaps.

## Status

| Phase | Status |
|-------|--------|
| 1 — Fix injury discount PA/IP threshold floor | done (2026-03-08) |
| 2 — Injury discount holdout validation | done (2026-03-08), no-go |
| 3 — Breakout/bust valuation integration | not started |
| 4 — Combined adjustment validation | not started |

## Phase 1: Fix injury discount PA/IP threshold floor

Fix a bug where the injury discount drops borderline players from the valuation pool entirely, making injury-adjusted evaluations incomparable to baseline.

### Context

The injury discount in `discount_projections` scales counting stats (including PA and IP) by `max(0, 1 - expected_days_lost / 183)`. The ZAR model then filters players below `min_pa=200` / `min_ip=30`. Players originally just above these thresholds get pushed below by the discount and are excluded entirely — they receive no valuation at all, not even $0. This drops ~45-53 players from the evaluation set, making apples-to-apples accuracy comparisons impossible.

The fix targets `ZarInjuryRiskModel` (the first-class `zar-injury-risk` system created by the Valuation System Unification roadmap), which delegates to `ZarModel.predict` after discounting projections.

### Steps

1. In `ZarInjuryRiskModel.predict`, apply the PA/IP eligibility filter **before** calling `discount_projections`, so the player pool is determined by the original (undiscounted) projections.
2. Alternatively, floor discounted PA/IP at the eligibility threshold so players stay in the pool but with reduced counting stats. Either approach preserves the same player set as the uncorrected baseline.
3. Add a test confirming that `fbm predict zar-injury-risk` produces the same player count as `fbm predict zar` for the same season/league.
4. Regenerate `zar-injury-risk` valuations for 2024 and 2025 holdout seasons and verify matched player counts now equal the `zar` versions (883 and 769 respectively).

### Acceptance criteria

- `fbm predict zar-injury-risk` produces the same player count as `fbm predict zar` for a given season/league.
- Existing injury discount behavior is preserved for players well above the threshold — only the edge case changes.
- All existing ZAR and `zar-injury-risk` tests pass.

### Gate

**Automatic pass** — this is a bug fix, not a speculative improvement. Proceed to phase 2 after merging.

---

## Phase 2: Injury discount holdout validation

Re-evaluate the injury discount against holdout seasons now that the player-count bug is fixed, producing a true apples-to-apples comparison.

### Context

Our initial evaluation (2026-03-07) showed injury adjustment improving 2024 MAE by 5% (3.90 vs 4.12 on the filtered set) but flat on 2025 (4.63 vs 4.63). However, those results were distorted by the threshold bug — the injury-adjusted set excluded ~50 players. With phase 1 fixed, we can get a clean signal.

### Steps

1. Generate baseline and injury-adjusted valuations for holdout seasons:
   ```bash
   fbm predict zar --season 2024 --param league=h2h --param projection_system=steamer --version holdout
   fbm predict zar-injury-risk --season 2024 --param league=h2h --param projection_system=steamer --version holdout
   fbm predict zar --season 2025 --param league=h2h --param projection_system=steamer --version holdout
   fbm predict zar-injury-risk --season 2025 --param league=h2h --param projection_system=steamer --version holdout
   ```
2. Evaluate all runs against actuals and record MAE and Spearman ρ:
   ```bash
   fbm valuations evaluate --season 2024 --system zar --version holdout
   fbm valuations evaluate --season 2024 --system zar-injury-risk --version holdout
   fbm valuations evaluate --season 2025 --system zar --version holdout
   fbm valuations evaluate --season 2025 --system zar-injury-risk --version holdout
   ```
3. Inspect the top 20 mispricings (`--top 20`): are the biggest pitcher overvaluations reduced? Are any batter valuations harmed?
4. Test with different `--seasons-back` values (3 vs 5) to see if the injury lookback window matters.

### Acceptance criteria

- Evaluation run on both holdout seasons with identical player counts (phase 1 fix verified).
- Results table with MAE, ρ, and top mispricings documented.

### Gate: go/no-go

**Go** if injury adjustment improves MAE on at least one holdout season without degrading ρ by more than 0.01 on either season. **No-go** if MAE worsens on both seasons or ρ drops significantly. If no-go, skip phase 4 (combined) and proceed directly to phase 5 (SGP).

### Results (2026-03-08)

**Note:** Phase 1's `_floor_playing_time` had a secondary bug — it floored PA/IP for *all* players below threshold, not just those pushed below by the injury discount. This inflated the injury-risk pool to ~3× the baseline. Fixed in this phase by checking original (pre-discount) PA/IP before flooring.

#### Summary table

| System | Season | n | MAE | ρ | ΔMAE | Δρ |
|--------|--------|---|-----|---|------|-----|
| zar (baseline) | 2024 | 883 | 3.92 | 0.7139 | — | — |
| zar-injury-risk (5yr) | 2024 | 883 | 3.88 | 0.7011 | −0.04 | −0.0128 |
| zar-injury-risk (3yr) | 2024 | 883 | 3.87 | 0.7054 | −0.05 | −0.0085 |
| zar (baseline) | 2025 | 769 | 4.31 | 0.6941 | — | — |
| zar-injury-risk (5yr) | 2025 | 769 | 4.53 | 0.6828 | +0.22 | −0.0113 |
| zar-injury-risk (3yr) | 2025 | 769 | 4.45 | 0.6873 | +0.14 | −0.0068 |

#### Top mispricings analysis

**2024:** The injury discount slightly reduced some pitcher overvaluations (Kevin Gausman $74→$49, Pablo López $62→$51) but introduced new distortions — it pushed Shohei Ohtani from $16→$3 (actual $73) and Aaron Judge from $55→$33 (actual $99), worsening undervaluation of elite batters who happen to have injury history. The biggest mispricings (Spencer Strider $102→$0, Ronald Acuña $95→$0) were unchanged since those are healthy-season projection errors, not injury-related.

**2025:** The discount amplified pitcher overvaluation rather than reducing it — Logan Gilbert rose from $60→$85 (actual $12), and new entries like Ryan Walker ($44→$0) and Emmanuel Clase ($40→$0) appeared in the top mispricings. The discount appears to redistribute value among pitchers rather than systematically reducing overvaluation.

#### seasons_back comparison

The 3-year lookback consistently outperforms the 5-year lookback: lower MAE on both seasons and smaller ρ degradation. If the injury discount were adopted, 3 years would be the preferred window — more recent injury history is more predictive.

#### Decision: **No-go**

The injury discount fails the gate criteria:

1. **MAE worsens on 2025** (+0.14 to +0.22), indicating the discount is not a reliable improvement.
2. **ρ drops beyond the 0.01 threshold** on 2024 with 5yr lookback (−0.0128) and on 2025 with both lookbacks (−0.0113 / −0.0068 barely under threshold).
3. **Root cause:** The simple multiplicative discount (scale counting stats by health fraction) is too blunt — it penalizes batters with minor injury history (Ohtani, Judge) as much as injury-prone pitchers, distorting relative rankings without systematically improving accuracy.

The injury discount should not be adopted as-is. Alternative approaches (replacement-level padding via `zar-replacement-padded`, or position-specific injury adjustments) may fare better. Phase 4 (combined validation) should proceed without `zar-injury-risk` unless a redesigned discount is developed.

---

## Phase 3: Breakout/bust valuation integration

Integrate breakout/bust classifier probabilities into ZAR valuations to discount bust-risk players and boost breakout candidates.

### Context

The breakout/bust classifier (completed 2026-03-05) outputs calibrated probabilities: `p_breakout`, `p_bust`, `p_neutral` per player. Currently these are used only for standalone reports (`fbm report breakout-candidates`, `fbm report bust-risks`). The second-largest bucket of valuation errors comes from breakout misses — players like Seth Lugo ($0 predicted, $38 actual) and Bryce Miller ($0 predicted, $36 actual) who vastly outperform projections.

Integrating P(bust) as a value discount and P(breakout) as a value boost could reduce these errors. The challenge is calibrating the magnitude: too aggressive and we introduce new errors; too conservative and we gain nothing.

### Steps

1. Create a `ZarBreakoutBustModel` following the same pattern as `ZarInjuryRiskModel` — a first-class model registered as `zar-breakout-bust` that loads breakout/bust predictions, applies an adjustment formula, and delegates to `ZarModel.predict` with `valuation_system="zar-breakout-bust"`.
2. Design the adjustment formula. Starting point: `adjusted_value = value * (1 - bust_discount * p_bust) + breakout_bonus * p_breakout`, where `bust_discount` and `breakout_bonus` are tunable parameters.
3. Run a parameter sweep on a development season (e.g., 2023) to find reasonable values for `bust_discount` and `breakout_bonus`. Avoid overfitting — use coarse grid (e.g., 0.25, 0.5, 0.75, 1.0).
4. Generate and evaluate on holdout seasons:
   ```bash
   fbm predict zar-breakout-bust --season 2024 --param league=h2h --param projection_system=steamer --version holdout
   fbm predict zar-breakout-bust --season 2025 --param league=h2h --param projection_system=steamer --version holdout
   fbm valuations evaluate --season 2024 --system zar-breakout-bust --version holdout
   fbm valuations evaluate --season 2025 --system zar-breakout-bust --version holdout
   ```
5. Inspect specifically: are previously-missed breakouts now valued higher? Are bust-risk pitchers reduced? Are false-positive breakout boosts introducing new errors?

### Acceptance criteria

- `fbm predict zar-breakout-bust` produces and persists valuations with `system="zar-breakout-bust"`.
- Adjustment formula is tested with synthetic data (known P(breakout)/P(bust) values produce expected value shifts).
- Holdout evaluation results documented with MAE, ρ, and top mispricings.

### Gate: go/no-go

**Go** if breakout/bust adjustment improves MAE on at least one holdout season without degrading ρ by more than 0.01. **No-go** if it worsens both MAE and ρ, or if the parameter sweep shows high sensitivity (small changes in discount/bonus cause large accuracy swings, indicating overfitting). If no-go, document findings.

---

## Phase 4: Combined adjustment validation

Compare all available ZAR variants head-to-head on holdout seasons to pick the production default.

### Context

By this point, several ZAR variants may exist — each attacking pitcher overvaluation from a different angle:

- **`zar`** — baseline, no adjustments.
- **`zar-injury-risk`** — simple injury discount (scale counting stats by health fraction).
- **`zar-breakout-bust`** — breakout/bust probability adjustment (from phase 3).
- **`zar-replacement-padded`** — fills missed-time PA/IP with replacement-level production instead of zeroing them ([roadmap](zar-replacement-padded.md)).
- **`zar-distributional`** — runs ZAR across multiple playing-time scenarios weighted by probability ([roadmap](zar-distributional.md)).

Not all of these will be ready at the time phase 4 runs — include whichever systems are available. The goal is a single head-to-head comparison on the same player set to pick the winner.

### Steps

1. Generate holdout valuations for every available system:
   ```bash
   for sys in zar zar-injury-risk zar-breakout-bust zar-replacement-padded zar-distributional; do
     fbm predict $sys --season 2024 --param league=h2h --param projection_system=steamer --version holdout
     fbm predict $sys --season 2025 --param league=h2h --param projection_system=steamer --version holdout
   done
   ```
2. Evaluate all systems against actuals:
   ```bash
   for sys in zar zar-injury-risk zar-breakout-bust zar-replacement-padded zar-distributional; do
     fbm valuations evaluate --season 2024 --system $sys --version holdout --top 20
     fbm valuations evaluate --season 2025 --system $sys --version holdout --top 20
   done
   ```
3. Build a comparison matrix: MAE, ρ, and top-pitcher error for each system × season.
4. Inspect for double-penalty effects in models that stack adjustments (e.g., `zar-replacement-padded` with breakout/bust). If promising, test a combined model.
5. Adopt the best-performing system as the production default. Regenerate 2026 valuations.

### Acceptance criteria

- Comparison matrix covering all available ZAR variants on both holdout seasons.
- Production 2026 valuations regenerated with the winning system.
- Top pitcher overvaluations demonstrably reduced compared to baseline `zar`.

### Gate: go/no-go

**Go** if any variant improves over baseline `zar` on MAE without degrading ρ by more than 0.01 on either holdout season. Adopt the best variant. **No-go** if no variant clears that bar; in that case, document findings — the pitcher inflation problem may require fundamentally different inputs (in-season data, more granular injury models) rather than valuation-engine changes.

## Ordering

- **Phase 1** has no dependencies — it is a standalone bug fix and should be done first.
- **Phases 2 and 3** depend on phase 1 (for clean evaluation baselines) but are independent of each other. They can be implemented in either order or in parallel.
- **Phase 4** depends on phases 2 and 3. If either phase's gate is no-go, phase 4 is skipped or simplified to adopt the passing phase's result.

The go/no-go gates create natural off-ramps: if injury adjustment alone solves the problem (phase 2 go, phase 3 no-go), there's no need to build breakout integration. If all phases here fail to move the needle, the next step would be a separate roadmap for SGP or distributional valuation — not more tweaks to the current ZAR framework.
