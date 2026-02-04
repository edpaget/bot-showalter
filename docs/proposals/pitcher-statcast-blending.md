# Pitcher Statcast Blending (xERA / xBA-against)

**Status: Implemented** — `PitcherStatcastAdjuster` in `marcel_full_statcast_pitching` pipeline.

## What

Blend Statcast pitcher quality metrics (xERA, xBA-against) with Marcel pitcher rates for hits and earned runs, mirroring the `StatcastRateAdjuster` pattern already used for batters.

## Why

ERA correlation is 0.174 and WHIP correlation is 0.282 — the two weakest stats in the projection system. Pitcher normalization improved these by regressing BABIP/LOB% toward league averages, but it treats all pitchers as having the same underlying BABIP skill. Statcast pitch-level data provides direct measurement of pitcher contact management quality that predicts future ERA better than past ERA does.

## Evaluation Results (3-year backtest, 2022-2024)

| Metric | marcel_full_statcast_babip | marcel_full_statcast_pitching | Change |
|--------|---------------------------|-------------------------------|--------|
| ERA Corr | 0.209 | **0.247** | **+18.2%** |
| ERA RMSE | 1.210 | 1.211 | +0.1% |
| WHIP Corr | **0.293** | 0.266 | -9.2% |
| WHIP RMSE | **0.208** | 0.217 | +4.3% |
| Pitch Spearman | 0.413 | **0.418** | +1.2% |
| Pitch Top-20 | 0.350 | **0.367** | +4.9% |

ERA correlation improved substantially (+18.2%), and pitching rank accuracy improved (Spearman +1.2%, top-20 precision +4.9%). However, WHIP degraded (-9.2% correlation, +4.3% RMSE). The hit rate blend from xBA-against may over-correct WHIP components. Decoupling h and er blend weights is the primary follow-up optimization target.

## Pipeline Fit

`PitcherStatcastAdjuster` stage inserted after `PitcherNormalizationAdjuster` and before `StatcastRateAdjuster` in the `marcel_full_statcast_pitching` pipeline:

```
ParkFactor -> PitcherNorm -> PitcherStatcast -> StatcastRate -> BatterBabip -> Rebaseline -> Aging
```

Shares the `CachedStatcastDataSource` and `PlayerIdMapper` with batter stages. Uses `PitcherStatcastDataSource` protocol and `StatcastPitcherStats` dataclass.

## Implementation Details

- **Config:** `PitcherStatcastConfig(blend_weight=0.30, min_pa_for_blend=200)`
- **Hit rate derivation:** `x_h = xba * (1 - bb - hbp)` — xBA-against times AB/BF ratio
- **ER rate derivation:** `x_er = xera / 27.0` — convert per-9-innings to per-out
- **Blend formula:** `rate_new = w * x_rate + (1 - w) * rate`
- **Passthrough:** Batters, pitchers without Statcast data, below PA threshold, missing required rates
- **Diagnostics stored:** `pitcher_xera`, `pitcher_xba_against`, `pitcher_statcast_blended`, `pitcher_blend_weight`

## Follow-up Optimizations

1. **Decouple h and er blend weights.** The WHIP degradation suggests the h blend weight should be lower than the er blend weight. Grid search over h_weight 0.10-0.30 and er_weight 0.20-0.50.
2. **Add xSLG-against for HR rate blending.** The current implementation does not blend HR rate from barrel%-against (only h and er). Adding HR rate blending from xSLG/barrel data could improve W projections.
3. **Incorporate Stuff+ when available.** Stuff+ provides pitch-level quality measurement that xERA may not fully capture.

## Data Requirements

- Statcast pitcher expected stats via pybaseball's `statcast_pitcher_expected_stats()` (available 2015+)
- Statcast pitcher barrel/exit velocity data via `statcast_pitcher_exitvelo_barrels()`
- Per-pitcher: xERA, xBA-against, xSLG-against, xwOBA-against, barrel%-against, hard-hit%-against

## Key References

- Baseball Savant "Expected Stats" methodology
- Tango, T. "Using xwOBA Against to Project Pitcher ERA"
- Petriello, M. "What Stuff+ Tells Us About Pitcher Quality" (MLB.com, 2023)
- Healey, G. "Predicting Pitcher Performance with Pitch-Level Data" (SABR Analytics, 2022)
