# Pitcher Statcast Blend Weight Tuning

## What

Decouple the `PitcherStatcastAdjuster` blend weights for hits (h) and earned runs (er), and grid search for optimal values. Currently both rates use a single `blend_weight=0.30`.

## Why

The `marcel_full_statcast_pitching` evaluation (3-year backtest, 2022-2024) revealed a trade-off: ERA correlation improved +18.2% (0.209 to 0.247), but WHIP correlation degraded -9.2% (0.293 to 0.266). Since WHIP = (H + BB) / IP, the hit rate blend from xBA-against is the likely cause — it shifts h rates toward Statcast-derived values that improve ERA prediction but add noise to WHIP.

The er blend (from xERA/27) is clearly beneficial for ERA. The h blend (from xBA * ab_per_bf) has a mixed signal: it helps ERA (earned runs depend on hits allowed) but hurts WHIP (WHIP directly measures hit rate, which may already be well-calibrated after pitcher normalization).

## Proposed Change

Replace the single `blend_weight` in `PitcherStatcastConfig` with two parameters:

```python
@dataclass(frozen=True)
class PitcherStatcastConfig:
    h_blend_weight: float = 0.15      # lower — h rate already normalized
    er_blend_weight: float = 0.35     # higher — xERA signal is strong
    min_pa_for_blend: int = 200
```

Update `_blend_player` to use per-stat weights:

```python
weights = {"h": self._config.h_blend_weight, "er": self._config.er_blend_weight}
for stat in BLENDED_STATS:
    w = weights[stat]
    rates[stat] = w * sc_rates[stat] + (1.0 - w) * rates[stat]
```

## Grid Search Space

| Parameter | Range | Step |
|-----------|-------|------|
| h_blend_weight | 0.00 - 0.30 | 0.05 |
| er_blend_weight | 0.15 - 0.50 | 0.05 |

Total: 7 x 8 = 56 combinations. With 3-year backtest, this is ~168 pipeline evaluations. At ~2s per evaluation, roughly 6 minutes.

**Primary metric:** Pitching Spearman rho (overall rank accuracy).
**Secondary constraints:** ERA correlation >= 0.240, WHIP correlation >= 0.280.

## Expected Outcome

The sweet spot should have er_blend_weight > h_blend_weight. The hypothesis is:
- Setting h_blend_weight to 0.10-0.15 preserves the ERA benefit from hits (fewer hits = fewer runs) while reducing WHIP distortion.
- Setting er_blend_weight to 0.30-0.40 keeps the strong xERA signal.
- The net effect should recover most of the WHIP regression while keeping ERA correlation near 0.247.

## Implementation Steps

1. Add `h_blend_weight` and `er_blend_weight` to `PitcherStatcastConfig`, deprecate `blend_weight`.
2. Update `_blend_player` to use per-stat weights.
3. Update tests to cover per-stat weight behavior.
4. Add grid search configuration for the two-weight space.
5. Run grid search, update evaluation summary with optimal weights.

## Pipeline Fit

No new stages. This is a configuration change to `PitcherStatcastAdjuster` with a grid search to find optimal values.
