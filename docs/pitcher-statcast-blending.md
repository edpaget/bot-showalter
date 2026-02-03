# Pitcher Statcast Blending (Stuff+ / xERA)

## What

Blend Statcast pitcher quality metrics (xERA, xFIP, Stuff+) with Marcel pitcher rates, mirroring the `StatcastRateAdjuster` pattern already used for batters.

## Why

ERA correlation is 0.174 and WHIP correlation is 0.282 — the two weakest stats in the projection system. Pitcher normalization improved these by regressing BABIP/LOB% toward league averages, but it treats all pitchers as having the same underlying BABIP skill. Statcast pitch-level data (velocity, movement, spin, extension) provides direct measurement of pitcher "stuff" quality that predicts future ERA better than past ERA does. A pitcher whose Stuff+ increased due to a new pitch or velocity gain is undervalued by historical stats alone.

## Expected Impact

ERA and WHIP are the primary targets. Pitcher Spearman rho (currently 0.329) should improve as the model better separates pitchers with elite stuff from those who outperformed their peripherals.

## Pipeline Fit

New `PitcherStatcastAdjuster` stage inserted after `PitcherNormalizationAdjuster` and before `RebaselineAdjuster`. It blends Statcast-derived expected rates with the normalized Marcel rates for hits and earned runs.

Shares the existing `StatcastDataSource` infrastructure. Requires a new `statcast_pitcher_expected_stats()` method on the data source protocol.

## Data Requirements

- Statcast pitcher expected stats via pybaseball's `statcast_pitcher_expected_stats()` (available 2015+)
- Per-pitcher: xERA, xBA-against, xSLG-against, xwOBA-against, barrel%-against
- Optionally: Stuff+ (pitch-level quality metric, available from Baseball Savant 2020+)

## Key References

- Baseball Savant "Expected Stats" methodology
- Tango, T. "Using xwOBA Against to Project Pitcher ERA"
- Petriello, M. "What Stuff+ Tells Us About Pitcher Quality" (MLB.com, 2023)
- Healey, G. "Predicting Pitcher Performance with Pitch-Level Data" (SABR Analytics, 2022)

## Implementation Sketch

1. Extend `StatcastDataSource` protocol with `pitcher_expected_stats(year) -> list[StatcastPitcherStats]`
2. Create `StatcastPitcherStats` dataclass: `player_id, xera, xba_against, xslg_against, xwoba_against, barrel_rate_against, pa`
3. Derive expected H rate from xBA-against: `x_h = xba_against * ab_per_bf`
4. Derive expected HR rate from barrel%-against: `x_hr = barrel_rate * league_hr_per_barrel * bip_rate`
5. Derive expected ER from xERA: `x_er = xera / 9 * (outs / 3)` or reconstruct from components
6. Blend: `final_rate = w * statcast_rate + (1-w) * normalized_marcel_rate`
7. Default blending weight of 0.25-0.35 (pitcher Statcast metrics are noisier than batter metrics)
8. Store diagnostics: `pitcher_xera`, `pitcher_xwoba_against`, `pitcher_blend_weight`
