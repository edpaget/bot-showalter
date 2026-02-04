# Stolen Base Rule Change Adjustment

## What

Re-calibrate stolen base projections to account for the 2023 MLB rule changes (larger bases, pitch clock, limited pickoff attempts) that caused a structural ~30% increase in league-wide stolen base rates.

## Why

The rebaseline stage handles year-over-year league rate changes mechanically, but the regression constants and aging curves for SB/CS were tuned on pre-2023 data where the base-stealing environment was fundamentally different. Key issues:

- **Regression constants:** SB regression at 600 PA was tuned when league SB rates were ~0.020/PA. Post-rule-change rates are ~0.026/PA. The same regression PA implies different levels of trust in player skill vs. league context.
- **Aging curves:** Speed peaks at age 25 with sharp decline (0.035/year). But the rule changes disproportionately benefit older, slower runners (larger bases reduce the distance advantage of pure speed). The aging penalty may be too aggressive post-2023.
- **Attempt rate vs. success rate:** The rule changes increase attempt rates more than success rates. Players who previously didn't run now attempt steals, but their success rates are lower. The current model doesn't distinguish between attempt frequency and conversion skill.

SB correlation is 0.678 — good but with room for improvement, especially given the structural break in 2023.

## Expected Impact

Improved SB projections for the 2024+ evaluation years. The structural break means pre-2023 backtest years may show different optimal parameters than post-2023 years.

## Pipeline Fit

No new stage required. Changes to:
1. `regression_constants.py` — potentially different SB/CS regression PA for post-2023
2. `aging_curves.py` — potentially softer SB aging decline post-2023
3. `RegressionConfig` — add era-aware regression constants

Alternatively, a dedicated `StolenBaseAdjuster` stage that applies rule-change corrections to SB/CS rates based on the projection year.

## Data Requirements

- League-wide SB rates and attempt rates pre/post rule change (2021-2024)
- Per-player SB attempt rates vs. success rates (FanGraphs sprint speed + SB data)
- Sprint speed data from Statcast (available 2015+) for true speed measurement

## Key References

- MLB 2023 rule changes documentation
- Arthur, R. "The Stolen Base Boom" (FiveThirtyEight / The Athletic)
- Tango, T. "Adjusting for Rule Changes in Projections" (The Book Blog)
- Baseball Savant sprint speed leaderboards

## Implementation Sketch

1. Analyze pre/post SB rate distributions to quantify the structural shift
2. Grid search SB/CS regression PA separately for 2023+ projection years
3. Evaluate whether the SB aging curve slope should change post-2023:
   - Compare age-SB curves for 2019-2022 vs. 2023-2024
   - If older players benefit more, reduce `old_rate` for speed stats
4. Optionally incorporate sprint speed as a direct input:
   - `sprint_speed_adj = (player_speed - lg_avg_speed) * speed_to_sb_coeff`
   - Blend with historical SB rate
5. Backtest on 2023 and 2024 separately to validate the adjustment
