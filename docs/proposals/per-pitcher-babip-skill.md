# Per-Pitcher BABIP Skill Model

## What

Replace the flat league-mean BABIP regression target in `PitcherNormalizationAdjuster` with a pitcher-specific expected BABIP derived from batted-ball profile (GB%, FB%, IFFB%, line drive rate).

## Why

The current pitcher normalization regresses all pitchers toward the same league BABIP (~.300). This is correct on average but systematically misprices pitchers with extreme batted-ball profiles. Ground-ball pitchers sustain lower BABIPs (.280-.290) because ground balls are converted to outs more often than fly balls. Fly-ball pitchers with high infield-fly rates also suppress BABIP. The flat regression penalizes these pitchers and inflates their projected hit rates.

From the evaluation summary, ERA correlation is 0.174 — still the weakest metric. Pitcher normalization delivered the single largest improvement (+12.3% ERA correlation), but the remaining gap is partly explained by treating all pitchers' BABIP skill as identical.

## Expected Impact

Moderate improvement to ERA and WHIP correlations. The effect is concentrated on pitchers at the extremes of the ground-ball/fly-ball spectrum (roughly the top and bottom 20% of GB%).

## Pipeline Fit

Modification to the existing `PitcherNormalizationAdjuster` rather than a new stage. The `_compute_expected_babip()` method would accept batted-ball rates and compute a pitcher-specific target instead of returning the flat league mean.

Alternatively, a separate `PitcherBabipSkillAdjuster` stage that runs after normalization and shifts the hit rate toward the batted-ball-implied BABIP.

## Data Requirements

- Batted-ball rates per pitcher-season via pybaseball's `pitching_stats()`:
  - GB% (ground ball rate)
  - FB% (fly ball rate)
  - LD% (line drive rate)
  - IFFB% (infield fly ball rate)
- Historical BABIP by batted-ball profile bucket for calibration

## Key References

- Tango, T. "DIPS Theory and Pitcher BABIP" (The Book Blog)
- Lichtman, M. "BABIP and Batted Ball Types" (The Hardball Times)
- Studeman, D. "Ground Ball Pitchers and BABIP" (The Hardball Times)
- Carleton, R. "How Pitchers Can Influence BABIP" (Baseball Prospectus)

## Implementation Sketch

1. Fetch batted-ball profile data for each pitcher (GB%, FB%, LD%, IFFB%)
2. Compute pitcher-specific expected BABIP using linear model:
   ```
   x_babip = base_babip + gb_coeff * (gb_rate - lg_gb_rate)
                        + iffb_coeff * (iffb_rate - lg_iffb_rate)
                        + ld_coeff * (ld_rate - lg_ld_rate)
   ```
   Approximate coefficients from literature: GB -0.10, IFFB -0.15, LD +0.20
3. Clamp x_babip to reasonable range [.250, .340]
4. Pass pitcher-specific target to normalization instead of flat league mean
5. Pitchers below PA threshold or missing batted-ball data fall back to league mean
6. Store `pitcher_x_babip` and `pitcher_gb_rate` in metadata for diagnostics
