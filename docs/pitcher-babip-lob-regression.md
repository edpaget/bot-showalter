# Pitcher BABIP and LOB% Regression

## What

Explicitly regress pitcher BABIP toward the league mean (~.300) and LOB% toward a K-adjusted expectation (~73% baseline, adjusted upward for high-K pitchers) rather than relying solely on the general regression constant for hits and earned runs.

## Why

Pitcher BABIP and LOB% (strand rate) are the two most volatile components of ERA. The current system regresses all pitching stats equally, but BABIP and LOB% have very low year-to-year correlations for pitchers. Pitchers with extreme BABIP or LOB% in their recent history carry that noise forward into projections, producing ERA estimates that are systematically too optimistic or pessimistic.

## Pipeline Fit

New `PitcherNormalizationAdjuster` inserted after `RateComputer` and before `RebaselineAdjuster`. It adjusts the `h` and `er` rates to reflect expected BABIP and LOB% rather than observed values.

## Data Requirements

- Pitcher BABIP and LOB% per season (derivable from existing stats: H, HR, BB, HBP, IP, ER, SO)
- FIP components for K-adjusted LOB% baseline

## Key References

- Tango, T. "DIPS Theory" (original research)
- Lichtman, M. "The Relationship Between BABIP and Strikeout Rate" (The Book Blog)
- Studeman, D. "Pitcher BABIP and Regression to the Mean" (The Hardball Times)

## Implementation Sketch

1. Compute observed BABIP per pitcher: `(H - HR) / (BF - HR - SO - BB - HBP)`
2. Regress BABIP toward .300 (or K-adjusted mean): `expected_babip = w * observed + (1-w) * .300`
3. Recompute expected H from expected BABIP
4. Compute expected LOB% from K-adjusted baseline: `expected_lob = 0.73 + 0.1 * (K% - league_K%)`
5. Recompute expected ER from expected LOB%
6. Replace raw `h` and `er` rates with normalized values
