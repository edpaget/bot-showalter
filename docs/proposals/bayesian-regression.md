# Bayesian / Hierarchical Regression

## What

Replace the flat regression-to-the-mean formula with a hierarchical Bayesian approach that sets per-player prior strength based on player characteristics (age, experience, Statcast profile), rather than applying the same regression PA to all players uniformly.

## Why

The current regression formula `(n * rate + reg_pa * lg_rate) / (n + reg_pa)` treats 200 PA from a rookie identically to 200 PA from a 10-year veteran. In reality:

- **Veterans** have a strong established talent level — their true ability is well-estimated even from a small recent sample. They should regress less toward the league mean and more toward their own career baseline.
- **Rookies** have high uncertainty — their small MLB sample should be regressed heavily, but toward a prior informed by minor league performance, draft position, or Statcast data rather than the flat league average.
- **Breakout candidates** with Statcast improvements (higher exit velocity, new pitch) should regress less because the underlying skill has genuinely changed.

The flat regression constant is the single biggest modeling simplification in the current system. Per-player priors would improve projections most for players at the extremes: young players, players returning from injury with limited recent data, and players with recent skill changes.

## Expected Impact

Broad improvement across all stats, concentrated on players with limited recent sample sizes. The largest gains would appear in:
- HR, OBP for young hitters (less over-regression)
- ERA for pitchers with changed pitch mixes (less regression toward outdated baseline)
- Overall rank accuracy (Spearman rho) by better separating true talent from noise

## Pipeline Fit

New `BayesianRateComputer` implementing the `RateComputer` protocol as a replacement for `StatSpecificRegressionRateComputer`. It computes per-player regression weights rather than using fixed constants.

## Data Requirements

- Career stats (already available via the data source's multi-year history)
- Player experience / service time (derivable from years of MLB data)
- Optionally: Statcast year-over-year changes for detecting true skill shifts

## Key References

- Tango, T. et al. "The Book" (Chapter 2: Regression to the Mean)
- Albert, J. "Bayesian Estimation of Batting Averages" (Chance, 2006)
- Silver, N. "PECOTA's Bayesian Framework" (Baseball Prospectus methodology)
- Brown, L. "In-Season Prediction of Batting Averages: A Bayesian Approach" (Annals of Applied Statistics, 2008)

## Implementation Sketch

1. Define per-player prior strength as a function of experience:
   ```
   effective_reg_pa(stat) = base_reg_pa(stat) * experience_modifier
   where experience_modifier = max(0.3, 1.0 - 0.1 * years_of_mlb_data)
   ```
   A 7+ year veteran gets 0.3x the regression of a rookie.

2. Define per-player prior center (what to regress toward):
   ```
   prior_center = w_career * career_rate + (1 - w_career) * league_rate
   where w_career = min(1.0, career_pa / (career_pa + career_reg_pa))
   ```
   Veterans regress toward their own career rate; rookies toward the league mean.

3. Optionally detect skill changes using Statcast deltas:
   ```
   if abs(statcast_metric_change) > threshold:
       reduce effective_reg_pa by change_discount (e.g., 0.7x)
   ```
   This lets breakout signals propagate faster.

4. Final projection:
   ```
   rate = (recent_pa * recent_rate + eff_reg_pa * prior_center) / (recent_pa + eff_reg_pa)
   ```

5. Store diagnostics: `effective_regression_pa`, `prior_center`, `experience_modifier`
