# Pitching Projection Improvements

## Overview

Close the accuracy gap between `marcel_gb` and commercial projection systems (Steamer, ZiPS) for pitching. The 2021-2024 backtesting evaluation reveals that `marcel_gb` pitching trails Steamer significantly on ERA (0.245 vs 0.439 correlation), WHIP (0.329 vs 0.434), and W (0.481 vs 0.598), while batting performance is competitive. The improvements are staged from config-only changes through new data source integrations.

---

## Problem Statement

### 2021-2024 Evaluation Baseline (avg across 4 years)

**Batting** — marcel_gb is competitive:

| Stat | marcel_gb | Steamer | ZiPS | Gap to Best |
|------|-----------|---------|------|-------------|
| HR   | 0.678     | 0.677   | 0.682 | -0.004 |
| R    | 0.577     | 0.630   | 0.612 | -0.053 |
| RBI  | 0.583     | 0.633   | 0.607 | -0.050 |
| SB   | 0.736     | 0.697   | 0.746 | -0.010 |
| OBP  | 0.535     | 0.574   | 0.558 | -0.039 |
| Rank rho | 0.603 | 0.618   | 0.612 | -0.015 |
| **Top-20** | **0.525** | 0.500 | 0.475 | **+0.025** |

**Pitching** — marcel_gb trails significantly:

| Stat | marcel_gb | Steamer | ZiPS | Gap to Best |
|------|-----------|---------|------|-------------|
| W    | 0.481     | 0.598   | 0.490 | -0.117 |
| K    | 0.699     | 0.728   | 0.680 | -0.029 |
| ERA  | 0.245     | 0.439   | 0.321 | **-0.194** |
| WHIP | 0.329     | 0.434   | 0.369 | **-0.105** |
| NSVH | 0.724     | 0.782   | 0.717 | -0.058 |
| Rank rho | 0.412 | 0.514   | 0.431 | -0.102 |
| Top-20 | 0.388  | 0.388   | 0.338 | +0.000 |

### Root Causes

The current pipeline has uneven coverage across pitching stat groups:

| Stat Group | Stages That Touch It | Depth |
|---|---|---|
| SO, BB | Regression + Statcast + GB residual + Aging | Deep |
| H, ER | Regression + BABIP normalization + 25% xERA blend | Moderate |
| W, SV, HLD | Regression + Aging only | Shallow |

ERA and WHIP are derived from H, ER, BB, and IP. The two largest contributors to error are:
1. **H rates**: Pitcher xBA-against blending is disabled (`h_blend_weight=0.0`), so hits allowed depend entirely on BABIP regression to league mean — ignoring contact quality differences between pitchers.
2. **ER rates**: Only 25% xERA blending, and the LOB% model is a crude linear approximation from K% that ignores groundball tendencies.

W and NSVH have no context modeling at all — they're purely extrapolated from historical rates with aging curves, with no awareness of team quality or bullpen role.

---

## Current Pipeline (Pitching Path)

```
3-year stats
  → StatSpecificRegressionRateComputer (per-out regression)
  → ParkFactorAdjuster (universal)
  → PitcherNormalizationAdjuster (BABIP regression + LOB% from K%)
  → PitcherStatcastAdjuster (25% ER from xERA, 0% H from xBA)
  → [StatcastRateAdjuster — SKIPPED for pitchers]
  → [BatterBabipAdjuster — SKIPPED for pitchers]
  → GBResidualAdjuster (XGBoost SO/BB only, conservative mode)
  → RebaselineAdjuster (universal)
  → ComponentAgingAdjuster (stat-specific decay curves)
  → MarcelPlayingTime (IP → outs)
  → StandardFinalizer (rate → counting stats, ERA/WHIP/G)
```

### Key Configuration Defaults

```python
# pitcher_statcast_adjuster.py
PitcherStatcastConfig(
    h_blend_weight=0.0,       # xBA-against disabled
    er_blend_weight=0.25,     # 25% xERA
    min_pa_for_blend=200,
)

# presets.py — marcel_gb
GBResidualConfig(
    pitcher_allowed_stats=("so", "bb"),  # Only K/BB get ML corrections
    pitcher_min_pa=100,
)

# pitcher_normalization.py
PitcherNormalizationConfig(
    league_babip=0.300,
    babip_regression_weight=1.0,  # Full regression to mean
    lob_baseline=0.73,
    lob_k_sensitivity=0.1,       # Linear K% → LOB%
)
```

---

## Improvement Phases

### Phase 1: Config-Only Changes (No New Code)

Tune existing stage parameters to unlock signal that's currently suppressed.

#### 1a. Expand GB Residual to H and ER

The XGBoost model already has pitcher Statcast features (exit velo, barrel rate, GB%) that are directly predictive of hits and earned runs. Currently only SO and BB get ML corrections.

**Change** in `presets.py`:
```python
# Before
GBResidualConfig(
    pitcher_allowed_stats=("so", "bb"),
)

# After
GBResidualConfig(
    pitcher_allowed_stats=("so", "bb", "h", "er"),
)
```

**Expected impact**: Direct improvement to ERA and WHIP correlation. The residual model was trained on all stats — we're just un-gating the predictions.

**Risk**: The conservative mode was chosen to avoid introducing errors. Need to evaluate whether H/ER residuals help or hurt across all 4 backtest years before committing.

**Evaluation**: Run `evaluate 2024 --engine marcel_gb --years 2021,2022,2023,2024` before and after, compare pitching ERA/WHIP correlation.

#### 1b. Enable xBA-Against H Blending

Pitcher xBA-against from Statcast captures contact quality differences that BABIP regression to league mean erases.

**Change** in `RegressionConfig` or `PitcherStatcastConfig`:
```python
# Before
PitcherStatcastConfig(h_blend_weight=0.0, er_blend_weight=0.25)

# After — start conservative
PitcherStatcastConfig(h_blend_weight=0.15, er_blend_weight=0.25)
```

**Expected impact**: WHIP improvement. Pitchers who consistently suppress hard contact (low xBA-against) currently get no credit for it in H projections.

**Risk**: xBA-against can be noisy for low-IP pitchers. The `min_pa_for_blend=200` threshold already gates this, but we should grid-search `h_blend_weight` in [0.05, 0.10, 0.15, 0.20] to find the sweet spot.

#### 1c. Increase xERA Blend Weight

With BABIP normalization already stabilizing the base ER rate, a higher xERA blend is less likely to overfit.

**Grid search**: `er_blend_weight` in [0.25, 0.30, 0.35, 0.40].

**Evaluation**: Same backtest framework. Track both ERA correlation and ERA RMSE — correlation can improve even if RMSE worsens (systematic bias vs noise).

---

### Phase 2: LOB% Model Enhancement (Small Code Change)

#### Problem

The current LOB% estimate is a linear function of K%:

```python
# pitcher_normalization.py
lob = baseline + k_sensitivity * (k_pct - league_k_pct)
```

This misses that LOB% is also heavily influenced by:
- **Groundball rate**: GB pitchers strand more runners (grounders have lower wOBA than fly balls)
- **HR/FB rate**: Fly ball pitchers allow more damage per baserunner

Both signals are already available — the GB residual stage loads batted ball profiles.

#### Proposed Change

Add GB% as a term in the LOB% formula in `PitcherNormalizationAdjuster`:

```python
# Enhanced LOB% model
def _estimate_lob_pct(
    self,
    k_pct: float,
    gb_pct: float | None,
    league_k_pct: float,
    league_gb_pct: float,
    config: PitcherNormalizationConfig,
) -> float:
    lob = config.lob_baseline
    lob += config.lob_k_sensitivity * (k_pct - league_k_pct)
    if gb_pct is not None:
        lob += config.lob_gb_sensitivity * (gb_pct - league_gb_pct)
    return max(0.65, min(0.82, lob))
```

**New config fields**:
```python
@dataclass
class PitcherNormalizationConfig:
    # ... existing fields ...
    lob_gb_sensitivity: float = 0.15     # LOB% increase per unit GB% above average
    league_gb_pct: float = 0.43          # ~MLB average GB%
```

**Data flow**: GB% needs to be available in metadata by the time `PitcherNormalizationAdjuster` runs. Options:
1. Compute from batted ball data loaded in the rate computer stage
2. Load from pitcher Statcast data (already fetched by `PitcherStatcastAdjuster`, but that runs *after* normalization)

Option 2 requires reordering stages or pre-loading GB%. The simpler path is to have the rate computer store a `gb_pct` metadata field from the historical stats, since pybaseball `pitching_stats()` includes GB%.

**Expected impact**: ERA improvement — LOB% is the strongest predictor of the gap between FIP and ERA.

---

### Phase 3: Team Context for Wins (New Data Source)

#### Problem

W correlation is 0.481 vs Steamer's 0.598 — the largest single-stat gap. Wins depend heavily on:
- Team quality (run support for starters)
- Bullpen quality (protecting leads)
- Usage patterns (tandem starters, opener strategy)

The current pipeline has zero team context.

#### Approach: Preseason Win Total Scaling

Use Vegas preseason win totals (or prior year W%) to scale W projections:

```python
class TeamWinContextAdjuster:
    """Scales W projections based on team win environment."""

    def __init__(self, team_win_shares: dict[str, float]):
        """
        Args:
            team_win_shares: Map of team → expected win fraction (0.0 to 1.0).
                             E.g. {"NYY": 0.580, "OAK": 0.420}
        """
        self._win_shares = team_win_shares
        self._league_avg = sum(team_win_shares.values()) / len(team_win_shares)

    def adjust_pitching(
        self,
        rates: dict[str, float],
        metadata: dict[str, object],
    ) -> dict[str, float]:
        team = metadata.get("team", "")
        team_wpct = self._win_shares.get(str(team), self._league_avg)
        scale = team_wpct / self._league_avg

        adjusted = dict(rates)
        if "w" in adjusted:
            adjusted["w"] = adjusted["w"] * scale
        return adjusted
```

#### Data Source Options

| Source | Availability | Freshness | Effort |
|--------|-------------|-----------|--------|
| Prior year W-L record | pybaseball `team_batting()` | Free, instant | Low |
| FanGraphs preseason W projections | FanGraphs API | Requires scraping | Medium |
| Vegas futures (best signal) | Various odds APIs | Most accurate but costs $ | High |

**Recommended start**: Prior year W% with regression toward .500 (e.g., `adjusted_wpct = 0.3 * 0.500 + 0.7 * prior_wpct`). This is free, requires no new dependencies, and captures 70%+ of team quality signal.

**Expected impact**: W correlation improvement of 0.05-0.08. Won't close the full gap to Steamer (0.598) since they likely use projected lineups and depth charts, but should narrow it.

---

### Phase 4: Role-Based NSVH Modeling (New Logic)

#### Problem

NSVH (saves + holds) depends almost entirely on whether a pitcher *gets* save/hold opportunities. A dominant middle reliever in a team with an established closer gets few saves regardless of skill.

Current behavior: Raw SV/HLD rates are extrapolated with aging curves. A pitcher who had 30 saves last year is projected for ~30 again, even if they changed teams or lost the closer role.

#### Approach: Role Classification + Opportunity Caps

```python
class RelieverRole(Enum):
    CLOSER = "closer"
    SETUP = "setup"
    MIDDLE = "middle"
    LONG = "long"
    STARTER = "starter"

def classify_role(metadata: dict[str, object]) -> RelieverRole:
    """Classify pitcher role from historical stats."""
    is_starter = metadata.get("is_starter", False)
    if is_starter:
        return RelieverRole.STARTER

    games = float(metadata.get("games_per_year", [0])[-1])  # Most recent year
    ip_per_game = float(metadata.get("ip_per_year", [0])[-1]) / max(games, 1)

    # Use SV rate to distinguish closer from setup
    sv_rate = metadata.get("target_rates", {}).get("sv", 0)
    hld_rate = metadata.get("target_rates", {}).get("hld", 0)

    if sv_rate > hld_rate and sv_rate > 0.01:
        return RelieverRole.CLOSER
    if hld_rate > 0.005:
        return RelieverRole.SETUP
    if ip_per_game > 1.5:
        return RelieverRole.LONG
    return RelieverRole.MIDDLE

# Opportunity caps by role (per 70 appearances)
ROLE_SV_CAPS = {
    RelieverRole.CLOSER: 35,
    RelieverRole.SETUP: 3,
    RelieverRole.MIDDLE: 1,
    RelieverRole.LONG: 1,
    RelieverRole.STARTER: 0,
}
ROLE_HLD_CAPS = {
    RelieverRole.CLOSER: 5,
    RelieverRole.SETUP: 25,
    RelieverRole.MIDDLE: 15,
    RelieverRole.LONG: 5,
    RelieverRole.STARTER: 0,
}
```

This is a simplification — real role changes (new closer after trade deadline) are hard to predict. But capping projected saves for non-closers and holds for non-setup men would reduce the noise that comes from extrapolating a small-sample fluke season.

**Expected impact**: NSVH correlation improvement of 0.02-0.04. The bigger gains require roster/depth chart data which is out of scope for now.

---

### Phase 5: Reliever vs Starter Differentiation (Refinement)

#### Problem

The pipeline uses a single `is_starter` boolean (GS/G >= 0.333) but doesn't differentiate within relievers. Modern bullpen usage creates distinct archetypes with different projection profiles:

- **High-leverage setup**: 65+ appearances, 1.0 IP/G, elite K rates
- **Long relief/swingman**: 40-50 appearances, 2.0+ IP/G, moderate rates
- **Opener/bulk**: Hybrid role, hard to project

#### Proposed Enhancement

Replace the binary `is_starter` with a continuous `starter_fraction` and role-specific regression:

```python
# In StatSpecificRegressionRateComputer
starter_fraction = gs / max(g, 1)
if starter_fraction >= 0.8:
    regression_outs = config.starter_regression_outs  # Current defaults
elif starter_fraction <= 0.2:
    regression_outs = config.reliever_regression_outs  # Tighter regression
else:
    # Blend for swingmen
    regression_outs = (
        starter_fraction * config.starter_regression_outs
        + (1 - starter_fraction) * config.reliever_regression_outs
    )
```

Relievers should regress more aggressively on ERA-family stats (small samples) but less on K rate (true skill shows faster in short stints).

---

## Implementation Plan

### Phase 1: Config Tuning — ~1 session

1. Create a parameter sweep script that runs backtesting across config combinations
2. Test `pitcher_allowed_stats` expansion: `("so", "bb")` vs `("so", "bb", "h", "er")` vs `("so", "bb", "h", "er", "hr")`
3. Grid search `h_blend_weight` in [0.0, 0.05, 0.10, 0.15, 0.20]
4. Grid search `er_blend_weight` in [0.25, 0.30, 0.35, 0.40]
5. Evaluate each combination across 2021-2024, select best config
6. Update `marcel_gb_pipeline()` defaults

### Phase 2: LOB% Model — ~1 session

1. Add `gb_pct` to pitcher metadata in rate computer (from pybaseball GB% field)
2. Write failing tests for enhanced LOB% calculation
3. Add `lob_gb_sensitivity` and `league_gb_pct` to `PitcherNormalizationConfig`
4. Implement GB%-aware LOB% in `PitcherNormalizationAdjuster`
5. Re-run backtesting, compare ERA correlation

### Phase 3: Team Win Context — ~1-2 sessions

1. Write `TeamWinContextAdjuster` as a new pipeline stage
2. Data source: prior year team W% from pybaseball with regression to .500
3. Wire into pipeline builder: `.with_team_context()`
4. Add to `marcel_gb` pipeline
5. Backtest W correlation improvement

### Phase 4: Role-Based NSVH — ~1-2 sessions

1. Implement `RelieverRoleClassifier` using GS/G ratio and SV/HLD rates
2. Add role-based opportunity caps to finalizer
3. Write tests for role classification edge cases
4. Backtest NSVH correlation improvement

### Phase 5: Reliever Regression Refinement — ~1 session

1. Add `reliever_regression_outs` to `RegressionConfig`
2. Implement blended regression based on starter fraction
3. Backtest impact on reliever ERA/WHIP accuracy

---

## Success Criteria

Target metrics (2021-2024 avg, correlation):

| Stat | Current | Target | Steamer Reference |
|------|---------|--------|-------------------|
| ERA  | 0.245   | 0.350+ | 0.439 |
| WHIP | 0.329   | 0.400+ | 0.434 |
| W    | 0.481   | 0.540+ | 0.598 |
| K    | 0.699   | 0.720+ | 0.728 |
| NSVH | 0.724   | 0.750+ | 0.782 |
| Rank rho | 0.412 | 0.470+ | 0.514 |

Closing 50% of the gap to Steamer across all pitching stats would be a strong outcome. Full parity is unlikely without proprietary data (scouting reports, projected lineups, depth charts).

---

## Open Questions

1. **GB residual expansion risk**: Does adding H/ER to allowed stats help consistently across all 4 backtest years, or does it help some years and hurt others? Need per-year breakdown before committing.

2. **Stage ordering for GB%**: Should we reorder `PitcherNormalizationAdjuster` after `PitcherStatcastAdjuster` to get Statcast-derived GB%, or compute GB% from historical stats in the rate computer? The latter is simpler but less accurate for pitchers who changed approach.

3. **Reliever sample size**: Relievers have far fewer IP per season than starters. Should regression constants be role-aware (more regression for relievers) even before Phase 5?

4. **Role volatility**: Closer roles change mid-season. How much weight should we give to most-recent-year role vs multi-year pattern? A pitcher who closed for half a season and set up for the other half is hard to classify.

5. **Wins model ceiling**: Even with team context, W is inherently noisy (run timing, bullpen performance). Is 0.55+ correlation realistic, or should we focus effort on stats with higher signal (ERA, K, WHIP)?
