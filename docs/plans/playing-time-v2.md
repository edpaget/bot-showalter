# Playing Time Model v2 — Roadmap

## Goal

Improve the playing-time model from R² ≈ 0.20 (batters) / 0.25 (pitchers) to a level where it meaningfully improves Marcel projections when used as the PT source. The current model has the right architecture (OLS + aging curves + distributional output) but is feature-light — it cannot distinguish a star player from a bench player, a starter from a reliever, or a minor IL stint from a season-ending injury. This roadmap adds targeted feature engineering, model methodology improvements, and validation infrastructure in prioritized phases.

## Current State

The v1 playing-time model (Phases 1–6 complete) uses:
- **Features:** `age`, `pa_1/2/3`, `war_1/2`, `il_days_1/2/3`, `il_stints_1/2`, plus derived `il_days_3yr`, `il_recurrence`, `pt_trend`, `age_pt_factor`
- **Method:** OLS via `numpy.linalg.lstsq`, separate batter/pitcher models
- **Aging:** Piecewise-linear curve fit via delta method
- **Distributions:** Residual bucketing into 4 groups (young/old × healthy/injured)
- **Integration:** Marcel reads PT projections from DB when available

### Known Weaknesses

1. **No performance thresholding.** A 5-WAR player and a 0-WAR player are treated identically beyond WAR's linear contribution. In reality, above-average players are near-guaranteed full-time roles.
2. **No starter/reliever distinction.** Pitcher IP varies 3× between starters and relievers, but the model has no explicit role feature.
3. **IL treated as binary.** A 10-day IL stint for a bruised finger and a 120-day IL stint for Tommy John surgery get the same `il_recurrence = 1` flag.
4. **No feature interactions.** Age × injury, WAR × trend, and similar combinations are invisible to the linear model.
5. **No regularization.** Correlated features (pa_1, pa_2, pa_3) cause coefficient instability.
6. **No cross-validation.** R² is computed on training data only — likely overstated.
7. **Coarse distributions.** Only 4 residual buckets; distributions don't vary with continuous features like WAR or exact age.

---

## Phases

### Phase 1 — Feature Engineering

Add high-signal features that are derivable from existing data. No model architecture changes — just richer inputs to the same OLS regression.

#### 1a. WAR thresholding

**File:** `features/transforms/playing_time.py`

Add a new transform that produces binary indicator features from WAR:

```python
def make_war_threshold_transform() -> ...:
    """
    Inputs: war_1
    Outputs: war_above_2, war_above_4, war_below_0
    """
```

- `war_above_2`: 1.0 if `war_1 >= 2.0`, else 0.0 — above-average players keep jobs
- `war_above_4`: 1.0 if `war_1 >= 4.0`, else 0.0 — stars get full-time roles
- `war_below_0`: 1.0 if `war_1 < 0.0`, else 0.0 — replacement-level players lose PT

These binary features let the linear model capture the nonlinear relationship between WAR and playing time that a raw linear `war_1` term cannot.

#### 1b. IL severity tiers

**File:** `features/transforms/playing_time.py`

Add a transform that categorizes IL days into severity levels:

```python
def make_il_severity_transform() -> ...:
    """
    Inputs: il_days_1
    Outputs: il_minor, il_moderate, il_severe
    """
```

- `il_minor`: 1.0 if `0 < il_days_1 <= 20`, else 0.0
- `il_moderate`: 1.0 if `20 < il_days_1 <= 60`, else 0.0
- `il_severe`: 1.0 if `il_days_1 > 60`, else 0.0

A 10-day IL stint for a sore wrist has a very different PT impact than a 90-day IL stint for a torn ACL. The current model treats all IL days linearly.

#### 1c. Starter ratio feature (pitchers)

**File:** `models/playing_time/features.py`

Add a derived transform for pitchers:

```python
def make_starter_ratio_transform() -> ...:
    """
    Inputs: gs_1, g_1
    Outputs: starter_ratio
    """
```

- `starter_ratio = gs_1 / g_1` when `g_1 > 0`, else 0.0

This is the single most important missing feature for pitchers. A starter with `gs/g = 1.0` projects for ~180 IP; a reliever with `gs/g = 0.0` projects for ~60 IP. The current model has `gs_1` and `g_1` as separate features but no ratio, so the model must learn the relationship from linear terms alone.

#### 1d. Team PA/IP share

**File:** `features/transforms/playing_time.py`

Add a transform that computes the player's share of their team's total PA or IP from the prior season:

```python
def make_team_pt_share_transform(pt_column: str) -> ...:
    """
    Inputs: {pt_column}_1, team_{pt_column}_1
    Outputs: team_pt_share
    """
```

- `team_pt_share = pa_1 / team_pa_1` (batters) or `ip_1 / team_ip_1` (pitchers)
- When `team_pa_1` is 0 or None, default to 0.0

This captures "established starter vs. platoon vs. bench" without needing to model the full roster. A player who took 12% of their team's PA last year is an everyday player; one who took 3% is a bench bat. This is a cheap, non-circular way to encode team context — it uses observed prior-season data rather than trying to predict future roster competition.

**File:** `models/playing_time/features.py`

Add `team_pa_1` and `team_ip_1` as base features. These require a new feature source that aggregates team-level PA/IP from batting/pitching stats joined through roster stints. The SQL generation will need a subquery:

```sql
LEFT JOIN (
    SELECT rs.player_id, rs.season,
           SUM(bs.pa) AS team_pa
    FROM roster_stint rs
    JOIN batting_stats bs ON bs.team_id = rs.team_id AND bs.season = rs.season
    GROUP BY rs.player_id, rs.season
) team1 ON team1.player_id = spine.player_id AND team1.season = spine.season - 1
```

If this join proves too complex for the feature DSL, compute `team_pt_share` directly in a derived transform that queries team totals at transform time, or pre-compute team totals as a separate feature source.

#### 1e. Interaction terms

**File:** `features/transforms/playing_time.py`

Add interaction features that capture important conditional relationships:

```python
def make_pt_interaction_transform() -> ...:
    """
    Inputs: war_1, pt_trend, age, il_recurrence
    Outputs: war_trend, age_il_interact
    """
```

- `war_trend = war_1 × pt_trend` — high-WAR players with increasing PT are likely to keep it; low-WAR players with decreasing PT are likely to lose more
- `age_il_interact = max(0, age - 30) × il_recurrence` — recurring injuries are more concerning for older players

#### 1f. Wire features into feature builders

**File:** `models/playing_time/features.py`

Register the new transforms in `build_batting_pt_derived_transforms()` and `build_pitching_pt_derived_transforms()`. Update `build_batting_pt_training_features()` and `build_pitching_pt_training_features()` accordingly.

#### 1g. Tests

- Each transform produces correct outputs for known inputs.
- Edge cases: `war_1 = None`, `g_1 = 0`, `il_days_1 = 0`, negative WAR, `team_pa_1 = 0`.
- `team_pt_share` computes correct ratio and defaults to 0.0 when team total is missing.
- Feature column lists include the new features.
- End-to-end: train → predict cycle with new features produces valid output.

---

### Phase 2 — Ridge Regression

Replace OLS with ridge regression to stabilize coefficients and prevent overfitting with correlated features.

#### 2a. Add regularization to engine

**File:** `models/playing_time/engine.py`

Modify `fit_playing_time()` to support ridge regression:

```python
def fit_playing_time(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    target_column: str,
    player_type: str,
    alpha: float = 1.0,  # regularization strength; 0 = OLS
) -> PlayingTimeCoefficients:
```

Implementation: change `np.linalg.lstsq(X, y)` to the closed-form ridge solution `β = (X'X + αI)⁻¹ X'y`. The intercept column is not regularized (set diagonal entry to 0 for it).

Add `alpha` to `PlayingTimeCoefficients` for reproducibility.

#### 2b. Alpha selection via cross-validation

**File:** `models/playing_time/engine.py`

```python
def select_alpha(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    target_column: str,
    player_type: str,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    n_folds: int = 5,
) -> float:
```

Simple k-fold cross-validation: split rows into k folds by season (not random — avoids leakage), fit ridge on k-1 folds, evaluate on the held-out fold, pick the alpha with lowest mean RMSE.

#### 2c. Wire into model training

**File:** `models/playing_time/model.py`

In `train()`, call `select_alpha()` before `fit_playing_time()`. Report chosen alpha in `TrainResult.metrics`. Allow `alpha` override via `config.model_params["alpha"]`.

#### 2d. Tests

- Ridge with `alpha=0` produces same results as current OLS.
- Ridge with large `alpha` shrinks coefficients toward zero.
- `select_alpha()` returns a valid alpha from the candidate list.
- Cross-validation splits by season, not randomly.
- Serialization round-trips the alpha value.

---

### Phase 3 — Cross-Validation and Diagnostics

Add proper evaluation infrastructure so we can measure improvement reliably.

#### 3a. Holdout evaluation in training

**File:** `models/playing_time/model.py`

When `config.seasons` contains ≥4 seasons, automatically hold out the last season for evaluation. Report both training R² and holdout R²/RMSE in `TrainResult.metrics`:

```python
metrics = {
    "r_squared_batter_train": ...,
    "r_squared_batter_holdout": ...,
    "rmse_batter_holdout": ...,
    "r_squared_pitcher_train": ...,
    "r_squared_pitcher_holdout": ...,
    "rmse_pitcher_holdout": ...,
}
```

#### 3b. Coefficient report

**File:** `models/playing_time/engine.py`

Add a function that returns a diagnostic summary:

```python
def coefficient_summary(coefficients: PlayingTimeCoefficients) -> list[dict[str, Any]]:
    """Return feature name, coefficient, and standardized coefficient for inspection."""
```

Print this during training so we can verify coefficients have sensible signs and magnitudes (e.g., `il_days_1` should be negative, `pa_1` should be positive, `war_above_4` should be positive).

#### 3c. Feature ablation

**File:** `models/playing_time/engine.py`

```python
def ablation_study(
    rows: list[dict[str, Any]],
    feature_groups: dict[str, list[str]],
    base_features: list[str],
    target_column: str,
    player_type: str,
    holdout_rows: list[dict[str, Any]],
    alpha: float = 1.0,
) -> list[dict[str, Any]]:
    """Train with cumulative feature groups and report holdout R²/RMSE for each."""
```

Train the model with incremental feature groups and measure holdout performance for each:

- Baseline (v1 features only)
- \+ WAR thresholds
- \+ IL severity
- \+ starter ratio (pitchers) / team PT share (batters)
- \+ interaction terms
- All v2 features

Returns a list of `{"group": str, "features": list[str], "r_squared": float, "rmse": float}` dicts. This directly answers "which features are worth keeping?" early in development rather than waiting until Phase 6.

#### 3d. Wire into CLI

Print coefficient summary, holdout metrics, and ablation results during `fbm train playing_time`. No new CLI commands needed — just richer output. Ablation runs automatically when `config.model_params.get("ablation", False)` is set.

#### 3e. Tests

- Holdout R² is reported when ≥4 seasons provided.
- Holdout R² is not reported when <4 seasons provided.
- Coefficient summary includes all features with correct names.
- Ablation study returns one entry per feature group with valid R² values.
- Ablation with a single group matches standalone training result.

---

### Phase 4 — Improved Aging Curve

Replace the piecewise-linear aging curve with a quadratic spline that better captures the accelerating decline after age 32.

#### 4a. Quadratic aging curve

**File:** `models/playing_time/aging.py`

Add a new aging curve type:

```python
@dataclass(frozen=True)
class QuadraticAgingCurve:
    peak_age: float
    improvement_rate: float    # linear improvement per year before peak
    decline_linear: float      # linear decline per year after peak
    decline_quadratic: float   # quadratic acceleration per year² after peak
    player_type: str
```

Factor formula:
```
if age < peak: factor = 1.0 + (peak - age) × improvement_rate
if age > peak: factor = 1.0 - years_past × decline_linear - years_past² × decline_quadratic
```

This captures the observation that decline accelerates — a 33-year-old loses fewer games than a 38-year-old per year of aging.

#### 4b. Fit quadratic curve

**File:** `models/playing_time/aging.py`

```python
def fit_quadratic_aging_curve(
    rows: list[dict[str, Any]],
    player_type: str,
    current_column: str,
    prior_column: str,
    ...
) -> QuadraticAgingCurve:
```

Use the same delta method for peak age detection. For the decline side, fit a 2nd-degree polynomial to the age-specific mean deltas (ages > peak) using least squares.

#### 4c. Backward compatibility

Keep the existing `AgingCurve` dataclass and `compute_age_pt_factor()`. Add a new `compute_quadratic_age_pt_factor()`. The model selects which to use based on `config.model_params["aging_curve"]` (default: `"quadratic"`).

#### 4d. Tests

- Quadratic curve factor is ≥ piecewise-linear factor at all ages (the quadratic captures faster decline at extreme ages).
- Factor at peak age is 1.0.
- Factor is monotonically decreasing after peak.
- Falls back to piecewise-linear defaults when insufficient data.

---

### Phase 5 — Richer Distributional Output

Expand residual buckets and add continuous variance modeling.

#### 5a. Finer residual buckets

**File:** `models/playing_time/engine.py`

Expand the 2×2 grid (young/old × healthy/injured) to a richer grouping:

```python
def _bucket_key_v2(age, il_days_1, war_1) -> str:
    age_bin = "young" if age < 28 else ("prime" if age < 33 else "old")
    il_bin = "healthy" if il_days_1 == 0 else ("minor_il" if il_days_1 <= 30 else "major_il")
    war_bin = "high_war" if war_1 >= 2 else "low_war"
    return f"{age_bin}_{il_bin}_{war_bin}"
```

This creates up to 18 buckets (3 × 3 × 2). With `min_bucket_size=20`, sparse buckets fall back to the `"all"` bucket.

#### 5b. Variance model

**File:** `models/playing_time/engine.py`

As an alternative to discrete buckets, fit a simple variance model:

```python
def fit_variance_model(
    rows: list[dict[str, Any]],
    coefficients: PlayingTimeCoefficients,
    target_column: str,
) -> VarianceCoefficients:
    """Fit log(residual²) ~ age + il_days_1 + war_1 to model heteroscedastic variance."""
```

This lets the distribution width vary continuously with features rather than jumping between discrete buckets. Use whichever approach yields better calibrated intervals.

#### 5c. Tests

- Finer buckets produce different percentiles for meaningfully different player profiles.
- Variance model predicts wider distributions for older, injury-prone players.
- Distribution calibration: ~10% of outcomes fall below p10, ~90% below p90.

---

### Phase 6 — Evaluation and Tuning

End-to-end evaluation of the improved model against Marcel's internal formula and the v1 model.

#### 6a. A/B comparison script

Write a comparison workflow (or document the CLI commands) that:

1. Trains v2 playing-time model
2. Runs Marcel with v2 PT
3. Runs Marcel with internal formula (baseline)
4. Compares RMSE on counting stats (HR, R, RBI, SB, SO, W, SV, IP) for top-300 players

#### 6b. Hyperparameter tuning

Tune `alpha` (ridge penalty), `min_pa`/`min_ip` (spine filter thresholds), and `lags` (number of historical seasons). Use Phase 3's ablation results to decide final feature set. Document final chosen values.

---

## Phase Order

```
Phase 1 (feature engineering)
  ↓
Phase 2 (ridge regression)    Phase 3 (cross-validation)
  [independent of each other, both depend on Phase 1]
  ↓                            ↓
Phase 4 (aging curve)         Phase 5 (distributions)
  [independent, depend on Phases 2-3]
  ↓                            ↓
Phase 6 (evaluation + tuning)
  [depends on all prior phases]
```

Phases 1–3 are expected to deliver the bulk of the improvement. Phases 4–5 refine accuracy and output quality. Phase 6 validates the full pipeline.

---

## Files to Modify/Create

| Phase | File | Change |
|-------|------|--------|
| 1 | `features/transforms/playing_time.py` | Add WAR threshold, IL severity, interaction transforms |
| 1 | `models/playing_time/features.py` | Register new transforms, update feature column lists |
| 2 | `models/playing_time/engine.py` | Ridge regression, alpha selection |
| 2 | `models/playing_time/model.py` | Wire alpha selection into train() |
| 3 | `models/playing_time/model.py` | Holdout evaluation, coefficient reporting, ablation wiring |
| 3 | `models/playing_time/engine.py` | Coefficient summary, feature ablation study |
| 4 | `models/playing_time/aging.py` | Quadratic aging curve |
| 5 | `models/playing_time/engine.py` | Finer buckets, variance model |
| 6 | — | A/B comparison, hyperparameter tuning |

## Out of Scope

- **Gradient boosting / XGBoost.** The statcast model uses XGBoost, but playing time has fewer features and benefits more from interpretability. If ridge + feature engineering doesn't reach acceptable R², we can revisit.
- **Team-level roster constraints.** Redistributing PT within teams requires a separate constraint-optimization system.
- **In-season updates.** This model projects pre-season full-season totals.
- **Contract/service time effects.** Hard to quantify systematically.
