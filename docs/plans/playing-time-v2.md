# Playing Time Model v2 — Roadmap

## Goal

Produce well-calibrated distributional playing-time projections (PA for batters, IP for pitchers) that capture downside risk from injuries, aging, and role changes. The point estimate is secondary — Marcel's weighted-average baseline is hard to beat for top-of-roster players. The value of this model lies in its distributional output: fantasy managers need to understand P10/P25 downside scenarios, not just a point forecast.

## Current State

Phases 1–3 are complete. The model uses:

- **Features:** `age`, `pa_1/2/3` (or `ip_1/2/3`, `g_1/2/3`, `gs_1`), `war_1/2`, plus derived `war_above_2/4`, `war_below_0`, `war_trend`, `pt_trend`, `age_pt_factor`
- **Method:** Ridge regression with season-based CV for alpha selection
- **Aging:** Piecewise-linear curve fit via delta method
- **Distributions:** Residual bucketing into 4 groups (young/old × healthy/injured)
- **Diagnostics:** Holdout R²/RMSE, coefficient report, cumulative feature ablation with per-step alpha selection
- **Integration:** Marcel reads PT projections from DB when available

### Completed Phases

1. **Phase 1 — Feature Engineering.** Added WAR thresholds, IL severity tiers, starter ratio, interaction terms. IL features were later removed (zero predictive power per ablation).
2. **Phase 2 — Ridge Regression.** Replaced OLS with ridge; added `select_alpha()` CV.
3. **Phase 3 — Cross-Validation and Diagnostics.** Added holdout evaluation, coefficient report, and cumulative feature ablation study.

### Key Finding

The point-estimate model (R² ≈ 0.22 batters, 0.26 pitchers) does **not** meaningfully improve on Marcel's native playing-time formula for top-300 players:

| Stat | Pool | PT Model RMSE | Marcel RMSE |
|------|------|--------------|-------------|
| PA | All players | **135.9** | 192.8 |
| PA | Top 300 | 209.2 | **197.5** |
| IP | All players | **46.5** | 46.6 |
| IP | Top 300 | 57.9 | **52.3** |

The model wins for the full pool (separating part-timers from regulars) but loses where it matters for fantasy. Playing time for established players is dominated by stochastic events (injuries, trades, role changes) that no amount of historical feature engineering will predict. Further investment in point-estimate accuracy is not worthwhile.

The remaining value is in **distributional output** — the residual buckets that produce P10–P90 ranges. These are currently coarse (4 buckets) and don't vary with continuous features.

---

## Remaining Phase — Richer Distributional Output

### Goal

Replace the coarse 4-bucket residual system with finer buckets and a continuous variance model, producing well-calibrated prediction intervals that vary meaningfully with player profile.

### 4a. Finer residual buckets

**File:** `models/playing_time/engine.py`

Expand the 2×2 grid (young/old × healthy/injured) to a richer grouping that includes WAR:

```python
def _bucket_key_v2(age, il_days_1, war_1) -> str:
    age_bin = "young" if age < 28 else ("prime" if age < 33 else "old")
    il_bin = "healthy" if il_days_1 == 0 else ("minor_il" if il_days_1 <= 30 else "major_il")
    war_bin = "high_war" if war_1 >= 2 else "low_war"
    return f"{age_bin}_{il_bin}_{war_bin}"
```

This creates up to 18 buckets (3 × 3 × 2). With `min_bucket_size=20`, sparse buckets fall back to the `"all"` bucket. Note: IL data is still gathered for bucketing even though it's excluded from the regression.

### 4b. Variance model

**File:** `models/playing_time/engine.py`

As an alternative to discrete buckets, fit a simple heteroscedastic variance model:

```python
def fit_variance_model(
    rows: list[dict[str, Any]],
    coefficients: PlayingTimeCoefficients,
    target_column: str,
) -> VarianceCoefficients:
    """Fit log(residual²) ~ age + il_days_1 + war_1 to model heteroscedastic variance."""
```

This lets the distribution width vary continuously with features rather than jumping between discrete buckets. Use whichever approach yields better calibrated intervals.

### 4c. Calibration evaluation

Add a calibration check: for each percentile (P10, P25, P50, P75, P90), what fraction of outcomes actually fall below that threshold? Well-calibrated intervals should show ~10% below P10, ~50% below P50, etc.

```python
def calibration_check(
    rows: list[dict[str, Any]],
    coefficients: PlayingTimeCoefficients,
    residual_buckets: ResidualBuckets,
    target_column: str,
) -> dict[str, float]:
    """Return observed coverage fractions for each percentile."""
```

### 4d. Tests

- Finer buckets produce different percentiles for meaningfully different player profiles (young healthy high-WAR vs old injured low-WAR).
- Variance model predicts wider distributions for older, injury-prone players.
- Distribution calibration: ~10% of outcomes fall below P10, ~90% below P90 (within tolerance).
- Bucket fallback works correctly when a bucket has too few samples.
- Both approaches (finer buckets, variance model) round-trip through serialization.

---

## Out of Scope

- **Further point-estimate feature engineering.** Ablation shows diminishing returns; Marcel wins for top-300.
- **Gradient boosting / XGBoost for PT.** The problem is fundamentally noisy, not model-complexity-limited.
- **Team-level roster constraints.** Redistributing PT within teams requires a separate constraint-optimization system.
- **In-season updates.** This model projects pre-season full-season totals.
- **Contract/service time effects.** Hard to quantify systematically.
