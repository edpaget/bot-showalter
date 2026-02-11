# Prediction Intervals & Uncertainty — Implementation Plan

## Overview

Add Monte Carlo simulation-based uncertainty quantification to the projection pipeline. Uncertainty is carried as variance metadata through pipeline stages, then materialized into N simulated seasons at finalization time. Existing pipeline behavior is fully preserved — all changes are additive.

**User decisions:**
- **Representation:** MC samples (N simulated seasons per player)
- **Type strategy:** Wrapper dataclass around existing frozen projections
- **Scope:** Pipeline only; downstream (valuation/draft) deferred with proposal sketch
- **Propagation:** Monte Carlo simulation in the finalizer

**Key dependency:** `scipy` — add as an explicit dependency (`scipy>=1.12.0`). Already installed transitively via scikit-learn. Used for `scipy.stats.beta` (rate sampling), `scipy.stats.lognorm` (opportunity sampling), and `scipy.stats.qmc.LatinHypercube` (variance reduction).

---

## Phase 1: Core Types & MC Simulation Engine

**Goal:** Define uncertainty types and sampling primitives, independent of any pipeline stage.

### New file: `src/fantasy_baseball_manager/pipeline/uncertainty.py`

```python
@dataclass(frozen=True)
class UncertaintyConfig:
    n_samples: int = 1000
    random_seed: int | None = 42
    use_lhs: bool = True       # use Latin Hypercube Sampling via scipy.stats.qmc

@dataclass(frozen=True)
class BattingProjectionWithUncertainty:
    projection: BattingProjection          # original point estimate
    samples: np.ndarray                    # shape (n_samples, n_stats)
    stat_names: tuple[str, ...]
    def percentile(self, stat: str, q: float) -> float: ...
    def interval(self, stat: str, coverage: float = 0.90) -> tuple[float, float]: ...
    def std(self, stat: str) -> float: ...

@dataclass(frozen=True)
class PitchingProjectionWithUncertainty:
    # same shape as batting wrapper
```

**Sampling functions** (backed by `scipy.stats`):

| Function | Distribution | Purpose |
|----------|-------------|---------|
| `draw_rate_samples(means, variances, n, rng)` | `scipy.stats.beta` — fit alpha/beta from mean+variance via method of moments; naturally bounded [0,1], no clipping needed | Draw per-stat rate samples |
| `draw_opportunity_samples(mean, var, n, rng)` | `scipy.stats.lognorm` — parameterized from mean+variance; non-negative, right-skewed | Draw PA/IP samples |

Both functions accept an optional `scipy.stats.qmc.LatinHypercube` engine for stratified sampling (better convergence — ~500 LHS samples ≈ 2000 naive MC samples). When provided, uniform LHS draws are mapped through the distribution's `.ppf()` (inverse CDF). When not provided, falls back to `.rvs()` with a seeded `numpy.random.Generator`.

**Beta parameterization** from mean (μ) and variance (σ²):
```
alpha = μ * (μ * (1 - μ) / σ² - 1)
beta  = (1 - μ) * (μ * (1 - μ) / σ² - 1)
```
Guard: if σ² ≥ μ(1-μ) (variance exceeds Bernoulli limit), clamp to a wide Beta(1, 1) = Uniform[0,1].

**Log-normal parameterization** from mean (μ) and variance (σ²):
```
sigma_ln = sqrt(log(1 + σ² / μ²))
mu_ln    = log(μ) - sigma_ln² / 2
```

**Propagation helpers** (used by adjusters in Phase 3):

| Function | Formula | Used by |
|----------|---------|---------|
| `propagate_multiplicative(vars, multipliers)` | `var_new = var * mult^2` | Aging, park factors, rebaseline |
| `propagate_blend(var_a, var_b, weight_b, stats)` | `var_new = (1-w)^2 var_a + w^2 var_b` | Statcast, MTL, BABIP, contextual blends |
| `propagate_additive(var_base, var_correction, stats)` | `var_new = var_base + var_correction` | GB residual, skill change |

### Variance Estimator Protocols

Every place where variance is *originated* (not propagated through a transform) uses one of these protocols. This makes the ad hoc assumptions swappable for experimentation.

```python
class RateVarianceEstimator(Protocol):
    """Estimates per-stat variance at the rate computation stage (Marcel, etc.)."""
    def estimate_rate_variances(
        self,
        rates: dict[str, float],
        opportunities_by_year: Sequence[float],
        weights: Sequence[float],
        regression_pa: float,
    ) -> dict[str, float]: ...

class SecondaryVarianceEstimator(Protocol):
    """Estimates variance of a secondary source's rates for blend adjusters."""
    def estimate_secondary_variances(
        self,
        rates: dict[str, float],
        sample_size: float,
    ) -> dict[str, float]: ...

class OpportunityVarianceEstimator(Protocol):
    """Estimates variance of projected playing time (PA or outs)."""
    def estimate_opportunity_variance(
        self,
        projected_opportunities: float,
        historical_opportunities: Sequence[float],
        injury_factor: float,
    ) -> float: ...
```

**Default implementations** (the ad hoc estimates from the original plan):

| Protocol | Default Implementation | Formula |
|----------|----------------------|---------|
| `RateVarianceEstimator` | `BetaBinomialRateVariance` | `rate * (1-rate) / (eff_n + 1)` |
| `SecondaryVarianceEstimator` | `BetaBinomialSecondaryVariance` | `rate * (1-rate) / (sample_size + 1)` |
| `OpportunityVarianceEstimator` | `MixtureModelOpportunityVariance` | base CV + injury mixture (Phase 4 formula) |

**Alternative implementations** (can be swapped in for experimentation):

| Protocol | Alternative | Basis |
|----------|------------|-------|
| `RateVarianceEstimator` | `EmpiricalResidualRateVariance` | Backtest Marcel against actuals, compute per-stat residual variance |
| `SecondaryVarianceEstimator` | `MTLLearnedSecondaryVariance` | Uses `MultiTaskNet.get_learned_variances()` directly (Phase 6a) |
| `SecondaryVarianceEstimator` | `ConformalSecondaryVariance` | Uses conformal half-width from held-out calibration (Phase 6b) |
| `OpportunityVarianceEstimator` | `HistoricalVolatilityOpportunityVariance` | Fit from historical `actual_PA - projected_PA` errors per PA tier and age |

**Bundle for pipeline-level wiring:**

```python
@dataclass(frozen=True)
class UncertaintyEstimators:
    rate: RateVarianceEstimator
    secondary: SecondaryVarianceEstimator
    opportunity: OpportunityVarianceEstimator

    @staticmethod
    def defaults() -> UncertaintyEstimators:
        return UncertaintyEstimators(
            rate=BetaBinomialRateVariance(),
            secondary=BetaBinomialSecondaryVariance(),
            opportunity=MixtureModelOpportunityVariance(),
        )
```

This is wired through the pipeline builder. Stages that originate variance accept the relevant estimator via constructor DI, matching the existing pattern (see `StatcastRateAdjuster.__init__`).

### Modify: `src/fantasy_baseball_manager/pipeline/types.py`

Add two optional keys to `PlayerMetadata`:

```python
rate_variances: dict[str, float]       # per-stat posterior variance of the rate
opportunities_variance: float           # variance of projected PA or outs
```

### Tests: `tests/pipeline/test_uncertainty.py`

**Sampling & wrapper types:**
- `TestDrawRateSamples`: correct shape, zero-var → constant, Beta bounds respected (no clipping), mean/var recovery, seeded reproducibility, extreme rates (near 0 or 1) produce valid Beta params
- `TestDrawRateSamplesLHS`: Latin Hypercube sampling produces tighter mean/var recovery than naive MC at same sample count
- `TestDrawOpportunitySamples`: non-negative, log-normal mean recovery, max cap, LHS variant
- `TestBattingProjectionWithUncertainty`: percentile (via `np.percentile` on samples), interval, std
- `TestPitchingProjectionWithUncertainty`: same
- `TestPropagateMultiplicative`: mult^2 formula, passthrough for missing stats
- `TestPropagateBlend`: quadrature formula, non-blended stats pass through
- `TestPropagateAdditive`: sum formula

**Default variance estimators:**
- `TestBetaBinomialRateVariance`: more PA → less variance, rate near 0.5 → max variance, zero rate → zero variance, known exact values
- `TestBetaBinomialSecondaryVariance`: same shape, sample_size=0 → wide, sample_size=10000 → narrow
- `TestMixtureModelOpportunityVariance`: healthy vs. injury-prone, volatile vs. stable history, default CV fallback, custom parameters respected

**Estimator protocol conformance:** verify each default implementation satisfies its protocol (useful for catching signature drift)

**Files:** 1 new source, 1 modified (`types.py`), 1 new test

---

## Phase 2: Marcel Rate Uncertainty

**Goal:** Compute per-stat posterior variance from the Marcel regression and store in `rate_variances` metadata. This is where uncertainty originates.

### Math

Marcel's `weighted_rate()` is equivalent to Bayesian shrinkage:
```
posterior_rate = (Σ w_i * stat_i + regression_pa * league_rate) / (Σ w_i * opp_i + regression_pa)
```

Posterior variance (beta-binomial approximation):
```
Var(rate) = rate * (1 - rate) / (effective_n + 1)
where effective_n = Σ(w_i * opp_i) + regression_pa
```

More PA + more regression → smaller variance. Rookies get wide intervals. Veterans get tight intervals.

### New function in: `src/fantasy_baseball_manager/marcel/weights.py`

```python
def rate_posterior_variance(
    *, rate: float, opportunities: Sequence[float],
    weights: Sequence[float], regression_pa: float,
) -> float:
    effective_n = sum(w * o for w, o in zip(weights, opportunities)) + regression_pa
    return rate * (1.0 - rate) / (effective_n + 1.0)
```

### Modify: `src/fantasy_baseball_manager/pipeline/stages/rate_computers.py`

Accept optional `RateVarianceEstimator` via constructor DI:

```python
class MarcelRateComputer:
    def __init__(self, rate_variance_estimator: RateVarianceEstimator | None = None) -> None:
        self._rate_variance_estimator = rate_variance_estimator
```

In `compute_batting_rates()`, after the `raw_rates` loop:

```python
if self._rate_variance_estimator is not None:
    rate_variances = self._rate_variance_estimator.estimate_rate_variances(
        rates=raw_rates,
        opportunities_by_year=pa_per_year,
        weights=weights,
        regression_pa=MARCEL_REGRESSION_PA,
    )
    metadata["rate_variances"] = rate_variances
```

Same pattern for `compute_pitching_rates()` using `outs_per_year` and `MARCEL_REGRESSION_OUTS`.

When `rate_variance_estimator` is `None`, no variance metadata is produced — pipeline behaves identically to today.

### Modify: `src/fantasy_baseball_manager/pipeline/stages/stat_specific_rate_computer.py`

Same pattern — accept `RateVarianceEstimator | None` via constructor. Uses per-stat `regression_pa` from `self._batting_regression[stat]`.

### Tests

- `tests/marcel/test_weights_variance.py` — unit tests for `rate_posterior_variance()`
  - More PA → less variance
  - More regression → less variance
  - Rate near 0.5 → max variance
  - Zero rate → zero variance
  - Known exact values
- Update `tests/pipeline/stages/test_rate_computers.py` — verify `rate_variances` in metadata
- Update `tests/pipeline/stages/test_stat_specific_rate_computer.py` — same

**Files:** 1 modified utility, 2 modified rate computers, 2-3 test files

---

## Phase 3: Adjuster Uncertainty Propagation

**Goal:** Each adjuster transforms `rate_variances` alongside rates. If `rate_variances` is absent, adjusters behave identically to today (backward compatible).

### Adjuster Classification

| Adjuster | File | Transform | Propagation Rule |
|----------|------|-----------|-----------------|
| `ComponentAgingAdjuster` | `component_aging.py` | `rate * mult` | `var * mult^2` |
| `ParkFactorAdjuster` | `park_factor_adjuster.py` | `rate * park_mult` | `var * park_mult^2` |
| `RebaselineAdjuster` | `adjusters.py` | `rate * (target/avg)` | `var * (target/avg)^2` |
| `MarcelAgingAdjuster` | `adjusters.py` | `rate * age_mult` | `var * age_mult^2` |
| `StatcastRateAdjuster` | `statcast_adjuster.py` | `w*sc + (1-w)*marcel` | blend propagation |
| `PitcherStatcastAdjuster` | `pitcher_statcast_adjuster.py` | blend | blend propagation |
| `BatterBabipAdjuster` | `batter_babip_adjuster.py` | blend | blend propagation |
| `PitcherBabipSkillAdjuster` | `pitcher_babip_skill_adjuster.py` | blend | blend propagation |
| `PitcherNormalizationAdjuster` | `pitcher_normalization.py` | regression blend | blend propagation |
| `MTLBlender` | `mtl_blender.py` | blend | blend propagation (Phase 6 adds MTL var) |
| `ContextualBlender` | `contextual_blender.py` | blend | blend propagation |
| `EnsembleAdjuster` | `ensemble.py` | multi-source blend | blend propagation |
| `GBResidualAdjuster` | `gb_residual_adjuster.py` | `rate + residual` | additive propagation |
| `SkillChangeAdjuster` | `skill_change_adjuster.py` | `rate + delta` | additive propagation |
| `PlayerIdentityEnricher` | `identity_enricher.py` | no rate change | **no change needed** |

### Implementation pattern (same for all)

Each adjuster already constructs a new `PlayerMetadata` via `{**p.metadata}`. The change is to also read `rate_variances`, transform it, and write it back:

```python
# Example: multiplicative adjuster (component_aging.py)
existing_vars = p.metadata.get("rate_variances", {})
new_vars = propagate_multiplicative(existing_vars, {stat: mult for stat, mult in multipliers.items()})
new_metadata: PlayerMetadata = {**p.metadata}
if new_vars:
    new_metadata["rate_variances"] = new_vars
```

For blend adjusters, the secondary source's variance is obtained from a `SecondaryVarianceEstimator` injected via constructor:

```python
class StatcastRateAdjuster:
    def __init__(
        self,
        feature_store: FeatureStore,
        config: StatcastBlendConfig | None = None,
        secondary_variance: SecondaryVarianceEstimator | None = None,  # new
    ) -> None: ...
```

In `_blend_player()`:
```python
if self._secondary_variance is not None and existing_vars:
    sec_vars = self._secondary_variance.estimate_secondary_variances(
        rates=sc_rates, sample_size=statcast.pa,
    )
    new_vars = propagate_blend(existing_vars, sec_vars, w, set(BLENDED_STATS))
    metadata["rate_variances"] = new_vars
```

When `secondary_variance` is `None`, blend adjusters skip variance propagation entirely — backward compatible. Same pattern for all blend adjusters; each passes its own source's sample size.

### Approach: implement iteratively

Start with `ComponentAgingAdjuster` as the template (multiplicative — no estimator needed, just the propagation helper), verify with tests, then do `StatcastRateAdjuster` as the blend template (requires `SecondaryVarianceEstimator`), then apply to all remaining adjusters.

### Tests

For each modified adjuster, add test cases:
1. When `rate_variances` absent → metadata passes through unchanged
2. When present → output has correctly transformed variances
3. Verify formula matches expected values

**Files:** ~13 modified adjuster files, ~13 test file updates

---

## Phase 4: Playing Time Uncertainty

**Goal:** Produce `opportunities_variance` in metadata using a pluggable `OpportunityVarianceEstimator`.

### Modify: `src/fantasy_baseball_manager/pipeline/stages/enhanced_playing_time.py`

Accept optional `OpportunityVarianceEstimator` via constructor DI:

```python
class EnhancedPlayingTimeProjector:
    def __init__(
        self,
        config: PlayingTimeConfig | None = None,
        opportunity_variance: OpportunityVarianceEstimator | None = None,  # new
    ) -> None: ...
```

In `_project_batter()` and `_project_pitcher()`, after computing the final projected opportunities:

```python
if self._opportunity_variance is not None:
    pt_metadata["opportunities_variance"] = self._opportunity_variance.estimate_opportunity_variance(
        projected_opportunities=projected_pa_final,
        historical_opportunities=pa_per_year,
        injury_factor=injury_factor,
    )
```

When `opportunity_variance` is `None`, no variance metadata is produced.

### Default: `MixtureModelOpportunityVariance`

The default implementation uses:
```
Var(PA) = base_variance + injury_mixture_variance

base_variance:
  - If 2+ years of PA history: stdev(pa_per_year)^2
  - Else: (projected_pa * default_cv)^2

injury_mixture_variance:
  p_healthy = injury_factor
  p_injured = 1 - p_healthy
  Var_mixture = p_healthy * p_injured * (projected_pa * injury_reduction)^2
```

The magic numbers (`default_cv=0.20`, `injury_reduction=0.50`) are fields on the dataclass, making them explicit and tunable without subclassing:

```python
@dataclass(frozen=True)
class MixtureModelOpportunityVariance:
    default_cv: float = 0.20        # CV when < 2 years history
    injury_reduction: float = 0.50  # fraction of PA lost on major injury

    def estimate_opportunity_variance(self, ...) -> float: ...
```

### Tests: `tests/pipeline/test_uncertainty.py` (estimator unit tests)

- `TestMixtureModelOpportunityVariance`:
  - Healthy player (injury_factor=1.0) → injury term is zero, only base variance
  - Injury-prone player (injury_factor=0.7) → higher total variance
  - Volatile PA history → higher base variance
  - Single year of data → uses default CV
  - Custom `default_cv` / `injury_reduction` respected

### Tests: `tests/pipeline/stages/test_enhanced_playing_time_uncertainty.py`

- Variance stored in metadata when estimator provided
- No variance metadata when estimator is `None`
- End-to-end: variance values match direct estimator call

**Files:** 1 modified, 1-2 new/updated tests

---

## Phase 5: Monte Carlo Finalizer

**Goal:** New finalizer that reads `rate_variances` and `opportunities_variance`, runs MC simulation, produces wrapper types.

### New file: `src/fantasy_baseball_manager/pipeline/stages/mc_finalizer.py`

```python
class MonteCarloFinalizer:
    def __init__(
        self,
        config: UncertaintyConfig | None = None,
        delegate: StandardFinalizer | None = None,
    ) -> None: ...

    def finalize_batting(self, players: list[PlayerRates]) -> list[BattingProjectionWithUncertainty]:
        point_estimates = self._delegate.finalize_batting(players)
        # For each player: draw rate samples, draw PA samples, multiply
        ...

    def finalize_pitching(self, players: list[PlayerRates]) -> list[PitchingProjectionWithUncertainty]:
        # Same pattern; also derive ERA/WHIP samples from component samples
        ...
```

**MC simulation per player:**
1. Generate base uniform samples via `scipy.stats.qmc.LatinHypercube(d=n_rate_stats+1)` — one dimension per rate stat plus one for opportunities. LHS ensures stratified coverage across the joint space.
2. Map rate dimensions through `scipy.stats.beta(a, b).ppf(u)` using per-stat alpha/beta from mean+variance. Naturally bounded [0,1].
3. Map opportunity dimension through `scipy.stats.lognorm(s, scale).ppf(u)`, clipped to `[0, cap]`.
4. Multiply: `counting_stat_samples = rate_samples * opportunity_samples`
5. Derive composite stats (H = 1B+2B+3B+HR, AB = PA-BB-HBP-SF-SH, etc.)
6. Stack into `(n_samples, n_stats)` array

**Pitching-specific:** ERA and WHIP samples are derived per-sample from component stats (ER, H, BB, IP), preserving their natural correlation structure.

### Pipeline integration

Add a method to `ProjectionPipeline` in `engine.py` to expose intermediate `PlayerRates`:

```python
def compute_player_rates_batters(self, batting_source, team_source, year) -> list[PlayerRates]:
    """Run pipeline up to (but not including) finalization."""
    players = self.rate_computer.compute_batting_rates(...)
    for adjuster in self.adjusters:
        players = adjuster.adjust(players)
    players = self.playing_time.project(players)
    return players
```

This lets callers run the standard pipeline, then pass rates to `MonteCarloFinalizer`:

```python
rates = pipeline.compute_player_rates_batters(src, team_src, 2026)
mc_projections = MonteCarloFinalizer(config).finalize_batting(rates)
```

### Modify: `src/fantasy_baseball_manager/pipeline/engine.py`

Add `compute_player_rates_batters()` and `compute_player_rates_pitchers()`.

### Tests: `tests/pipeline/stages/test_mc_finalizer.py`

- Point estimate matches `StandardFinalizer` output
- Sample shape is `(n_samples, n_stats)`
- Zero variance → all samples identical
- High variance → wider intervals
- Seeded reproducibility
- Interval contains point estimate (for reasonable coverage)
- Pitching: ERA/WHIP samples derived correctly

**Files:** 1 new finalizer, 1 modified engine, 1 new test

---

## Phase 6: ML Model Uncertainty

**Goal:** Extract uncertainty from trained ML models and feed into `rate_variances`.

### 6a: MTL Learned Variances → `MTLLearnedSecondaryVariance`

The MTL model **already has** per-stat `_log_vars` (line 135 of `ml/mtl/model.py`). Expose them and wrap in a `SecondaryVarianceEstimator`.

**Modify:** `src/fantasy_baseball_manager/ml/mtl/model.py`

```python
# On MultiTaskNet:
def get_learned_variances(self) -> dict[str, float]:
    """Return per-stat variance (sigma^2 = exp(log_var))."""
    with torch.no_grad():
        return {stat: float(torch.exp(self._log_vars[stat]).item()) for stat in self.target_stats}
```

**New estimator:** `MTLLearnedSecondaryVariance` (in `uncertainty.py`)

```python
class MTLLearnedSecondaryVariance:
    """SecondaryVarianceEstimator that uses MTL's learned per-stat variances.

    Ignores sample_size — variance comes from the model's loss function, not
    from a beta-binomial approximation.
    """
    def __init__(self, model: MultiTaskNet) -> None:
        self._variances = model.get_learned_variances()

    def estimate_secondary_variances(
        self, rates: dict[str, float], sample_size: float,
    ) -> dict[str, float]:
        return {stat: self._variances.get(stat, 0.0) for stat in rates}
```

**Modify:** `src/fantasy_baseball_manager/pipeline/stages/mtl_blender.py`

The MTL blender already accepts `SecondaryVarianceEstimator | None` from Phase 3. In Phase 6, the pipeline builder wires in `MTLLearnedSecondaryVariance` instead of the default `BetaBinomialSecondaryVariance`. No code changes to the blender itself — just different DI wiring.

### 6b: LightGBM Conformal Prediction → `ConformalSecondaryVariance`

**New file:** `src/fantasy_baseball_manager/ml/conformal.py`

```python
@dataclass
class ConformalCalibration:
    """Calibrated conformal prediction from held-out residuals."""
    calibration_scores: np.ndarray  # |y - f(x)| on held-out data
    coverage: float = 0.90

    @staticmethod
    def from_model(
        X_cal: np.ndarray, y_cal: np.ndarray,
        model: StatResidualModel, coverage: float = 0.90,
    ) -> ConformalCalibration:
        preds = model.predict(X_cal)
        return ConformalCalibration(
            calibration_scores=np.abs(y_cal - preds),
            coverage=coverage,
        )

    def prediction_half_width(self) -> float:
        return float(np.quantile(self.calibration_scores, self.coverage))
```

**New estimator:** `ConformalSecondaryVariance` (in `uncertainty.py`)

Wraps a `ConformalCalibration` into the `SecondaryVarianceEstimator` protocol. Converts half-width to variance using the calibration distribution directly (no Gaussian assumption):

```python
class ConformalSecondaryVariance:
    def __init__(self, calibration: ConformalCalibration) -> None:
        self._var = float(np.var(calibration.calibration_scores))

    def estimate_secondary_variances(
        self, rates: dict[str, float], sample_size: float,
    ) -> dict[str, float]:
        return {stat: self._var for stat in rates}
```

**Modify:** `src/fantasy_baseball_manager/pipeline/stages/gb_residual_adjuster.py`

The GB adjuster already uses additive propagation from Phase 3. In Phase 6, it accepts a `ConformalSecondaryVariance` (or any `SecondaryVarianceEstimator`) to provide the correction's variance. This replaces the beta-binomial fudge for GB residuals.

### 6c: MC Dropout (optional, low priority)

Current MTL dropout rates are 0.05/0.0 — too low for meaningful epistemic uncertainty. Skip for now; revisit if dropout rates are increased.

### Tests

- `tests/ml/test_mtl_learned_variances.py` — variance extraction, consistency with log_var
- `tests/ml/test_conformal.py` — calibration, half-width, `from_model` factory
- `tests/pipeline/test_uncertainty.py` — `MTLLearnedSecondaryVariance` and `ConformalSecondaryVariance` unit tests
- `tests/pipeline/stages/test_mtl_blender_uncertainty.py` — blend propagation with MTL estimator vs. beta-binomial estimator

**Files:** 1 new conformal module, 2 modified (MTL model, uncertainty.py), 1 modified (GB adjuster), 3-4 test files

---

## Phase 7: Downstream Integration (Proposal Sketch — Not Implemented)

### Risk-Aware Valuation

Run z-score/SGP on each MC sample to get a distribution of player values:

```python
@dataclass(frozen=True)
class PlayerValueWithUncertainty:
    expected_value: float      # mean across samples
    floor_value: float         # 10th percentile
    ceiling_value: float       # 90th percentile
    value_std: float           # std of total value
```

**Challenge:** Z-scores are relative (pool-dependent). Options:
- Fix pool mean/std from point estimates, vary only the player's stat (fast, approximate)
- Full re-computation per sample (slow, exact) — batch-vectorize with numpy

### Risk-Aware Draft Strategy

```python
@dataclass(frozen=True)
class UncertaintyDraftStrategy:
    risk_aversion: float = 0.5  # 0 = ceiling-seeker, 1 = floor-seeker
    def score(self, pv: PlayerValueWithUncertainty) -> float:
        return (1 - self.risk_aversion) * pv.ceiling_value + self.risk_aversion * pv.floor_value
```

Use cases:
- **Floor picks** (risk_aversion=1.0): safe in H2H leagues, protect category leads
- **Ceiling picks** (risk_aversion=0.0): upside gambles in roto, tournament DFS
- **Portfolio optimization**: minimize roster-level variance while maintaining expected value

### Correlation Structure (Future)

Current plan assumes independent rate distributions. To capture stat correlations (HR↔RBI, K↔AVG), estimate the correlation matrix from historical residuals and draw from a multivariate normal. This would improve the realism of joint distributions for portfolio optimization.

---

## Builder & Preset Wiring

The `UncertaintyEstimators` bundle is threaded through the pipeline builder so stages get their estimators via DI. Presets provide sensible defaults:

```python
# In builder or presets:
def with_uncertainty(
    self,
    config: UncertaintyConfig | None = None,
    estimators: UncertaintyEstimators | None = None,
) -> Self:
    estimators = estimators or UncertaintyEstimators.defaults()
    # Wire estimators into rate computers, adjusters, playing time
    ...
```

Callers experimenting with different uncertainty models only change the estimator bundle:

```python
pipeline = (
    PipelineBuilder()
    .with_uncertainty(estimators=UncertaintyEstimators(
        rate=EmpiricalResidualRateVariance(backtest_data),
        secondary=BetaBinomialSecondaryVariance(),
        opportunity=HistoricalVolatilityOpportunityVariance(pa_errors),
    ))
    .build()
)
```

---

## Summary

| Phase | New Files | Modified Files | Dependencies | Scope |
|-------|-----------|---------------|-------------|-------|
| 1: Types & MC engine | 1 src, 1 test | 1 (`types.py`) | None | Foundation + protocols + default estimators |
| 2: Marcel variance | 1 test | 3 (weights, 2 rate computers) | Phase 1 | Rate origin (uses `RateVarianceEstimator` via DI) |
| 3: Adjuster propagation | — | ~13 adjusters, ~13 tests | Phase 2 | Pipeline flow (blend adjusters use `SecondaryVarianceEstimator` via DI) |
| 4: PT uncertainty | 1 test | 1 (`enhanced_playing_time.py`) | Phase 1 | Playing time (uses `OpportunityVarianceEstimator` via DI) |
| 5: MC finalizer | 1 src, 1 test | 1 (`engine.py`) | Phases 1,2,4 | Finalization |
| 6: ML uncertainty | 1 src, 3-4 tests | 3 (MTL, uncertainty.py, GB) | Phase 3 | Alt estimator implementations (`MTLLearnedSecondaryVariance`, `ConformalSecondaryVariance`) |
| 7: Downstream | — | — | Phases 5,6 | Proposal only |

**Phase execution order:** 1 → 2 → 3 (parallel with 4) → 5 → 6

### Risks & Mitigations

1. **Beta-binomial approximation** assumes per-PA Bernoulli trials (OK for HR/PA, BB/PA; less precise for R/PA, RBI/PA). Mitigate: calibrate against historical residual variances in a later pass.
2. **Beta parameterization edge cases** — when variance ≥ mean*(1-mean), method-of-moments yields invalid alpha/beta. Mitigate: clamp to Beta(1,1) (uniform) as a safe fallback; log a warning.
3. **Phase 3 is the largest phase** (~13 adjuster files). Mitigate: implement with shared helpers; each adjuster is a ~5-10 line mechanical change. Can be split into sub-PRs by adjuster category.
4. **scipy dependency** — already installed transitively via scikit-learn; add as explicit dependency to make the requirement visible.

### Verification

- `uv run pytest` after each phase
- `uv run ty check src tests` before each commit
- After Phase 5: run full pipeline on a sample player, verify intervals are plausible (e.g., HR p10/p90 bracket the point estimate, wider for low-PA players)
- After Phase 6: compare MTL-blended intervals to Marcel-only intervals — MTL should narrow intervals for stats it predicts well
