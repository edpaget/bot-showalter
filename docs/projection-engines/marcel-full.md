# Marcel Full

The comprehensive projection pipeline, implemented as `marcel_full`. Combines all proven adjustments: park factors, Statcast blending, pitcher normalization, and BABIP skill adjustment.

## Status

**Active** - Recommended for production use when ML model dependencies are not desired.

## Pipeline Stages

```
StatSpecificRegressionRateComputer
    → ParkFactorAdjuster
    → PitcherNormalizationAdjuster
    → PitcherStatcastAdjuster
    → StatcastRateAdjuster
    → BatterBabipAdjuster
    → RebaselineAdjuster
    → ComponentAgingAdjuster
    → MarcelPlayingTime
    → StandardFinalizer
```

## Key Features

### 1. Park Factor Adjustment

Adjusts rates based on ballpark effects using FanGraphs park factors:

- HR rates adjusted for park HR factor
- Hit rates adjusted for park hit factor
- Dampened to prevent over-correction (regression_weight=1.0, years_to_average=1)

**Impact**: Marginal batting rank improvement (Spearman +0.004). Park factors are intentionally conservative due to methodology limitations.

### 2. Pitcher BABIP/LOB% Normalization

The single highest-impact adjustment in the entire system. Regresses two highly luck-dependent metrics:

**BABIP Regression** (weight: 0.50)
- Pitchers have limited control over balls in play
- Historical BABIP regresses toward league mean (~.300)

**LOB% Regression** (weight: 0.60)
- Left-on-base percentage is heavily luck-driven
- Regresses toward a K%-adjusted expectation: `0.73 + 0.1 * (K% - league_K%)`
- High-K pitchers sustain slightly higher LOB%

After regression, hit and earned run rates are recomputed from the normalized components.

**Impact**:
- ERA RMSE: -8.9% (1.422 → 1.295)
- ERA Correlation: +12.3% (0.155 → 0.174)
- WHIP Correlation: +11.5% (0.253 → 0.282)
- Pitching Rank Spearman: +7.9% (0.305 → 0.329)

### 3. Pitcher Statcast Blending

Blends Statcast expected stats (xBA-against, xERA) with Marcel rates:

- Captures pitch quality and contact management
- Blend weight: 0.30 for both h and er rates

**Impact**:
- ERA Correlation: +18.2% (0.209 → 0.247)
- Pitching Rank Spearman: +1.2%
- Note: WHIP correlation degrades slightly (-9.2%)

### 4. Batter Statcast Contact Quality

Blends Statcast expected stats (xBA, xSLG, xwOBA) with Marcel batter rates:

- Barrel rate provides direct power quality measure
- Hit-type decomposition improves singles/doubles/triples estimates

**Impact**:
- HR Correlation: +1.1% (0.658 → 0.665)
- OBP Correlation: +1.9% (0.535 → 0.545)
- Batting Rank Spearman: +0.9%

### 5. Batter BABIP Skill Adjustment

Derives expected BABIP from Statcast data and adjusts singles rate:

- Uses xBA and barrel rate to estimate expected BABIP
- Adjusts singles rate toward target (weight: 0.5)

**Impact**: Essentially neutral at default weight. Safe to include; weight may benefit from tuning.

## Configuration

```python
from fantasy_baseball_manager.pipeline.presets import marcel_full_pipeline
from fantasy_baseball_manager.pipeline.stages.regression_config import RegressionConfig

# Default configuration
pipeline = marcel_full_pipeline()

# Custom regression constants
custom_config = RegressionConfig(...)
pipeline = marcel_full_pipeline(config=custom_config)
```

## Performance

3-year backtest (2022-2024), min 200 PA batters, min 50 IP pitchers:

### Batting

| Metric | marcel | marcel_full | Change |
|--------|--------|-------------|--------|
| HR RMSE | 7.803 | 7.261 | -6.9% |
| HR Correlation | 0.637 | **0.665** | **+4.4%** |
| OBP Correlation | 0.526 | **0.545** | **+3.6%** |
| Rank Spearman | 0.558 | **0.586** | **+5.0%** |

### Pitching

| Metric | marcel | marcel_full | Change |
|--------|--------|-------------|--------|
| ERA RMSE | 1.422 | 1.211 | -14.8% |
| ERA Correlation | 0.155 | **0.247** | **+59.4%** |
| WHIP Correlation | 0.253 | 0.266 | +5.1% |
| Rank Spearman | 0.305 | **0.418** | **+37.0%** |

The pitching improvements are dramatic, driven primarily by BABIP/LOB% normalization.

## Data Sources

| Component | Source | Cache |
|-----------|--------|-------|
| Historical Stats | PyBaseball | Session |
| Park Factors | FanGraphs | Persistent |
| Statcast xStats | PyBaseball | Session |
| Pitcher Batted Ball | PyBaseball | Session |

## Limitations

While marcel_full delivers strong accuracy, it still:

1. **Uses statistical adjustments only** - No ML-based residual corrections
2. **Has linear playing time model** - Doesn't account for injuries, role changes
3. **Park factor methodology is conservative** - Rate-division approach has known issues

For the highest accuracy, use `marcel_gb` which adds ML residual corrections.

## When to Use

- Production projections without ML dependencies
- When training ML models is not practical
- As the base for custom pipeline experimentation
- When explainability is important (all adjustments are statistical)

## Implementation

```python
# src/fantasy_baseball_manager/pipeline/presets.py

def marcel_full_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """Kitchen-sink pipeline with all adjusters enabled."""
    cfg = config or RegressionConfig()
    return (
        PipelineBuilder("marcel_full", config=cfg)
        .with_park_factors()
        .with_pitcher_normalization()
        .with_pitcher_statcast()
        .with_statcast()
        .with_batter_babip()
        .build()
    )
```

## Related Documentation

- [Pitcher Normalization Evaluation](../evaluations/pitcher-normalization.md)
- [Park Factors Analysis](../evaluations/park-factors-savant.md)
- [Evaluation Summary](../evaluations/evaluation-summary.md)
- [Gradient Boosting Model](./gradient-boosting-residual-model.md) - Next level: marcel_gb
