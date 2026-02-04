# Marcel

The modern baseline projection pipeline, implemented as `marcel`. Improves on Marcel Classic with stat-specific regression and component aging.

## Status

**Active** - Recommended as the modern baseline for projections without external data dependencies.

## Pipeline Stages

```
StatSpecificRegressionRateComputer → RebaselineAdjuster → ComponentAgingAdjuster → MarcelPlayingTime → StandardFinalizer
```

## Key Improvements Over Marcel Classic

### 1. Stat-Specific Regression Constants

Different stats stabilize at different rates. The `StatSpecificRegressionRateComputer` uses per-stat regression constants instead of a universal value:

**Batting Regression (PA)**

| Stat | Regression PA | Rationale |
|------|---------------|-----------|
| HR rate | 1200 | Power is moderately stable |
| K rate | 150 | Strikeout rate stabilizes quickly |
| BB rate | 500 | Walk rate is fairly stable |
| SB/CS rate | 600 | Speed signals are reliable |
| Singles/Doubles/Triples | 1200 | Contact outcomes are noisy |

**Pitching Regression (Outs)**

| Stat | Regression Outs | Rationale |
|------|-----------------|-----------|
| K rate | 30 | Strikeout ability is highly stable |
| BB rate | 170 | Walk rate is fairly stable |
| HR rate | 200 | HR rate is moderately noisy |
| H rate | 200 | BABIP-dependent, very noisy |
| ER rate | 150 | Heavily BABIP/LOB dependent |

Lower regression constants mean the system trusts the player's actual performance more; higher constants mean more regression to league average.

### 2. Component Aging

Instead of applying one aging multiplier to all stats, `ComponentAgingAdjuster` applies different curves per stat category:

- **Speed stats** (SB, triples) decline faster with age
- **Power stats** (HR, ISO) have a later peak and slower decline
- **Plate discipline** (BB%, K%) is more stable across ages

This captures the reality that a 35-year-old may still hit for power but has likely lost significant speed.

## Configuration

```python
from fantasy_baseball_manager.pipeline.presets import marcel_pipeline
from fantasy_baseball_manager.pipeline.stages.regression_config import RegressionConfig

# Default configuration
pipeline = marcel_pipeline()

# Custom regression constants
custom_config = RegressionConfig(
    batting_regression_pa={"hr": 1000, "so": 100, ...},
    pitching_regression_outs={"so": 25, "bb": 150, ...},
)
pipeline = marcel_pipeline(config=custom_config)
```

## Performance

4-year backtest (2021-2024), min 200 PA batters, min 50 IP pitchers:

### Batting

| Metric | marcel_classic | marcel | Change |
|--------|----------------|--------|--------|
| HR Correlation | 0.635 | 0.637 | +0.3% |
| SB RMSE | 6.724 | 6.618 | -1.6% |
| SB Correlation | 0.666 | **0.678** | **+1.8%** |
| OBP Correlation | 0.530 | 0.526 | -0.8% |
| Rank Spearman | 0.558 | 0.558 | - |

The SB improvement (+1.8% correlation) is the primary gain, driven by the lower SB regression constant trusting speed signals more.

### Pitching

| Metric | marcel_classic | marcel | Change |
|--------|----------------|--------|--------|
| K RMSE | 38.942 | 38.829 | -0.3% |
| K Correlation | 0.690 | **0.692** | **+0.3%** |
| ERA RMSE | 1.437 | 1.422 | -1.0% |
| ERA Correlation | 0.153 | 0.155 | +1.3% |
| Rank Spearman | 0.302 | 0.305 | +1.0% |

The K rate improvement reflects the very low regression constant (30 outs) correctly identifying strikeout ability as highly stable.

## Limitations

While improved over Marcel Classic, this pipeline still:

1. **Ignores park effects** - A player moving from Coors Field to Petco Park will be over-projected
2. **Ignores Statcast data** - Contact quality (barrel rate, exit velocity) is not considered
3. **Has weak ERA projections** - ERA correlation of 0.155 is still poor due to BABIP/LOB noise

These limitations are addressed by `marcel_full` and `marcel_gb`.

## When to Use

- As the default simple projection system
- When Statcast data is unavailable
- When you need fast projections without ML model dependencies
- As a baseline for A/B testing new adjustments

## Implementation

```python
# src/fantasy_baseball_manager/pipeline/presets.py

def marcel_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """Modern baseline: per-stat regression + component aging."""
    cfg = config or RegressionConfig()
    return ProjectionPipeline(
        name="marcel",
        rate_computer=StatSpecificRegressionRateComputer(
            batting_regression=cfg.batting_regression_pa,
            pitching_regression=cfg.pitching_regression_outs,
        ),
        adjusters=(RebaselineAdjuster(), ComponentAgingAdjuster()),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )
```

## Related Documentation

- [Regression Config](../src/fantasy_baseball_manager/pipeline/stages/regression_config.py) - Default regression constants
- [Component Aging](../src/fantasy_baseball_manager/pipeline/stages/component_aging.py) - Per-stat aging curves
- [Evaluation Summary](../evaluations/evaluation-summary.md) - Full backtest results
