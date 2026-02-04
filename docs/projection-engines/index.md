# Projection Engines Overview

This project implements several projection pipelines based on the Marcel methodology, progressively enhanced with modern adjustments. Each pipeline builds on the previous one, adding data sources and corrections that improve accuracy.

## Available Pipelines

| Pipeline | Status | Best For | Description |
|----------|--------|----------|-------------|
| `marcel_classic` | Active | Baseline comparison | Original Marcel with uniform aging |
| `marcel` | Active | Simple projections | Modern baseline with per-stat regression |
| `marcel_full` | Active | Production (no ML) | Full adjustments: park, Statcast, normalization |
| `marcel_gb` | Active | **Best accuracy** | marcel_full + ML residual corrections |

## Quick Start

```python
from fantasy_baseball_manager.pipeline.presets import build_pipeline

# Get the best accuracy pipeline
pipeline = build_pipeline("marcel_gb")

# Or for simpler projections without ML dependencies
pipeline = build_pipeline("marcel_full")
```

## Pipeline Architecture

All pipelines follow the same stage-based architecture:

```
Historical Stats → Rate Computer → Adjusters → Playing Time → Finalizer → Projections
```

1. **Rate Computer**: Calculates weighted rates from historical seasons (3-year lookback)
2. **Adjusters**: Apply corrections (park factors, Statcast blending, aging, etc.)
3. **Playing Time**: Projects plate appearances or innings pitched
4. **Finalizer**: Converts rates to counting stats

## Performance Summary

Based on 2021-2024 backtesting (339 batters/year, 200+ PA):

| Pipeline | HR Corr | SB Corr | ERA Corr | Rank Spearman |
|----------|---------|---------|----------|---------------|
| marcel_classic | 0.635 | 0.666 | 0.153 | 0.558 |
| marcel | 0.637 | 0.678 | 0.155 | 0.558 |
| marcel_full | 0.665 | 0.678 | 0.247 | 0.586 |
| marcel_gb | 0.678 | 0.736 | 0.247 | 0.603 |

Key findings:
- **Pitcher normalization** (BABIP/LOB% regression) delivers the largest single improvement: ERA RMSE -8.9%
- **Statcast blending** improves HR correlation +1.1% and OBP correlation +1.9%
- **GB residual model** adds +4.5% to rank accuracy, the most important fantasy metric

## Detailed Documentation

- [Marcel Classic](./marcel-classic.md) - Original Marcel methodology
- [Marcel](./marcel.md) - Modern baseline with stat-specific regression
- [Marcel Full](./marcel-full.md) - Kitchen-sink pipeline with all adjustments
- [Gradient Boosting Residual Model](./gradient-boosting-residual-model.md) - ML-enhanced projections

## Methodology Notes

All pipelines use a 3-year lookback with Marcel weighting (5-4-3 for most recent to oldest year). The key differentiator is how they handle:

1. **Regression to mean**: Flat vs. stat-specific constants
2. **External data**: Park factors, Statcast expected stats
3. **Pitcher normalization**: BABIP and LOB% regression
4. **Aging adjustments**: Uniform vs. component-specific curves
5. **Residual correction**: Statistical vs. ML-based

See individual engine documentation for implementation details and performance characteristics.
