# Gradient Boosting Residual Model

The gradient boosting (GB) residual model improves Marcel projections by predicting systematic errors using machine learning. It analyzes Statcast data and player characteristics to identify where Marcel tends to over- or under-project specific stats.

## Available Pipelines

| Pipeline | Description |
|----------|-------------|
| `marcel_classic` | Original Marcel with uniform aging |
| `marcel` | Modern baseline with per-stat regression and component aging |
| `marcel_full` | Kitchen-sink: park factors, Statcast xStats, BABIP adjustments |
| `marcel_gb` | **Best accuracy**: marcel_full + GB residual corrections |

## How It Works

1. **Training**: The model learns from historical projection errors (residuals = actual - projected) using features like barrel rate, exit velocity, chase rate, and age.

2. **Inference**: For each player, the model predicts expected residuals and adjusts the Marcel projection accordingly.

3. **Rate Conversion**: Residuals are predicted as counting stats (e.g., +5 HR) and converted to rate adjustments before being applied.

## Features Used

The batter model uses 25 features:

| Category | Features |
|----------|----------|
| Marcel Rates | hr, so, bb, singles, doubles, triples, sb, iso |
| Statcast | xba, xslg, xwoba, barrel_rate, hard_hit_rate |
| Swing Decision | chase_rate, whiff_rate, chase_minus_league_avg, whiff_minus_league_avg, chase_x_whiff, discipline_score, has_skill_data |
| Demographics | age, age_squared, opportunities |
| Derived | xba_minus_marcel_avg, barrel_vs_hr_ratio |

## Configuration

The `GBResidualConfig` class controls adjuster behavior:

```python
@dataclass(frozen=True)
class GBResidualConfig:
    model_name: str = "default"
    batter_min_pa: int = 100          # Min Statcast PA to apply adjustments
    pitcher_min_pa: int = 100
    min_rate_denominator_pa: int = 300  # Prevents extreme adjustments for low-PA players
    min_rate_denominator_ip: int = 100
    batter_allowed_stats: tuple[str, ...] | None = None  # None = all stats
    pitcher_allowed_stats: tuple[str, ...] | None = None
```

### Conservative Mode

By default, `marcel_gb` only adjusts HR and SB for batters. This preserves OBP accuracy while improving power/speed predictions:

```python
GBResidualConfig(
    batter_allowed_stats=("hr", "sb"),
    pitcher_allowed_stats=("so", "bb"),
)
```

## Performance

Evaluation across 2021-2024 seasons (n=1356 batters, 200+ PA):

| Metric | marcel_full | marcel_gb |
|--------|-------------|-----------|
| HR correlation | 0.652 | **0.678** (+4%) |
| SB correlation | 0.691 | **0.736** (+6.5%) |
| OBP correlation | 0.516 | **0.535** (+3.7%) |
| Rank Spearman | 0.577 | **0.603** (+4.5%) |
| Top-20 precision | 0.487 | **0.525** (+7.8%) |

The GB model provides the best overall rank accuracy, which is the most important metric for fantasy baseball.

## Why Conservative Mode?

Early experiments with adjusting all stats (hr, so, bb, singles, doubles, triples, sb) degraded OBP predictions significantly:

| Stat | Mean Adjustment | Max Adjustment |
|------|-----------------|----------------|
| singles | +13.88 | +44.40 |
| bb | +7.31 | +26.58 |
| hr | +4.55 | +15.29 |
| sb | +1.33 | +13.52 |

Singles adjustments were too aggressive because BABIP (batting average on balls in play) is inherently noisy and hard to predict. By limiting adjustments to HR and SB—stats where the model shows clear predictive value—we get the best of both worlds.

## Training Models

To retrain the GB models:

```bash
uv run fantasy-baseball-manager ml train --years 2021,2022,2023,2024 --name default
```

To inspect a trained model:

```bash
uv run fantasy-baseball-manager ml info default --type batter
```

## Implementation Details

### Minimum Rate Denominator

Players with low historical PA averages (e.g., a player with PA history of [377, 9, 0]) would get extreme rate adjustments when dividing residuals by average PA. The `min_rate_denominator_pa` config (default 300) prevents this:

```python
opportunities = max(avg_historical_pa, config.min_rate_denominator_pa)
rate_adjustment = residual / opportunities
```

### Pipeline Stage Order

The GB adjuster runs after Statcast adjustments but before aging:

1. Rate computation (Marcel weights)
2. Park factors
3. Statcast adjustments (xStats)
4. **GB residual adjustments**
5. Rebaseline
6. Component aging
7. Playing time projection
8. Finalization

This ensures GB adjustments are made on park- and Statcast-adjusted rates, then aging is applied afterward.
