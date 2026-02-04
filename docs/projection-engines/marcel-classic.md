# Marcel Classic

The original Marcel projection method, implemented as `marcel_classic`. This is the baseline against which all other pipelines are compared.

## Status

**Active** - Maintained for backward compatibility and baseline comparisons.

## Pipeline Stages

```
MarcelRateComputer → RebaselineAdjuster → MarcelAgingAdjuster → MarcelPlayingTime → StandardFinalizer
```

## Methodology

Marcel (named after a monkey who could supposedly make projections) is a simple, transparent projection system created by Tom Tango. It uses three core principles:

### 1. Weighted Historical Rates

The rate computer calculates weighted averages from the last 3 seasons:

| Year | Weight |
|------|--------|
| Most recent (Y-1) | 5 |
| Two years ago (Y-2) | 4 |
| Three years ago (Y-3) | 3 |

### 2. Regression to Mean

All players regress toward league average based on their sample size. Marcel Classic uses **universal regression constants**:

| Player Type | Regression Constant |
|-------------|---------------------|
| Batters | 1200 PA |
| Pitchers | 134 outs |

The regression formula:

```
projected_rate = (player_rate * player_PA + league_rate * regression_PA) / (player_PA + regression_PA)
```

A player with 600 PA and a .300 AVG regresses more toward league average than a player with 1200 PA and the same .300 AVG.

### 3. Uniform Aging Adjustment

A single aging multiplier applies to all stats based on the player's age relative to peak (29):

- **Under 29**: +0.6% per year below age 29
- **Over 29**: -0.3% per year above age 29

This assumes all skills decline uniformly with age.

## Configuration

```python
from fantasy_baseball_manager.pipeline.presets import marcel_classic_pipeline

pipeline = marcel_classic_pipeline()
# name: "marcel_classic"
# years_back: 3
```

## Performance

4-year backtest (2021-2024), min 200 PA batters, min 50 IP pitchers:

### Batting

| Metric | Value |
|--------|-------|
| HR RMSE | 7.795 |
| HR Correlation | 0.635 |
| SB RMSE | 6.724 |
| SB Correlation | 0.666 |
| OBP RMSE | 0.031 |
| OBP Correlation | 0.530 |
| Rank Spearman | 0.558 |
| Top-20 Precision | 0.513 |

### Pitching

| Metric | Value |
|--------|-------|
| K Correlation | 0.690 |
| ERA RMSE | 1.437 |
| ERA Correlation | 0.153 |
| WHIP RMSE | 0.230 |
| WHIP Correlation | 0.256 |
| Rank Spearman | 0.302 |
| Top-20 Precision | 0.312 |

## Limitations

Marcel Classic has known limitations that later pipelines address:

1. **Universal regression constants** treat all stats equally. In reality, some stats (K%, BB%) stabilize faster than others (BABIP, HR/FB%).

2. **Uniform aging** assumes power, speed, and contact decline at the same rate. Speed actually declines faster with age than plate discipline.

3. **No external data** - ignores park effects, Statcast quality metrics, and pitcher BABIP/LOB luck.

4. **Pitcher ERA weakness** - ERA correlation of 0.153 is notably poor because ERA depends heavily on BABIP and LOB%, which are largely luck-driven and regress strongly to the mean.

## When to Use

- As a baseline for evaluating new pipeline features
- When you need the simplest, most transparent projections
- When external data sources (Statcast, park factors) are unavailable

## Implementation

```python
# src/fantasy_baseball_manager/pipeline/presets.py

def marcel_classic_pipeline() -> ProjectionPipeline:
    """Original Marcel method: MarcelRateComputer + uniform aging."""
    return ProjectionPipeline(
        name="marcel_classic",
        rate_computer=MarcelRateComputer(),
        adjusters=(RebaselineAdjuster(), MarcelAgingAdjuster()),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )
```

## References

- [Tom Tango's original Marcel description](http://www.tangotiger.net/archives/stud0346.shtml)
- Evaluation data: `docs/evaluations/marcel.json`
