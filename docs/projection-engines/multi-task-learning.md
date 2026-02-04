# Multi-Task Learning Neural Network

The multi-task learning (MTL) model uses a PyTorch neural network to predict multiple correlated baseball stats simultaneously. Unlike the gradient boosting model which predicts residuals from Marcel, MTL can operate as a standalone rate predictor or be blended with Marcel projections.

## Available Pipelines

| Pipeline | Description |
|----------|-------------|
| `mtl` | Standalone MTL predictions (experimental) |
| `marcel_mtl` | Marcel + MTL blend (70% Marcel, 30% MTL) |

## Architecture

The MTL model uses a shared trunk with stat-specific output heads:

```
Input Features (25 batter / 21 pitcher)
    │
[Shared Trunk]
    Linear(n_features, 64) → ReLU → Dropout(0.05)
    Linear(64, 32) → ReLU
    │
[Stat-Specific Heads] (one per stat)
    Linear(32, 16) → ReLU → Linear(16, 1)
```

**Target stats**:
- Batters: HR, SO, BB, singles, doubles, triples, SB (7 stats)
- Pitchers: H, ER, SO, BB, HR (5 stats)

**Loss function**: Uncertainty-weighted multi-task MSE (learns per-stat weights automatically)

## Two Usage Modes

### Mode 1: Standalone (`mtl`)

Uses the neural network directly for rate prediction. Falls back to Marcel for players without sufficient Statcast data.

```python
from fantasy_baseball_manager.pipeline.presets import build_pipeline
pipeline = build_pipeline("mtl")
```

**Status**: Experimental. Underperforms Marcel on most metrics due to limited training signal (see Findings below).

### Mode 2: Ensemble (`marcel_mtl`)

Blends Marcel rates with MTL predictions:
```
blended_rate = 0.7 * marcel_rate + 0.3 * mtl_rate
```

```python
pipeline = build_pipeline("marcel_mtl")
```

**Status**: Shows promise for pitcher projections.

## Performance (2024 Evaluation)

### Batting (n=348, 200+ PA)

| Metric | marcel | marcel_mtl | marcel_gb |
|--------|--------|------------|-----------|
| HR RMSE | 7.13 | **7.11** | 8.73 |
| HR Corr | **0.656** | 0.648 | 0.692 |
| SB RMSE | 7.51 | **7.50** | **7.31** |
| SB Corr | **0.689** | 0.680 | **0.739** |
| OBP Corr | 0.478 | 0.385 | **0.505** |
| Rank ρ | 0.540 | 0.547 | **0.581** |

### Pitching (n=318, 50+ IP)

| Metric | marcel | marcel_mtl | marcel_gb |
|--------|--------|------------|-----------|
| ERA RMSE | 1.383 | **1.182** | 1.210 |
| WHIP RMSE | 0.230 | **0.208** | 0.236 |
| ERA Corr | 0.153 | **0.174** | 0.194 |
| K Corr | **0.686** | 0.675 | **0.695** |
| Top-20 | 0.300 | **0.400** | 0.350 |

**Key finding**: `marcel_mtl` achieves the best ERA and WHIP RMSE of any pipeline, and significantly better pitching Top-20 precision than baseline Marcel.

## Training

```bash
# Train on 2016-2023 data (8 years provides ~3000 samples)
uv run fantasy-baseball-manager ml train-mtl \
    --years 2016,2017,2018,2019,2020,2021,2022,2023 \
    --name default \
    --validate
```

Models are saved to `~/.fantasy_baseball/models/mtl/`.

## Configuration

### Architecture (`MTLArchitectureConfig`)

```python
@dataclass(frozen=True)
class MTLArchitectureConfig:
    shared_layers: tuple[int, ...] = (64, 32)
    head_hidden_size: int = 16
    dropout_rates: tuple[float, ...] = (0.05, 0.0)  # Light dropout
    use_batch_norm: bool = False  # Disabled - causes train/test mismatch
```

### Training (`MTLTrainingConfig`)

```python
@dataclass(frozen=True)
class MTLTrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001  # Very low to preserve variance
    patience: int = 25
    val_fraction: float = 0.15
```

### Blender (`MTLBlenderConfig`)

```python
@dataclass(frozen=True)
class MTLBlenderConfig:
    model_name: str = "default"
    mtl_weight: float = 0.3  # 30% MTL, 70% Marcel
    min_pa: int = 100
```

## Technical Findings

### Why Standalone MTL Underperforms Marcel

1. **Variance compression**: The neural network regresses predictions heavily toward the mean (HR std: 3.0 vs actual 9.2). This causes systematic underprediction of superstars (e.g., Aaron Judge: MTL=9.0 HR, Actual=58).

2. **Limited temporal signal**: MTL uses 1 year of prior Statcast features, while Marcel uses 3 years of weighted historical stats. More years of history provides better signal for rate prediction.

3. **Regression to mean correlation**: Actual-vs-error correlation of -0.94 for HR indicates extreme mean regression. Marcel's -0.61 is more calibrated.

### Data Leakage Fix (Critical)

The original implementation had a training bug where target-year actual rates were used as "marcel rate" features. This caused the model to appear accurate during development but fail in production.

**Fixed**: Features now correctly use prior-year (Y-1) rates, matching inference behavior.

### Why Batch Normalization Disabled

With small batch sizes (~64) and limited samples (~3000), batch norm statistics during training differ significantly from inference, causing prediction drift. Layer normalization or no normalization works better.

### OBP Degradation

The MTL blend hurts OBP correlation (0.478 → 0.385) because it affects singles and BB predictions, which are noisy and hard to predict from Statcast features. The GB model avoids this by only adjusting HR and SB.

## Recommendations

1. **Use `marcel_gb` for overall best accuracy** - It remains the top performer for fantasy-relevant rank correlation.

2. **Consider `marcel_mtl` for pitcher-heavy leagues** - The 15% improvement in ERA RMSE and 10% in WHIP RMSE may be valuable.

3. **Future work**: Train MTL to predict residuals (like GB) rather than raw rates. This would combine Marcel's stable baseline with MTL's ability to capture complex feature interactions.

## Features Used

Same as the GB model (25 batter features, 21 pitcher features):

| Category | Features |
|----------|----------|
| Prior Year Rates | hr, so, bb, singles, doubles, triples, sb |
| Statcast | xba, xslg, xwoba, barrel_rate, hard_hit_rate |
| Swing Decision | chase_rate, whiff_rate, discipline_score |
| Demographics | age, age_squared, opportunities |
| Derived | marcel_iso, xba_minus_marcel_avg, barrel_vs_hr_ratio |

## Implementation Notes

### Pipeline Stage Order (marcel_mtl)

1. Rate computation (Marcel weights)
2. Park factors
3. Pitcher normalization
4. Statcast adjustments
5. BABIP adjustments
6. **MTL blending**
7. Rebaseline
8. Component aging
9. Playing time projection
10. Finalization

### Uncertainty-Weighted Loss

The model learns per-stat loss weights automatically using homoscedastic uncertainty:

```python
loss_i = (1/2σ²_i) * MSE_i + log(σ_i)
```

This allows stats with different scales (e.g., HR rate ~0.03 vs SO rate ~0.22) to be balanced automatically during training.
