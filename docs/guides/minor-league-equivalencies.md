# Minor League Equivalencies (ML MLE)

The ML MLE model uses gradient boosting to predict MLB performance from minor league statistics. Unlike traditional MLE which applies fixed translation factors (AAA × 0.90, AA × 0.80), the ML model learns stat-specific, level-aware, and context-dependent translations from historical call-up data.

## Available Pipelines

| Pipeline | Description |
|----------|-------------|
| `mle` | Marcel with ML-based minor league translations for limited-history players |

## Architecture

The MLE system integrates as a specialized rate computer in the projection pipeline:

```
MiLB Stats (MLB Stats API)
        ↓
[CachedMinorLeagueDataSource] ← Cache
        ↓
[AggregatedMiLBStats] (PA-weighted across levels)
        ↓
[MLEBatterFeatureExtractor] → 32 features
        ↓
[MLEGradientBoostingModel] (7 LightGBM models)
        ↓
[MLERateComputer] (blends MLE + Marcel)
        ↓
PlayerRates → standard pipeline stages
```

### Rate Computer Logic

For each player:
1. If MLB PA ≥ 200: Use Marcel rates (sufficient MLB history)
2. If MLB PA < 200 AND qualifying MiLB data:
   - Extract features from prior-year MiLB stats
   - Get MLE predictions
   - Blend with Marcel: `(mlb_pa × marcel + milb_pa × mle) / total_pa`
3. If no qualifying MiLB data: Fall back to Marcel

**Qualifying MiLB criteria**: ≥200 PA at AAA or AA in prior year

### Feature Set (32 features)

| Category | Features | Count |
|----------|----------|-------|
| Rate stats | hr, so, bb, hit, singles, doubles, triples, sb, iso, avg, obp, slg | 12 |
| Age | age, age_squared, age_for_level | 3 |
| Level one-hot | level_aaa, level_aa, level_high_a, level_single_a | 4 |
| Level distribution | pct_at_aaa, pct_at_aa, pct_at_high_a, pct_at_single_a | 4 |
| Sample size | total_pa, log_pa | 2 |
| Statcast | xba, xslg, xwoba, barrel_rate, hard_hit_rate, sprint_speed, has_statcast | 7 |

**Multi-level handling**: When a player plays at multiple levels (e.g., 300 PA at AA + 200 PA at AAA), stats are aggregated PA-weighted.

**Age-for-level**: Captures "advanced for level" signal (e.g., 22-year-old at AAA is young → higher upside).

## Performance (2024 Holdout Evaluation)

**Test set**: 60 players who debuted in 2024 with ≥100 MLB PA.

### Per-Stat Results

| Stat | ML MLE RMSE | Traditional MLE RMSE | Improvement | Spearman ρ |
|------|-------------|----------------------|-------------|------------|
| **SB** | 0.020 | 0.082 | **+75.6%** | 0.45 |
| **SO** | 0.049 | 0.084 | **+41.3%** | 0.52 |
| **BB** | 0.034 | 0.042 | +19.0% | 0.48 |
| Doubles | 0.017 | 0.019 | +12.2% | 0.31 |
| Triples | 0.006 | 0.006 | +7.1% | 0.22 |
| Singles | 0.034 | 0.037 | +6.8% | 0.38 |
| HR | 0.013 | 0.011 | -19.4% | 0.41 |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Average RMSE improvement vs traditional MLE | **20.4%** |
| Stats where ML MLE wins | 6/7 |
| Average RMSE improvement vs no translation | **24.9%** |

### Why Each Stat Performs Differently

**SB (+75.6%)**: Stolen base translation varies dramatically by player profile. The ML model learns that speed + opportunity context matters more than raw SB rate. Traditional MLE's uniform factor misses these nuances entirely.

**SO (+41.3%)**: Strikeout rate translation varies by level and age. Younger players at AAA often have more swing-and-miss but also more upside. The model captures that a 23-year-old's K rate at AAA translates differently than a 27-year-old's.

**BB (+19.0%)**: Walk rate is relatively stable across levels, but the model learns contextual adjustments for age and level that fixed factors miss.

**HR (-19.4%)**: Traditional MLE was slightly better here. Possible reasons:
1. Power translates relatively consistently (the 0.90 factor is reasonable)
2. Sample size effects (only 60 test samples)
3. Missing Statcast features (exit velocity, launch angle would help)

## Training

```bash
# Train on 2021-2023 call-up data
uv run fantasy-baseball-manager ml train-mle \
    --years 2021,2022,2023 \
    --validation-years 2024 \
    --name default
```

Models are saved to `~/.fantasy_baseball/models/mle/`.

**Training data**: ~400-600 samples (call-ups with ≥200 MiLB PA + ≥100 MLB PA).

**Hyperparameters** (tuned for small dataset):
- `n_estimators`: 100
- `max_depth`: 4
- `learning_rate`: 0.1
- `min_child_samples`: 20
- `subsample`: 0.8

## Configuration

### Rate Computer (`MLERateComputerConfig`)

```python
@dataclass(frozen=True)
class MLERateComputerConfig:
    model_name: str = "default"  # Model name in MLEModelStore
    min_milb_pa: int = 200       # Minimum MiLB PA to qualify
    mlb_pa_threshold: int = 200  # Use MLE if player has < this MLB PA
```

### Training (`MLETrainingConfig`)

```python
@dataclass
class MLETrainingConfig:
    min_samples: int = 50           # Skip stat if fewer training samples
    hyperparameters: MLEHyperparameters = field(default_factory=MLEHyperparameters)
    min_milb_pa: int = 200          # MiLB qualification threshold
    min_mlb_pa: int = 100           # MLB target qualification threshold
    max_prior_mlb_pa: int = 200     # Exclude players with prior MLB experience
```

## Usage

### Standalone Pipeline

```python
from fantasy_baseball_manager.pipeline.presets import build_pipeline

pipeline = build_pipeline("mle")
projections = pipeline.project_batters(data_source, projection_year=2025)
```

### With PipelineBuilder

```python
from fantasy_baseball_manager.pipeline.builder import PipelineBuilder

pipeline = (
    PipelineBuilder("custom_mle")
    .with_mle_rate_computer(mlb_pa_threshold=150)
    .with_park_factors()
    .with_statcast()
    .with_enhanced_playing_time()
    .build()
)
```

### Checking MLE Application

```python
for player in projections:
    if player.metadata.get("mle_applied"):
        print(f"{player.name}: MLE from {player.metadata['mle_source_level']}")
        print(f"  Marcel rates: {player.metadata['marcel_rates']}")
        print(f"  MLE rates: {player.metadata['mle_rates']}")
```

## Technical Findings

### Why ML Beats Traditional MLE

1. **Stat-specific translations**: K% and BB% translate better than HR% and SB%. Fixed factors assume uniform translation.

2. **Age-for-level signal**: A 22-year-old at AA has more upside than a 26-year-old. The model learns this from historical call-ups.

3. **Level distribution matters**: Being promoted mid-season (50% AA, 50% AAA) is a signal of talent that fixed factors ignore.

4. **Sample size weighting**: More MiLB PA = more reliable features. The model weights appropriately.

### Limitations

1. **Small training set**: ~400-600 samples limits model complexity. Strong regularization required.

2. **No pitcher support**: Currently batters only. Pitchers fall back to Marcel.

3. **Missing Statcast**: MiLB Statcast only available at AAA since 2023. Model uses indicator variable to handle missing data.

4. **Recent call-ups only**: Training data is 2021-2024 (COVID canceled 2020 MiLB season).

### Player ID Mapping

MiLB data uses MLBAM IDs, while MLB data uses FanGraphs IDs. The system uses `SfbbMapper` to translate between ID systems during training and evaluation.

## Recommendations

1. **Use `mle` for prospect-heavy leagues**: The 20%+ improvement in translation accuracy helps identify breakout rookies.

2. **Combine with `marcel_gb` for best overall accuracy**: MLE handles rookies, GB handles established players.

3. **Monitor HR predictions**: Traditional MLE is slightly better for HR. Consider ensemble approach.

4. **Future work**:
   - Add MiLB Statcast features (exit velocity, launch angle) for HR improvement
   - Implement pitcher MLE
   - Ensemble MLE + traditional for robust predictions

## Module Structure

```
src/fantasy_baseball_manager/minors/
├── __init__.py
├── types.py              # MinorLeagueLevel, MiLBStatcastStats, MLEPrediction
├── data_source.py        # MinorLeagueDataSource protocol + MLB Stats API impl
├── cached_data_source.py # CachedMinorLeagueDataSource with TTL
├── features.py           # MLEBatterFeatureExtractor (32 features)
├── model.py              # MLEStatModel, MLEGradientBoostingModel
├── training.py           # MLEModelTrainer orchestration
├── training_data.py      # AggregatedMiLBStats, MLETrainingDataCollector
├── evaluation.py         # MLEEvaluator, baselines, reporting
├── persistence.py        # MLEModelStore for model serialization
└── rate_computer.py      # MLERateComputer (pipeline integration)
```

**Test coverage**: 141 tests across all minors modules.
