# ML Valuation Function — Implementation Plan

## Overview

Replace the fixed equal-weight z-score valuation with a learned valuation function that maps stat projections to fantasy value. The model learns the market's valuation function from consensus projections paired with ADP, then applies that function to our projections to surface genuine projection disagreements — not valuation quirks.

**Core insight:** Train on *consensus* projections (Steamer/ZiPS) → ADP. Apply to *our* projections (Marcel) at inference time. Differences between our output and ADP now reflect only projection alpha, since the valuation function is shared.

**Implementation strategy:** Staged approach — ridge regression baseline first, then GBM, then neural network. Each stage is independently useful and validates the approach before adding complexity.

**Data available:** 9 years of FantasyPros ADP (2015-2026), Steamer projections (2015-2026), ZiPS projections (2014-2026). With ~800 players/year × 2 systems × 9+ years, we have ~14,000 potential training samples — well above the original proposal's ~4,500 target.

**Key dependency:** The `build_multi_year_dataset()` function in `adp/training_dataset.py` already joins projections with ADP by normalized name, producing `BatterTrainingRow` (16 fields) and `PitcherTrainingRow` (17 fields).

---

## Background: Why ML Over Fixed Weights?

The z-score valuator treats all categories equally, but the draft market does not. Analysis of our top-50 rankings vs. ADP reveals systematic biases:

- **Speed overvalued:** Perdomo (+43), Turang (+29), Abrams (+52) — high-SB, low-power profiles rank much higher in our system than ADP
- **Category correlations ignored:** HR/R/RBI are correlated (power hitters score in all three), while SB anti-correlates with power. Equal weighting doesn't account for this.
- **Positional scarcity not captured:** Catcher production is scarce; OF is deep. Fixed weights can't learn this from data.

The market's valuation function is non-linear and context-dependent. An ML model can learn these patterns from historical projection-to-ADP mappings.

### What Major Projection Systems Use

None of the major systems (PECOTA, ZiPS, Steamer, THE BAT X, ATC) use deep learning for their core methodology. They rely on weighted historical averages, comparable-player analysis, and regression-based adjustments. The consensus is that careful feature engineering outperforms algorithmic sophistication at baseball-scale sample sizes.

However, *valuation* (mapping stats to draft value) is a different problem than *projection* (predicting future stats). The valuation function is smoother, lower-dimensional, and better-suited to ML — we're learning a mapping from ~15 features to 1 target, with ~14K samples. This is well within the regime where even simple ML models excel.

### Approach Comparison

| Approach | Strengths | Weaknesses | Fit for our problem |
|----------|-----------|------------|-------------------|
| **Ridge regression** | Fully interpretable; coefficients = category weights; analytical prediction intervals; trains instantly | Cannot capture non-linear effects (diminishing returns, stat interactions) | Excellent baseline; may be sufficient |
| **GBM (LightGBM)** | Best empirical performer on our data sizes (proven by marcel_gb); SHAP interpretability; handles mixed features | No natural strategy embedding; can't extrapolate beyond training distribution | Strong choice; proven in our codebase |
| **Neural network** | Can learn strategy embeddings; captures arbitrary non-linear interactions | Risk of overfitting at ~14K samples; less interpretable; more engineering | Worth trying after GBM validates the approach |
| **LambdaMART (ranking)** | Optimizes for ranking quality directly (NDCG); ADP is fundamentally a rank | Loses auction-dollar interpretation; less common | Good variant to try within GBM stage |
| **Random forest** | Native uncertainty via tree variance | Lower accuracy than GBM for tabular data | Skip — GBM dominates here |
| **Bayesian linear** | Principled uncertainty; equivalent to ridge with Gaussian priors | Same linearity limitation as ridge | Consider as ridge extension if intervals are needed |

---

## Phase 1: Ridge Regression Baseline

**Goal:** Learn interpretable category weights from the market. Validate the separation-of-concerns approach (train on consensus, infer with our projections). Establish accuracy baselines.

### Why start here

Ridge regression coefficients directly answer "how does the market weight HR vs. SB?" — the core question motivating this work. If learned weights alone fix the speed-overvaluation bias, we may not need more complexity. Ridge also provides analytical prediction intervals and trains in milliseconds, enabling rapid iteration.

### New module: `src/fantasy_baseball_manager/valuation/ml_valuation.py`

Core types shared across all stages:

```python
@dataclass(frozen=True)
class ValuationModelConfig:
    target: str = "log_adp"           # "adp", "log_adp", or "adp_rank"
    min_pa: int = 50
    min_ip: float = 20.0
    test_years: tuple[int, ...] = (2025, 2026)

@dataclass(frozen=True)
class TrainedValuationModel:
    """Wrapper returned by all model types, with common interface."""
    model: Any                        # sklearn/lightgbm/torch model
    feature_names: tuple[str, ...]
    scaler: StandardScaler | None     # fitted feature scaler
    config: ValuationModelConfig
    training_years: tuple[int, ...]
    validation_metrics: dict[str, float]
    player_type: str                  # "batter" or "pitcher"

    @property
    def is_fitted(self) -> bool: ...

    def predict_value(self, features: dict[str, float]) -> float:
        """Single-player prediction. Returns predicted value (lower = better)."""

    def predict_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Batch prediction for efficiency."""
```

### Feature extraction

Extract features from `BatterTrainingRow` / `PitcherTrainingRow`:

**Batter features (~11):**
`pa, hr, r, rbi, sb, bb, so, obp, slg, war` + position one-hot or ordinal encoding

**Pitcher features (~12):**
`ip, w, sv, hld, so, bb, h, er, hr, era, whip, war`

Position encoding: ordinal by scarcity (C=1, SS=2, 2B=3, 3B=4, OF=5, 1B/DH=6) for batters. SP/RP binary for pitchers.

### Training target

Use `log(ADP)` rather than raw ADP. The ADP scale is highly non-linear (pick 1 vs. 10 matters far more than 200 vs. 210). Log transform compresses the tail and makes the target more normally distributed, improving regression performance.

Alternative: `1/ADP` (inverse ADP, a "value" measure) — worth testing.

### Training pipeline

```python
def train_ridge_valuation(
    rows: list[BatterTrainingRow] | list[PitcherTrainingRow],
    config: ValuationModelConfig,
    alpha: float = 1.0,
) -> TrainedValuationModel:
    """Train ridge regression on consensus projections → log(ADP)."""
    # 1. Extract feature matrix X, target y = log(adp)
    # 2. Split by year: train on years not in config.test_years, test on test_years
    # 3. StandardScaler on features (fit on train only)
    # 4. Ridge(alpha=alpha).fit(X_train, y_train)
    # 5. Evaluate on test set (Spearman rank correlation, RMSE, top-50 precision)
    # 6. Return TrainedValuationModel with metrics
```

### Evaluation metrics

| Metric | What it measures |
|--------|-----------------|
| Spearman rank correlation | Overall ranking quality |
| RMSE on log(ADP) | Prediction accuracy |
| Top-50 precision@50 | Accuracy where it matters most |
| Speed-player bias | Mean rank delta for high-SB players vs. ADP |
| Coefficient analysis | Learned category weights (interpretability) |

### Inference integration

```python
def valuate_batting_ml(
    projections: list[BattingProjection],
    model: TrainedValuationModel,
) -> list[PlayerValue]:
    """Apply learned valuation function to our projections."""
    # 1. Extract features from BattingProjection (same features as training)
    # 2. model.predict_batch(X) → predicted log(ADP)
    # 3. Convert to PlayerValue objects, using predicted value as total_value
    # 4. Sort by total_value (ascending = more valuable)
```

This returns `list[PlayerValue]` — the same type as z-score and SGP valuators — so it slots into the existing CLI and draft pipeline.

### CLI command

```
uv run fantasy valuate 2026 --method ml-ridge --top 50
```

Follow the existing `valuate` CLI pattern. Add `--method` flag to select valuation method (zscore, sgp, ml-ridge, ml-gbm, ml-nn).

### Tests

- `tests/valuation/test_ml_valuation.py`:
  - Feature extraction from training rows matches expected shape
  - Ridge model trains and produces valid predictions (all finite, monotonic with key stats)
  - Log-ADP transform and inverse are consistent
  - `predict_value` and `predict_batch` agree
  - Model with zero alpha matches OLS
  - Returns `PlayerValue` objects with correct structure
  - Coefficient signs are sensible (more HR → lower ADP, more ERA → higher ADP)

### Deliverables

- Feature extraction utilities (shared across all stages)
- Ridge training pipeline
- Evaluation harness (shared across all stages)
- Inference → `PlayerValue` conversion
- Coefficient analysis / interpretability report
- CLI integration

**Files:** 1-2 new source files, 1 new test file, 1 modified CLI

---

## Phase 2: Gradient Boosted Trees (LightGBM)

**Goal:** Capture non-linear stat interactions that ridge misses. Validate whether non-linearity matters for valuation.

### Why GBM

Our own backtesting shows GBM outperforms neural approaches at our data scale (marcel_gb beat marcel_mtl for batters). GBM handles tabular data well, provides SHAP-based interpretability, and we already have LightGBM infrastructure (`ml/residual_model.py`). The `StatResidualModel` pattern (hyperparameters dataclass, fit/predict interface, feature importances) can be adapted for valuation.

### New in `ml_valuation.py`

```python
def train_gbm_valuation(
    rows: list[BatterTrainingRow] | list[PitcherTrainingRow],
    config: ValuationModelConfig,
    hyperparameters: ModelHyperparameters | None = None,
) -> TrainedValuationModel:
    """Train LightGBM on consensus projections → log(ADP)."""
    # 1. Same feature extraction as ridge
    # 2. Same train/test split by year
    # 3. LGBMRegressor with conservative hyperparameters
    # 4. Early stopping on validation set
    # 5. SHAP feature importance analysis
    # 6. Return TrainedValuationModel
```

### LambdaMART ranking variant

Also try `objective='lambdarank'` — this optimizes for ranking quality (NDCG) directly rather than regression accuracy. Since we care about rank order more than exact ADP prediction, this may outperform regression.

```python
def train_lambdarank_valuation(
    rows: list[BatterTrainingRow] | list[PitcherTrainingRow],
    config: ValuationModelConfig,
) -> TrainedValuationModel:
    """Train LambdaMART ranking model."""
    # LGBMRanker with lambdarank objective
    # Group by year (rankings are relative within a year)
```

### Hyperparameters

Follow the conservative settings from `residual_model.py`:

```python
n_estimators=200
max_depth=4
learning_rate=0.05
min_child_samples=20
subsample=0.8
```

Small max_depth and high min_child_samples prevent overfitting on ~14K samples.

### SHAP analysis

After training, compute SHAP values to understand:
- Which stats drive valuation most (expect HR, SB, OBP, ERA near the top)
- Interaction effects (does HR matter more when OBP is also high?)
- Position effects (is catcher scarcity captured?)
- Non-linear patterns (diminishing returns on SB?)

### Evaluation

Same metrics as Phase 1, plus:
- **Ridge vs. GBM comparison** — if GBM doesn't meaningfully beat ridge, the valuation function is approximately linear and we don't need more complexity
- **SHAP consistency** — do SHAP importances align with ridge coefficients?

### Tests

- `tests/valuation/test_ml_valuation.py` (extend):
  - GBM trains and produces valid predictions
  - Feature importances are non-negative and sum to ~1
  - GBM predictions are rank-correlated with ridge predictions (sanity check)
  - LambdaMART produces valid rankings
  - Early stopping works (fewer trees than n_estimators)

**Files:** Extend Phase 1 source and test files. No new files needed.

---

## Phase 3: Neural Network with Strategy Embeddings

**Goal:** Learn strategy-conditioned valuation that adjusts category emphasis without manual weight tuning.

### When to attempt this

Only proceed if Phase 2 demonstrates that:
1. The learned valuation meaningfully outperforms z-score (Spearman > 0.85 on test set)
2. Non-linearity matters (GBM beats ridge by a meaningful margin)
3. Strategy conditioning would add value (i.e., we want different valuations for different draft strategies)

If ridge alone solves the speed-overvaluation problem, the neural network adds complexity without proportional benefit.

### Architecture

```python
class ValuationNetwork(nn.Module):
    """Feed-forward network with optional strategy conditioning."""

    def __init__(
        self,
        n_stat_features: int,      # ~11 for batters, ~12 for pitchers
        n_positions: int,           # position embedding dimension
        n_strategies: int = 4,      # number of strategy types
        strategy_dim: int = 8,      # strategy embedding dimension
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
    ) -> None: ...

    def forward(
        self,
        stats: torch.Tensor,       # (batch, n_stat_features)
        position: torch.Tensor,    # (batch,) — position index
        strategy: torch.Tensor,    # (batch,) — strategy index
    ) -> torch.Tensor:             # (batch, 1) — predicted value
```

**Layer structure:**
1. Stat features → Linear → ReLU (learns stat interactions)
2. Position → Embedding(n_positions, position_dim)
3. Strategy → Embedding(n_strategies, strategy_dim)
4. Concatenate [stat_hidden, position_emb, strategy_emb]
5. Dense → ReLU → Dense → output

### Strategy conditioning

Define strategy types as an enum:

```python
class DraftStrategy(Enum):
    BALANCED = 0       # Standard overall ADP
    POWER = 1          # Emphasize HR/R/RBI
    SPEED = 2          # Emphasize SB
    PUNT_SAVES = 3     # De-emphasize closer value
```

**Training approach:** Since we don't have ADP from different league formats, train on `BALANCED` (all ADP data) and use the strategy embedding to modulate at inference time. The strategy embedding is initialized to zero for BALANCED, meaning it has no effect during training. At inference, non-BALANCED strategies shift the embedding to emphasize/de-emphasize categories.

**Alternative:** If we can collect ADP from points leagues vs. category leagues, or from NFBC (deep) vs. Yahoo (shallow), these format differences provide natural strategy labels for training.

### Training

```python
def train_nn_valuation(
    rows: list[BatterTrainingRow] | list[PitcherTrainingRow],
    config: ValuationModelConfig,
    architecture: NNArchitectureConfig | None = None,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 10,
) -> TrainedValuationModel:
    """Train neural network on consensus projections → log(ADP)."""
    # 1. Feature extraction + position encoding
    # 2. Year-based train/val/test split
    # 3. StandardScaler on stat features
    # 4. Train with AdamW, MSE loss, early stopping
    # 5. Learning rate scheduling (ReduceLROnPlateau)
    # 6. Return TrainedValuationModel
```

### Model persistence

Follow the contextual model pattern: save model state_dict + config + scaler + feature_names as a checkpoint dict via `torch.save()`. Store in `~/.fantasy_baseball/models/valuation/`.

### Interpretability

- **Gradient-based feature importance:** `d(output) / d(feature)` averaged across players
- **SHAP values:** Use DeepSHAP or KernelSHAP for individual explanations
- **Strategy embedding analysis:** Project embeddings to see which strategies are similar
- **Comparison to ridge coefficients:** Validate that the NN learns consistent category importance

### Tests

- `tests/valuation/test_ml_valuation.py` (extend):
  - Network forward pass produces correct output shape
  - Training reduces loss over epochs
  - Strategy embedding for BALANCED ≈ no effect (predictions match no-strategy model)
  - Model saves and loads correctly (round-trip)
  - Predictions are rank-correlated with GBM predictions

**Files:** Extend Phase 1-2 source and test files. Possibly 1 new file for the network architecture.

---

## Phase 4: Evaluation & Integration

**Goal:** Select the best model, integrate into the draft pipeline, and validate end-to-end.

### Model selection

Run all three models (ridge, GBM, NN) on the same test set. Compare:

| Metric | Ridge | GBM | GBM-Rank | NN |
|--------|-------|-----|----------|-----|
| Spearman vs. ADP | | | | |
| RMSE log(ADP) | | | | |
| Top-50 precision | | | | |
| Speed-player bias | | | | |
| Training time | | | | |
| Interpretability | | | | |

Select the model with the best Spearman correlation and acceptable interpretability. If ridge is within 0.02 of GBM, prefer ridge for simplicity.

### Pipeline integration

The ML valuator implements the same interface as z-score/SGP, returning `list[PlayerValue]`. It slots into the draft ranking pipeline as an alternative valuation method.

```python
# In valuation CLI or draft pipeline:
if method == "ml":
    model = load_valuation_model(player_type="batter")
    values = valuate_batting_ml(projections, model)
elif method == "zscore":
    values = zscore_batting(projections, categories)
```

### Alpha analysis

The whole point: compare our ML-valuated rankings (from Marcel projections) to ADP. The remaining differences are pure projection disagreements:

```
Our ranking = MarcelProjection → LearnedValuation → rank
ADP ranking = ConsensusProjection → MarketValuation → rank

Delta = Our ranking - ADP ranking
```

Large positive deltas = players we think are overvalued by the market (our projections are lower).
Large negative deltas = players we think are undervalued (our projections are higher).

**Key validation:** The speed-player overvaluation bias should disappear. If Perdomo still shows +43 after ML valuation, it means we genuinely project him higher than consensus — a projection disagreement worth investigating — rather than a valuation artifact.

### CLI additions

```
# Train a valuation model
uv run fantasy valuation train --method ridge --years 2015-2024 --test-years 2025,2026

# Evaluate model
uv run fantasy valuation evaluate --method ridge --test-years 2025,2026

# Compare methods
uv run fantasy valuation compare --methods ridge,gbm,nn --test-years 2025,2026

# Show projection alpha (our rankings vs ADP)
uv run fantasy valuation alpha --engine marcel --top 50

# Rank players using ML valuation
uv run fantasy valuate 2026 --method ml --top 50
```

### Tests

- `tests/valuation/test_ml_valuation_integration.py`:
  - End-to-end: train on multi-year data, predict on held-out year, verify Spearman > 0.7
  - `valuate_batting_ml` returns `PlayerValue` objects sorted by value
  - ML valuations are rank-correlated with z-score valuations (sanity)
  - Alpha analysis: known projection disagreements surface correctly

**Files:** Modified CLI, possibly 1 new integration test file

---

## Implementation Order

```
Phase 1: Ridge Regression Baseline
├── Feature extraction utilities
├── Ridge training pipeline
├── Evaluation harness
├── Inference → PlayerValue
├── CLI: train + evaluate + valuate
└── Coefficient analysis

Phase 2: GBM (depends on Phase 1 infra)
├── LightGBM training (reuses feature extraction + evaluation)
├── LambdaMART ranking variant
├── SHAP analysis
└── Ridge vs. GBM comparison

Phase 3: Neural Network (depends on Phase 2 validation)
├── ValuationNetwork architecture
├── Strategy embeddings
├── Training loop with early stopping
├── Model persistence
└── Interpretability analysis

Phase 4: Integration (depends on best model from 1-3)
├── Model selection report
├── Pipeline integration
├── Alpha analysis
└── CLI additions
```

**Phase 1 can start immediately.** Phases 1-2 share feature extraction and evaluation code — build these as reusable utilities from the start. Phase 3 is conditional on Phase 2 results. Phase 4 ties everything together.

### Estimated scope

| Phase | New src files | Modified files | New test files |
|-------|--------------|---------------|---------------|
| 1 | 1-2 | 1 (CLI) | 1 |
| 2 | 0 (extend Phase 1) | 0 | 0 (extend Phase 1) |
| 3 | 0-1 (extend or new arch file) | 0 | 0 (extend Phase 1) |
| 4 | 0 | 1-2 (CLI, pipeline) | 1 |

---

## Risks & Mitigations

1. **Overfitting to ADP noise.** ADP reflects market consensus, which includes irrational biases (name recognition, recency bias). Mitigate: regularization (ridge alpha, GBM max_depth), year-based cross-validation, inspect outlier predictions.

2. **Name-matching failures in training data.** The `build_multi_year_dataset()` join uses normalized names, which can fail on name changes, accent characters, or ambiguous names. Mitigate: review `unmatched_adp` / `unmatched_batting` diagnostics per year; consider adding MLBAM ID join path.

3. **ADP distribution shift.** The 2023 stolen base rule change dramatically altered SB values. A model trained on 2015-2022 may learn outdated SB weights. Mitigate: weight recent years more heavily; add a `year` feature; consider training only on 2023+ for initial deployment.

4. **Strategy embeddings without strategy-labeled data.** Without ADP from different league formats, the strategy embedding is untrained. Mitigate: start with manual strategy vectors (not learned); validate by checking that "power" strategy increases HR importance as expected.

5. **Consensus-projection availability.** Training requires Steamer/ZiPS projections paired with ADP. Historical CSVs exist but ADP columns may be empty for some years. Mitigate: the FantasyPros ADP data is independent of projection CSVs; join by name rather than by column.

---

## Success Criteria

1. **ML valuation Spearman rank correlation with ADP > 0.85** on held-out years (when using consensus projections as input)
2. **Speed-player bias reduced by > 50%** — mean rank delta for top-10 SB players drops from ~35 to < 18
3. **Ridge coefficient signs are all sensible** — more HR → lower ADP, higher ERA → higher ADP, etc.
4. **Projection alpha is actionable** — when feeding Marcel projections through the learned valuation, the top-10 "undervalued" and "overvalued" players pass a sniff test
5. **No regression in draft outcomes** — ML-valuated rankings produce competitive draft results in mock simulations vs. z-score rankings
