# Breakout / Bust Classifier Roadmap

Train a classification model to predict which players will significantly outperform (breakout) or underperform (bust) their ADP, using historical feature patterns. Unlike the talent-delta report (which identifies regression candidates from recent performance changes), this classifier learns multi-factor patterns — age curves, Statcast quality-of-contact changes, platoon shifts, and playing-time volatility — that historically precede breakout or bust seasons. The output is a ranked list of breakout and bust candidates for the upcoming draft.

This roadmap depends on: ADP integration (done), feature infrastructure (done — SQL assembler, feature library, feature groups), GBM training infrastructure (done — `models/gbm_training.py`), projections (done), valuations (done).

## Status

| Phase | Status |
|-------|--------|
| 1 — Label generation | not started |
| 2 — Classification model | not started |
| 3 — Calibration and evaluation | not started |
| 4 — CLI commands | not started |

## Phase 1: Label Generation

Define what constitutes a "breakout" and "bust" in terms of ADP-relative performance, and build the labeled dataset from historical seasons.

### Context

The feature infrastructure and GBM training pipeline already exist for projection models. This model needs a different target variable: instead of predicting raw stats, it predicts whether a player beats or misses their ADP-implied value by a significant margin. The first step is defining that margin and generating historical labels.

### Steps

1. Define domain types in `src/fantasy_baseball_manager/domain/breakout_bust.py`:
   - `OutcomeLabel` enum: `BREAKOUT`, `BUST`, `NEUTRAL`.
   - `LabeledSeason` frozen dataclass: `player_id: int`, `season: int`, `player_type: str`, `adp_rank: int`, `adp_pick: float`, `actual_value_rank: int`, `rank_delta: int`, `label: OutcomeLabel`.
   - `LabelConfig` frozen dataclass: `breakout_threshold: int = 30` (beat ADP rank by 30+), `bust_threshold: int = -30` (missed ADP rank by 30+), `min_adp_rank: int = 300` (only label players drafted in top 300).
2. Build `generate_labels()` in `src/fantasy_baseball_manager/services/breakout_bust.py`:
   - Accepts `adp: list[ADP]`, `actual_valuations: list[Valuation]`, `config: LabelConfig`.
   - For each player with both an ADP record and end-of-season valuation, compute `rank_delta = adp_rank - actual_value_rank` (positive = beat ADP = breakout direction).
   - Assign `BREAKOUT` if `rank_delta >= breakout_threshold`, `BUST` if `rank_delta <= bust_threshold`, else `NEUTRAL`.
   - Returns `list[LabeledSeason]`.
3. Build `assemble_labeled_dataset()` that joins labels with pre-season features:
   - Uses the existing feature assembler to pull features for `season - 1` (the information available before the draft).
   - Returns a DataFrame with features + label column.
4. Write tests verifying label assignment math and that features come from the prior season (no data leakage).

### Acceptance criteria

- Labels are generated for seasons where both ADP and actual valuations exist (likely 2016-2025).
- No feature from season N is used to label season N (strict temporal separation).
- Class distribution is reported: roughly 15-20% breakout, 15-20% bust, 60-70% neutral for reasonable thresholds.
- `LabelConfig` thresholds are configurable.

## Phase 2: Classification Model

Train a GBM classifier on the labeled dataset using the existing training infrastructure, with temporal cross-validation.

### Context

The `gbm_training.py` module already supports GBM training with temporal expanding CV, sample weights, and hyperparameter tuning. This phase adapts it for a classification target instead of regression. The model predicts `P(breakout)` and `P(bust)` for each player.

### Steps

1. Define model types in `domain/breakout_bust.py`:
   - `BreakoutPrediction` frozen dataclass: `player_id: int`, `player_name: str`, `player_type: str`, `position: str`, `p_breakout: float`, `p_bust: float`, `p_neutral: float`, `top_features: list[tuple[str, float]]` (SHAP or feature importance for this prediction).
2. Register a new model `breakout-bust` in the model registry following the pattern in `models/`:
   - Training target: multi-class classification (BREAKOUT / BUST / NEUTRAL).
   - Use LightGBM `multiclass` objective or treat as two binary classifiers (breakout vs. not, bust vs. not).
   - Feature set: reuse relevant feature groups from statcast-gbm-preseason (quality-of-contact, age, sprint speed, platoon splits, consistency metrics) plus ADP-derived features (ADP rank, position scarcity at ADP, ADP trend if available).
   - Temporal expanding CV: train on seasons 1..N-1, validate on season N.
3. Implement `train()` and `predict()` methods:
   - `train()` uses `gbm_training.py` utilities adapted for classification (log-loss instead of RMSE, stratified folds).
   - `predict()` outputs `list[BreakoutPrediction]` with calibrated probabilities.
4. Add per-prediction feature attribution using LightGBM's built-in feature importance or SHAP values to populate `top_features`.
5. Write tests for training on synthetic data, verifying output probabilities sum to 1.0 and temporal CV respects time ordering.

### Acceptance criteria

- Model trains on historical data with temporal CV (no future leakage).
- Predictions produce calibrated probabilities (`p_breakout + p_bust + p_neutral ≈ 1.0`).
- Log-loss on held-out seasons beats a naive base-rate classifier.
- `top_features` provides interpretable explanations for each prediction.

## Phase 3: Calibration and Evaluation

Evaluate the classifier's real-world usefulness: does it actually identify breakouts and busts at a rate better than chance, and are the probabilities well-calibrated?

### Context

A model that says "30% breakout probability" should be right about 30% of the time. Calibration and precision/recall at various thresholds determine whether the model is actionable for draft decisions.

### Steps

1. Build `evaluate_classifier()` in `services/breakout_bust.py`:
   - Accepts held-out `LabeledSeason` data and `BreakoutPrediction` predictions.
   - Computes: precision, recall, F1 for each class at various probability thresholds.
   - Computes calibration metrics: reliability diagram data (predicted probability vs. observed frequency in decile bins).
   - Computes lift: how much more likely is a player flagged as breakout to actually break out vs. base rate?
2. Build `historical_backtest()` that runs walk-forward evaluation:
   - For each season 2018-2025: train on prior seasons, predict current season, score against actuals.
   - Aggregate metrics across all test seasons.
3. Define actionability thresholds: "flag as breakout candidate if `p_breakout > X`" where X is tuned for precision >= 0.4 (at least 40% of flagged breakouts actually break out).
4. Write tests for evaluation metric computation with known inputs.

### Acceptance criteria

- Walk-forward backtest produces per-season and aggregate precision/recall.
- Lift over base rate is >= 1.5 for the top-20 breakout candidates (they break out 1.5x more often than average).
- Calibration reliability diagram shows reasonable alignment (not wildly overconfident).
- Actionability thresholds are documented and justified.

## Phase 4: CLI Commands

Expose breakout/bust predictions and evaluation through the CLI.

### Steps

1. Add `fbm report breakout-candidates --season <year>`:
   - Prints players ranked by `p_breakout` descending.
   - Columns: name, position, ADP rank, p_breakout, top contributing features.
   - Supports `--player-type batter|pitcher`, `--min-probability <float>`, `--top <n>`.
2. Add `fbm report bust-risks --season <year>`:
   - Prints players ranked by `p_bust` descending.
   - Same column structure and filters as breakout candidates.
3. Add `fbm evaluate breakout-bust --season <year>` for backtesting on a specific held-out season.
4. Add `fbm train breakout-bust` and `fbm predict breakout-bust` following existing model CLI patterns.
5. Register commands in `cli/app.py`.

### Acceptance criteria

- `fbm report breakout-candidates` prints a ranked list with probabilities and feature explanations.
- `fbm report bust-risks` prints a ranked list of players likely to disappoint.
- Training and prediction follow the existing model lifecycle (`prepare` / `train` / `predict`).
- Evaluation report includes precision, recall, and lift metrics.

## Ordering

Phase 1 is independent — it only needs ADP and valuation data that already exist. Phase 2 depends on phase 1 (needs labels) and the existing GBM infrastructure. Phase 3 depends on phase 2 (needs trained model). Phase 4 depends on all prior phases. The feature ablation roadmap (already in progress) may improve feature selection for this model but is not a blocker.
