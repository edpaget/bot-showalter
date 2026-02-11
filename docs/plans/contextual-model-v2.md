# Contextual Model v2 Roadmap

## Overview

The contextual transformer has completed its first full train-evaluate cycle with all architectural fixes (CLS token, rate targets, bug audit, game-local attention). Results show the model learns real per-game signal — pitcher fine-tuning beats baseline on all 4 stats (21–30% MSE reduction) — but season-level projections still underperform Marcel GB due to mean collapse and lack of player identity signal. This roadmap focuses on the highest-leverage improvements to close the gap: residual learning, player identity features, fine-tuning stability, and batter-specific attention.

## Current State

**What works:**
- Pre-training (MGM) and fine-tuning pipeline end-to-end functional
- CLS aggregation token provides cross-game information flow
- Rate targets with 5-game averaging smooth single-game noise
- Pitcher fine-tuning beats context-mean baseline on SO, H, BB, HR
- Evaluation harness compares contextual vs Marcel GB vs Steamer

**What doesn't work:**
- Season-level correlation collapses (pitcher K corr 0.658 vs Marcel's 0.696, ERA 0.193 vs 0.206)
- Batter model worse than baseline on SO and HR — net negative at season level
- No player identity signal — player tokens carry zero features, all differentiation comes from pitch data
- Full backbone unfreezing during fine-tuning risks catastrophic forgetting, especially for batters
- OBP correlation collapsed to 0.230 (from Marcel's 0.505) due to compounding SO/BB errors

**Key files:**
- `src/fantasy_baseball_manager/contextual/training/finetune.py` — fine-tune trainer
- `src/fantasy_baseball_manager/contextual/training/dataset.py` — training sample construction, `compute_rate_targets()`
- `src/fantasy_baseball_manager/contextual/model/model.py` — transformer + prediction head
- `src/fantasy_baseball_manager/contextual/model/tensorizer.py` — token construction
- `src/fantasy_baseball_manager/contextual/model/mask.py` — attention patterns
- `src/fantasy_baseball_manager/contextual/adapter.py` — prediction-to-rate conversion
- `src/fantasy_baseball_manager/pipeline/stages/contextual_rate_computer.py` — pipeline integration
- `src/fantasy_baseball_manager/contextual/training/config.py` — all config dataclasses

## Phases

### Phase 1: Residual Learning

**Goal:** Reframe the prediction task from absolute rates to Marcel residuals, making the model an adjustment layer rather than a standalone projection system.

**Rationale:** The model learns per-game signal but can't express it as absolute season projections that differentiate players. Predicting residuals (actual - Marcel) is an easier task: smaller target variance, zero-centered, and the model only needs to detect deviations from a strong prior. Conservative near-zero predictions default to Marcel — the model can only help, not hurt.

**Scope:**
- Add `target_mode="residuals"` to `FineTuneConfig` alongside existing `"counts"` and `"rates"` modes
- Modify `compute_rate_targets()` in `dataset.py` to accept Marcel rates per player and compute `target = actual_rate - marcel_rate`
- Build a data pipeline that pairs each training window with the corresponding Marcel projection for that player-year
- Update the adapter to add residual predictions back onto Marcel rates at inference time
- Add a `residual_mode` flag to `ContextualRateComputerConfig` that controls the inference-time addition
- Re-train pitcher and batter models with residual targets
- Evaluate against Marcel GB and Steamer

**Key files:** `dataset.py`, `config.py`, `adapter.py`, `contextual_rate_computer.py`, `finetune.py`

**Dependencies:** Marcel projections accessible during training data construction. The Marcel pipeline already exists; need to generate and cache Marcel rates for each training season (2015–2022).

### Phase 2: Player Historical Stat Features

**Goal:** Give the model direct access to player identity through career/recent stat summaries injected into player tokens.

**Rationale:** Root cause #3 from the evaluation — player tokens are initialized with all-zero features. Two pitchers with identical recent pitch data but vastly different career numbers get the same prediction. Injecting career K%, BB%, HR%, wOBA (and age) into the player token gives the model a strong prior for each player. This pairs naturally with residual learning: the model sees "this pitcher has a career 25% K rate" and adjusts from Marcel based on recent pitch-level trends.

**Scope:**
- Define a set of player summary features (career and recent-season rates for the target stats, plus age)
- Extend the tensorizer to populate player token numeric features from a player stats lookup
- Build a data source that provides historical stats per player-year for training
- Update `GameSequenceBuilder` or the fine-tune dataset to attach player stats to each training sample
- Extend `ContextualRateComputerConfig` to enable/disable player features
- Re-train and evaluate

**Key files:** `tensorizer.py`, `dataset.py`, `config.py`, `model.py` (embedding dimension may change)

**Dependencies:** Player historical stats available per player-year. The Marcel pipeline computes weighted historical rates — these could be reused directly.

### Phase 3: Gradual Unfreezing

**Goal:** Protect lower-layer pitch-physics representations from corruption during fine-tuning while allowing upper layers to adapt.

**Rationale:** The batter model's SO and HR regression below baseline suggests catastrophic forgetting — the backbone is being damaged during fine-tuning on weak batter signal. The current config has `freeze_backbone=False`, unfreezing everything at once. Gradual unfreezing (ULMFiT-style) is well-established to outperform both full-freeze and full-unfreeze in low-data regimes.

**Scope:**
- Add an unfreezing schedule to `FineTuneConfig` (e.g., `unfreeze_schedule: list[int]` — epochs at which to unfreeze each layer group, top-down)
- Modify the fine-tune trainer to manage `requires_grad` per layer group across epochs
- Define layer groups: prediction head (always unfrozen) → top transformer layer → middle layers → bottom layers → embedder
- Default schedule: head-only for first N epochs, then progressively unfreeze from top
- Compare against current full-unfreeze baseline for both pitcher and batter models

**Key files:** `finetune.py`, `config.py`

**Dependencies:** None — purely a training strategy change. Should be evaluated alongside Phase 1 and 2 improvements.

### Phase 4: Batter Context Window Tuning

**Goal:** Validate whether the current 30-game batter context window is optimal, or if expanding to 40–50 games improves batter signal.

**Rationale:** A pitcher sees ~100 batters across 10 starts; a batter sees ~40 PA across 10 games. The config already uses asymmetric windows (batter=30, pitcher=10), but the batter fine-tuning results suggest 30 games may still not provide enough event exposure. This is trivial to experiment with.

**Scope:**
- Run batter fine-tuning with context windows of 30, 40, and 50 games
- Compare per-stat MSE/MAE against baseline for each window size
- Update defaults if a larger window improves results
- Verify positional encoding handles the longer sequences (max sequence length implications)

**Key files:** `config.py`, potentially `model.py` (positional encoding range)

**Dependencies:** None. Can be run as a hyperparameter sweep independently of other phases.

### Phase 5: Event-Weighted Attention

**Goal:** Bias batter-side attention toward PA outcome events (HR, SO, BB, hits) and away from intermediate pitches (fouls, balls, called strikes).

**Rationale:** For batter projection, the signal is overwhelmingly in PA outcomes, not in pitch-by-pitch count progression. Uniform attention dilutes outcome signal with noise from foul balls. Adding a learnable bias toward PA-ending events gives the model a structural hint about where to focus.

**Scope:**
- Add a learnable scalar attention bias per event type in the attention mask computation
- Flag PA-ending pitch positions in the tensorized sequence (already available via `pa_event`)
- Apply the bias to attention logits before softmax, only for batter-perspective sequences
- Train and evaluate the impact on batter fine-tuning metrics

**Key files:** `mask.py`, `model.py`, `tensorizer.py`

**Dependencies:** Phases 1–3 should be evaluated first. This is a targeted batter-specific optimization — worth pursuing only if batters are still the weak link after residual learning and player features.

### Phase 6: Ensemble Integration

**Goal:** Tune the Marcel-contextual blend weight via cross-validation to extract maximum pipeline value from the improved contextual model.

**Rationale:** Even if the contextual model doesn't beat Marcel standalone, it may capture complementary signal. The `marcel_contextual` pipeline already supports weighted blending (default 70/30). With residual learning, the natural ensemble is additive: `final = marcel + weight * contextual_residual`. Cross-validation on held-out years determines the optimal weight.

**Scope:**
- Implement cross-validation weight tuning using the evaluation harness across 2021–2024
- Test blend weights from 0.0 to 1.0 in increments of 0.1 for both pitcher and batter
- Allow per-perspective and potentially per-stat blend weights
- Update the `marcel_contextual` pipeline defaults with the tuned weights
- Final evaluation comparing tuned ensemble vs Marcel GB vs Steamer

**Key files:** `pipeline/presets.py`, `evaluation/harness.py`, `pipeline/stages/contextual_rate_computer.py`

**Dependencies:** Phases 1–2 completed and models re-trained. The ensemble is only worth tuning once the contextual model is contributing positive signal.

## Open Questions

- **Marcel rate availability during training:** Generating Marcel projections for each training season (2015–2022) requires running the Marcel pipeline retrospectively. How expensive is this, and should the results be cached to disk?
- **Player feature set for Phase 2:** Should features be career-weighted (Marcel-style 5/4/3 weighting) or simple recent-season averages? Career-weighted is more principled but adds a Marcel dependency to the training data pipeline.
- **Batter model viability:** If residual learning + player features still can't make the batter model contribute positively, should we abandon batter contextual predictions entirely and focus the model on pitchers only?
- **Re-pretraining:** Phases 2 and 5 change the input representation (player token features, attention biases). Do these require re-pretraining the backbone, or can fine-tuning alone adapt?
