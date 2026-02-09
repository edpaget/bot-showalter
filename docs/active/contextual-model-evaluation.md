# Contextual Model Evaluation & Improvement Plan

Parent doc: [contextual-event-embeddings.md](contextual-event-embeddings.md)

## Status

First end-to-end evaluation of the fine-tuned contextual transformer against established projection systems. The model underperforms across the board, producing predictions that collapse toward the population mean rather than differentiating players. This doc captures the evaluation results, root cause analysis, and a prioritized improvement plan.

---

## Evaluation Results (2024 Season)

### Batting

| Stat | Contextual | Marcel GB | Steamer | ZiPS |
|------|-----------|-----------|---------|------|
| HR Corr | **0.526** | 0.692 | 0.692 | 0.689 |
| HR RMSE | 7.90 | 8.73 | 7.33 | 7.00 |
| R Corr | 0.542 | 0.542 | 0.624 | 0.595 |
| RBI Corr | 0.567 | 0.567 | 0.621 | 0.599 |
| SB Corr | 0.669 | 0.739 | 0.669 | 0.741 |
| OBP Corr | **0.167** | 0.505 | 0.550 | 0.537 |
| Rank rho | 0.499 | 0.581 | 0.608 | 0.609 |
| Top-20 | 0.300 | 0.550 | 0.500 | 0.450 |

### Pitching

| Stat | Contextual | Marcel GB | Steamer | ZiPS |
|------|-----------|-----------|---------|------|
| W Corr | 0.472 | 0.472 | 0.606 | 0.498 |
| K Corr | **0.606** | 0.696 | 0.726 | 0.683 |
| ERA Corr | **0.141** | 0.206 | 0.368 | 0.262 |
| WHIP Corr | **0.114** | 0.237 | 0.406 | 0.341 |
| NSVH Corr | 0.719 | 0.719 | 0.774 | 0.768 |
| Rank rho | 0.346 | 0.420 | 0.484 | 0.443 |
| Top-20 | 0.200 | 0.400 | 0.300 | 0.350 |

### Key Observations

- Stats falling back to Marcel (R, RBI, SB for batters; W, NSVH for pitchers) match Marcel GB exactly, confirming the fallback path works correctly.
- Stats where the contextual model replaces Marcel are **uniformly worse**: HR correlation drops from 0.692 to 0.526, OBP from 0.505 to 0.167, K from 0.696 to 0.606, ERA from 0.206 to 0.141.
- The model appears to predict a narrow range near the population mean, destroying inter-player variance. RMSE can be competitive (HR RMSE 7.90 vs Marcel's 8.73) while correlation is poor — classic mean-prediction behavior.

### Raw Fine-Tuning Output (Pitcher)

```
Val loss: 1.2981
  so: MSE=2.0422  MAE=0.9967
  h: MSE=2.1858  MAE=1.0635
  bb: MSE=0.6954  MAE=0.6269
  hr: MSE=0.2691  MAE=0.3371
```

These are per-game errors. Without comparing against the context-mean baseline (added in the same session, but not yet used for a training run), we can't confirm whether the model is beating the naive "predict the recent average" strategy. This is the first thing to verify.

---

## Root Cause Analysis

### 1. Player Token Isolation (Critical — Architectural)

**File:** `src/.../contextual/model/mask.py:44-57`

Each `[PLAYER]` token can only attend to pitch events **within its own game**. Player tokens cannot see other player tokens or pitches from other games. The attention rules are:

- Pitch tokens attend to all non-padding positions (cross-game visibility)
- Player tokens attend only to same-game, non-player, non-padding positions + self

During fine-tuning, the model predicts via `preds.mean(dim=1)` — averaging the independent per-game player token outputs. Each player token is a self-contained single-game summarizer with no way to aggregate information across the context window.

**Impact:** The model learns "given one game's pitches, what stats does the average player produce next?" rather than "given this player's recent game sequence, what will they do next?" Cross-game trends (hot streaks, fatigue, pitch mix evolution) are invisible at the player-token level.

**Note:** The paper's architecture has the same constraint — player embeddings attend only to their own game's events. The difference is the paper used this for per-game prediction *within the context window*, while our fine-tuning predicts the *next* game. The information bottleneck is that each player token summarizes one game in isolation, and the simple mean-pooling across games discards temporal ordering entirely.

### 2. Single-Game Count Targets Are Extremely Noisy

**File:** `src/.../contextual/training/dataset.py:352-371`

Training targets are raw counting stats from a single game: HR ∈ {0,1,2,3}, SO ∈ {0,1,2,3,4}, BB ∈ {0,1,2,3}. For pitchers:

- Most games have 0 HR allowed — predicting the mean (~0.3) minimizes MSE
- SO varies widely (2-12 per start) — high variance even for the same pitcher
- The MSE loss heavily penalizes rare high-count games, pushing predictions toward the safe mean

This is the regression-to-the-mean problem: when per-sample noise dominates signal, MSE-optimal predictions collapse toward the population average.

### 3. No Player Identity Signal

**File:** `src/.../contextual/model/tensorizer.py:131-138`

The `[PLAYER]` token is initialized with all-zero features. The model has no player ID embedding, no historical stats, no age, no handedness of the batter/pitcher themselves (only the opponent's handedness is encoded per-pitch via `stand` and `p_throws`).

All player differentiation must come from pitch-level features (velocity, movement, launch data), which is an indirect and lossy signal for projecting season-level stats. Two pitchers facing similar opposition with similar pitch mixes will get nearly identical predictions regardless of their actual skill levels.

### 4. Rate Conversion Amplifies Compression

**File:** `src/.../contextual/adapter.py:113-116`

Per-game predictions are divided by average outs/game (or PA/game) to produce rates:

```python
rates["so"] = preds.get("so", 0.0) / denom
```

If the model predicts K/game in a narrow band (say 4.5–5.5 for all pitchers), dividing by ~18 outs/game yields K rates of 0.25–0.31. A 1.0 K/game spread in predictions becomes a 0.06 spread in rates — much smaller than the actual player variance.

### 5. Train/Inference Data Mismatch

**Training** (`dataset.py`): Uses sliding windows over 2015-2022 seasons with `context_window=10` games, predicting game N+1's stats from games 0..N-1.

**Inference** (`contextual_rate_computer.py:68,111`): Uses `data_year = year - 1` (2023 data for 2024 projections), taking the last N games of that season as context.

The training signal spans full seasons with many windows per player. Inference uses a single window from the end of a potentially very different season. Players who had second-half surges or slumps will be poorly represented by this mismatch.

---

## Infrastructure Fixes Completed

### Baseline Comparison in Fine-Tuning Output

**Commit:** `cb8e029` — `feat(finetune): add context-mean baseline comparison to eval output`

Added a context-window-mean baseline to validation metrics. For each sample, the baseline predicts the average of the context games' stats. The CLI now shows:

```
hr: MSE=0.2691  MAE=0.3371  (baseline MSE=0.2900  MAE=0.3500)
```

This answers the threshold question: is the model learning *anything* beyond the recent average?

### Model Loading Fix

**Commit:** `7ae82ad` — `fix(persistence): add map_location="cpu" to all torch.load calls`

Added `map_location="cpu"` to all `torch.load` calls in `persistence.py` so models trained on CUDA can load on CPU-only machines.

### Checkpoint Naming Fix

**Commit:** `e5ee8f1` — `fix(finetune): include perspective in checkpoint names`

Checkpoint names now include the perspective (`finetune_{perspective}_best`, `finetune_{perspective}_latest`, `finetune_{perspective}_epoch_N`) so batter and pitcher models don't overwrite each other.

### Prediction Variance Logging

Added `_log_prediction_variance()` to `ContextualEmbeddingRateComputer`. After computing rates, logs per-stat distribution (min, max, std, median, p25, p75) across all contextual players. This runs automatically when the pipeline executes and will confirm or refute the mean-collapse hypothesis.

---

## Improvement Plan

### Phase 1: Verify Signal Exists (No Architecture Changes)

**Goal:** Determine whether the model learns any player-discriminating signal at all, or whether the architecture structurally prevents it.

#### 1a. Re-run fine-tuning with baseline comparison

Re-train both batter and pitcher models using the updated code that reports baseline metrics. If the model's MSE/MAE doesn't beat the context-mean baseline, the model is not learning — skip to Phase 2.

#### 1b. Fix checkpoint naming — DONE

Checkpoint names now include perspective. See "Infrastructure Fixes Completed" above.

#### 1c. Inspect prediction variance — DONE

`_log_prediction_variance()` added to `ContextualEmbeddingRateComputer`. See "Infrastructure Fixes Completed" above. Output will appear in logs when running the projection pipeline with contextual models loaded.

### Phase 2: Architectural Fixes (Cross-Game Aggregation)

**Goal:** Enable the model to learn cross-game player patterns.

#### 2a. Add a CLS aggregation token

Insert a single `[CLS]` token at position 0 that can attend to all non-padding positions across all games. Use this for prediction instead of averaging per-game player tokens. This is the simplest change that enables cross-game information flow.

**Changes:**
- `tensorizer.py`: Insert CLS token at position 0 with its own game_id
- `mask.py`: CLS attends to all non-padding positions; all tokens can attend to CLS
- `model.py`: Extract CLS hidden state for prediction instead of averaging player tokens
- `finetune.py`: Use CLS output directly (no mean pooling)

#### 2b. Two-stage aggregation

More ambitious: encode each game independently via the existing per-game player tokens, then pass the sequence of per-game embeddings through a lightweight second-stage transformer (2 layers) that aggregates temporal patterns across games. The second stage sees the ordered game summaries and can learn trends.

**Tradeoff:** More parameters, more complexity, but preserves the pre-trained per-game encoding and adds temporal modeling on top.

### Phase 3: Target and Loss Improvements

**Goal:** Reduce target noise and give the model a better optimization landscape.

#### 3a. Train on rate targets instead of counts

Instead of predicting raw counts (HR=1, SO=7), predict per-PA or per-out rates. This normalizes for game length and reduces variance.

#### 3b. Multi-game averaged targets

Instead of predicting the next single game, predict the average stats over the next K games (e.g., K=3 or K=5). This smooths the target distribution and increases the signal-to-noise ratio.

**Tradeoff:** Fewer training samples (need K additional games after each window).

#### 3c. Auxiliary contrastive loss

Add a contrastive term that pushes same-player embeddings closer and different-player embeddings apart. This provides an explicit "player identity" learning signal without requiring player ID features.

### Phase 4: Feature Enrichment

**Goal:** Give the model direct access to information that currently requires indirect inference.

#### 4a. Player historical stat features

Inject summary statistics (career K%, BB%, HR%, wOBA) into the `[PLAYER]` token instead of all-zeros. This gives the model a strong prior about player quality.

#### 4b. Player ID embeddings

Add a learned embedding per player (by MLBAM ID). This requires a vocabulary of known players and won't generalize to unseen players, but provides the strongest possible identity signal for the training population.

#### 4c. Matchup features

Add opponent-quality features (team wOBA against, park factor) to provide context about the difficulty of each game. Currently, a 7-K game against a bad lineup looks the same as a 7-K game against a great one.

### Phase 5: Ensemble Integration

**Goal:** Even if the contextual model can't beat Marcel standalone, extract its unique signal via ensemble.

#### 5a. Marcel-contextual blend tuning

The `marcel_contextual` pipeline defaults to 70/30 Marcel/contextual. With an improved contextual model, tune this weight via cross-validation on held-out years. Even a small positive contribution at 10-20% weight could improve the overall system.

#### 5b. Residual learning

Instead of predicting raw stats, train the contextual model to predict **residuals from Marcel** — the amount by which a player will over/underperform their Marcel projection. This is a much easier learning task (smaller target variance, zero-centered) and directly complements the existing pipeline.

---

## Prioritized Next Steps

1. ~~**Fix checkpoint naming**~~ — DONE (`e5ee8f1`)
2. **Re-train with baseline comparison** — answers whether model learns anything (1 training run)
3. ~~**Add prediction variance logging**~~ — DONE (prediction variance logging)
4. **Run projection pipeline and inspect variance logs** — confirms mean-collapse hypothesis
5. **Implement CLS aggregation token** (Phase 2a) — highest expected impact
6. **Switch to rate targets** (Phase 3a) — reduces noise, simple change
7. **Evaluate residual learning** (Phase 5b) — may be the fastest path to pipeline value
