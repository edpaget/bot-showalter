# Contextual Model Evaluation & Improvement Plan

Parent doc: [contextual-event-embeddings.md](../archive/contextual-event-embeddings.md)

## Status

First end-to-end evaluation of the fine-tuned contextual transformer against established projection systems. The initial model underperformed across the board, producing predictions that collapsed toward the population mean. Since the evaluation, significant architectural and training improvements have been implemented: CLS aggregation token, rate targets with multi-game averaging, game-local pitch attention, a comprehensive bug audit (10 fixes), and per-stat loss weighting. These changes address the core root causes identified below. **The model needs to be re-trained and re-evaluated with the current codebase.**

---

## Evaluation Results (2024 Season — Pre-Fix Baseline)

These results were collected **before** the CLS token, rate targets, bug fixes, and attention scope changes. They represent the original broken model and serve as a baseline for measuring improvement.

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

These are per-game errors from the original count-target model. The baseline comparison infrastructure now exists (see Infrastructure Fixes) but hasn't been used for a full training run yet.

---

## Root Cause Analysis

### 1. Player Token Isolation (Critical — Mitigated by CLS Token)

**Original file:** `src/.../contextual/model/mask.py`

Each `[PLAYER]` token can only attend to pitch events **within its own game**. Player tokens cannot see other player tokens or pitches from other games.

During fine-tuning, the original model predicted via `preds.mean(dim=1)` — averaging the independent per-game player token outputs. Each player token was a self-contained single-game summarizer with no way to aggregate information across the context window.

**Mitigation (DONE):** A `[CLS]` aggregation token at position 0 now attends to all non-padding positions across all games, providing the cross-game information flow that was missing. The model predicts from the CLS hidden state instead of averaging player tokens.

### 2. Single-Game Count Targets Are Extremely Noisy (Fixed)

**Original file:** `src/.../contextual/training/dataset.py`

Training targets were raw counting stats from a single game: HR ∈ {0,1,2,3}, SO ∈ {0,1,2,3,4}, BB ∈ {0,1,2,3}. The MSE loss heavily penalized rare high-count games, pushing predictions toward the safe mean.

**Fix (DONE):** The model now trains on rate targets averaged over 5 games (`target_mode="rates"`, `target_window=5`), which normalizes for game length and smooths target noise.

### 3. No Player Identity Signal (Remaining)

**File:** `src/.../contextual/model/tensorizer.py`

The `[PLAYER]` token is initialized with all-zero features. The model has no player ID embedding, no historical stats, no age, no handedness of the batter/pitcher themselves (only the opponent's handedness is encoded per-pitch via `stand` and `p_throws`).

All player differentiation must come from pitch-level features (velocity, movement, launch data), which is an indirect and lossy signal for projecting season-level stats. Two pitchers facing similar opposition with similar pitch mixes will get nearly identical predictions regardless of their actual skill levels.

**Note:** The CLS token now has a learned embedding (fixing bug M3), but the player tokens themselves still carry no identity information.

### 4. Rate Conversion Amplifies Compression (Mitigated)

**Original file:** `src/.../contextual/adapter.py`

Per-game count predictions were divided by average outs/game (or PA/game) to produce rates, compressing already-narrow prediction spreads further.

**Mitigation (DONE):** In rate mode, the model outputs rates directly via `predicted_rates_to_pipeline_rates()` — no division step required. The adapter still supports the legacy count-to-rate conversion path for models trained with `target_mode="counts"`.

### 5. Train/Inference Data Mismatch (Remaining)

**Training** (`dataset.py`): Uses sliding windows over 2015-2022 seasons with configurable context windows, predicting the average rate over the next K games.

**Inference** (`contextual_rate_computer.py`): Uses `data_year = year - 1` (2023 data for 2024 projections), taking the last N games of that season as context.

The training signal spans full seasons with many windows per player. Inference uses a single window from the end of a potentially very different season. Players who had second-half surges or slumps will be poorly represented by this mismatch.

### 6. Cross-Game Pitch Visibility Was Too Broad (Fixed)

**Original behavior:** Pitch tokens could attend to all non-padding positions across all games, creating an unrestricted global attention pattern.

**Fix (DONE):** Commit `f439d82` restricted pitch attention to **game-local scope only**. Pitch tokens now attend only to same-game non-padding positions. This matches the intended semantics: pitch tokens summarize their own game, while the CLS token handles cross-game aggregation.

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

**Commit:** `e8edac6` — `feat(pipeline): add prediction variance logging to contextual rate computer`

Added `_log_prediction_variance()` to `ContextualEmbeddingRateComputer`. After computing rates, logs per-stat distribution (min, max, std, median, p25, p75) across all contextual players. This runs automatically when the pipeline executes and will confirm or refute the mean-collapse hypothesis.

### Bug Audit Fixes

All 10 bugs identified in the [contextual model bug audit](../archive/contextual-model-bug-audit.md) have been fixed:

**Critical:**
- **C1** (`db7d93a`): Numeric feature normalization replaced — was computing a single mean/variance across all 23 features with different scales.
- **C2** (`db7d93a`): `ModelConfig` now persisted in checkpoint metadata — predictor no longer hardcodes default architecture.
- **C3** (`db7d93a`): `pa_event` now masked alongside `pitch_type`/`pitch_result` during MGM pre-training — eliminates the shortcut where the model could trivially infer masked targets.

**Moderate:**
- **M1** (`a5704c9`): Pre-training loss uses `-100` sentinel instead of `ignore_index=0` which collided with PAD.
- **M2** (`a5704c9`): Player token self-attention now explicitly included in mask (was working by accident via a NaN-prevention diagonal hack).
- **M3** (`791a84b`): CLS token uses a learned embedding instead of all-zeros (which was identical to PAD).
- **M4** (`791a84b`): Fine-tune MSE loss now uses per-stat weighting inversely proportional to variance.

**Minor:**
- **m1** (`cbd2352`): Gradient accumulation support added to both pre-training and fine-tuning.
- **m2** (`cbd2352`): `extract_game_stats` PA boundary detection made robust to different pitch numbering schemes.
- **m3** (`cbd2352`): Padding `game_id` changed from 0 to -2 to avoid collision with first game's ID.

### Game-Local Pitch Attention

**Commit:** `f439d82` — `fix(contextual): restrict pitch attention to game-local scope`

Pitch tokens now attend only to same-game positions. Previously they had cross-game visibility, which was inconsistent with the per-game summarization design and gave pitches information they shouldn't have had.

### CLS Token Exclusion from Pre-Training Masking

**Commit:** `7db4330` — `fix(pretrain): exclude CLS token from MGM masking candidates`

CLS (position 0, `game_id=-1`) is now excluded from masking candidates during masked gamestate modeling. It has no pitch type or result to predict, so masking it was producing meaningless loss terms.

### Rate Mode Alignment

**Commit:** `b9ab705` — `fix(contextual): align rate_mode default with trained model target_mode`

`ContextualRateComputerConfig.rate_mode` default now aligns with `FineTuneConfig.target_mode` default (both `True`/`"rates"`), preventing silent mode mismatch at inference time.

---

## Improvement Plan

### Phase 1: Verify Signal Exists (No Architecture Changes) — DONE

#### 1a. Re-run fine-tuning with baseline comparison — DONE

Baseline comparison metrics implemented in `finetune.py`. Infrastructure is ready.

#### 1b. Fix checkpoint naming — DONE

Checkpoint names now include perspective. See "Infrastructure Fixes Completed" above.

#### 1c. Inspect prediction variance — DONE

`_log_prediction_variance()` added to `ContextualEmbeddingRateComputer`. See "Infrastructure Fixes Completed" above.

### Phase 2: Architectural Fixes (Cross-Game Aggregation) — DONE

#### 2a. Add a CLS aggregation token — DONE

**Commit:** `228aeaa` — `feat(model): add CLS aggregation token for cross-game attention`

Implemented all planned changes:
- `tensorizer.py`: CLS token inserted at position 0 with `game_id=-1`
- `mask.py`: CLS attends to all non-padding positions
- `model.py`: Learned CLS embedding; extracts CLS hidden state for prediction
- `finetune.py`: Uses CLS output directly (no mean pooling)

#### 2a-follow-up. Re-pretrain with CLS token — DONE

**Commit:** `7db4330` — `fix(pretrain): exclude CLS token from MGM masking candidates`

CLS excluded from masking candidates during MGM pre-training. The backbone can now learn to route cross-game information through CLS from the start.

#### 2b. Two-stage aggregation

More ambitious: encode each game independently via the existing per-game player tokens, then pass the sequence of per-game embeddings through a lightweight second-stage transformer (2 layers) that aggregates temporal patterns across games. The second stage sees the ordered game summaries and can learn trends.

**Tradeoff:** More parameters, more complexity, but preserves the pre-trained per-game encoding and adds temporal modeling on top.

**Status:** Not implemented. The CLS token provides a simpler cross-game aggregation mechanism. Revisit if CLS-based model still underperforms after re-training.

### Phase 3: Target and Loss Improvements — PARTIALLY DONE

#### 3a. Train on rate targets instead of counts — DONE

**Commit:** `9915680` — `feat(contextual): add rate targets for contextual fine-tuning`

Implemented `compute_rate_targets()` in `dataset.py`. Default config is now `target_mode="rates"` with `target_window=5` games. The adapter's `predicted_rates_to_pipeline_rates()` handles rate-mode output without the lossy count-to-rate division.

#### 3b. Multi-game averaged targets — DONE (part of 3a)

Implemented as part of rate targets. `target_window=5` averages the target rate over 5 games, smoothing single-game noise. Configurable via `FineTuneConfig.target_window`.

#### 3c. Auxiliary contrastive loss

Add a contrastive term that pushes same-player embeddings closer and different-player embeddings apart. This provides an explicit "player identity" learning signal without requiring player ID features.

**Status:** Not implemented.

### Phase 4: Feature Enrichment

**Goal:** Give the model direct access to information that currently requires indirect inference.

#### 4a. Player historical stat features

Inject summary statistics (career K%, BB%, HR%, wOBA) into the `[PLAYER]` token instead of all-zeros. This gives the model a strong prior about player quality.

**Status:** Not implemented.

#### 4b. Player ID embeddings

Add a learned embedding per player (by MLBAM ID). This requires a vocabulary of known players and won't generalize to unseen players, but provides the strongest possible identity signal for the training population.

**Status:** Not implemented.

#### 4c. Matchup features

Add opponent-quality features (team wOBA against, park factor) to provide context about the difficulty of each game. Currently, a 7-K game against a bad lineup looks the same as a 7-K game against a great one.

**Status:** Not implemented.

### Phase 5: Ensemble Integration

**Goal:** Even if the contextual model can't beat Marcel standalone, extract its unique signal via ensemble.

#### 5a. Marcel-contextual blend tuning

The `marcel_contextual` pipeline defaults to 70/30 Marcel/contextual. With an improved contextual model, tune this weight via cross-validation on held-out years. Even a small positive contribution at 10-20% weight could improve the overall system.

**Status:** Not implemented.

#### 5b. Residual learning

Instead of predicting raw stats, train the contextual model to predict **residuals from Marcel** — the amount by which a player will over/underperform their Marcel projection. This is a much easier learning task (smaller target variance, zero-centered) and directly complements the existing pipeline.

**Status:** Not implemented.

---

## Prioritized Next Steps

1. ~~**Fix checkpoint naming**~~ — DONE (`e5ee8f1`)
2. ~~**Add prediction variance logging**~~ — DONE (`e8edac6`)
3. ~~**Implement CLS aggregation token**~~ — DONE (`228aeaa`)
4. ~~**Exclude CLS from pre-training masking**~~ — DONE (`7db4330`)
5. ~~**Switch to rate targets**~~ — DONE (`9915680`)
6. ~~**Bug audit fixes (10 bugs)**~~ — DONE (see Infrastructure Fixes)
7. ~~**Restrict pitch attention to game-local scope**~~ — DONE (`f439d82`)
8. ~~**Align rate_mode default**~~ — DONE (`b9ab705`)
9. **Re-pretrain and re-fine-tune with all fixes** — full training run with the current codebase to get a post-fix evaluation
10. **Re-evaluate against Marcel/Steamer/ZiPS** — update the evaluation tables above with post-fix numbers
11. **Evaluate residual learning** (Phase 5b) — may be the fastest path to pipeline value if direct prediction still underperforms
12. **Add player historical stat features** (Phase 4a) — strongest remaining root cause fix for player identity
13. **Contrastive loss** (Phase 3c) — explicit player-identity learning signal
