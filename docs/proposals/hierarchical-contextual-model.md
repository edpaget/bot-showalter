# Hierarchical Contextual Model

## Goal

Replace the current single-stage contextual transformer with a two-stage hierarchical architecture that encodes games independently before composing them, providing a structurally sound inductive bias for multi-game player performance prediction.

## Current Architecture

The existing `ContextualPerformanceModel` uses a single transformer encoder over a flat sequence of pitch events from multiple games:

```
[CLS] [PLAYER_G1] pitch pitch ... [PLAYER_G2] pitch pitch ... → Transformer → CLS embedding → Head
```

### Problems with the current approach

1. **Pitch tokens attend cross-game.** Pitch tokens use fully-connected attention to all non-padding positions, meaning a pitch in game 3 directly attends to pitches in game 1. This creates noisy cross-game interactions that the model must learn to ignore.

2. **Player tokens are underutilized.** Player tokens are restricted to game-local attention, but since pitch tokens already carry cross-game information after layer 1, the hierarchical separation is undermined.

3. **CLS bottleneck.** The CLS token must aggregate information from hundreds of individual pitches across all games into a single `d_model` vector. It competes with direct pitch-level cross-game attention as a communication pathway, rather than being the sole aggregation point.

4. **Quadratic scaling.** A player with 5 games of 100 pitches each produces ~500 tokens. Self-attention cost is O(500^2) = 250k pairs. This limits how many games of context we can include.

## Proposed Architecture

A two-stage hierarchical transformer that respects game boundaries structurally:

```
Stage 1 (Intra-Game):     [CLS_G] pitch pitch ... → Game Encoder → game_embedding
Stage 2 (Cross-Game):     [CLS] game_emb_1 game_emb_2 ... → Context Encoder → CLS embedding → Head
```

### Stage 1: Intra-Game Encoder

A transformer encoder that processes each game independently:

- **Input:** `[GAME_CLS] + pitch events` for a single game
- **Attention:** Fully-connected within the game (no masking complexity)
- **Output:** The `[GAME_CLS]` hidden state serves as the game-level embedding
- **Weight sharing:** The same encoder processes all games (shared parameters)

This encoder can be pretrained with the existing MGM (Masked Gamestate Modeling) objective on individual games. The per-game `[GAME_CLS]` replaces the current per-game `[PLAYER]` token.

### Stage 2: Cross-Game Context Encoder

A smaller transformer that operates over game-level embeddings:

- **Input:** `[CLS] + game_embedding_1 + game_embedding_2 + ...`
- **Sequence length:** Number of games (typically 5-20), not number of pitches
- **Positional encoding:** Encodes temporal ordering of games (can use sinusoidal or learned, based on days-between-games for non-uniform spacing)
- **Output:** The `[CLS]` hidden state feeds the prediction head

### Prediction Head

Unchanged from current design — `PerformancePredictionHead` takes the final CLS embedding and produces `n_targets` predictions.

## Module Design

### New modules

| Module | Responsibility |
|--------|---------------|
| `model/intra_game_encoder.py` | Stage 1 transformer + per-game CLS extraction |
| `model/cross_game_encoder.py` | Stage 2 transformer over game embeddings |
| `model/hierarchical_model.py` | Top-level model composing both stages + head |

### Modified modules

| Module | Change |
|--------|--------|
| `model/tensorizer.py` | Tensorize per-game (list of games → list of per-game tensors) instead of flat sequence |
| `model/config.py` | Add stage 2 config (n_layers, n_heads, d_model for cross-game encoder) |
| `training/` | Update training loops for two-stage pretraining + fine-tuning |

### Removed modules

| Module | Reason |
|--------|--------|
| `model/mask.py` | No longer needed — stage 1 is fully-connected within each game, stage 2 is fully-connected over game embeddings |

## Training Strategy

### Phase 1: Intra-Game Pretraining (MGM)

Same as current pretraining, but each training example is a single game rather than a multi-game sequence. The MGM head predicts masked pitch type and result within the game. This is simpler and more data-efficient since every game in the dataset is an independent training example.

### Phase 2: Fine-Tuning

Two options:

**Option A: End-to-end fine-tuning.** Unfreeze both stages. The intra-game encoder is initialized from pretrained weights, the cross-game encoder is initialized randomly. Train with the performance prediction loss. This is simpler but slower to converge.

**Option B: Frozen stage 1.** Freeze the pretrained intra-game encoder and only train the cross-game encoder + head. Pre-extract game embeddings for the entire dataset, then train stage 2 as a lightweight model over cached embeddings. This is much faster and allows rapid experimentation with the cross-game architecture.

Option B is recommended for initial development; Option A can be explored later for marginal gains.

## Computational Benefits

| Metric | Current (flat) | Proposed (hierarchical) |
|--------|---------------|------------------------|
| Attention pairs (5 games, 100 pitches each) | O(500^2) = 250k | 5 x O(100^2) + O(5^2) = 50k |
| Max games in context (2048 seq_len) | ~8-10 games | ~200 games (limited by stage 2 seq_len) |
| MGM training examples per PlayerContext | 1 | N (one per game) |
| Fine-tuning with frozen encoder | Not possible | Pre-extract embeddings, train stage 2 only |

## Temporal Encoding Consideration

Games are not uniformly spaced — a player might have games on consecutive days or with a week gap (off days, injury, All-Star break). The cross-game encoder's positional encoding should reflect this. Options:

1. **Days-since-first-game** fed through sinusoidal encoding (analogous to current pitch-level positional encoding)
2. **Learned relative position** based on binned days-between-games
3. **Concatenate game metadata** (date, opponent, home/away) as additional features alongside the game embedding before stage 2

Option 1 is simplest and sufficient for an initial implementation.

## Migration Path

1. **Fix game-local attention** in the current single-stage model (prerequisite — validates the inductive bias)
2. **Extract the intra-game encoder** from the existing model (embedder + transformer with game-local attention)
3. **Build the cross-game encoder** as a new small transformer
4. **Compose into `HierarchicalContextualModel`** with the existing prediction heads
5. **Update tensorizer** to produce per-game tensor batches
6. **Update training loops** for two-phase training

Step 1 is valuable on its own and should be done regardless of whether the full hierarchical refactor proceeds.
