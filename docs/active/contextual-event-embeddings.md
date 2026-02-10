# Contextual Event Embeddings for Player Projection

## Paper Reference

Heaton & Mitra, "Learning Contextual Event Embeddings to Predict Player Performance in the MLB" (Penn State / Leibniz U Hannover). Code: https://github.com/cheat16/contextual_performance_prediction

## What

Train a transformer model on pitch-level play-by-play sequences to learn contextual representations of baseball events, then fine-tune those representations to project player performance. The core idea: instead of treating all singles (or strikeouts, or walks) as identical counting-stat increments, learn embeddings that capture *how* events happen — game state, pitch characteristics, batted ball quality, base-runner movement — and use that richer signal for projection.

This is the "bag-of-events to contextual embeddings" transition, analogous to NLP's bag-of-words to BERT evolution.

## Why

Our current pipeline (Marcel, Statcast adjusters, gradient boosting residuals) operates entirely on aggregated counting stats and rate metrics. This leaves several information sources on the table:

1. **Event context is discarded.** A strikeout with bases loaded and two outs in a one-run game is the same as a strikeout in a 10-0 blowout with nobody on. Context matters for understanding a player's true ability profile.

2. **Sequencing within games is lost.** Whether a pitcher's velocity faded in the 6th inning, whether a batter adjusted pitch selection after the first at-bat — none of this survives aggregation into season-level rates.

3. **Statcast data is underused.** We currently blend xwOBA-derived adjustments as a scalar correction. A contextual model consumes pitch speed, spin, location, exit velocity, and launch angle *per event*, preserving the full joint distribution rather than compressing to summary statistics.

4. **Short-sample projections are weak.** For in-season projections (matchup analysis, start/sit, ROS updates), 10 games of contextual play-by-play could provide better signal than the same 10 games reduced to counting stats.

The paper demonstrates competitive-with-sportsbooks predictions from just 10 games of context, trained on only 6 years of data. The pitcher strikeout model achieved R^2=0.21 and MAE=1.85 (vs best sportsbook R^2=0.23, MAE=1.80), while making 50% more predictions. Batter predictions were weaker (R^2=0.04-0.06) but the authors attribute this to batters participating in ~1/9th the events per game.

## How It Works (Paper Summary)

### Data Representation

Each pitch in a game is represented as a vector combining:

- **Gamestate delta** (learnable embedding): change in base occupancy, ball-strike count, outs, score
- **Pitch type** (learnable embedding): fastball, slider, changeup, etc.
- **Event type** (learnable embedding): strikeout, single, flyout, etc.
- **Statcast numerics**: pitch speed, spin rate, location (x/y), exit velocity, launch angle, distance
- **Player sabermetrics** (~5% of vector): pitcher/batter/matchup stats at 15-day, season, and career scopes

A game is a sequence of these pitch vectors. Multiple games are concatenated to form the full input sequence (N=10 games).

### Architecture

Standard transformer encoder (multi-head attention layers). Two input modes:

- **Team-batting sequence**: N games of a team's offensive play, from which individual batter embeddings are derived
- **Pitcher sequence**: N games of an individual pitcher's starts

Special **player embeddings** are prepended to the event sequence. Attention masking ensures each player embedding only attends to events that player participated in.

### Training

**Phase 1 — Pre-training (Masked Gamestate Modeling):** Analogous to BERT's masked language modeling. 15% of pitch vectors are randomly masked; the model predicts the missing gamestate and event type. This teaches the model the structure and progression of baseball games.

**Phase 2 — Fine-tuning:** Given player embeddings from the previous N games, predict per-game stats (strikeouts, hits, walks) via a linear projection head.

### Key Results (2021 Season)

| Target | Pitcher R^2 | Pitcher MAE | Batter R^2 | Batter MAE |
|--------|-------------|-------------|------------|------------|
| Strikeouts | 0.21 | 1.85 | 0.06 | 0.57 |
| Hits | 0.07 | 1.72 | 0.04 | 0.57 |
| Walks | 0.04 | 0.98 | 0.02 | 0.29 |

Pitcher strikeout predictions were competitive with 3 major US sportsbooks while covering 50% more games. The embedding visualizations confirmed the model learns contextually meaningful representations: singles are differentiated by outs, base occupancy, and runner movement; strikeouts encode leverage situations.

## Pipeline Fit

This doesn't slot cleanly into the existing 4-stage pipeline (rate compute → adjust → playing time → finalize). Instead, it operates as an **alternative projection engine** — a fundamentally different approach that produces player-level stat predictions directly from play-by-play data.

### Integration Options

#### Option A: Standalone Projection Engine (Recommended)

Register as a new `ProjectionPipeline` entry alongside Marcel. The transformer produces raw stat predictions per game; a thin adapter layer converts these to season-rate projections compatible with downstream valuation.

```
play-by-play data → Transformer → per-game predictions → aggregate to season rates
                                                          ↓
                                    existing valuation/ranking pipeline
```

This fits the existing `ProjectionPipeline` interface if we implement a `ContextualEmbeddingRateComputer` that:
- Fetches play-by-play data via a new `DataSource[PitchEvent]`
- Runs the transformer model to produce player embeddings
- Converts embeddings to rate predictions via the fine-tuned head
- Returns `list[PlayerRates]` like any other `RateComputer`

The remaining pipeline stages (adjusters, playing time, finalizer) would still apply, though some adjusters (like Statcast blending) would be redundant since the model already consumes Statcast features directly.

#### Option B: Embedding-Based Rate Adjuster

Use the pre-trained model to produce player embeddings, then feed those as features into the existing gradient boosting residual model or as a new `RateAdjuster`. This is lower-risk but captures less of the model's potential.

#### Option C: Ensemble with Marcel

Run both Marcel and the contextual model independently, then blend predictions. This provides the stability of Marcel with the contextual signal from the transformer. A learned or fixed blending weight per stat category controls the mix.

### Recommendation

Start with **Option A** as a standalone engine for evaluation, then explore **Option C** for production use. The ensemble approach is likely to outperform either model alone since they exploit different information (Marcel: decades of counting stats; contextual: recent play-by-play detail).

## Data Requirements

### Primary: Pitch-Level Play-by-Play

- Source: Statcast via `pybaseball.statcast()` — available from 2015+
- Volume: ~700K pitches/season, ~4.2M pitches for 6 training years
- Fields needed per pitch: pitch type, speed, spin rate, location (plate_x, plate_z), zone, batted ball exit velocity, launch angle, hit distance, events, description, game state (balls, strikes, outs, runners on base, score)
- Storage: ~2-3 GB as parquet per season

### Secondary: Player Sabermetrics

- Already available via existing `PybaseballDataSource` (batting/pitching season stats)
- Need to segment to 15-day/season/career windows for model input (new processing)
- Historical matchup stats (batter vs pitcher) — available via pybaseball or Statcast filtering

### Derived: Gamestate Sequences

Processing pipeline to transform raw Statcast pitch data into model-ready sequences:

1. Group pitches by game → half-inning → plate appearance
2. Compute gamestate deltas (base changes, count changes, outs, score)
3. Map pitch types and event types to vocabulary indices (for learnable embeddings)
4. Attach Statcast numerics and player sabermetric vectors
5. Concatenate N games into a single sequence per player/team

## New Components

### Data Layer (Implemented)

| Component | Type | Description |
|-----------|------|-------------|
| `PitchEvent` | frozen dataclass | Single pitch with 30 fields (Statcast + gamestate) |
| `GameSequence` | frozen dataclass | Ordered `tuple[PitchEvent, ...]` for one game from one player's perspective |
| `PlayerContext` | frozen dataclass | N games of sequences for one player |
| `Vocabulary` | frozen dataclass | Token-to-index mapping with `<PAD>`/`<UNK>` fallback (5 vocabs defined) |
| `GameSequenceBuilder` | service | `StatcastStore` → `list[GameSequence]` with batter/pitcher perspective |
| `PitchSequenceDataSource` | `DataSource[PlayerContext]` | Wraps builder with caching and game window truncation |
| `SequenceCache` | service | File-based parquet cache for processed sequences |

### Model Layer

| Component | Type | Description |
|-----------|------|-------------|
| `EventEmbedder` | nn.Module | Maps pitch features to dense vectors (learned + numeric) |
| `GamestateTransformer` | nn.Module | Transformer encoder with player embedding slots |
| `MaskedGamestateHead` | nn.Module | Pre-training prediction head (gamestate + event) |
| `PerformancePredictionHead` | nn.Module | Fine-tuning head (player embedding → stat predictions) |
| `ContextualPerformanceModel` | nn.Module | Full model combining above components |

### Pipeline Integration

| Component | Type | Description |
|-----------|------|-------------|
| `ContextualEmbeddingRateComputer` | `RateComputer` | Wraps trained model, produces `list[PlayerRates]` |
| `ContextualProjectionEngine` | config | Pipeline config with this rate computer + appropriate adjusters |

## Implementation Phases

### Phase 1: Data Pipeline

Build the pitch-level data acquisition and sequence construction:

- `StatcastPitchDataSource` implementing `DataSource[PitchEvent]`
- `GameSequenceBuilder` that groups, orders, and featurizes pitch data
- Caching layer for processed sequences (these are expensive to rebuild)
- Vocabulary definitions for pitch types, event types, gamestate deltas

### Phase 2: Pre-Training

Implement and train the masked gamestate model:

- `EventEmbedder` with learnable embeddings + numeric projection
- `GamestateTransformer` with player embedding attention masking
- MGM training loop with 15% masking
- Validation metrics: masked prediction accuracy by event type

### Phase 3: Fine-Tuning

Fine-tune for per-game stat prediction:

- `PerformancePredictionHead` (linear projection from player embeddings)
- Training on historical games with known outcomes
- Per-stat evaluation: MSE, MAE, R^2 against actuals
- Comparison to Marcel and sportsbook lines where available

### Phase 4: Pipeline Integration

Wire the trained model into the projection pipeline:

- `ContextualEmbeddingRateComputer` implementing `RateComputer` protocol
- Adapter to convert per-game predictions to season-rate `PlayerRates`
- Register as a new `ProjectionPipeline` engine
- Backtest using existing evaluation framework (2021-2024)

### Phase 5: Ensemble (Optional)

Blend with Marcel:

- Per-stat blending weights learned on validation set
- Fall back to Marcel-only for players with insufficient play-by-play data
- Evaluate ensemble vs individual models

## Risks and Limitations

**Data volume.** The model needs pitch-level data from 2015+ (Statcast era). This is only ~10 years of data, and earlier years have sparser Statcast fields. Marcel benefits from 40+ years of counting stats.

**Batter signal is weak.** The paper's batter R^2 values were 0.02-0.06 — barely above zero. Batters participate in far fewer events per game than pitchers. This may limit the model's usefulness for batter projections unless we substantially increase the context window beyond 10 games.

**Compute cost.** Transformer training on millions of pitch sequences is GPU-intensive. Inference is lighter but still heavier than Marcel's arithmetic. We'd need to decide whether to run inference live or pre-compute embeddings.

**Granularity mismatch.** The paper predicts per-game stats (pitcher Ks in tonight's game). Our pipeline projects full-season rates. Aggregating per-game predictions to season projections introduces additional modeling choices (how many games? which opponents?). For pre-season projections this requires simulating a schedule or averaging over opponent distributions.

**Rookies and cold starts.** Players without MLB play-by-play history have no input for the model. The paper notes this limitation and suggests using minor league data as a future direction — aligns with our MLE work.

**Reproducibility.** The paper's GitHub repo provides code but the trained model weights and exact dataset construction may need adaptation. We should plan to train from scratch on our own data pipeline.

## Relation to Existing Proposals

- **Statcast Contact Quality** (`statcast-contact-quality.md`): The contextual model subsumes this — it consumes exit velocity, launch angle, and barrel rate per-pitch rather than as season aggregates. If we build this, the scalar Statcast blending adjuster becomes redundant for the contextual engine.

- **Player Embeddings** (`ml-experimental-approaches.md` #1): The player embeddings from the pre-trained transformer *are* player embeddings — learned from game context rather than stat profiles. These could be extracted and used for similarity analysis, comp players, and prospect projection even without the fine-tuning step.

- **Sequence Models for Career Trajectories** (`ml-experimental-approaches.md` #2): Complementary. The contextual model operates on pitch-level sequences within games; career trajectory models operate on season-level sequences across years. Could compose: contextual embeddings as input features to a career trajectory model.

- **Multi-Task Learning** (`ml-experimental-approaches.md` #3): The fine-tuning head already predicts multiple stats (Ks, hits, walks) simultaneously. Could extend to predict full stat lines with shared representations.

- **In-Season Proposals** (`in-season/`): This model is particularly well-suited for in-season use (matchup analysis, start/sit) since it operates on recent game sequences. A pitcher's last 10 starts capture form, velocity trends, and pitch mix changes that season-level rates miss.

## Related Documents

- [Alternative Training Strategies](../guides/contextual-training-strategies.md) — other pre-training objectives (ELECTRA, contrastive, autoregressive), fine-tuning approaches, and batter-specific improvements
- [Cloud GPU Training Guide](../guides/contextual-cloud-gpu-training.md) — provider options, setup, and code changes for training on cloud infrastructure

## Key References

- Heaton & Mitra, "Learning Contextual Event Embeddings to Predict Player Performance in the MLB"
- Vaswani et al., "Attention Is All You Need" (Transformer architecture)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (masked pre-training)
- Silver & Huffman, "Baseball Predictions and Strategies Using Explainable AI" (PA-level prediction baseline)
- Bailey, "Forecasting Batting Averages in MLB" (Statcast + PECOTA blending)

---

## Implementation Plan

High-level breakdown grounded in the existing codebase. Each phase will get its own detailed
plan before implementation begins.

### Current State

- **Statcast pitch data downloaded**: 2015–2025, ~7.7M rows, 841 MB of monthly parquet files
  in `~/.fantasy_baseball/statcast/{season}/statcast_{season}_{MM}.parquet` via `StatcastStore`.
- **PyTorch available**: Already in `pyproject.toml` dependencies.
- **ML training infrastructure**: `MTLTrainer`, `ModelStore`, feature extractors, train/val
  splitting, early stopping — all reusable patterns.
- **Pipeline integration patterns**: `RateComputer` protocol, `PipelineBuilder` fluent API,
  `PIPELINES` registry in `presets.py`, Marcel fallback precedent in `MTLRateComputer`.
- **DataSource[T]**: Established protocol with `Context`-aware year binding, `Result` error
  handling, and `ALL_PLAYERS` sentinel.
- **Ensemble precedent**: `MTLBlender` already blends MTL predictions with Marcel rates using
  per-stat weights — directly reusable pattern for contextual + Marcel blending.
- **Evaluation harness**: Backtest framework (2021–2024) with correlation, RMSE, and rank
  metrics against `marcel_gb` baseline (HR=0.678, SB=0.736, rank ρ=0.603).

### Phase 1 — Pitch Event Data Pipeline ✅ COMPLETE

**Goal**: Transform raw Statcast parquet into model-ready pitch sequences.

**Status**: Implemented and validated. 82 unit tests passing, smoke-tested against real 2024
Statcast data (Ohtani: 178 games, 3,173 pitches, 17.8 avg pitches/game).

**Deliverables**:

| Component | Location | Description |
|-----------|----------|-------------|
| `PitchEvent` | `contextual/data/models.py` | Frozen dataclass — 30 fields: identity, pitch categorical/continuous, batted ball, gamestate, context, PA outcome, run expectancy |
| `GameSequence` | `contextual/data/models.py` | Ordered `tuple[PitchEvent, ...]` for one game from one player's perspective, with game metadata |
| `PlayerContext` | `contextual/data/models.py` | N games of sequences for one player with player identity |
| `Vocabulary` | `contextual/data/vocab.py` | Token-to-index mappings with `<PAD>`=0 and `<UNK>`=1 fallback. 5 vocabs: `PITCH_TYPE_VOCAB` (21), `PITCH_RESULT_VOCAB` (17), `PA_EVENT_VOCAB` (27), `BB_TYPE_VOCAB` (6), `HANDEDNESS_VOCAB` (4) |
| `GameSequenceBuilder` | `contextual/data/builder.py` | `StatcastStore` → `list[GameSequence]` — groups by `(game_pk, player)`, sorts by `at_bat_number`/`pitch_number`, supports batter/pitcher perspective |
| `PitchSequenceDataSource` | `contextual/data/source.py` | `DataSource[PlayerContext]` — wraps builder, keyed by `Player.mlbam_id` + context year, optional game window truncation |
| `SequenceCache` | `contextual/data/cache.py` | File-based parquet cache at `{cache_dir}/sequences/{season}/{perspective}/{player_id}.parquet` |

**Key decisions made**:
- **Gamestate encoding**: Absolute (not deltas). Deltas are trivially computed at tensor-creation time (Phase 2); absolutes are canonical, testable, and decoupled from model architecture.
- **Missing values**: `None` throughout, no imputation. Tensor layer handles masking in Phase 2.
- **Cache backend**: File-based parquet (not SQLite). Volume too large for TEXT columns; parquet is typed and compressed.
- **Sequence grouping**: By `(game_pk, player)`, not half-inning. Half-inning is implicit in `inning`/`is_top` fields.
- **Player sabermetrics**: Deferred to Phase 2/3. Keeps Phase 1 focused on pitch-level features.
- **Vocabulary source**: Exhaustive enumeration of 2015–2024 values, forward-compatible via `<UNK>` fallback.
- **bat_score/fld_score**: Derived from `home_score`/`away_score` + `inning_topbot` (perspective-normalized).

### Phase 2 — Model Architecture

**Goal**: Implement the transformer model with player embedding slots and both training heads.

**What exists**: `MultiTaskNet` in `ml/mtl/model.py` provides a pattern for multi-head PyTorch
models with uncertainty-weighted loss and `predict()` / `from_params()` serialization. Not
directly reusable (feed-forward, not sequential), but the outer structure is a template.

**Deliverables**:

| Component | Location | Description |
|-----------|----------|-------------|
| `EventEmbedder` | `contextual/model/embedder.py` | `nn.Module` — maps `PitchEvent` to dense vector: learnable embeddings (pitch type, event type, gamestate delta) concatenated with projected Statcast numerics |
| `GamestateTransformer` | `contextual/model/transformer.py` | `nn.Module` — transformer encoder with positional encoding and player embedding slots prepended to sequence |
| `PlayerAttentionMask` | `contextual/model/mask.py` | Builds attention mask ensuring each player embedding only attends to its own events |
| `MaskedGamestateHead` | `contextual/model/heads.py` | Pre-training head — predicts masked gamestate delta + event type from surrounding context |
| `PerformancePredictionHead` | `contextual/model/heads.py` | Fine-tuning head — linear projection from player embedding to per-game stat vector |
| `ContextualPerformanceModel` | `contextual/model/model.py` | Top-level `nn.Module` composing embedder + transformer + swappable head |
| `ModelConfig` | `contextual/model/config.py` | Hyperparameters: embed dims, num layers, num heads, dropout, numeric projection size, max sequence length |

**Key decisions to make in detailed plan**:
- Embedding dimensions for each categorical feature (pitch type, event type, gamestate)
- Transformer depth/width tradeoffs (paper doesn't specify exact architecture)
- Whether to use rotary or sinusoidal positional encoding (games have natural ordering)
- Player embedding initialization strategy (random vs. warm-start from season stats)
- Numeric feature normalization (per-feature z-score vs. learned projection)

### Phase 3 — Pre-Training (Masked Gamestate Modeling)

**Goal**: Train the transformer on the masked gamestate objective to learn contextual
representations of baseball events.

**What exists**: `MTLTrainer` pattern (data collection → train/val split → training loop with
early stopping → model persistence). `ModelStore` / `MTLModelStore` for saving/loading trained
models with metadata.

**Deliverables**:

| Component | Location | Description |
|-----------|----------|-------------|
| `MGMDataset` | `contextual/training/dataset.py` | `torch.utils.data.Dataset` — loads `PlayerContext` sequences, applies 15% random masking, yields (input, mask_targets) |
| `MGMTrainer` | `contextual/training/pretrain.py` | Training loop: masked gamestate modeling with cross-entropy loss on masked positions |
| `PreTrainingConfig` | `contextual/training/config.py` | Epochs, batch size, learning rate schedule, masking ratio, warmup steps, checkpoint interval |
| `ContextualModelStore` | `contextual/persistence.py` | Save/load model checkpoints with training metadata (extends existing persistence pattern) |
| `pretrain` CLI command | `contextual/cli.py` | `uv run python -m fantasy_baseball_manager contextual pretrain --seasons 2015,2016,...` |

**Training plan**:
- Training data: 2015–2022 (~5.6M pitches), validation: 2023 (~760K pitches), held out: 2024–2025
- Masked prediction accuracy by event type as validation metric
- Checkpoint every N epochs; resume from checkpoint on interrupt
- GPU training expected; estimate ~6–12 hours on single consumer GPU

**Key decisions to make in detailed plan**:
- Masking strategy (uniform random vs. structured — e.g., mask entire plate appearances)
- Loss weighting between gamestate delta prediction and event type prediction
- Learning rate schedule (linear warmup + cosine decay vs. reduce-on-plateau)
- Whether to pre-train on team-batting and pitcher sequences jointly or separately
- Minimum sequence length filtering (discard rain-shortened games?)

### Phase 4 — Fine-Tuning

**Goal**: Fine-tune the pre-trained model to predict per-game player statistics from the
learned player embeddings.

**Deliverables**:

| Component | Location | Description |
|-----------|----------|-------------|
| `FineTuneDataset` | `contextual/training/dataset.py` | Yields (N prior games as context, next-game stat targets) per player |
| `FineTuneTrainer` | `contextual/training/finetune.py` | Freeze or slow-learn transformer weights, train prediction head on per-game stats |
| `FineTuneConfig` | `contextual/training/config.py` | Context window (N games), target stats, freeze strategy, learning rate |
| `finetune` CLI command | `contextual/cli.py` | `uv run python -m fantasy_baseball_manager contextual finetune --base-model <path>` |

**Target stats** (aligned with existing pipeline):
- Pitchers: SO, H, BB, ER, HR (same as `MTLTrainer` pitcher targets)
- Batters: HR, SO, BB, H, 2B, 3B, SB (same as `MTLTrainer` batter targets)

**Evaluation**:
- Per-stat MSE, MAE, R² on held-out 2024 games (same metrics as `ml/validation.py`)
- Compare to Marcel per-game predictions (baseline: season rate × 1 game of opportunity)
- Compare to paper's reported R² values as sanity check

**Key decisions to make in detailed plan**:
- Context window size (paper uses N=10 games — is that optimal for our use case?)
- Freeze strategy (full freeze of transformer vs. discriminative learning rates)
- Pitcher vs. batter model separation (paper trains separate models — follow or unify?)
- How to handle batters' low event frequency (paper's primary weakness: R²=0.02–0.06)
- Sliding window vs. fixed-window training data construction

### Phase 5 — Pipeline Integration

**Goal**: Wire the trained model into the projection pipeline as a first-class `RateComputer`
so it produces `list[PlayerRates]` compatible with all downstream stages.

**What exists**: `MTLRateComputer` is the exact template — loads a pre-trained model,
extracts features, produces rates, falls back to Marcel for insufficient data. `PipelineBuilder`
has `with_mtl_rate_computer()` as a pattern for registration.

**Deliverables**:

| Component | Location | Description |
|-----------|----------|-------------|
| `ContextualEmbeddingRateComputer` | `pipeline/stages/contextual_rate_computer.py` | Implements `RateComputer` — loads fine-tuned model, fetches N recent games via `PitchSequenceDataSource`, runs inference, aggregates per-game predictions to season rates |
| `PerGameToSeasonAdapter` | `contextual/adapter.py` | Converts per-game stat predictions → per-PA/per-out rates for `PlayerRates`. Handles schedule simulation or opponent-averaged aggregation for pre-season projections |
| `PipelineBuilder.with_contextual()` | `pipeline/builder.py` | Builder method to register the contextual rate computer |
| `contextual` pipeline preset | `pipeline/presets.py` | Standalone pipeline: contextual rate computer + park factors + pitcher normalization |

**Fallback behavior** (following `MTLRateComputer` pattern):
- Players with < N games of Statcast play-by-play → fall back to Marcel
- Metadata flag: `contextual_predicted: bool` for downstream introspection
- Rookies without MLB pitch data → Marcel (or MLE if available)

**Adjuster compatibility**:
- Park factors and pitcher normalization still apply (stadium/league effects)
- Statcast blending adjuster is **redundant** — model already consumes pitch-level Statcast
- GB residual adjuster may still add value (different signal) — evaluate empirically

**Key decisions to make in detailed plan**:
- Per-game → season aggregation strategy (mean over N predictions? schedule-weighted?)
- How many games of context required before trusting the model over Marcel
- Whether to skip Statcast blending adjuster automatically when contextual is the rate computer
- In-season vs. pre-season inference paths (recent games available vs. simulated schedule)

### Phase 6 — Ensemble & Evaluation

**Goal**: Blend contextual predictions with Marcel for production use and evaluate against
the `marcel_gb` baseline across the full backtest suite.

**What exists**: `MTLBlender` (`pipeline/stages/mtl_blender.py`) implements per-stat weighted
blending of two rate sources with configurable weights and minimum PA thresholds. Evaluation
harness runs 2021–2024 backtests with correlation, RMSE, and rank metrics.

**Deliverables**:

| Component | Location | Description |
|-----------|----------|-------------|
| `ContextualBlender` | `pipeline/stages/contextual_blender.py` | `RateAdjuster` — blends contextual rates with Marcel rates per stat (follows `MTLBlender` pattern) |
| `BlenderConfig` | `contextual/config.py` | Per-stat blend weights, minimum games threshold for contextual inclusion |
| `marcel_contextual` pipeline preset | `pipeline/presets.py` | Marcel base + contextual blender + existing adjusters |
| Backtest results | `docs/projection-engines/contextual-event-embeddings.md` | Performance table in projection-engines index format |
| `PipelineBuilder.with_contextual_blender()` | `pipeline/builder.py` | Builder method for the ensemble configuration |

**Evaluation plan**:
- Run existing evaluation harness against `contextual` (standalone) and `marcel_contextual` (ensemble)
- Primary comparison: `marcel_gb` (current best — HR=0.678, SB=0.736, rank ρ=0.603)
- Per-stat breakdown to identify where contextual signal helps vs. hurts
- Pitcher vs. batter analysis (paper suggests pitchers benefit more)
- Learned blend weights (optimize on 2021–2023 validation, evaluate on 2024)

**Success criteria**:
- Ensemble matches or exceeds `marcel_gb` on at least 3 of 5 headline metrics
- Pitcher strikeout correlation improvement ≥ 3% (model's strongest signal per paper)
- No regression > 2% on any individual metric vs. `marcel_gb`
- Batter projections at least match Marcel (low bar given paper's weak batter R²)

### Module Structure

All new code lives under `src/fantasy_baseball_manager/contextual/`:

```
contextual/
├── __init__.py
├── cli.py                    # pretrain / finetune / evaluate commands
├── config.py                 # all configuration dataclasses
├── persistence.py            # model checkpoint save/load
├── adapter.py                # per-game predictions → season rates
├── data/
│   ├── __init__.py
│   ├── models.py             # PitchEvent, GameSequence, PlayerContext
│   ├── vocab.py              # pitch type / event type / gamestate vocabularies
│   ├── builder.py            # GameSequenceBuilder (raw parquet → sequences)
│   ├── source.py             # DataSource[PlayerContext]
│   └── cache.py              # processed sequence disk cache
├── model/
│   ├── __init__.py
│   ├── config.py             # ModelConfig (architecture hyperparameters)
│   ├── embedder.py           # EventEmbedder
│   ├── transformer.py        # GamestateTransformer
│   ├── mask.py               # PlayerAttentionMask
│   ├── heads.py              # MaskedGamestateHead, PerformancePredictionHead
│   └── model.py              # ContextualPerformanceModel (top-level)
└── training/
    ├── __init__.py
    ├── config.py             # PreTrainingConfig, FineTuneConfig
    ├── dataset.py            # MGMDataset, FineTuneDataset
    ├── pretrain.py           # MGMTrainer
    └── finetune.py           # FineTuneTrainer
```

Pipeline integration touches existing files:
- `pipeline/stages/contextual_rate_computer.py` (new)
- `pipeline/stages/contextual_blender.py` (new)
- `pipeline/builder.py` (add `with_contextual()`, `with_contextual_blender()`)
- `pipeline/presets.py` (add `contextual`, `marcel_contextual` presets)

### Phase Dependencies

```
Phase 1 (Data Pipeline) ✅
  └─► Phase 2 (Model Architecture) — needs PitchEvent/GameSequence types for tensor shapes
      └─► Phase 3 (Pre-Training) — needs model + data pipeline
          └─► Phase 4 (Fine-Tuning) — needs pre-trained weights
              └─► Phase 5 (Pipeline Integration) — needs fine-tuned model
                  └─► Phase 6 (Ensemble & Evaluation) — needs working standalone engine
```

Phases are strictly sequential — each depends on the previous. No parallelism between
phases, though within each phase we follow TDD (tests first, then implementation).
