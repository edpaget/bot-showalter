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

### Data Layer

| Component | Type | Description |
|-----------|------|-------------|
| `PitchEvent` | dataclass | Single pitch with all features (Statcast + gamestate) |
| `GameSequence` | dataclass | Ordered sequence of `PitchEvent` for one game |
| `PlayerContext` | dataclass | N games of sequences for one player/team |
| `StatcastPitchDataSource` | `DataSource[PitchEvent]` | Fetches pitch-level data via pybaseball |
| `GameSequenceBuilder` | service | Transforms raw pitch data into model-ready sequences |

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

## Key References

- Heaton & Mitra, "Learning Contextual Event Embeddings to Predict Player Performance in the MLB"
- Vaswani et al., "Attention Is All You Need" (Transformer architecture)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (masked pre-training)
- Silver & Huffman, "Baseball Predictions and Strategies Using Explainable AI" (PA-level prediction baseline)
- Bailey, "Forecasting Batting Averages in MLB" (Statcast + PECOTA blending)
