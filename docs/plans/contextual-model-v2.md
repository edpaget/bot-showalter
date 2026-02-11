# Contextual Model v2 Roadmap — Hierarchical Architecture

## Overview

The contextual transformer completed its first full train-evaluate cycle with all architectural fixes (CLS token, rate targets, bug audit, game-local attention). Results show the model learns real per-game signal — pitcher fine-tuning beats baseline on all 4 stats (21-30% MSE reduction) — but season-level projections underperform Marcel GB due to **mean collapse**: the model has no player identity signal and cannot differentiate players, so predictions regress toward league-average rates.

The root cause is architectural. The current model asks a single flat transformer to simultaneously learn pitch physics (per-pitch), interpret PA outcomes (per-plate-appearance), track game-level trends (per-game), and project future performance (per-player, over weeks/months). These operate at fundamentally different granularities. The `[CLS]` token must attend through hundreds of pitch tokens to extract all of this — a task it fails at.

This roadmap replaces the flat single-model approach with a **hierarchical architecture** that processes data at three natural levels of baseball (pitch → plate appearance → game), with player identity conditioning the levels where it matters. Research on entity identity in transformer models (Temporal Fusion Transformer, batter2vec, Wide & Deep) supports this design: hybrid approaches combining learned embeddings with explicit features consistently outperform either alone, and entity identity should condition the network rather than being concatenated as a passive input.

## Design Decisions

### Hierarchical processing over flat sequences

The current model processes ~600 pitch tokens in a flat sequence for a 30-game batter context. Most individual pitches are noise for projection — foul ball #3 in a 9-pitch AB tells you nothing about future HR rate. The signal lives in PA outcomes and their progression over time.

The batter2vec paper (Alcorn, 2018) demonstrated that PA-level sequences capture rich player information invisible to traditional statistics — qualitative relationships, stylistic groupings, and matchup dynamics. By compressing pitch sequences into PA embeddings first, we reduce noise while retaining signal, then model PA sequences at the level where batter2vec showed the information lives.

Each level in the hierarchy compresses at a natural baseball boundary: pitches → PA outcome (what happened and how), PAs → game (how did this game go for this player), games → projection (what does the recent trend predict). Each level is independently testable.

### Separation of generic physics from player-specific interpretation

The pitch-level encoder (Level 1) learns **universal pitch physics** with no player identity conditioning. A 95mph fastball with 18 inches of vertical break behaves the same regardless of who threw it. The pre-trained MGM weights already learned this — they were trained across all pitchers and batters without identity signals.

Player identity is introduced at Level 2 (PA → game), where it conditions the interpretation of PA outcomes. A strikeout PA means something different for Luis Arraez (alarming) than for Joey Gallo (Tuesday). The same generic PA embedding gets interpreted differently depending on who the player is. This avoids re-pretraining while placing identity exactly where it matters.

The information flow is:
- **Level 1:** "What happened physically?" (universal)
- **Level 2:** "What does that mean for THIS player?" (identity-conditioned)
- **Level 3:** "What does the trend predict?" (identity-conditioned)

### Hybrid identity: stat features + archetype embeddings

Research consistently shows hybrid approaches outperform pure embeddings or pure features (Wide & Deep at Google, TFT vs DeepAR, Guo & Berkhahn entity embeddings). We use both:

- **Multi-horizon stat features** (career, 3yr, 1yr, 30-day rates): Recomputed at inference time, naturally encode trajectory (a player whose 30-day K-rate diverges from their career rate looks different from a stable player). Prevent the model from having to rediscover basic player identity from the training signal.
- **Archetype embeddings**: Capture latent player type — clusters of players who share trait combinations that stats alone may not encode. Low-dimensional, transfer across players, stable prior even with small samples.

Stat feature dropout during training (randomly zeroing stat features) prevents shortcut learning where the model ignores pitch context and just learns Marcel-style regression-to-the-mean from the stat inputs.

### MLE integration for cold-start

The existing Minor League Equivalencies pipeline translates MiLB stats into the same feature space as MLB stats. Every player from Single-A up can receive stat features and an archetype assignment — the identity module sees rate stats regardless of source, with no special-casing for rookies.

### Player identity first, residual learning second

The mean collapse is fundamentally an information problem, not a target-framing problem. Changing targets to residuals doesn't help if the model can't tell players apart. Building player identity first is the more diagnostic experiment: if the model can't use identity to differentiate predictions, residual learning won't fix that.

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
- Flat pitch sequence forces CLS to simultaneously learn pitch physics, PA interpretation, game trends, and projection

**Key files:**
- `src/fantasy_baseball_manager/contextual/training/finetune.py` — fine-tune trainer
- `src/fantasy_baseball_manager/contextual/training/dataset.py` — training sample construction, `compute_rate_targets()`
- `src/fantasy_baseball_manager/contextual/model/model.py` — transformer + prediction head
- `src/fantasy_baseball_manager/contextual/model/tensorizer.py` — token construction
- `src/fantasy_baseball_manager/contextual/model/mask.py` — attention patterns
- `src/fantasy_baseball_manager/contextual/adapter.py` — prediction-to-rate conversion
- `src/fantasy_baseball_manager/pipeline/stages/contextual_rate_computer.py` — pipeline integration
- `src/fantasy_baseball_manager/contextual/training/config.py` — all config dataclasses
- `src/fantasy_baseball_manager/minors/` — MLE pipeline (data sources, features, model, rate computer)

**Relevant research:**
- batter2vec (Alcorn, 2018) — learned 64-dim player embeddings from at-bat sequences, captured relationships invisible to stats
- Temporal Fusion Transformer (Lim et al., 2019) — entity identity conditions the entire network through gating, outperforms passive concatenation
- Wide & Deep (Cheng et al., 2016) — hybrid features + embeddings outperforms either alone at Google Play scale
- Entity Embeddings (Guo & Berkhahn, 2016) — learned embeddings boost all downstream ML methods when used as features
- Embedding Collapse (ICML 2024) — entity embeddings collapse to low-rank when scaled naively, multi-embedding designs mitigate

## Target Architecture

```
Player Identity (conditions Levels 2 and 3, TFT-style gating):
  stat_features: [career rates, 3yr rates, 1yr rates, 30d rates, age, handedness]
                 (from MLB stats or MLE-translated MiLB stats — same feature space)
  → MLP → stat_repr (d=64)

  archetype_id: [cluster assignment derived from stat profile]
  → nn.Embedding(n_archetypes, d=32) → archetype_repr

  identity_repr = concat(stat_repr, archetype_repr)  (d=96)

Level 1 — Pitch → PA (pre-trained transformer, NOT conditioned on identity):
  Per game: full pitch sequence → transformer (game-local attention, frozen)
  → pool tokens by PA boundary → PA embeddings (d=256 from transformer d_model)
  Learns: universal pitch physics, contact quality, pitch sequencing effects
  Preserves: cross-PA context within a game (fatigue, pitch mix evolution)

Level 2 — PA → Game (learned, conditioned on identity):
  Per game: sequence of PA embeddings → identity-conditioned attention → game embedding
  Learns: how PA outcomes relate to THIS player's baseline and tendencies
  Example: "strikeout on breaking balls" means different things for different players

Level 3 — Game → Projection (learned, conditioned on identity):
  Sequence of game embeddings → identity-conditioned attention → prediction head
  Learns: temporal trends, how recent game sequence predicts future performance
  Short sequence (~10-30 games), clean signal

Prediction Head:
  → projected rates (or Marcel residuals after Phase 3)
```

### Identity conditioning mechanism (TFT-style)

At Levels 2 and 3, the identity vector conditions processing through gated residual networks rather than simple concatenation:
- Identity initializes attention query bias (the model "asks different questions" for different player types)
- Identity gates which PA/game features the model attends to (a power pitcher's games are read differently than a finesse pitcher's)
- This prevents the identity signal from being a passive input the model can ignore

### Stat feature dropout

During training, stat features are randomly zeroed with probability p=0.2. This forces the model to extract useful signal from the pitch/PA context even when stat features provide an easy shortcut, preventing the model from learning Marcel-style regression through extra steps.

## Phases

### Phase 1: Player Identity Foundation

**Goal:** Build the data infrastructure for multi-horizon player stat features and player archetype clustering.

**Rationale:** Prerequisite for identity conditioning at Levels 2 and 3. Stat features must be available per player-year for training data construction, and archetypes must be derived from clustering all players in a unified stat space.

**Scope:**
- Define a `PlayerStatProfile` dataclass holding multi-horizon rate stats: career, 3yr, 1yr, and 30-day rates for each target stat (SO, BB, HR, H for pitchers; SO, BB, HR, H, 2B, 3B for batters), plus age and handedness
- Build a `PlayerStatProfileBuilder` that computes profiles from historical batting/pitching data, handling cases where a horizon has insufficient data (fall back to next longer horizon)
- Generate and cache stat profiles for each player-year in the training range (2015-2022 training, 2023-2024 eval)
- Build player archetype clustering: run k-means (or similar) on stat profiles across all player-years to define ~15-25 archetypes per player type (batter/pitcher), store cluster centroids and assignments
- Add archetype assignment at inference time: given a new stat profile, assign the nearest archetype
- Validate: inspect archetype clusters for face validity (do the groupings make baseball sense?)

**Key files (new):** `contextual/identity/stat_profile.py`, `contextual/identity/archetypes.py`

**Dependencies:** Historical batting/pitching stats already available through the Marcel pipeline data sources.

### Phase 2: Hierarchical Model Architecture

**Goal:** Replace the flat single-model architecture with the three-level hierarchy (pitch → PA → game) with identity conditioning at Levels 2 and 3.

**Rationale:** This is the core architectural change. It solves three problems simultaneously: attention dilution (projection model works over ~10-30 game embeddings, not ~600 pitch tokens), player identity (conditioning at Levels 2-3), and noise reduction (pitch-level noise compressed out at PA boundaries).

This phase is built incrementally to isolate the value of each level:

**Phase 2a: Game-level baseline (Levels 0 + 3 only)**
- Build the identity module: MLP over stat features (d=64) + archetype embedding (d=32) → identity_repr (d=96)
- Use mean-pooled game-level pitch features as a simple context representation (no PA-level processing yet)
- Build the projection layer (Level 3): small attention over game representations, conditioned on identity
- Prediction head on top
- Train and evaluate — this is the minimal architecture that tests whether identity + game context produces player-differentiated predictions
- Key diagnostic: does prediction variance across players increase? Do high-K pitchers get higher K-rate predictions?

**Phase 2b: Add PA-level processing (Level 2)**
- Add PA boundary detection to the tensorizer: identify PA-ending pitch positions (already available via `pa_event`) and group pitch tokens by PA
- Pool pitch tokens within each PA boundary from the frozen pre-trained transformer → PA embeddings
- Build Level 2: small attention layer over PA embeddings per game, conditioned on identity → game embeddings
- Replace the mean-pooled game features from 2a with the learned game embeddings
- Train and evaluate — measure improvement over Phase 2a. This tests whether PA-level sequence modeling adds value over aggregate game features (the batter2vec hypothesis)

**Phase 2c: Identity conditioning refinement**
- Implement TFT-style gated residual networks for identity conditioning at Levels 2 and 3
- Add stat feature dropout (p=0.2) during training
- Compare against simple concatenation-based conditioning from 2a/2b
- Ablate: measure pitch context contribution with and without stat dropout to verify the pitch tower isn't being ignored

**Key files:** `model.py` (new hierarchical architecture), `tensorizer.py` (PA boundary extraction), `dataset.py`, `finetune.py`, `config.py`, `adapter.py`

**Dependencies:** Phase 1 (stat profiles and archetypes available).

### Phase 3: Residual Learning

**Goal:** Reframe the prediction task from absolute rates to Marcel residuals, making the model an adjustment layer on top of a strong prior.

**Rationale:** With player identity and hierarchical processing in place, residual learning completes the picture. The identity module provides "who is this player," the pitch/PA/game hierarchy provides "what has been happening recently," and the model predicts "how should we adjust Marcel's projection given this evidence." Near-zero residual predictions default to Marcel — the model can only help, not hurt.

**Scope:**
- Add `target_mode="residuals"` to `FineTuneConfig` alongside existing `"counts"` and `"rates"` modes
- Modify `compute_rate_targets()` in `dataset.py` to accept Marcel rates per player and compute `target = actual_rate - marcel_rate`
- Build a data pipeline that pairs each training window with the corresponding Marcel projection for that player-year
- Update the adapter to add residual predictions back onto Marcel rates at inference time
- Add `residual_mode` flag to `ContextualRateComputerConfig`
- Re-train pitcher and batter models with residual targets
- Evaluate against absolute-rate hierarchical model, Marcel GB, and Steamer

**Key files:** `dataset.py`, `config.py`, `adapter.py`, `contextual_rate_computer.py`, `finetune.py`

**Dependencies:** Phase 2 (hierarchical model operational). Marcel projections accessible during training data construction — generate and cache Marcel rates for each training season (2015-2022).

### Phase 4: MLE Rookie Integration

**Goal:** Extend the identity module to handle rookies and prospects using MLE-translated minor league stats, providing a unified player identity representation for all players.

**Rationale:** The existing MLE pipeline translates MiLB stats into MLB-equivalent rates — the same feature space the identity module already consumes. Rookies can receive stat features and archetype assignments without any model changes.

**Scope:**
- Extend `PlayerStatProfileBuilder` to accept MLE-translated rates as a fallback when MLB history is insufficient (< 200 MLB PA)
- Map MLE rate predictions into the same `PlayerStatProfile` format used by MLB players
- Assign archetypes to prospects from their MLE-translated stat profiles (same clustering, same nearest-centroid assignment)
- Add source tracking metadata to profiles: `source="mlb"`, `source="mle"`, `source="blended"` (for players with partial MLB history)
- Validate on known recent call-ups: do MLE-sourced profiles produce reasonable predictions compared to their actual MLB performance?
- Test that archetype assignments for prospects make baseball sense (e.g., a power-hitting AAA prospect clusters with MLB power hitters)

**Key files:** `contextual/identity/stat_profile.py`, `minors/rate_computer.py`, `contextual/adapter.py`

**Dependencies:** Phase 2 (identity module built). Existing MLE pipeline (`src/fantasy_baseball_manager/minors/`).

### Phase 5: Fine-Tuning Stability

**Goal:** Determine whether gradually unfreezing the pitch transformer (Level 1) improves PA embeddings for the projection task.

**Rationale:** Level 1 starts frozen with pre-trained MGM weights. These weights encode general pitch physics but were not trained for projection. Selectively unfreezing upper transformer layers may allow the PA embeddings to become more projection-relevant. However, the hierarchical architecture already insulates Level 1 from the projection task's gradient signal (Levels 2-3 absorb most of the adaptation), so unfreezing may not be necessary.

**Scope:**
- Add an unfreezing schedule to `FineTuneConfig` (e.g., `unfreeze_schedule: list[int]` — epochs at which to unfreeze each layer group, top-down)
- Modify the fine-tune trainer to manage `requires_grad` per layer group across epochs
- Define layer groups: Levels 2-3 + prediction head (always unfrozen) → top transformer layer → middle layers → bottom layers → embedder
- Compare: fully-frozen Level 1 vs gradual top-down unfreezing vs full unfreeze
- Evaluate for both pitcher and batter models
- If frozen performs comparably, keep it frozen (simpler, faster training)

**Key files:** `finetune.py`, `config.py`

**Dependencies:** Phase 2 (hierarchical architecture). Can be evaluated independently of Phases 3-4.

### Phase 6: Batter-Specific Improvements

**Goal:** Address batter-specific weaknesses through context window tuning and event-weighted attention at the PA level.

**Rationale:** Batter predictions have consistently lagged pitcher predictions. The hierarchical architecture should help (PA-level modeling aligns with batter2vec findings), but additional batter-specific tuning may be needed. Context window size and PA-level attention weighting are the two most likely levers.

**Scope:**
- **Context window tuning:** Run batter fine-tuning with context windows of 30, 40, and 50 games. With the hierarchical architecture, longer windows are cheaper — the sequence length at Level 3 grows linearly with games, not with total pitches. Compare per-stat MSE/MAE for each window size.
- **PA-outcome attention bias at Level 2:** Add a learnable bias toward PA outcome type (HR, SO, BB, hit, out) in the Level 2 attention computation. This lets the model learn that HR PAs are more informative than routine groundout PAs for projecting future HR rate.

**Key files:** `config.py`, `model.py`

**Dependencies:** Phases 1-3 evaluated. Only pursue if batters are still the weak link after the hierarchical architecture and residual learning.

### Phase 7: Ensemble Integration

**Goal:** Tune the contextual-Marcel blend via cross-validation to maximize pipeline value.

**Rationale:** With residual learning (Phase 3), the natural ensemble is additive: `final = marcel + weight * contextual_residual`. Cross-validation on held-out years determines the optimal weight. Even if the contextual model doesn't beat Marcel standalone, it may capture complementary signal.

**Scope:**
- Implement cross-validation weight tuning using the evaluation harness across 2021-2024
- Test blend weights from 0.0 to 1.0 in increments of 0.1 for both pitcher and batter
- Allow per-perspective and potentially per-stat blend weights
- Update the `marcel_contextual` pipeline defaults with the tuned weights
- Final evaluation comparing tuned ensemble vs Marcel GB vs Steamer

**Key files:** `pipeline/presets.py`, `evaluation/harness.py`, `pipeline/stages/contextual_rate_computer.py`

**Dependencies:** Phases 1-3 completed and models re-trained. The ensemble is only worth tuning once the contextual model is contributing positive signal.

## Open Questions

- **Archetype count and method:** k-means is the obvious starting point but may not capture the shape of player clusters well. Should we try GMM or HDBSCAN? How do we validate cluster quality beyond face validity?
- **Stat feature horizons:** Career/3yr/1yr/30d is the initial proposal. Are all four horizons needed, or do some dominate? Ablation study after Phase 2a can answer this empirically.
- **PA boundary pooling strategy:** Mean pooling over pitch tokens within a PA is simplest. Alternatives: attention pooling (let the model learn to weight pitches within a PA), or using the last token (PA outcome) as the PA representation. The right choice may differ for pitchers vs batters.
- **Level 2 sequence length:** For pitchers, a single game may have ~25 PAs faced. For batters, ~4 PAs per game. Should Level 2 process PAs within a single game (short sequences) or across multiple games (longer, more signal)? Starting with per-game and measuring is the pragmatic choice.
- **Pitcher MLE:** The MLE pipeline currently only supports batters. Phase 4 will work for batter prospects immediately but pitcher prospects will fall back to Marcel-only stat features. Extending MLE to pitchers is a separate initiative.
- **Batter model viability:** If the hierarchical architecture + residual learning still can't make the batter model contribute positively, should we abandon batter contextual predictions entirely and focus on pitchers only?
- **Re-pretraining:** If frozen Level 1 representations aren't useful for producing PA embeddings, re-pretraining with a PA-level objective (predict PA outcome from pitch sequence) could help — but this is a large investment and should only be considered after Phase 5 experiments.
- **Embedding collapse monitoring:** The ICML 2024 research on embedding collapse in recommendation models suggests monitoring the rank of identity and PA embeddings during training. If embeddings collapse to low-rank, multi-embedding designs or contrastive regularization may be needed.
