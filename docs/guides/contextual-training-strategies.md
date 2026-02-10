# Alternative Training Strategies for Contextual Event Embeddings

Parent doc: [contextual-event-embeddings.md](../archive/contextual-event-embeddings.md)

The main design commits to BERT-style masked gamestate modeling (MGM) pre-training followed by linear fine-tuning. This doc catalogs alternative strategies worth evaluating, organized by the problem they address.

---

## Alternative Pre-Training Objectives

### Next-Event Prediction (Autoregressive / GPT-style)

Train left-to-right to predict the next pitch/event given all prior events, instead of masking 15% and predicting bidirectionally.

**Why consider it:** Baseball is inherently sequential — each pitch causally follows from prior game state. An autoregressive model also naturally produces a forward-simulation capability useful for schedule-averaged season projections (Phase 5's per-game → season aggregation problem). Instead of averaging per-game predictions over an assumed schedule, the model can simulate plate appearances against a distribution of opponent profiles.

**Tradeoff:** Unidirectional context is strictly less information than bidirectional for embedding quality. A single at the start of a game can't attend to the strikeout that ends it. For pure embedding quality, MGM likely wins. For downstream simulation and projection, autoregressive may be more directly useful.

**Implementation cost:** Low — swap the masking + reconstruction loss for a causal attention mask and next-token cross-entropy loss. The `GamestateTransformer` already supports attention masking; this is a different mask pattern.

### Replaced Token Detection (ELECTRA-style)

Train a small generator network to produce plausible-but-wrong pitch events, then train the main model as a discriminator: for every position, predict whether the token is real or replaced.

**Why consider it:** Far more sample-efficient than MGM. MGM learns from only the 15% of masked tokens per batch. ELECTRA learns from 100% of tokens (every position gets a real/fake label). The original ELECTRA paper demonstrated matching BERT quality at ~1/4 the compute budget. This directly addresses the limited-data concern (only ~10 years of Statcast data, vs. NLP corpora with billions of tokens).

**Tradeoff:** Requires training and maintaining a separate generator network. The generator needs to be good enough to produce challenging replacements (random replacements are too easy to detect) but not so good that it produces indistinguishable events. Adds hyperparameter tuning complexity (generator size, replacement ratio).

**Implementation cost:** Medium — need a second smaller transformer as generator, a two-phase training loop (generator step, then discriminator step), and a binary classification head replacing the MGM reconstruction head.

### Contrastive Pre-Training (SimCLR / InfoNCE)

Define positive pairs (two game windows from the same player in a similar time period) and negative pairs (different players, different eras). Train via contrastive loss to produce player embeddings where similar performance profiles cluster together.

**Why consider it:** Directly optimizes for the representation quality that matters downstream — player embeddings that capture performance similarity — rather than the proxy task of reconstructing masked events. Contrastive learning has shown strong results in domains where the downstream task is representation-based rather than generative.

**Tradeoff:** Defining good positive/negative pairs requires domain judgment. Two windows of the same player 3 years apart (pre/post injury, aging curve) may not be a valid positive pair. Negative mining strategy matters a lot — hard negatives (similar players, different outcomes) teach more than easy negatives (pitcher vs. position player).

**Implementation cost:** Medium — needs a pair/batch construction strategy in the dataset, a projection head for the contrastive space, and an InfoNCE or NT-Xent loss function. No masking infrastructure needed.

### Multi-Objective Pre-Training

Combine MGM with auxiliary objectives trained simultaneously:

| Auxiliary Objective | What It Teaches |
|---|---|
| **Pitch sequence ordering** — shuffle pitches within a PA, predict correct order | Pitch sequencing patterns, count leverage |
| **Inning prediction** — predict which inning a pitch occurred in | Fatigue effects, bullpen usage, leverage awareness |
| **Player identity prediction** — from masked sequences, predict pitcher/batter ID | Individual style signatures, pitch repertoires |
| **Score differential prediction** — predict run differential at end of inning | Clutch context, high-leverage situations |

**Why consider it:** Each auxiliary task provides complementary training signal from the same data. No additional data collection needed. The model learns richer representations because it must encode multiple aspects of game context simultaneously.

**Tradeoff:** More loss terms to balance. Risk of one objective dominating training if losses aren't weighted carefully. The existing `MultiTaskNet` pattern with uncertainty-weighted loss (`MTLTrainer`) provides a precedent for multi-task loss balancing.

**Implementation cost:** Low per-objective — each is a small additional head on the transformer output. The training loop needs multi-task loss aggregation, but the project already has this pattern in `MTLTrainer`.

---

## Alternative Fine-Tuning Strategies

### Adapter Layers

Insert small bottleneck modules (down-project → nonlinearity → up-project) into frozen transformer layers instead of using a single linear projection head.

**Why consider it:** The planned linear head may lack capacity to extract batter signal, which the doc identifies as the model's primary weakness (R²=0.02–0.06). Adapters add expressiveness while preserving pre-trained representations. In NLP, adapters match full fine-tuning performance with ~3% of trainable parameters.

**Implementation:** Add `AdapterModule(d_model, bottleneck_dim)` after each transformer layer's feed-forward block. Freeze all original parameters; train only adapter weights + prediction head.

### Gradual Unfreezing (ULMFiT-style)

Instead of a binary freeze/unfreeze decision, unfreeze transformer layers one at a time from top to bottom across training epochs.

**Why consider it:** Consistently outperforms both full-freeze and full-fine-tune in low-data regimes. Lower layers capture general patterns (pitch physics, game structure) that shouldn't change; upper layers capture task-specific abstractions that benefit from tuning. The Phase 4 plan already flags "freeze strategy" as an open decision — this is a well-studied answer.

**Implementation:** Trivial — set `requires_grad=False` on layer groups and toggle them on a schedule. No new modules needed.

### Prefix Tuning

Prepend a small number of learnable "soft prompt" vectors to the input sequence at fine-tune time, keeping the entire transformer frozen.

**Why consider it:** Very parameter-efficient. The learned prefix can encode task-specific context (e.g., "predict strikeouts" vs. "predict walks") without separate models or heads. A single pre-trained model serves multiple prediction targets with different prefixes.

**Tradeoff:** Less expressive than adapter layers. Works best when pre-training and fine-tuning domains are closely aligned (they are here — both are baseball sequences).

**Implementation cost:** Low — add `nn.Parameter(num_prefix_tokens, d_model)` and prepend to the input sequence before the transformer forward pass.

---

## Strategies Targeting Weak Batter Signal

The paper's batter R² values (0.02–0.06) are the model's primary weakness. These strategies specifically address the root cause: batters participate in ~1/9th the events per game that pitchers do.

### Asymmetric Context Windows

Use N=10 games for pitchers but N=30–50 games for batters, roughly equalizing the total event count each player type contributes to their embedding.

**Why consider it:** The paper uses N=10 for both, but a pitcher sees ~100 batters across 10 starts while a batter sees ~40 PAs. Equalizing event exposure is the most direct fix. The `PitchSequenceDataSource` already supports configurable game windows.

**Implementation cost:** Trivial — pass different `max_games` to the data source based on player type. May need to adjust positional encoding range if the model was designed for shorter sequences.

### Event-Weighted Attention

Add an auxiliary attention bias so that plate appearance outcomes (final pitch of each PA) receive higher weight than intermediate pitches in the batter's sequence.

**Why consider it:** For batter projection, the signal is overwhelmingly in PA outcomes (single, homer, strikeout), not in the pitch-by-pitch ball-strike count progression. Uniform attention over all pitches dilutes the outcome signal with noise from foul balls and called strikes.

**Implementation:** Add a learnable scalar bias to attention logits for positions flagged as PA-ending events. Small change to `PlayerAttentionMask`.

### Masked Plate Appearance Prediction

Replace per-pitch MGM with a batter-specific pre-training task: mask entire plate appearance outcomes and predict the PA result from the pitch sequence leading up to it.

**Why consider it:** Forces the model to learn the relationship between pitch sequences and outcomes — exactly the signal needed for batter projection. Standard MGM treats predicting a "ball" call the same as predicting a "home run", but these carry vastly different projection-relevant information.

**Implementation cost:** Medium — need PA-level masking logic in the dataset and a PA-outcome classification head. Could run alongside standard MGM as a multi-objective setup.

---

## Architecture Variations

### Hierarchical Transformer

Two levels of encoding: a pitch-level encoder within each plate appearance, then a PA-level encoder across the game/context window.

```
Pitches in PA → Pitch Encoder → PA embedding
PA embeddings across games → Game Encoder → Player embedding
```

**Why consider it:** Matches baseball's natural hierarchy (pitch → PA → inning → game). Dramatically reduces sequence lengths — a game becomes ~35 PA tokens instead of ~300 pitch tokens. Shorter sequences mean faster training, larger effective context windows (more games fit), and better long-range attention (transformer attention is O(n²) in sequence length).

**Tradeoff:** More complex architecture. The two-level encoding loses some cross-PA pitch-level interactions (a pitch in one PA can't directly attend to a pitch in another PA). Whether those interactions matter for projection is an empirical question.

**Implementation cost:** High — requires a second transformer module, a pooling strategy to collapse pitch sequences into PA embeddings, and modified data loading to preserve the PA hierarchy.

### Rotary Position Embeddings with Game Boundaries

Use RoPE (Rotary Position Embeddings) for within-game relative positions, with special `[GAME_SEP]` tokens between concatenated games.

**Why consider it:** Standard sinusoidal positional encoding treats position 150 (start of game 2) the same as position 150 in a single long game. RoPE encodes relative positions, so the model learns that "3 pitches ago in the same game" is different from "3 pitches ago but in a different game". Game boundary tokens make game transitions explicit.

**Implementation cost:** Low — RoPE is a drop-in replacement for sinusoidal encoding (`torch` ecosystem has implementations). Game separator tokens are a vocabulary addition.

---

## Data Strategies

### Curriculum Learning

Start training on high-event-count players (aces who pitch 200+ innings, everyday lineup hitters) where signal is strongest, then gradually mix in lower-event players (relievers, platoon bats, bench players).

**Why consider it:** Stabilizes early training. The model first learns clear performance patterns from high-sample players, then generalizes to noisier lower-sample players. Curriculum learning has shown consistent improvements in noisy-label and imbalanced-data regimes.

**Implementation:** Sort training data by player event count. In early epochs, sample only from the top quartile. Linearly expand the sampling pool across epochs until all players are included.

### Synthetic Schedule Simulation

Rather than the mean-over-N-predictions aggregation approach planned for Phase 5, train a separate head that takes the player embedding plus an opponent-distribution vector and directly predicts season-level stats.

**Why consider it:** Sidesteps the per-game → season aggregation gap entirely. The Phase 5 plan acknowledges this gap as a key open decision. A season-projection head trained on (player embedding, schedule features) → season stats avoids compounding per-game prediction errors across 162 games.

**Tradeoff:** Requires defining opponent-distribution features (e.g., average opposing pitcher/batter quality, park factor distribution, division schedule weighting). More modeling assumptions than simple averaging.

### Cross-League Pre-Training

Pre-train on combined MLB + MiLB data. AAA Statcast data has been available since 2023; lower levels have partial coverage.

**Why consider it:** Increases training data volume. Also directly addresses the rookie cold-start problem (documented in Risks): players without MLB history would have MiLB sequences available. Aligns with the existing MLE work referenced in the main doc.

**Tradeoff:** MiLB data quality is lower (fewer tracked fields, different pitch tracking systems at some levels). Domain gap between MiLB and MLB competition levels could hurt if not accounted for (a MiLB home run is not the same signal as an MLB home run). Could add a league-level embedding to let the model learn this distinction.

---

## Prioritization

Given the documented risks and the goal of beating `marcel_gb` (HR=0.678, SB=0.736, rank rho=0.603), the highest-impact experiments ranked by expected value:

| Priority | Strategy | Addresses | Effort | Expected Impact |
|---|---|---|---|---|
| 1 | ELECTRA-style pre-training | Limited data volume | Medium | High — 4x sample efficiency from same data |
| 2 | Asymmetric context windows | Weak batter signal | Trivial | Medium — equalizes event exposure |
| 3 | Gradual unfreezing | Fine-tuning stability | Trivial | Medium — well-established improvement |
| 4 | Hierarchical transformer | Batter signal + training speed | High | High — but significant architecture change |
| 5 | Multi-objective pre-training | Representation quality | Low | Medium — complementary signal at no data cost |
| 6 | Adapter layers | Fine-tuning expressiveness | Low | Low-Medium — incremental over linear head |

Recommendation: implement MGM first (as planned), then run ELECTRA as the first comparison experiment. Add asymmetric context windows and gradual unfreezing as low-cost improvements to either approach. Consider the hierarchical transformer only if flat-sequence results plateau.

---

## Cloud GPU Training

Training these models locally is impractical for the full dataset. See [Cloud GPU Training Guide](contextual-cloud-gpu-training.md) for provider options, setup, and the code changes needed to run on cloud infrastructure.
