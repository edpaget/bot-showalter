# Experimental Machine Learning Approaches

## Overview

This document outlines experimental ML techniques that could provide larger gains than incremental adjustments to the current Marcel-based pipeline. These range from lower-risk extensions (gradient boosting on residuals) to more ambitious architectural changes (neural embeddings, sequence models).

The current system is grounded in interpretable Bayesian regression with linear blending of Statcast signals. These approaches explore non-linear modeling, learned representations, and uncertainty quantification.

---

## High-Potential Approaches

### 1. Player Embeddings via Neural Networks

#### What

Learn dense vector representations (embeddings) of players from their full stat profiles, Statcast data, and career trajectories. Players who are similar in embedding space can share information for projection purposes.

#### Why

Marcel treats each player independently except for regression toward league mean. Embeddings allow the model to recognize that a 25-year-old with a specific Statcast profile, plate discipline pattern, and minor league history resembles other players who went on to break out. This is especially powerful for:

- Players with limited MLB data (borrow strength from similar players)
- Identifying "development archetypes" (e.g., late-blooming power, early-peak speedster)
- Detecting regime changes when a player's embedding shifts suddenly between seasons

#### Implementation Sketch

1. Define input features: age, position, 3-year stat history, Statcast profile, physical attributes
2. Train an autoencoder or contrastive model to learn 32-64 dimensional embeddings
3. For projection, find k-nearest neighbors in embedding space
4. Blend neighbor trajectories weighted by similarity, or use embeddings as features in a downstream model

#### Data Requirements

- Multi-year player stat histories (already available)
- Statcast profiles (already integrated)
- Optional: minor league stats, physical measurements (height/weight), draft position

#### Key References

- Mikolov et al., "Distributed Representations of Words and Phrases" (Word2Vec methodology)
- MLB player similarity scores (Bill James methodology, for comparison)
- "Deep Learning for Baseball" (various blog posts on player embeddings)

#### Complexity

High. Requires neural network infrastructure, careful feature engineering, and validation that embeddings capture meaningful similarity.

---

### 2. Sequence Models for Career Trajectories

#### What

Use LSTMs or Transformers trained on season-by-season stat sequences to model career trajectories, replacing or augmenting multiplicative aging curves.

#### Why

Current aging curves assume a fixed functional form (peak age, linear decline rate) with position modifiers. Real careers are messier:

- Injury recovery follows non-linear patterns
- Breakouts and collapses have detectable signatures in prior seasons
- Position changes, swing changes, and role changes create discontinuities
- Some players age gracefully, others fall off cliffs

Sequence models can learn these patterns from data rather than assuming a parametric form.

#### Implementation Sketch

1. Represent each player-season as a feature vector (stats, age, Statcast, role)
2. Train LSTM/Transformer to predict next-season stats given history
3. Condition on player metadata (position, handedness) via embeddings
4. At inference, feed player's history and generate projection
5. Could ensemble with Marcel for stability

#### Data Requirements

- Complete career histories for training (decades of data available via Lahman/pybaseball)
- Consistent feature representation across eras (may need era adjustments)

#### Key References

- Hochreiter & Schmidhuber, "Long Short-Term Memory" (LSTM architecture)
- Vaswani et al., "Attention Is All You Need" (Transformer architecture)
- Time series forecasting literature (N-BEATS, Temporal Fusion Transformers)

#### Complexity

High. Sequence models require careful handling of variable-length histories, era effects, and can overfit on small datasets. May need regularization or pretraining.

---

### 3. Multi-Task Learning for Correlated Stats

#### What

Train a single neural network to predict all stats simultaneously with shared hidden layers, exploiting correlations between stats.

#### Why

Fantasy stats are deeply correlated:

- K% affects BABIP (fewer balls in play)
- Barrel rate drives HR, 2B, and SLG
- Sprint speed affects SB, triples, and infield hit rate
- BB% and K% together indicate plate discipline
- For pitchers: K% affects ERA, WHIP, LOB%

Current approach projects each stat somewhat independently. MTL forces the model to learn a shared representation that respects these correlations, providing regularization and potentially better generalization.

#### Implementation Sketch

1. Design network with shared trunk (2-3 hidden layers) and stat-specific heads
2. Loss function: weighted sum of per-stat losses (weights by fantasy importance or variance)
3. Train on (features → all stats) pairs
4. Shared layers learn general player quality; heads learn stat-specific mappings

#### Data Requirements

- Standard projection features (same as current pipeline)
- No additional data needed; this is an architectural change

#### Key References

- Caruana, "Multitask Learning" (original MTL paper)
- Ruder, "An Overview of Multi-Task Learning in Deep Neural Networks"

#### Complexity

Medium. Well-understood technique with good library support. Main challenge is tuning loss weights and architecture.

---

### 4. Bayesian Neural Networks for Uncertainty Quantification

#### What

Replace point estimates with full posterior distributions over projections using Bayesian neural networks or MC Dropout.

#### Why

For fantasy decision-making, knowing the variance of a projection is as important as the mean:

- High-variance players are better late-round picks (upside)
- Low-variance players are safer early picks (floor)
- Trade decisions depend on confidence intervals
- Weekly start/sit depends on ceiling vs floor

Current Marcel provides some implicit uncertainty (regression amount reflects certainty), but doesn't give usable distributions.

#### Implementation Sketch

1. Train neural network with dropout
2. At inference, run multiple forward passes with dropout enabled (MC Dropout)
3. Collect distribution of outputs; report mean and percentiles
4. Alternative: use variational inference (Bayes by Backprop) for true posteriors

#### Data Requirements

- Same as standard neural net approaches
- Validation set to calibrate uncertainty estimates

#### Key References

- Gal & Ghahramani, "Dropout as a Bayesian Approximation"
- Blundell et al., "Weight Uncertainty in Neural Networks"
- Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation"

#### Complexity

Medium. MC Dropout is simple to implement. Calibration of uncertainty estimates requires care.

---

### 5. Mixture of Experts by Player Archetype

#### What

Train multiple specialized models (experts) for different player archetypes, with a learned gating network that routes each player to the appropriate expert(s).

#### Why

A one-size-fits-all model may underfit the distinct dynamics of different player types:

- Power-over-contact hitters (high K%, high HR) age and project differently than contact-first hitters
- High-K pitchers have different BABIP/ERA dynamics than groundball pitchers
- Speedsters vs sluggers have different value trajectories

Mixture of experts can capture these differences while sharing statistical strength where appropriate.

#### Implementation Sketch

1. Define 4-8 expert networks (can be small MLPs)
2. Gating network takes player features, outputs soft assignment to experts
3. Final prediction is weighted average of expert outputs
4. Train end-to-end; gating learns to specialize experts

#### Data Requirements

- Standard features
- May benefit from explicit archetype labels for initialization (e.g., cluster players first)

#### Key References

- Jacobs et al., "Adaptive Mixtures of Local Experts"
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"

#### Complexity

Medium-High. Architecture is more complex, but interpretability is good (can inspect which expert handles which players).

---

### 6. Graph Neural Networks for Team Context

#### What

Model players as nodes in a graph with edges representing team relationships, lineup adjacency, and divisional matchups. Use GNNs to propagate contextual information.

#### Why

Player performance depends on context that current projections ignore:

- Lineup protection (hitters around you affect pitches you see)
- Team run environment (more RBI opportunities on high-scoring teams)
- Bullpen quality (affects pitcher W/L, hold opportunities)
- Division opponents (AL East vs AL Central difficulty)
- Manager tendencies (aggressiveness on bases, bullpen usage)

#### Implementation Sketch

1. Build graph: players as nodes, edges for teammates/lineup-adjacent/division-opponents
2. Node features: player stats and projections
3. Edge features: relationship type, playing time overlap
4. GNN layers propagate information (e.g., GraphSAGE, GAT)
5. Output layer predicts adjusted projections incorporating context

#### Data Requirements

- Team rosters and lineup data
- Historical lineup configurations
- Manager/team tendency stats

#### Key References

- Hamilton et al., "Inductive Representation Learning on Large Graphs" (GraphSAGE)
- Veličković et al., "Graph Attention Networks"

#### Complexity

High. Requires graph construction, GNN infrastructure, and careful validation that context effects are real and learnable.

---

### 7. Contrastive Learning for Comp Players

#### What

Learn player similarity in terms of future performance rather than current stats, using contrastive learning objectives.

#### Why

Traditional "comp" players are based on current stat similarity, but we care about future trajectory similarity. A player with unusual stats may have no good current-stat comps but may resemble other players who went on to similar outcomes.

#### Implementation Sketch

1. Define positive pairs: players whose next-year stats were similar
2. Define negative pairs: players whose next-year stats diverged
3. Train encoder to maximize similarity of positive pairs, minimize for negatives
4. For new player, find nearest neighbors in learned space
5. Project using neighbor trajectories

#### Data Requirements

- Multi-year stat histories
- Outcome data for contrastive supervision

#### Key References

- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
- Oord et al., "Representation Learning with Contrastive Predictive Coding"

#### Complexity

Medium-High. Contrastive learning is well-established but requires careful pair mining and negative sampling.

---

## Lower-Lift Experiments

### 8. Gradient Boosting on Marcel Residuals (Recommended Starting Point)

#### What

Keep Marcel as the base projection, but train XGBoost/LightGBM to predict Marcel's errors from additional features.

#### Why

This approach:

- Preserves the existing pipeline as a strong baseline
- Doesn't require architectural changes
- Can incorporate features Marcel doesn't use (swing decisions, pitch mix, trends)
- Provides feature importance for interpretability
- Has mature, well-understood tooling

If there are systematic patterns in Marcel's errors, gradient boosting will find them.

#### Implementation Sketch

1. Generate Marcel projections for historical seasons
2. Compute residuals: `actual - marcel_projected`
3. Assemble additional features not in Marcel:
   - Statcast swing decisions (chase rate, whiff rate, meatball swing%)
   - Sprint speed and acceleration
   - Pitch mix changes (for pitchers)
   - Batted ball direction (pull%, oppo%)
   - Month-by-month trends from prior year
   - Age × stat interactions
4. Train XGBoost to predict residuals from features
5. Final projection: `marcel + xgb_correction`
6. Inspect feature importances to understand what drives corrections

#### Data Requirements

- Historical Marcel projections and actuals (can generate from existing pipeline)
- Statcast swing/take data (pybaseball)
- Pitch-level data for pitch mix (pybaseball)

#### Key References

- Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System"
- Friedman, "Greedy Function Approximation: A Gradient Boosting Machine"

#### Complexity

Low-Medium. Standard ML workflow with good library support. Main work is feature engineering.

---

### 9. Gaussian Processes for Flexible Aging

#### What

Replace multiplicative aging curves with Gaussian Processes that learn smooth, non-parametric aging functions with uncertainty.

#### Why

Current aging curves assume a fixed functional form. GPs can:

- Learn arbitrary smooth functions from data
- Provide uncertainty estimates (wider bands for ages with less data)
- Pool information across players while allowing individual variation
- Handle the fact that aging is heterogeneous (some players age well, others don't)

#### Implementation Sketch

1. Define GP prior over age → stat-multiplier function
2. Fit GP to historical data (age, player features) → (actual / projected_without_age)
3. Use GP posterior mean as aging adjustment, posterior variance for uncertainty
4. Could fit separate GPs per stat or use multi-output GP

#### Data Requirements

- Historical projections and actuals across age ranges
- Player metadata for hierarchical modeling

#### Key References

- Rasmussen & Williams, "Gaussian Processes for Machine Learning"
- GPyTorch, GPflow libraries

#### Complexity

Medium. GPs are well-understood but can be computationally expensive. Sparse GP approximations help.

---

### 10. Minor League Transfer Learning

#### What

Pre-train a projection model on minor league data (more samples), then fine-tune on MLB data.

#### Why

Young MLB players have thin track records but may have multiple minor league seasons. Transfer learning can:

- Leverage MiLB signal for prospects
- Learn general "player development" patterns that transfer to MLB
- Improve projections for call-ups and rookies

Requires minor league equivalencies (MLE) to translate MiLB stats to MLB scale.

#### Implementation Sketch

1. Assemble MiLB stat histories with level indicators
2. Apply or learn minor league equivalency translations
3. Pre-train projection model on MiLB → next-year-MiLB or MiLB → MLB-debut
4. Fine-tune on MLB → MLB projection task
5. For players with MiLB history, include translated MiLB stats as features

#### Data Requirements

- Minor league statistics (available via various sources, some scraping required)
- Level-specific park/league factors for MLE

#### Key References

- Silver, N. "Minor League Equivalencies" (PECOTA methodology)
- Transfer learning literature (pre-train/fine-tune paradigm)

#### Complexity

Medium-High. MiLB data acquisition and MLE calculation add complexity. Transfer learning mechanics are straightforward.

---

## Recommendation

**Start with Gradient Boosting on Residuals (#8).** It has the best risk/reward ratio:

- Low implementation complexity
- Preserves existing pipeline
- Immediate interpretability via feature importance
- Clear evaluation against Marcel baseline
- If it doesn't help, you've learned that Marcel's errors aren't systematically predictable from available features

If that shows promise, **Player Embeddings (#1)** is the most interesting longer-term investment for building something fundamentally different from traditional projection systems.

---

## Evaluation Strategy

For any approach, evaluate using the existing backtesting framework:

1. Train on years Y-3 to Y-1, project year Y
2. Measure RMSE and Spearman rho vs actuals
3. Compare to Marcel baseline and current best pipeline
4. Repeat for multiple years (2021-2024) to assess consistency
5. Inspect failure modes: which players/stats improve or degrade?

Be wary of overfitting: ML models can memorize training data. Use proper train/validation/test splits and monitor for degradation on held-out years.
