Best Experimental Targets

  1. Temporal Fusion Transformer (from #2)

  Why it's interesting:
  - Combines attention mechanisms with recurrent layers
  - Learns which past seasons matter most (interpretable attention weights)
  - Handles both static features (position, handedness) and temporal sequences
  - The https://www.researchgate.net/publication/391148375_Pitcher_Performance_Pred
  iction_Major_League_Baseball_MLB_by_Temporal_Fusion_Transformer showed it beat
  RNNs and traditional systems
  - PyTorch Forecasting library has a solid implementation

  What you'd learn: Attention, temporal modeling, multi-horizon forecasting

  2. Player Embeddings + kNN/downstream model (from #1)

  Why it's interesting:
  - Fundamentally different paradigm—learn player similarity, then project via
  neighbors
  - The https://www.sloansportsconference.com/research-papers/batter-pitcher-2vec-s
  tatistic-free-talent-modeling-with-neural-player-embeddings work showed
  embeddings capture things stats miss
  - Can visualize embeddings (t-SNE/UMAP) to see if clusters make baseball sense
  - Contrastive learning variant adds another technique to explore

  What you'd learn: Embeddings, representation learning, similarity-based inference

  3. Multi-Task Neural Network (from #3)

  Why it's interesting:
  - Simpler architecture (good starting point)
  - Directly exploits the correlation structure in your stats
  - Easy to compare: same features as Marcel, but learned non-linear mappings
  - Can inspect shared layer activations

  What you'd learn: NN fundamentals, MTL loss weighting, architecture design

  Suggested Experiment Plan
  Approach: Multi-Task NN
  Complexity: Low
  Novelty vs Marcel: Medium
  Time Investment: Good first experiment
  ────────────────────────────────────────
  Approach: TFT
  Complexity: Medium-High
  Novelty vs Marcel: High
  Time Investment: Best baseball validation
  ────────────────────────────────────────
  Approach: Player Embeddings
  Complexity: Medium
  Novelty vs Marcel: High
  Time Investment: Most conceptually different
  I'd suggest:
  1. Start with Multi-Task NN — Get your NN infrastructure working, establish
  baseline
  2. Then TFT — Most likely to actually compete with Marcel based on research
  3. Player Embeddings — Most novel, good for handling prospects/thin data
