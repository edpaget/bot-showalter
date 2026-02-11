# Contextual Model v2 — In-Season Extensions

## What

Extend the contextual model v2 hierarchical architecture to support game-level, matchup-aware predictions suitable for in-season decisions (sit/start, streaming, waiver evaluation). The v2 roadmap produces season-level rate projections; these extensions add the matchup context, prediction horizon flexibility, and environmental features needed to power the downstream in-season tools (start-sit optimizer, matchup analyzer, ROS updater).

## Why

The v2 hierarchical architecture (pitch → PA → game → projection) is well-suited for in-season use: the 30-day stat window captures current form, Level 3 models game-sequence trends, and incremental inference is efficient (process one new game through Levels 1-2, re-run Level 3). But it models players in isolation. Sit/start decisions are fundamentally about matchups — the same batter is a strong start against a soft-tossing lefty and a clear sit against an elite righty. Without opponent context, park factors, and game-level prediction targets, the contextual model can't serve as the projection engine for the in-season tools that already exist as proposals.

## Relationship to Existing Work

This proposal bridges two bodies of work:

- **Contextual model v2 roadmap** (`docs/plans/contextual-model-v2.md`): Defines the hierarchical architecture and player identity system. This proposal extends it, not replaces it. All extensions depend on Phases 1-3 of the v2 roadmap being complete.
- **In-season tool proposals** (`docs/proposals/in-season/`): The start-sit optimizer, matchup analyzer, and ROS projector all consume player projections but don't specify where matchup-adjusted projections come from. These extensions provide that source.
- **Platoon splits proposal** (`docs/proposals/platoon-splits.md`): Proposes split-based rate projections using traditional stats. The matchup conditioning described here subsumes platoon effects in a richer representation — the contextual model learns platoon-like effects from the pitch/PA data itself, conditioned on opponent identity features.

## Extension 1: Matchup Conditioning

### Problem

The v2 plan conditions predictions solely on the player's own identity (stat features + archetype). A batter's expected HR rate against a high-spin fastball pitcher in Coors Field is very different from their rate against a sinker/slider pitcher in Oracle Park. The model has no way to express this.

### Design

Add an **opponent identity vector** as a second conditioning signal at Level 3. Reuse the same identity infrastructure from Phase 1 of the v2 roadmap:

- Compute a `PlayerStatProfile` for the opposing pitcher (for batter predictions) or opposing lineup aggregate (for pitcher predictions)
- Assign an archetype to the opponent using the same clustering
- Produce an `opponent_identity_repr` (d=96) using the same MLP + archetype embedding architecture

At Level 3, the game → projection attention receives two conditioning signals:
- `player_identity_repr`: "who is this player?" (existing)
- `opponent_identity_repr`: "who are they facing?" (new)

The combined conditioning captures platoon effects naturally — a left-handed opponent archetype conditions the model differently than a right-handed one — without requiring explicit split stat features.

**For pitcher predictions:** The opponent is the aggregate lineup. Compute a lineup-level stat profile (mean of expected starter stat profiles, weighted by lineup position) and assign a lineup archetype.

**For batter predictions:** The opponent is the starting pitcher. Use their stat profile and archetype directly.

### Training

Each training sample already has a date and player. Look up the opposing pitcher/lineup for that game from Retrosheet/Statcast game logs. Compute their stat profile as of that date (same windowing as the player's own profile). The matchup signal is learned end-to-end.

### Why This Before More Complex Interaction Models

A full batter-pitcher interaction model (cross-attention between both players' pitch sequences) would be powerful but requires paired training data, dramatically increases compute, and introduces a chicken-and-egg problem at inference (you need the opponent's context window too). Conditioning on opponent identity features captures the most important matchup effects (platoon, pitch-type tendency, K/BB tendency) at minimal architectural cost.

## Extension 2: Park and Environment Features

### Problem

Park factors shift HR rates by 20-30% and affect other stats meaningfully. For season-level projections this washes out across the schedule, but for game-level predictions it's a primary driver.

### Design

Add a small environment feature vector to the Level 3 prediction head input:

```
env_features = [
    park_hr_factor,      # e.g., 1.30 for Coors, 0.85 for Oracle
    park_h_factor,
    park_bb_factor,
    is_home,             # home/away indicator
    temperature_bucket,  # cold/mild/warm/hot (4 bins)
    is_dome,             # indoor venue flag
]
```

Park factors are already well-established (available from FanGraphs or ESPN park factors). These are concatenated to the Level 3 output before the prediction head — they modify the final prediction without interfering with the temporal modeling.

For season-level projections, pass league-average environment features (all factors = 1.0, temperature = mild) so the model's existing behavior is preserved.

### Data Source

Park factors: FanGraphs publishes multi-year park factors annually. Cache per-venue factors as a static lookup table, updated yearly.

Game-level weather: Retrosheet game logs include temperature for historical training data. For inference, weather APIs or pre-game data feeds provide day-of temperature.

## Extension 3: Multi-Horizon Prediction Heads

### Problem

The v2 plan uses a single prediction target: rate over the next N games (currently 5-game average). Sit/start needs single-game predictions; ROS projections need 30+ game horizons. These have fundamentally different variance profiles — a player's single-game HR probability is ~5-15% while their 30-game HR rate is much more stable and predictable.

### Design

Replace the single prediction head with multiple horizon-specific heads sharing the same backbone:

```
Level 3 output (game-sequence representation)
  → head_1game:  predict next-game rates  (high variance, emphasize matchup + recency)
  → head_7game:  predict next-week rates  (medium variance, weekly sit/start)
  → head_30game: predict next-30-day rates (lower variance, ROS blending)
```

Each head is a small MLP (2 layers) with its own parameters. All three share the same Level 1-2-3 backbone. During training, each sample produces three targets at different horizons (the actual rates over the next 1, 7, and 30 games). The total loss is a weighted sum.

### Variance-Aware Loss

Single-game targets are inherently noisy — a player either hit a HR or didn't. Use heteroscedastic loss that lets each head learn its own uncertainty:

```
loss_h = (pred_h - target_h)^2 / (2 * sigma_h^2) + log(sigma_h)
```

where `sigma_h` is a learned per-horizon, per-stat uncertainty parameter. This prevents the noisy single-game head from destabilizing the backbone.

### Inference

The downstream tool selects the appropriate head:
- Start-sit optimizer → `head_1game` (with matchup + park features)
- Weekly matchup analyzer → `head_7game`
- ROS projection updater → `head_30game`

## Extension 4: Rate-to-Counting-Stat Conversion

### Problem

Fantasy decisions are about counting stats and fantasy points, not rates. Predicting a 12% HR rate is useless without knowing how many PA the player will get in the prediction window. PA opportunity depends on lineup position, game pace, and whether the player is even in the starting lineup.

### Design

Add a lightweight PA/IP estimator that runs alongside the rate prediction:

**For batters:**
- Inputs: lineup position (1-9 or bench), team projected game total, opponent pitcher pace (pitches/game as proxy for game length)
- Output: expected PA for the game
- Method: lookup table from historical lineup-position-to-PA distributions, adjusted by game pace. No ML needed — this is well-characterized empirically (~4.5 PA for leadoff, ~3.8 PA for 8th, etc.).

**For pitchers:**
- Inputs: starter vs. reliever, recent workload (IP/start over last 5 starts), opponent lineup quality
- Output: expected IP for the game
- Method: similar lookup from recent usage patterns

Counting stat projections = predicted rate * expected PA/IP. This feeds directly into the matchup analyzer's category projection step and the start-sit optimizer's scoring.

### Data Source

Lineup position: available from pre-game lineup postings (MLB API publishes confirmed lineups ~2-4 hours before game time). For weekly projections, use recent lineup position trends.

## Extension 5: Inference Pipeline for Daily Updates

### Problem

In-season tools need predictions updated daily as new games complete. The v2 plan doesn't address inference cadence, data freshness, or how the prediction pipeline runs during the season.

### Design

Build an `InSeasonPredictionService` that orchestrates daily updates:

1. **After each day's games complete** (~1am ET):
   - Fetch new game data (pitch-level from Statcast, box scores for stats)
   - Process each new game through Levels 1-2 → produce new game embeddings
   - Append to each player's game embedding cache

2. **On prediction request** (e.g., "give me start-sit recommendations for Tuesday"):
   - Load the player's cached game embedding sequence
   - Load the opponent's identity features for the upcoming game
   - Load park/environment features
   - Run Level 3 + prediction heads → matchup-adjusted rate predictions at desired horizon
   - Convert rates to counting stats using PA/IP estimator

3. **Caching strategy:**
   - Game embeddings (Levels 1-2 output) are immutable once computed — cache aggressively
   - Stat profiles need recomputation as the 30-day window slides — recompute daily
   - Archetype assignments are stable within a season — recompute weekly or on significant stat changes

The key efficiency insight: Levels 1-2 (the expensive part — processing pitch sequences through the transformer) only run once per game. Level 3 (the cheap part — attention over ~30 game embeddings) runs on every prediction request with fresh matchup/environment context. This makes game-level predictions fast at inference time.

### Latency Target

Level 3 inference for a single player should complete in <100ms on CPU, making real-time lineup optimization feasible. The batch update (processing all games from a day through Levels 1-2) can run as a background job.

## Dependencies and Ordering

All extensions depend on the v2 roadmap Phases 1-3 being complete (identity foundation, hierarchical architecture, residual learning).

Suggested ordering:

1. **Extension 1 (Matchup conditioning)** — highest impact for sit/start, reuses existing identity infrastructure
2. **Extension 2 (Park/environment features)** — straightforward, low risk, meaningful for game-level accuracy
3. **Extension 3 (Multi-horizon heads)** — enables different downstream tools to use appropriate prediction granularity
4. **Extension 5 (Inference pipeline)** — required for any in-season use; could be built in parallel with 1-3 using the existing single-horizon model as a placeholder
5. **Extension 4 (Rate-to-counting conversion)** — needed for the final integration with start-sit and matchup analyzers, but simpler and can come last

Extensions 1 and 2 can be trained jointly (add matchup + environment features in the same training run). Extension 3 requires a separate training phase but uses the same data.

## Open Questions

- **Opponent identity granularity for pitchers:** Should the opponent be the full lineup, the top-6 hitters, or individual batter-by-batter? Full lineup is simplest but loses information about lineup construction. Individual batters would require per-batter predictions aggregated up, which is architecturally complex.
- **Training data for matchup conditioning:** Historical game logs provide the opposing pitcher/lineup, but constructing opponent stat profiles "as of that date" requires point-in-time stat computation for every opponent in every training game. This is a meaningful data engineering effort.
- **Multi-horizon target noise:** Single-game targets are binary events (HR or not). Should the 1-game head predict probability rather than rate, using binary cross-entropy loss instead of MSE? This changes the interpretation but may train more stably.
- **Inference without confirmed lineups:** For weekly projections, confirmed lineups aren't available yet. Should the model fall back to season-average matchup features, or use projected probable pitchers?
- **Interaction with ROS projections:** The ROS proposal uses Bayesian blending of pre-season projections with in-season actuals. The 30-game prediction head produces a similar output. Should the ROS projector consume the contextual model's 30-game predictions as its "in-season signal" instead of raw stats? This would give the ROS projector access to pitch-level context without building its own feature pipeline.
- **Retraining cadence:** Should the model be retrained mid-season as new data accumulates, or is pre-season training sufficient for the full year? The identity features update dynamically, but the model weights are fixed.
