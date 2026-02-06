# Skill Change Approach for Projection Adjustments

## Background: Why Gradient Boosting on Residuals Underperformed

The initial ML approach trained gradient boosting models to predict Marcel projection residuals (actual - projected) using Statcast features. This delivered marginal improvement for hitters and worse results for pitchers.

**Root cause**: Marcel residuals are mostly noise, not signal.

Marcel already captures the most predictive information (weighted historical performance regressed to the mean). What remains after Marcel is predominantly random variance—luck, injuries, BABIP fluctuation, sequencing effects. Training a model to predict noise yields noise.

Additional issues:
- **Feature redundancy**: Model features included Marcel rates, duplicating information Marcel already used
- **Statcast overlap**: xBA, xSLG, xwOBA correlate highly with actual results Marcel incorporated
- **Pitcher variance**: Pitching stats have higher variance than hitting, making residuals even noisier
- **Sample limitations**: Training on hundreds of samples with 18-21 features risks overfitting

## Alternative: Skill Change Detection

Instead of predicting performance residuals, detect when underlying skills have changed in ways Marcel's weighted-average approach would miss.

**Key insight**: Marcel assumes skills are stable (just regressing toward the mean). When skills genuinely change, Marcel's assumption breaks down.

### Conceptual Difference

| Aspect | Residual Prediction | Skill Change Detection |
|--------|---------------------|------------------------|
| Target | Counting stat error | Underlying skill delta |
| Signal quality | Noisy (luck + skill) | Cleaner (skill only) |
| Output | Continuous adjustment | Binary: changed or stable |
| Sample requirement | Large (ML training) | Small (two years of data) |
| Interpretability | Black box | Explicit rules |

### Approach

```
Features(Year N) vs Features(Year N-1) → Detect skill delta → Adjust projection
```

Most players receive no adjustment (skills stable). For the ~10-15% with genuine skill changes, apply targeted corrections. This avoids the failure mode of making small adjustments to everyone that net to noise.

## Skill Metrics by Predictive Signal

Research on year-over-year stability and predictive power of various metrics.

### Tier 1: Highly Predictive & Stable

#### Batters

| Metric | YoY Stability | Predictive Of | Notes |
|--------|---------------|---------------|-------|
| Barrel Rate | High | HR, ISO | r=0.61 with future HR. Premier power metric |
| Exit Velocity (90th pctl) | High | Power, wOBAcon | More stable than max EV |
| Chase Rate | Very High | K%, BB%, wOBA | Stabilizes quickly (~50 pitches). Hard to change |
| Hard Hit Rate | High | Power output | Stabilizes after ~50 balls in play |

#### Pitchers

| Metric | YoY Stability | Predictive Of | Notes |
|--------|---------------|---------------|-------|
| Fastball Velocity | Very High | K%, future ERA | Most predictive single skill (R²≈0.08) |
| Stuff+ (aggregate) | High | Next-year FIP | Better than FIP itself for YoY prediction |
| Whiff Rate | High | K% | R²≈0.75 correlation with strikeouts |
| Fastball Spin Rate | High | K%, ERA | Second most predictive after velocity |

### Tier 2: Moderately Predictive

**Batters:**
- Sprint Speed: Correlates year-to-year but stolen base success is volatile
- xwOBA: Useful but largely reflects same info as actuals
- Whiff Rate: Predictive for K% but not BB%

**Pitchers:**
- GB/FB Ratio: Stable skill, modest ERA impact
- Extension: Affects perceived velocity
- Slider Spin: Third most predictive (R²≈0.023)

### Tier 3: Low Signal (Avoid)

- BABIP (both): Mostly noise year-to-year
- HR/FB Rate (pitchers): High variance, weak YoY correlation
- Strand Rate: Largely luck
- xBA/xSLG alone: Correlate highly with what Marcel already sees

## Implementation Design

### Skill Delta Detection

```python
@dataclass
class SkillDelta:
    """Year-over-year skill change for a player."""
    player_id: str
    year: int

    # Batter skills
    exit_velo_90th_delta: float | None = None
    barrel_rate_delta: float | None = None
    chase_rate_delta: float | None = None
    hard_hit_rate_delta: float | None = None
    sprint_speed_delta: float | None = None

    # Pitcher skills
    fastball_velo_delta: float | None = None
    whiff_rate_delta: float | None = None
    stuff_plus_delta: float | None = None

    # Sample sizes for confidence
    pa_current: int = 0
    pa_prior: int = 0

    def is_significant(self, min_pa: int = 200) -> bool:
        """Require sufficient sample in both years."""
        return self.pa_current >= min_pa and self.pa_prior >= min_pa
```

### Adjustment Thresholds

Thresholds calibrated to filter noise while catching real changes:

```python
BATTER_THRESHOLDS = {
    "barrel_rate": {
        "threshold": 0.02,       # 2 percentage points
        "affects": ["hr", "doubles"],
        "adjustment_factor": 0.5,  # ISO adjustment multiplier
    },
    "exit_velo_90th": {
        "threshold": 1.5,        # mph
        "affects": ["hr"],
        "adjustment_factor": 0.007,  # ~1 HR per 100 PA per 1.5 mph
    },
    "chase_rate": {
        "threshold": 0.03,       # 3 pct points (rare to change)
        "affects": ["bb", "so"],
        "adjustment_factor": 0.15,  # BB/K rate adjustment
    },
    "hard_hit_rate": {
        "threshold": 0.03,       # 3 percentage points
        "affects": ["hr", "doubles"],
        "adjustment_factor": 0.3,
    },
}

PITCHER_THRESHOLDS = {
    "fastball_velo": {
        "threshold": 1.0,        # mph
        "affects": ["so", "er"],
        "adjustment_factor": {
            "so": 0.005,         # K/out rate per mph
            "er": -0.002,        # ER/out rate per mph
        },
    },
    "whiff_rate": {
        "threshold": 0.03,       # 3 percentage points
        "affects": ["so"],
        "adjustment_factor": 0.8,  # Direct K rate impact
    },
    "stuff_plus": {
        "threshold": 10,         # Stuff+ points
        "affects": ["so", "er", "bb"],
        "adjustment_factor": 0.003,
    },
}
```

### Adjuster Implementation

```python
class SkillChangeAdjuster:
    """Adjusts projections based on detected skill changes."""

    def __init__(
        self,
        statcast_source: StatcastDataSource,
        batter_thresholds: dict = BATTER_THRESHOLDS,
        pitcher_thresholds: dict = PITCHER_THRESHOLDS,
        min_pa: int = 200,
    ):
        self.statcast_source = statcast_source
        self.batter_thresholds = batter_thresholds
        self.pitcher_thresholds = pitcher_thresholds
        self.min_pa = min_pa

    def compute_delta(
        self,
        player_id: str,
        year: int,
    ) -> SkillDelta | None:
        """Compute skill deltas between year-1 and year-2."""
        current = self.statcast_source.get_stats(player_id, year - 1)
        prior = self.statcast_source.get_stats(player_id, year - 2)

        if current is None or prior is None:
            return None

        return SkillDelta(
            player_id=player_id,
            year=year,
            barrel_rate_delta=current.barrel_rate - prior.barrel_rate,
            exit_velo_90th_delta=current.ev_90th - prior.ev_90th,
            chase_rate_delta=current.chase_rate - prior.chase_rate,
            # ... etc
            pa_current=current.pa,
            pa_prior=prior.pa,
        )

    def adjust_batter(
        self,
        player: PlayerRates,
        delta: SkillDelta,
    ) -> PlayerRates:
        """Apply skill-based adjustments to batter projection."""
        if not delta.is_significant(self.min_pa):
            return player

        rates = dict(player.rates)
        adjustments_made = {}

        for skill, config in self.batter_thresholds.items():
            skill_delta = getattr(delta, f"{skill}_delta", None)
            if skill_delta is None:
                continue

            if abs(skill_delta) < config["threshold"]:
                continue  # Not a significant change

            # Apply adjustment
            factor = config["adjustment_factor"]
            for stat in config["affects"]:
                if stat in rates:
                    adjustment = skill_delta * factor
                    rates[stat] = rates[stat] + adjustment
                    adjustments_made[f"{skill}->{stat}"] = adjustment

        if not adjustments_made:
            return player

        return PlayerRates(
            player_id=player.player_id,
            name=player.name,
            year=player.year,
            age=player.age,
            rates=rates,
            opportunities=player.opportunities,
            metadata={
                **player.metadata,
                "skill_change_adjustments": adjustments_made,
            },
        )
```

## Key Design Principles

1. **Conservative by default**: Most players get no adjustment. Only act on clear evidence.

2. **High threshold for action**: The metrics with strongest signal are also most stable. When they change significantly, it's likely real.

3. **Transparent adjustments**: Each adjustment traces to a specific skill change, unlike black-box ML predictions.

4. **Sample size gates**: Require sufficient PA in both years before trusting any delta.

5. **Additive to Marcel**: This approach complements Marcel rather than replacing it. Marcel handles the baseline; skill change detection handles the edge cases.

## Data Requirements

To implement this approach, need year-over-year access to:

**Batters (from Statcast):**
- Exit velocity percentiles (especially 90th)
- Barrel rate
- Hard hit rate
- Chase rate (O-Swing%)
- Whiff rate (SwStr%)
- Sprint speed

**Pitchers (from Statcast + pitch-level data):**
- Fastball velocity (and velocity by pitch type)
- Spin rates by pitch type
- Stuff+ or equivalent aggregate metric
- Whiff rate
- Extension

## Relationship to Player Embeddings (ML Experimental Option 1)

The skill change approach relates to the Player Embeddings concept from `ml-experimental-approaches.md`. Option 1 mentions:

> "Detecting regime changes when a player's embedding shifts suddenly between seasons"

Skill change detection is a **lightweight, interpretable implementation of regime change detection**. Rather than detecting that a learned embedding vector shifted, we detect shifts in specific, researched skill metrics.

### Comparison

| Aspect | Skill Change | Player Embeddings |
|--------|--------------|-------------------|
| Regime detection | Explicit rules on known metrics | Implicit via embedding drift |
| Complexity | Low (thresholds on deltas) | High (neural network) |
| Interpretability | High ("barrel rate +3%") | Low (64-dim vector shifted) |
| Data needs | 2 years of Statcast | Full career histories |
| New players | Works with 1 prior year | Needs substantial training data |

### Integration Paths

**Path A: Skill Change as Stepping Stone**

Implement skill change first (low complexity), validate it improves projections, then use learnings to inform which features matter for embeddings.

**Path B: Skill Deltas as Embedding Features**

Feed skill change deltas into the embedding model as input features. The embedding model can then learn non-linear interactions between skill changes and player archetypes.

**Path C: Hybrid Override**

Use embeddings for base projections (borrowing strength from similar players), but apply skill change adjustments when there's clear evidence of regime change:

```python
def project(player: Player) -> Projection:
    # Embedding-based projection
    neighbors = find_similar_players(player.embedding)
    base_projection = blend_neighbor_trajectories(neighbors)

    # Skill change override
    delta = compute_skill_delta(player)
    if delta.has_significant_change():
        # Player is diverging from archetype - don't trust neighbors
        return apply_skill_adjustments(base_projection, delta)

    return base_projection
```

### Recommendation

Given that gradient boosting on residuals (Option 8) didn't deliver, implement skill change first:

1. Low complexity, interpretable, tests the core hypothesis
2. Validates which signals matter before building complex systems
3. If embeddings come later, use hybrid approach (Path C)

The skill change approach de-risks the embedding investment by identifying high-value features upfront.

## References

- [FanGraphs: League-Relative Statcast Forecasting](https://fantasy.fangraphs.com/how-league-relative-statcast-power-metrics-forecast-next-years-rates/)
- [Baseball Prospectus: Evaluation of Stuff Metrics](https://www.baseballprospectus.com/news/article/82426/prospectus-feature-an-updated-evaluation-of-hitting-and-pitching-including-stuff-metrics/)
- [Pitcher List: Plate Discipline Guide](https://pitcherlist.com/a-beginners-guide-to-understanding-plate-discipline-metrics-for-hitters/)
- [FanGraphs: Sprint Speed and Stolen Bases](https://fantasy.fangraphs.com/how-sprint-speed-relates-to-stolen-bases/)
- [SABR: Pitcher Development Metrics](https://sabrstatanalysis.blog/2019/08/01/old-vs-new-and-the-pitcher-development-battle/)
- [FantraxHQ: Barrel Rate Analysis](https://fantraxhq.com/statcast-101-barrel-rates-launch-angle/)
- [BaseballHQ: Chase Rate](https://www.baseballhq.com/articles/skills/batters/plate-discipline-chase-rate)
