# Skill Change Adjuster Implementation Plan

## Overview

Implement a pipeline adjuster that detects year-over-year skill changes and applies targeted projection adjustments. This replaces the gradient boosting residual approach with a rule-based system focused on interpretable, high-signal metrics.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ SkillDataSource │────>│ SkillDeltaComputer│────>│ SkillChangeAdjuster │
│ (new data layer)│     │  (compute deltas) │     │  (apply adjustments)│
└─────────────────┘     └──────────────────┘     └─────────────────────┘
        │                                                   │
        │ Fetches:                                          │ Implements:
        │ - Plate discipline                                │ - RateAdjuster protocol
        │ - Exit velo percentiles                           │ - Batter adjustments
        │ - Sprint speed                                    │ - Pitcher adjustments
        │ - Pitcher velocity/spin                           │
        v                                                   v
┌─────────────────┐                              ┌─────────────────────┐
│ Baseball Savant │                              │ ProjectionPipeline  │
│ FanGraphs       │                              │ (existing)          │
└─────────────────┘                              └─────────────────────┘
```

## Implementation Phases

### Phase 1: Skill Data Layer

**Goal**: Extend data sources to provide the metrics needed for skill change detection.

#### 1.1 Batter Skill Stats Data Model

Create `src/fantasy_baseball_manager/pipeline/skill_data.py`:

```python
@dataclass(frozen=True)
class BatterSkillStats:
    """Skill metrics for a batter in a single season."""
    player_id: str          # MLBAM ID
    year: int
    pa: int

    # Contact quality (Tier 1)
    barrel_rate: float      # barrels / BBE
    hard_hit_rate: float    # balls >= 95 mph / BBE
    exit_velo_avg: float    # average exit velocity
    exit_velo_90th: float   # 90th percentile exit velocity

    # Plate discipline (Tier 1)
    chase_rate: float       # O-Swing% (swings at pitches outside zone)
    whiff_rate: float       # SwStr% (swinging strikes / pitches)

    # Speed (Tier 2)
    sprint_speed: float     # ft/sec (from Statcast)


@dataclass(frozen=True)
class PitcherSkillStats:
    """Skill metrics for a pitcher in a single season."""
    player_id: str          # MLBAM ID
    year: int
    pa_against: int

    # Stuff (Tier 1)
    fastball_velo: float    # average fastball velocity
    fastball_spin: float    # average fastball spin rate
    whiff_rate: float       # SwStr%

    # Batted ball (Tier 2)
    barrel_rate_against: float
    hard_hit_rate_against: float
    gb_rate: float          # ground ball rate
```

#### 1.2 Data Source Implementation

**Option A: FanGraphs for plate discipline + Statcast for batted ball**

FanGraphs provides O-Swing%, SwStr%, etc. via pybaseball's `batting_stats` and `pitching_stats` with advanced=True.

```python
class FanGraphsSkillDataSource:
    """Fetches plate discipline and advanced metrics from FanGraphs."""

    def batter_skill_stats(self, year: int) -> list[BatterSkillStats]: ...
    def pitcher_skill_stats(self, year: int) -> list[PitcherSkillStats]: ...
```

**Option B: Baseball Savant leaderboards**

Savant has dedicated leaderboards for swing/take, expected stats, and sprint speed. May require custom scraping beyond pybaseball.

**Recommendation**: Start with Option A (FanGraphs via pybaseball) since it's more stable and already used in the codebase. Add Savant data incrementally.

#### 1.3 Tasks

- [ ] Define `BatterSkillStats` and `PitcherSkillStats` dataclasses
- [ ] Implement `FanGraphsSkillDataSource` using pybaseball
- [ ] Add caching layer (`CachedSkillDataSource`)
- [ ] Write unit tests with fake data sources
- [ ] Verify data availability for 2022-2024 seasons

---

### Phase 2: Skill Delta Computation

**Goal**: Compute year-over-year skill deltas and determine significance.

#### 2.1 Delta Data Model

```python
@dataclass(frozen=True)
class BatterSkillDelta:
    """Year-over-year skill changes for a batter."""
    player_id: str
    year: int  # projection year (deltas are year-2 to year-1)

    # Deltas (current - prior)
    barrel_rate_delta: float | None
    exit_velo_90th_delta: float | None
    chase_rate_delta: float | None
    whiff_rate_delta: float | None
    sprint_speed_delta: float | None

    # Sample sizes
    pa_current: int
    pa_prior: int

    def has_sufficient_sample(self, min_pa: int = 200) -> bool:
        return self.pa_current >= min_pa and self.pa_prior >= min_pa


@dataclass(frozen=True)
class PitcherSkillDelta:
    """Year-over-year skill changes for a pitcher."""
    player_id: str
    year: int

    fastball_velo_delta: float | None
    fastball_spin_delta: float | None
    whiff_rate_delta: float | None
    barrel_rate_against_delta: float | None

    pa_against_current: int
    pa_against_prior: int

    def has_sufficient_sample(self, min_pa: int = 200) -> bool:
        return self.pa_against_current >= min_pa and self.pa_against_prior >= min_pa
```

#### 2.2 Delta Computer

```python
class SkillDeltaComputer:
    """Computes skill deltas from two years of skill data."""

    def __init__(
        self,
        skill_source: SkillDataSource,
        id_mapper: PlayerIdMapper,
        min_pa: int = 200,
    ) -> None: ...

    def compute_batter_deltas(self, year: int) -> dict[str, BatterSkillDelta]:
        """Compute deltas for all batters with sufficient data."""
        # Fetches year-1 and year-2 skill stats
        # Returns dict keyed by FanGraphs player ID
        ...

    def compute_pitcher_deltas(self, year: int) -> dict[str, PitcherSkillDelta]:
        """Compute deltas for all pitchers with sufficient data."""
        ...
```

#### 2.3 Tasks

- [ ] Define `BatterSkillDelta` and `PitcherSkillDelta` dataclasses
- [ ] Implement `SkillDeltaComputer`
- [ ] Handle ID mapping (FanGraphs ↔ MLBAM)
- [ ] Write unit tests for delta computation
- [ ] Test edge cases (missing data, insufficient PA)

---

### Phase 3: Adjustment Logic

**Goal**: Apply projection adjustments based on skill deltas.

#### 3.1 Configuration

```python
@dataclass(frozen=True)
class SkillChangeConfig:
    """Thresholds and adjustment factors for skill change detection."""

    min_pa: int = 200

    # Batter thresholds (absolute delta required to trigger adjustment)
    barrel_rate_threshold: float = 0.02      # 2 percentage points
    exit_velo_90th_threshold: float = 1.5    # mph
    chase_rate_threshold: float = 0.03       # 3 percentage points
    whiff_rate_threshold: float = 0.03       # 3 percentage points
    sprint_speed_threshold: float = 0.5      # ft/sec

    # Pitcher thresholds
    fastball_velo_threshold: float = 1.0     # mph
    fastball_spin_threshold: float = 100     # rpm
    pitcher_whiff_threshold: float = 0.03    # 3 percentage points

    # Adjustment factors (how much to adjust per unit of skill change)
    # These map skill deltas to rate adjustments
    barrel_to_hr_factor: float = 0.5         # HR rate += delta * factor
    barrel_to_doubles_factor: float = 0.3
    exit_velo_to_hr_factor: float = 0.005    # per mph
    chase_to_bb_factor: float = -0.5         # lower chase = more BB
    chase_to_so_factor: float = 0.3          # lower chase = fewer SO
    whiff_to_so_factor: float = 0.7
    sprint_to_sb_factor: float = 0.02        # per ft/sec

    # Pitcher factors
    velo_to_so_factor: float = 0.003         # per mph (rate per out)
    velo_to_er_factor: float = -0.001        # per mph
    spin_to_so_factor: float = 0.00002       # per rpm
```

#### 3.2 Adjuster Implementation

```python
@dataclass
class SkillChangeAdjuster:
    """Pipeline adjuster that applies skill-change-based corrections."""

    delta_computer: SkillDeltaComputer
    config: SkillChangeConfig = field(default_factory=SkillChangeConfig)

    _batter_deltas: dict[str, BatterSkillDelta] | None = field(default=None, init=False)
    _pitcher_deltas: dict[str, PitcherSkillDelta] | None = field(default=None, init=False)
    _cached_year: int | None = field(default=None, init=False)

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        """Implements RateAdjuster protocol."""
        if not players:
            return []

        year = players[0].year
        self._ensure_deltas_loaded(year)

        return [self._adjust_player(p) for p in players]

    def _adjust_player(self, player: PlayerRates) -> PlayerRates:
        if self._is_batter(player):
            return self._adjust_batter(player)
        return self._adjust_pitcher(player)

    def _adjust_batter(self, player: PlayerRates) -> PlayerRates:
        delta = self._batter_deltas.get(player.player_id)
        if delta is None or not delta.has_sufficient_sample(self.config.min_pa):
            return player

        rates = dict(player.rates)
        adjustments = {}

        # Barrel rate → HR, doubles
        if delta.barrel_rate_delta is not None:
            if abs(delta.barrel_rate_delta) >= self.config.barrel_rate_threshold:
                hr_adj = delta.barrel_rate_delta * self.config.barrel_to_hr_factor
                rates["hr"] = rates.get("hr", 0) + hr_adj
                adjustments["barrel→hr"] = hr_adj

                dbl_adj = delta.barrel_rate_delta * self.config.barrel_to_doubles_factor
                rates["doubles"] = rates.get("doubles", 0) + dbl_adj
                adjustments["barrel→doubles"] = dbl_adj

        # Exit velocity → HR
        if delta.exit_velo_90th_delta is not None:
            if abs(delta.exit_velo_90th_delta) >= self.config.exit_velo_90th_threshold:
                hr_adj = delta.exit_velo_90th_delta * self.config.exit_velo_to_hr_factor
                rates["hr"] = rates.get("hr", 0) + hr_adj
                adjustments["ev90→hr"] = hr_adj

        # Chase rate → BB, SO (note: lower chase is better)
        if delta.chase_rate_delta is not None:
            if abs(delta.chase_rate_delta) >= self.config.chase_rate_threshold:
                bb_adj = delta.chase_rate_delta * self.config.chase_to_bb_factor
                rates["bb"] = rates.get("bb", 0) + bb_adj
                adjustments["chase→bb"] = bb_adj

                so_adj = delta.chase_rate_delta * self.config.chase_to_so_factor
                rates["so"] = rates.get("so", 0) + so_adj
                adjustments["chase→so"] = so_adj

        # ... similar for whiff_rate, sprint_speed

        if not adjustments:
            return player

        return PlayerRates(
            player_id=player.player_id,
            name=player.name,
            year=player.year,
            age=player.age,
            rates=rates,
            opportunities=player.opportunities,
            metadata={**player.metadata, "skill_adjustments": adjustments},
        )
```

#### 3.3 Tasks

- [ ] Define `SkillChangeConfig` with researched thresholds
- [ ] Implement `SkillChangeAdjuster` for batters
- [ ] Implement `SkillChangeAdjuster` for pitchers
- [ ] Ensure rates stay within valid bounds (non-negative, sum constraints)
- [ ] Add metadata tracking for adjustments made
- [ ] Write unit tests for each adjustment type
- [ ] Write integration tests with realistic data

---

### Phase 4: Pipeline Integration

**Goal**: Wire the skill change adjuster into the projection pipeline.

#### 4.1 Builder Updates

Update `src/fantasy_baseball_manager/pipeline/builder.py`:

```python
def with_skill_change_adjuster(
    self,
    skill_source: SkillDataSource,
    config: SkillChangeConfig | None = None,
) -> PipelineBuilder:
    """Add skill change adjustment stage."""
    delta_computer = SkillDeltaComputer(
        skill_source=skill_source,
        id_mapper=self._id_mapper,
    )
    adjuster = SkillChangeAdjuster(
        delta_computer=delta_computer,
        config=config or SkillChangeConfig(),
    )
    self._adjusters.append(adjuster)
    return self
```

#### 4.2 Preset Pipeline

Create a new preset in `presets.py`:

```python
def marcel_skill_change(container: ServiceContainer) -> ProjectionPipeline:
    """Marcel with skill change adjustments."""
    return (
        PipelineBuilder(container)
        .with_marcel_rates()
        .with_statcast_blend()
        .with_skill_change_adjuster(container.skill_data_source)
        .with_park_factors()
        .with_aging()
        .with_playing_time()
        .build("marcel_skill_change")
    )
```

#### 4.3 Tasks

- [ ] Add `SkillDataSource` to `ServiceContainer`
- [ ] Update `PipelineBuilder` with skill change adjuster method
- [ ] Create `marcel_skill_change` preset
- [ ] Update CLI to support new pipeline
- [ ] Document pipeline configuration options

---

### Phase 5: Validation & Tuning

**Goal**: Validate the approach improves projections and tune thresholds.

#### 5.1 Backtesting

Use the existing evaluation harness:

```bash
# Compare marcel_full vs marcel_skill_change
uv run python -m fantasy_baseball_manager.evaluation.cli \
    --pipeline marcel_full \
    --pipeline marcel_skill_change \
    --years 2022 2023 2024
```

#### 5.2 Analysis

- Track which players received adjustments
- Compare adjusted vs actual for those players specifically
- Analyze false positives (adjustment made, didn't help)
- Analyze false negatives (no adjustment, should have adjusted)

#### 5.3 Threshold Tuning

If needed, use grid search over threshold/factor combinations:

```python
# Pseudocode
for barrel_threshold in [0.015, 0.02, 0.025]:
    for barrel_factor in [0.4, 0.5, 0.6]:
        config = SkillChangeConfig(
            barrel_rate_threshold=barrel_threshold,
            barrel_to_hr_factor=barrel_factor,
        )
        rmse = backtest(config)
        results.append((config, rmse))
```

#### 5.4 Tasks

- [ ] Run backtest against 2022-2024 seasons
- [ ] Compare RMSE/correlation to marcel_full baseline
- [ ] Analyze adjustment hit rate (% of adjustments that helped)
- [ ] Tune thresholds if needed
- [ ] Document final configuration choices

---

## File Structure

```
src/fantasy_baseball_manager/
├── pipeline/
│   ├── skill_data.py              # NEW: Skill stats data models & sources
│   └── stages/
│       └── skill_change_adjuster.py   # NEW: Adjuster implementation
└── ...

tests/
├── pipeline/
│   ├── test_skill_data.py         # NEW: Data source tests
│   └── stages/
│       └── test_skill_change_adjuster.py  # NEW: Adjuster tests
└── ...
```

## Dependencies

- `pybaseball` - already installed, provides FanGraphs data
- No new dependencies required

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| FanGraphs data missing metrics | Fall back to Statcast-only metrics; implement graceful degradation |
| ID mapping gaps | Log warnings, pass through unadjusted; improve mapper over time |
| Thresholds too aggressive | Start conservative (high thresholds), tune down if needed |
| Thresholds too conservative | Monitor adjustment rate; if <5% of players adjusted, lower thresholds |
| Overfitting to backtest years | Hold out 2024 for final validation; use 2022-2023 for tuning |

## Success Criteria

1. **Accuracy**: RMSE on held-out year ≤ marcel_full baseline (don't make things worse)
2. **Hit rate**: >50% of adjusted players closer to actual than unadjusted projection
3. **Coverage**: 10-20% of players with sufficient data receive adjustments
4. **Interpretability**: Every adjustment traces to a specific skill change

## Timeline Estimate

| Phase | Scope |
|-------|-------|
| Phase 1 | Data layer - skill stats models and sources |
| Phase 2 | Delta computation |
| Phase 3 | Adjustment logic |
| Phase 4 | Pipeline integration |
| Phase 5 | Validation and tuning |

## Next Steps

1. Start with Phase 1: implement `BatterSkillStats` dataclass and `FanGraphsSkillDataSource`
2. Verify pybaseball provides O-Swing%, SwStr%, and other needed metrics
3. Proceed through phases sequentially, with tests at each phase
