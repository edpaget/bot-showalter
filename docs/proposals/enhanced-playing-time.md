# Enhanced Playing Time Model

## Overview

The `EnhancedPlayingTimeProjector` provides an alternative to the basic Marcel playing time formula. It applies multiplicative adjustments for injury risk, age-based decline, and role-based caps.

**Status:** Available but not enabled in default pipelines. Evaluation showed marginal benefit for rate stats but slight regression for counting stats.

## Algorithm

```
projected_pt = base_pt × injury_factor × age_factor × volatility_factor
projected_pt = min(projected_pt, role_cap)
```

Where `base_pt` is computed using the standard Marcel formula:
- Batters: `0.5 × PA_y1 + 0.1 × PA_y2 + 200`
- Starters: `0.5 × IP_y1 + 0.1 × IP_y2 + 60`
- Relievers: `0.5 × IP_y1 + 0.1 × IP_y2 + 25`

## Configuration

```python
from fantasy_baseball_manager.pipeline.stages.playing_time_config import PlayingTimeConfig

config = PlayingTimeConfig(
    # Injury proxy settings
    games_played_weight=0.08,   # penalty for missed games (0-1)
    min_games_pct=0.40,         # below this, apply max penalty

    # Age-based PT decline
    age_decline_start=35,       # age when decline begins
    age_decline_rate=0.01,      # 1% per year after threshold

    # Volatility adjustment
    volatility_threshold=0.25,  # >25% year-over-year swing triggers penalty
    volatility_penalty=0.0,     # disabled by default

    # Role caps
    batter_pa_cap=720,
    starter_ip_cap=220,
    reliever_ip_cap=90,
    catcher_pa_cap=580,
)
```

## Usage

### Enabling in PipelineBuilder

```python
from fantasy_baseball_manager.pipeline.builder import PipelineBuilder
from fantasy_baseball_manager.pipeline.stages.playing_time_config import PlayingTimeConfig

pipeline = (
    PipelineBuilder("my_pipeline")
    .with_park_factors()
    .with_enhanced_playing_time(config=PlayingTimeConfig())
    .build()
)
```

### Standalone Usage

```python
from fantasy_baseball_manager.pipeline.stages.enhanced_playing_time import (
    EnhancedPlayingTimeProjector,
)
from fantasy_baseball_manager.pipeline.stages.playing_time_config import PlayingTimeConfig

projector = EnhancedPlayingTimeProjector(config=PlayingTimeConfig())
projected_players = projector.project(players)
```

## Diagnostic Metadata

The projector adds the following metadata to each player:

| Field | Description |
|-------|-------------|
| `injury_factor` | Multiplier from injury proxy (0.92-1.0) |
| `age_pt_factor` | Multiplier from age decline (0.76-1.0) |
| `volatility_factor` | Multiplier from volatility (0.9 or 1.0) |
| `base_pa` / `base_ip` | Pre-adjustment Marcel projection |
| `projected_pa_before_cap` / `projected_ip_before_cap` | Post-adjustment, pre-cap value |

## Evaluation Results

Tested against basic Marcel PT over 2021-2024 seasons:

| Metric | Enhanced PT | Basic PT | Difference |
|--------|-------------|----------|------------|
| **Batting RMSE** | | | |
| HR | 7.98 | 7.80 | +2.3% |
| R | 22.08 | 21.31 | +3.6% |
| RBI | 22.23 | 21.47 | +3.5% |
| SB | 6.78 | 6.67 | +1.6% |
| **Pitching RMSE** | | | |
| W | 3.77 | 3.69 | +2.2% |
| K | 41.35 | 39.29 | +5.2% |
| ERA | 1.19 | 1.23 | **-3.3%** |
| WHIP | 0.20 | 0.21 | **-3.3%** |
| **Rank Correlation** | | | |
| Batting rho | 0.578 | 0.577 | +0.2% |
| Pitching rho | 0.411 | 0.409 | +0.5% |

### Key Findings

1. **Rate stats benefit**: ERA and WHIP predictions improve ~3% because the model correctly anticipates reduced playing time for injury-prone/older players.

2. **Counting stats regress**: HR, R, RBI, K predictions are 2-5% worse because enhanced PT systematically under-projects total accumulation.

3. **Marcel is already good**: The basic Marcel formula already incorporates historical playing time patterns through its weighted average, making additional adjustments marginal.

4. **Trade-off exists**: Enhanced PT helps rate stat accuracy at the cost of counting stat accuracy.

### Recommendation

- **Standard fantasy leagues**: Use basic Marcel PT (default)
- **Leagues weighting rate stats**: Consider enabling enhanced PT
- **DFS/daily formats**: Enhanced PT may help identify injury risk

## Extensibility

The design supports future integration with real injury data:

```python
class InjuryDataSource(Protocol):
    def days_on_il(self, player_id: str, year: int) -> int: ...

# Future: EnhancedPlayingTimeProjector could accept an injury_source
# parameter to use actual IL data instead of games-played proxy
```

## Files

| File | Description |
|------|-------------|
| `pipeline/stages/playing_time_config.py` | Configuration dataclass |
| `pipeline/stages/enhanced_playing_time.py` | Projector implementation |
| `tests/pipeline/stages/test_enhanced_playing_time.py` | Unit tests |
