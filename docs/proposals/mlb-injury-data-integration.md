# MLB Injury Data Integration

## What

Integrate MLB Stats API injury/transaction data to adjust playing time projections for players currently on the injured list or with significant injury history. This addresses a gap where the system ranks players like Corbin Burnes (60-day IL, out until the All-Star break) as if they'll pitch a full season.

## Why

The existing `EnhancedPlayingTimeProjector` uses games played as a backward-looking injury proxy, but it has two blind spots:

1. **Known current injuries**: A player placed on the 60-day IL before the season has a concrete, knowable timeline. The games-played proxy can't account for this because it relies on prior-year data.
2. **Injury recurrence**: The proxy treats a player who missed half a season identically whether it was a freak HBP or a recurring soft-tissue issue. Historical IL stint data lets us distinguish these.

For fantasy drafts, the impact is significant. A starting pitcher missing the first half loses ~50% of his IP projection, which cascades through K, W, ERA contribution, and total value.

## Data Source

### MLB Stats API — Transactions Endpoint

```
GET https://statsapi.mlb.com/api/v1/transactions
    ?startDate=2025-01-01
    &endDate=2026-12-31
    &transactionTypes=Injured List
```

**What it provides:**
- IL placements and activations (10-day, 15-day, 60-day)
- Injury descriptions (when reported by teams)
- Transaction dates with retroactive dating
- Player IDs (MLBAM IDs, which we already use for cross-referencing)

**Python wrapper:** [`MLB-StatsAPI`](https://github.com/toddrob99/MLB-StatsAPI) on PyPI (`statsapi` package).

**Rate limits:** No authentication required. No documented rate limits, but responses should be cached.

**Limitations:**
- No expected return dates (teams rarely commit to timelines)
- Injury descriptions are free-text, not categorized
- Minor league IL stints may be incomplete

### Supplementary: ESPN Injuries API (optional, phase 2)

```
GET https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries
```

Provides current injury status with informal return timelines (e.g., "mid-July"). Undocumented but functional. Could supplement the MLB API for return date estimation.

## Architecture

### New Module

```
src/fantasy_baseball_manager/
├── injuries/
│   ├── __init__.py
│   ├── models.py          # ILStint, PlayerInjuryRecord
│   ├── protocol.py        # InjurySource protocol
│   ├── mlb_api.py         # MLBInjurySource (Stats API client)
│   └── adjuster.py        # InjuryAdjuster (playing time modifier)
```

### Models

```python
@dataclass(frozen=True)
class ILStint:
    player_id: str          # MLBAM ID
    player_name: str
    il_type: str            # "10-Day", "15-Day", "60-Day"
    injury: str             # free-text description, e.g. "right elbow sprain"
    placed_date: date
    activated_date: date | None   # None = still on IL
    retroactive_date: date | None

@dataclass(frozen=True)
class PlayerInjuryRecord:
    player_id: str
    stints: tuple[ILStint, ...]
    is_currently_on_il: bool
    current_il_type: str | None
    total_days_on_il_last_3yr: int
    stint_count_last_3yr: int
```

### Protocol

```python
class InjurySource(Protocol):
    def fetch_current_il(self) -> tuple[ILStint, ...]:
        """Fetch all players currently on the injured list."""
        ...

    def fetch_player_history(
        self, player_id: str, years: int = 3,
    ) -> PlayerInjuryRecord:
        """Fetch IL history for a specific player."""
        ...
```

This follows the same protocol pattern as `ProjectionSource` and `ProjectionBlender` — consumers depend on the protocol, not the concrete MLB API implementation.

### Adjuster

`InjuryAdjuster` takes an `InjurySource` and adjusts playing time projections:

```python
class InjuryAdjuster:
    def __init__(self, source: InjurySource, season_start: date, season_end: date) -> None:
        ...

    def adjust(self, player_id: str, projected_pa_or_ip: float) -> AdjustedPlayingTime:
        ...
```

**Adjustment logic:**

1. **Currently on IL** — Estimate missed fraction of the season:
   - 60-day IL placed in March: assume return at All-Star break (~50% reduction)
   - 15-day IL placed in March: assume return by mid-April (~10% reduction)
   - If activated date exists, use actual missed days
   - Multiply projected PA/IP by the remaining-season fraction

2. **Injury history penalty** (stacks with existing games-played proxy):
   - 3+ IL stints in last 3 years: additional 5% reduction
   - 100+ days on IL in last 3 years: additional 5% reduction
   - These are conservative; the games-played proxy already captures most of this signal

The return value includes diagnostic metadata (same pattern as `EnhancedPlayingTimeProjector`):

```python
@dataclass(frozen=True)
class AdjustedPlayingTime:
    projected_value: float          # adjusted PA or IP
    il_reduction_factor: float      # 0.0-1.0, fraction of season available
    history_reduction_factor: float # 0.92-1.0, injury-prone penalty
    is_currently_on_il: bool
    il_type: str | None
    injury_description: str | None
```

### Integration Points

**1. Pipeline stage** — Add as an optional stage in `PipelineBuilder`, running after playing time projection:

```python
pipeline = (
    PipelineBuilder("draft_2026")
    .with_enhanced_playing_time(config=PlayingTimeConfig())
    .with_injury_adjustment(source=mlb_injury_source)  # new
    .build()
)
```

**2. Draft rank CLI** — Surface injury flags in `draft-rank` output. Players currently on IL should display the IL type and injury alongside their rank so the drafter has context.

**3. ROS blender** — During the season, the `BayesianBlender` can use IL status to zero out remaining projected stats for the duration of a stint when a player hits the IL mid-season.

### Caching

Follow the existing `CacheStore` pattern:

| Data | Namespace | Key | TTL | Rationale |
|------|-----------|-----|-----|-----------|
| Current IL roster | `injuries` | `current_il` | 6 hours | Changes with transactions |
| Player IL history | `injuries` | `history:{player_id}` | 7 days | Historical data is stable |

## Implementation Steps

1. **Models + protocol** — Define `ILStint`, `PlayerInjuryRecord`, `InjurySource` protocol, `AdjustedPlayingTime`
2. **MLB API client** — Implement `MLBInjurySource` against the transactions endpoint, with tests using recorded JSON fixtures
3. **Adjuster logic** — Implement `InjuryAdjuster` with current-IL and history-based adjustments, unit tested with synthetic data
4. **Caching layer** — Wrap `MLBInjurySource` with `CachedInjurySource` using existing `CacheStore`
5. **Pipeline integration** — Add `.with_injury_adjustment()` to `PipelineBuilder`
6. **CLI output** — Add IL status/injury columns to `draft-rank` display
7. **Evaluation** — Backtest against 2023-2025 seasons: compare projected vs actual PA/IP for players who started seasons on IL

## Open Questions

- **Return date estimation**: The MLB API doesn't provide expected return dates. For 60-day IL cases, should we default to a fixed assumption (e.g., minimum 60 days) or try to scrape estimated timelines from FanGraphs/ESPN?
- **Interaction with enhanced PT**: Should the injury adjuster replace the games-played proxy in `EnhancedPlayingTimeProjector`, or stack on top of it? Stacking risks double-penalizing; replacing means the adjuster needs to handle the no-data fallback.
- **Granularity of history penalty**: The proposed thresholds (3+ stints, 100+ days) are starting points. Worth tuning against historical data to see what actually predicts future missed time.
