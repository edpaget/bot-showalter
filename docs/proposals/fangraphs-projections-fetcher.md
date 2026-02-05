# FanGraphs Projections Fetcher

## What

Create a module to fetch and cache Steamer and ZiPS projections from FanGraphs, following the same pattern as the Yahoo ADP fetcher. This provides the consensus projection data needed to train the ML valuation model.

## Why

The ML valuation function proposal requires consensus projections (Steamer/ZiPS) as training inputs. By training on consensus projections paired with ADP, we learn the market's valuation function separately from projection disagreements. This module provides that data.

## FanGraphs API

FanGraphs exposes a public JSON API for projections:

```
https://www.fangraphs.com/api/projections?type={system}&stats={type}&pos=all&team=0&players=0&lg=all
```

**Projection systems:**
- `steamer` — Steamer (preseason)
- `zips` — ZiPS (preseason)
- `steameru` — Steamer update (in-season)
- `zipsdc` — ZiPS with depth chart playing time

**Stat types:**
- `bat` — Batting projections
- `pit` — Pitching projections

**Response format:** JSON array of player objects with fields:

| Field | Description |
|-------|-------------|
| `playerid` | FanGraphs player ID |
| `xMLBAMID` | MLB Advanced Media ID (for cross-referencing) |
| `PlayerName` | Display name |
| `Team` | Team abbreviation |
| `Pos` | Primary position |
| `G`, `PA`, `AB` | Games, plate appearances, at-bats |
| `H`, `1B`, `2B`, `3B`, `HR` | Hits breakdown |
| `R`, `RBI`, `BB`, `SO`, `HBP` | Counting stats |
| `SB`, `CS` | Stolen bases, caught stealing |
| `AVG`, `OBP`, `SLG`, `OPS` | Rate stats |
| `wOBA`, `wRC+`, `WAR` | Advanced metrics |

Pitching responses include: `W`, `L`, `GS`, `G`, `IP`, `H`, `ER`, `HR`, `BB`, `SO`, `ERA`, `WHIP`, `K/9`, `BB/9`, `FIP`, `WAR`, `SV`, `HLD`.

## Architecture

Follow the existing ADP pattern:

```
src/fantasy_baseball_manager/
├── projections/
│   ├── __init__.py
│   ├── models.py          # ProjectionEntry, ProjectionData
│   ├── protocol.py        # ProjectionSource protocol
│   └── fangraphs.py       # FanGraphsProjectionSource
└── cache/
    └── sources.py         # Add CachedProjectionSource
```

### Models

```python
@dataclass(frozen=True)
class BattingProjectionEntry:
    player_id: str           # FanGraphs ID
    mlbam_id: str | None     # For cross-referencing
    name: str
    team: str
    position: str
    pa: int
    h: int
    hr: int
    r: int
    rbi: int
    sb: int
    bb: int
    hbp: int
    obp: float
    # Additional fields as needed

@dataclass(frozen=True)
class PitchingProjectionEntry:
    player_id: str
    mlbam_id: str | None
    name: str
    team: str
    ip: float
    w: int
    sv: int
    hld: int
    so: int
    era: float
    whip: float
    # Additional fields as needed

@dataclass(frozen=True)
class ProjectionData:
    batting: tuple[BattingProjectionEntry, ...]
    pitching: tuple[PitchingProjectionEntry, ...]
    system: str              # "steamer", "zips", etc.
    fetched_at: datetime
```

### Protocol

```python
class ProjectionSource(Protocol):
    def fetch_projections(self) -> ProjectionData: ...
```

### Fetcher

The `FanGraphsProjectionSource` makes HTTP requests to the API endpoints and parses the JSON responses. No browser automation needed since this is a direct JSON API.

### Caching

Add `CachedProjectionSource` following the same wrapper pattern as `CachedADPSource`:
- Cache namespace: `"projections"`
- Cache key: `"{system}"` (e.g., "steamer", "zips")
- TTL: 7 days (projections update less frequently than ADP)

## Implementation Steps

1. Create `projections/models.py` with frozen dataclasses
2. Create `projections/protocol.py` with `ProjectionSource` protocol
3. Create `projections/fangraphs.py` with HTTP-based fetcher
4. Add `CachedProjectionSource` to `cache/sources.py`
5. Add CLI command to fetch/display projections
6. Write tests with recorded JSON fixtures

## Data Alignment

For training the ML valuation model, we need to align:
- FanGraphs projections (keyed by `playerid` or `xMLBAMID`)
- Yahoo ADP (keyed by name)
- Our Marcel projections (keyed by internal player ID)

The existing `player_id/mapper.py` module handles cross-system ID mapping. We'll extend it to support FanGraphs IDs if not already present.

## Cache Strategy

| Data | TTL | Rationale |
|------|-----|-----------|
| Steamer/ZiPS preseason | 7 days | Updated infrequently before season |
| Steamer update (in-season) | 24 hours | Updates with recent performance |
| ADP | 24 hours | Changes daily during draft season |

## Testing

- Unit tests with mocked HTTP responses
- Fixture files with sample JSON responses
- Round-trip serialization tests for cache
