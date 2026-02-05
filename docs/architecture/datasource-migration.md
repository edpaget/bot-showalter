# DataSource Migration Tracker

Tracking migration from legacy Protocol-based data sources to the unified `DataSource[T]` architecture.

## New Architecture Components (Complete)

| Component | Location | Status |
|-----------|----------|--------|
| `Context` | `context.py` | Done |
| `Result[T, E]` | `result.py` | Done |
| `DataSource[T]` | `data/protocol.py` | Done |
| `ALL_PLAYERS` sentinel | `data/protocol.py` | Done |
| `cached()` wrapper | `cache/wrapper.py` | Done |
| `Serializer` protocol | `cache/serialization.py` | Done |
| `Player` identity | `player/identity.py` | Done |

---

## Cached Wrappers to Deprecate

Located in `src/fantasy_baseball_manager/cache/sources.py`:

| Wrapper | Priority | Status | Notes |
|---------|----------|--------|-------|
| `CachedPositionSource` | Low | Pending | Yahoo-specific |
| `CachedRosterSource` | Low | Pending | Yahoo-specific, session-bound |
| `CachedDraftResultsSource` | Low | Pending | Yahoo-specific, multi-method |
| `CachedADPSource` | Medium | Pending | Simple single-method |
| `CachedProjectionSource` | Medium | Pending | Simple single-method |

### Serializers to Consolidate

These will be replaced by `DataclassListSerializer` or type-specific `Serializer` implementations:

- [ ] `_serialize_positions` / `_deserialize_positions`
- [ ] `_serialize_rosters` / `_deserialize_rosters`
- [ ] `_serialize_draft_results` / `_deserialize_draft_results`
- [ ] `_serialize_adp_data` / `_deserialize_adp_data`
- [ ] `_serialize_projection_data` / `_deserialize_projection_data`

---

## Data Source Protocols to Migrate

### High Priority

#### `StatsDataSource` (marcel/data_source.py)

**Current interface:**
```python
class StatsDataSource(Protocol):
    def batting_stats(self, year: int) -> list[BattingSeasonStats]: ...
    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]: ...
    def team_batting(self, year: int) -> list[BattingSeasonStats]: ...
    def team_pitching(self, year: int) -> list[PitchingSeasonStats]: ...
```

**New interface:**
```python
batting_source: DataSource[BattingSeasonStats]   # year from context
pitching_source: DataSource[PitchingSeasonStats]
team_batting_source: DataSource[BattingSeasonStats]
team_pitching_source: DataSource[PitchingSeasonStats]
```

**Implementation:** `PybaseballDataSource`

**Status:** Done

**Consumers:**
- `MarcelProjector`
- `RateComputer`
- Various pipeline stages

---

#### `MinorLeagueDataSource` (minors/data_source.py)

**Current interface:**
```python
class MinorLeagueDataSource(Protocol):
    def batting_stats(self, year: int, level: MinorLeagueLevel) -> list[MinorLeagueBatterSeasonStats]: ...
    def batting_stats_all_levels(self, year: int) -> list[MinorLeagueBatterSeasonStats]: ...
    def pitching_stats(self, year: int, level: MinorLeagueLevel) -> list[MinorLeaguePitcherSeasonStats]: ...
    def pitching_stats_all_levels(self, year: int) -> list[MinorLeaguePitcherSeasonStats]: ...
```

**Challenge:** Has `level` parameter in addition to `year`. Options:
1. Create separate sources per level
2. Add level to context
3. Keep level as parameter, only move year to context

**Implementation:** `FanGraphsMinorLeagueDataSource`

**Status:** Pending

**Consumers:**
- `MLERateComputer`
- MLE pipeline

---

#### `PlayerIdMapper` (player_id/mapper.py)

**Current interface:**
```python
class PlayerIdMapper(Protocol):
    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None: ...
    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None: ...
    def fangraphs_to_mlbam(self, fangraphs_id: str) -> str | None: ...
    def mlbam_to_fangraphs(self, mlbam_id: str) -> str | None: ...
```

**New interface:**
```python
PlayerMapper = Callable[[Player | Sequence[Player]], Result[Player | Sequence[Player], PlayerMapperError]]
```

Takes `Player` objects, returns enriched `Player` with additional IDs populated.

**Implementation:** `SfbbMapper`

**Status:** Pending

**Consumers:**
- Many pipeline stages
- Valuation
- Draft ranking

---

### Medium Priority

#### `ADPSource` (adp/protocol.py)

**Current interface:**
```python
class ADPSource(Protocol):
    def fetch_adp(self) -> ADPData: ...
```

**New interface:**
```python
adp_source: DataSource[ADPEntry]  # Returns Sequence[ADPEntry]
```

**Implementations:** `YahooADPSource`, `EspnADPScraper`

**Status:** Pending

**Notes:** Has registry pattern in `adp/registry.py`

---

#### `ProjectionSource` (projections/protocol.py)

**Current interface:**
```python
class ProjectionSource(Protocol):
    def fetch_projections(self) -> ProjectionData: ...
```

**New interface:**
```python
batting_projection_source: DataSource[BattingProjection]
pitching_projection_source: DataSource[PitchingProjection]
```

**Implementations:** `SteamerSource`, `ZipsSource`, `FanGraphsProjectionSource`

**Status:** Pending

---

### Low Priority (Yahoo-specific)

#### `RosterSource` (cache/sources.py)

**Current interface:**
```python
class RosterSource(Protocol):
    def fetch_rosters(self) -> LeagueRosters: ...
```

**Implementation:** `YahooRosterSource`

**Status:** Pending

**Notes:** Session-bound to Yahoo league, may not fit DataSource pattern well

---

#### `DraftResultsSource` (cache/sources.py)

**Current interface:**
```python
class DraftResultsSource(Protocol):
    def fetch_draft_results(self) -> list[YahooDraftPick]: ...
    def fetch_draft_status(self) -> DraftStatus: ...
    def fetch_user_team_key(self) -> str: ...
```

**Implementation:** `YahooDraftResultsSource`

**Status:** Pending

**Notes:** Multi-method protocol, may need different approach

---

#### `PositionSource` (cache/sources.py)

**Current interface:**
```python
class PositionSource(Protocol):
    def fetch_positions(self) -> dict[str, tuple[str, ...]]: ...
```

**Status:** Pending

---

## Migration Strategy

### Phase 1: Proof of Concept ✅
- [x] Migrate `StatsDataSource.batting_stats` to new pattern (`create_batting_source()`)
- [x] Update one consumer (`MarcelRateComputer.compute_batting_rates_v2`) to use new source
- [x] Verify caching works via `cached()` wrapper (tests in `test_batting_data_source.py`)

### Phase 2: Stats Sources
- [x] Complete `StatsDataSource` migration (all 4 methods)
- [ ] Migrate `MinorLeagueDataSource`
- [ ] Update all pipeline consumers

### Phase 3: Player Mapping
- [ ] Create `PlayerMapper` following DataSource pattern
- [ ] Migrate `SfbbMapper` to return enriched `Player` objects
- [ ] Update consumers to use `Player` type

### Phase 4: External Data
- [ ] Migrate `ADPSource`
- [ ] Migrate `ProjectionSource`
- [ ] Update registry patterns

### Phase 5: Yahoo Sources
- [ ] Evaluate if Yahoo sources fit DataSource pattern
- [ ] Migrate or leave as-is with clear documentation

### Phase 6: Cleanup
- [ ] Remove old cached wrapper classes
- [ ] Remove old serializer functions
- [ ] Update `ServiceContainer` to build new-style sources
- [ ] Archive old protocols (mark deprecated)

---

## Open Questions

1. **MinorLeagueDataSource level parameter**: Add to context or keep as parameter?
2. **Yahoo sources**: Do session-bound sources fit the DataSource pattern?
3. **DraftResultsSource**: Multi-method protocols - split into separate sources?
4. **Backward compatibility**: How long to maintain old protocols?

---

## Dependencies

```
Context ─────────────────────────────────────┐
Result ──────────────────────────────────────┤
DataSource[T] ───────────────────────────────┤
ALL_PLAYERS ─────────────────────────────────┼──► StatsDataSource migration
cached() ────────────────────────────────────┤
Serializer ──────────────────────────────────┤
Player ──────────────────────────────────────┘
                                             │
StatsDataSource ─────────────────────────────┼──► MinorLeagueDataSource
                                             │
Player + PlayerMapper ───────────────────────┼──► Consumer updates
                                             │
All sources migrated ────────────────────────┴──► Remove old wrappers
```
