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
| `CachedPositionSource` | Low | Done | Replaced by `cached_call()` + `PositionDictSerializer` |
| `CachedRosterSource` | Low | Done | Replaced by `cached_call()` + `LeagueRostersSerializer` |
| `CachedDraftResultsSource` | Low | Done | Replaced by `cached_call()` + `DataclassListSerializer` |
| `CachedADPSource` | Medium | Done | Replaced by `cached()` + `TupleFieldDataclassListSerializer` |
| `CachedProjectionSource` | Medium | Done | Replaced by `cached()` + `DataclassListSerializer` |

### Serializers to Consolidate

These will be replaced by `DataclassListSerializer` or type-specific `Serializer` implementations:

- [x] `_serialize_positions` / `_deserialize_positions`
- [x] `_serialize_rosters` / `_deserialize_rosters`
- [x] `_serialize_draft_results` / `_deserialize_draft_results`
- [x] `_serialize_adp_data` / `_deserialize_adp_data`
- [x] `_serialize_projection_data` / `_deserialize_projection_data`

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

**New interface:**
```python
milb_batting_source: DataSource[MinorLeagueBatterSeasonStats]   # year from context, all levels
milb_pitching_source: DataSource[MinorLeaguePitcherSeasonStats] # year from context, all levels
```

**Implementation:** `MinorLeagueBattingDataSource`, `MinorLeaguePitchingDataSource` (delegates to `MLBStatsAPIDataSource`)

**Status:** Done

**Resolution:** The new DataSource classes return all levels for the context's year (equivalent to `_all_levels` methods). Level-specific queries can be achieved by filtering the results or using the legacy protocol for specific-level fetches.

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
# SfbbMapper now implements callable interface:
mapper(player: Player) -> Ok[Player] | Err[PlayerMapperError]
mapper(players: list[Player]) -> Ok[list[Player]] | Err[PlayerMapperError]
```

Takes `Player` objects, returns enriched `Player` with fangraphs_id and mlbam_id populated.

**Implementation:** `SfbbMapper` (with callable `__call__` method)

**Status:** Done

**Resolution:** `SfbbMapper` now implements both the legacy `PlayerIdMapper` protocol (for backward compatibility) and the new callable interface. The callable interface enriches `Player` objects directly using `Player.with_ids()`. Legacy methods remain available for consumers not yet migrated.

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
adp_source: DataSource[ADPEntry]  # Returns list[ADPEntry]
```

**Implementations:**
- `YahooADPDataSource` (new-style, callable)
- `ESPNADPDataSource` (new-style, callable)
- `CompositeADPDataSource` (new-style, aggregates multiple sources)
- Legacy: `YahooADPScraper`, `ESPNADPScraper`, `CompositeADPSource`

**Factory functions:**
- `create_yahoo_adp_source() -> DataSource[ADPEntry]`
- `create_espn_adp_source() -> DataSource[ADPEntry]`
- `create_composite_adp_source(sources) -> DataSource[ADPEntry]`

**Registry functions:**
- `get_datasource(name) -> DataSource[ADPEntry]`
- `register_datasource(name, factory)`
- `list_datasources() -> tuple[str, ...]`

**Status:** Done

**Resolution:** The new DataSource classes (`YahooADPDataSource`, `ESPNADPDataSource`, `CompositeADPDataSource`) implement the callable interface returning `Ok[list[ADPEntry]] | Err[DataSourceError]`. Legacy classes remain available for backward compatibility. The registry supports both patterns via `get_source()` (legacy) and `get_datasource()` (new).

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

**Implementations:**
- `BattingProjectionDataSource` (new-style, wraps any ProjectionSource)
- `PitchingProjectionDataSource` (new-style, wraps any ProjectionSource)
- Legacy: `FanGraphsProjectionSource`, `CSVProjectionSource`

**Factory functions:**
- `create_batting_projection_source(system) -> DataSource[BattingProjection]`
- `create_pitching_projection_source(system) -> DataSource[PitchingProjection]`

**Status:** Done

**Resolution:** The new DataSource classes (`BattingProjectionDataSource`, `PitchingProjectionDataSource`) wrap any `ProjectionSource` implementation and implement the callable interface returning `Ok[list[T]] | Err[DataSourceError]`. Factory functions use `FanGraphsProjectionSource` by default. Legacy classes remain available for backward compatibility.

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

### Phase 2: Stats Sources ✅
- [x] Complete `StatsDataSource` migration (all 4 methods)
- [x] Migrate `MinorLeagueDataSource`
- [x] Update all pipeline consumers
  - [x] Add `MarcelRateComputer.compute_pitching_rates_v2()` (matches `_v2` pattern for batting)
  - [x] Update `RateComputer` protocol to use new signature
  - [x] Migrate other rate computer implementations
    - [x] `StatSpecificRegressionRateComputer`
    - [x] `PlatoonRateComputer`
    - [x] `MTLRateComputer`
    - [x] `MLERateComputer`
    - [x] `MLEAugmentedRateComputer`
  - [x] Restore `RateComputer` type on `ProjectionPipeline.rate_computer`

### Phase 3: Player Mapping
- [x] Create `PlayerMapper` following DataSource pattern (`PlayerMapperError`, callable interface on `SfbbMapper`)
- [x] Migrate `SfbbMapper` to return enriched `Player` objects (via `__call__` method)
- [x] Update consumers to use `Player` type

### Phase 4: External Data
- [x] Migrate `ADPSource`
- [x] Migrate `ProjectionSource`
- [x] Update registry patterns (for ADP)

### Phase 5: Yahoo Sources ✅ (Keep As-Is)

Evaluated — these protocols do not fit the `DataSource[T]` pattern and should remain as-is:

- [x] **`RosterSource`** — Returns singular `LeagueRosters` (not a list). Session-bound to Yahoo league. Single implementation (`YahooRosterSource`). Already cached via `cached_call()`. Structural mismatch with `DataSource[T]`.
- [x] **`DraftResultsSource`** — Multi-method protocol (3 methods returning unrelated types: `list[YahooDraftPick]`, `DraftStatus`, `str`). Only used in `draft/cli.py`. Cannot be meaningfully split into separate `DataSource[T]` instances.
- [x] **`PositionSource`** — Returns `dict[str, tuple[str, ...]]` (a mapping, not a list). Two lightweight implementations (`YahooPositionSource`, `CsvPositionSource`). Only used in `draft/cli.py`. Already cached via `cached_call()`.

### Phase 6: Cleanup
- [x] Remove `CachedADPSource` and `CachedProjectionSource` cached wrappers
- [x] Remove `_serialize_adp_data`/`_deserialize_adp_data` and `_serialize_projection_data`/`_deserialize_projection_data`
- [x] Migrate `draft/cli.py` to use `get_datasource()` + `cached()` for ADP
- [x] Migrate `pipeline/presets.py` to use `cached()` + `DataclassListSerializer` for projections
- [x] Update `ExternalProjectionAdapter` to accept `DataSource[T]` directly
- [x] Remove remaining old cached wrapper classes (`CachedPositionSource`, `CachedRosterSource`, `CachedDraftResultsSource`)
- [x] Remove remaining old serializer functions
- [x] Update `ServiceContainer` to build new-style sources
- [ ] Archive old protocols (mark deprecated) — see Phase 7

### Phase 7: Migrate Consumers Off Old Protocols

The new `DataSource[T]` implementations exist but ~90 call sites still reference the old
protocols in type annotations and method calls. This phase migrates those consumers.

#### 7a. `StatsDataSource` → `DataSource[T]` callables (~37 refs)

**Current state:** Rate computers already accept `DataSource[T]`. `pipeline/engine.py` has
internal `_adapt_batting`/`_adapt_pitching` adapter functions wrapping `StatsDataSource` into
`DataSource[T]`. All other consumers call methods directly: `data_source.batting_stats(year)`.

**Approach — bottom-up, guided by `ProjectionPipeline`:**

1. Change `ProjectionPipeline.project_batters` / `project_pitchers` signatures to accept
   `DataSource[T]` pairs directly instead of `StatsDataSource`:
   ```python
   # Before
   def project_batters(self, data_source: StatsDataSource, year: int) -> ...
   # After
   def project_batters(
       self,
       batting_source: DataSource[BattingSeasonStats],
       team_batting_source: DataSource[BattingSeasonStats],
       year: int,
   ) -> ...
   ```
   Remove the `_adapt_*` helper functions (no longer needed).

2. Update `ProjectionPipelineProtocol` (pipeline/protocols.py) to match.

3. Migrate direct `data_source.method(year)` consumers. Common call-site pattern:
   ```python
   # Before
   actuals = data_source.batting_stats(year)
   # After
   with new_context(year=year):
       actuals = batting_source(ALL_PLAYERS).unwrap()
   ```
   Key files (in dependency order):
   - [ ] `evaluation/actuals.py` — `actuals_as_projections()` takes 2 DataSource params
   - [ ] `evaluation/harness.py` — `evaluate_source()` takes 2 DataSource params
   - [ ] `evaluation/cli.py` — thread sources through CLI entry points
   - [ ] `ros/projector.py` — `ROSProjector.__init__` takes batting/pitching sources
   - [ ] `ml/training.py` — `ResidualModelTrainer` fields become DataSource params
   - [ ] `ml/mtl/dataset.py` — replace `StatsDataSource` field
   - [ ] `ml/mtl/trainer.py` — replace `StatsDataSource` field
   - [ ] `ml/cli.py` — thread sources through CLI entry points
   - [ ] `pipeline/engine.py` — update `ProjectionPipeline` signatures
   - [ ] `pipeline/protocols.py` — update `ProjectionPipelineProtocol`
   - [ ] `pipeline/source.py` — `PipelineProjectionSource` adapts pipeline
   - [ ] `services/container.py` — build DataSource instances instead of `PybaseballDataSource`
   - [ ] `minors/training.py`, `minors/evaluation.py`, `minors/training_data.py` — update MLE consumers

4. Remove `StatsDataSource` protocol, `PybaseballDataSource` class, and
   `CachedStatsDataSource` wrapper.

#### 7b. `MinorLeagueDataSource` → `DataSource[T]` callables (~14 refs)

**Status:** Done

**Resolution:** All consumers now accept `DataSource[MinorLeagueBatterSeasonStats]` instead of
`MinorLeagueDataSource`. Call sites use `new_context(year=...)` + `source(ALL_PLAYERS)` pattern.
`CachedMinorLeagueDataSource` replaced by `cached()` + `MiLBBatterStatsSerializer`.
`MinorLeagueDataSource` protocol removed.

**Completed:**

1. Updated consumers to accept `DataSource[MinorLeagueBatterSeasonStats]`:
   - [x] `minors/rate_computer.py` — `MLERateComputer.milb_source` and `MLEAugmentedRateComputer.milb_source`
   - [x] `minors/training_data.py` — `MLETrainingDataCollector.milb_source`
   - [x] `minors/training.py` — `MLEModelTrainer.milb_source` (passes to collector)
   - [x] `minors/evaluation.py` — `MLEEvaluator.milb_source` (passes to collector)
   - [x] `pipeline/builder.py` — `_resolve_milb_source()` returns `DataSource[T]`

2. Replaced `CachedMinorLeagueDataSource` with `cached()` + `MiLBBatterStatsSerializer`.

3. Removed `MinorLeagueDataSource` protocol and `CachedMinorLeagueDataSource`.

#### 7c. `PlayerIdMapper` → `Player` Identity Through the Pipeline (~40 refs)

**Status:** Done

**Resolution:** Added `player: Player | None` field to `PlayerRates` and created a
`PlayerIdentityEnricher` adjuster that runs first in the pipeline chain. The enricher
stamps `Player` objects (with MLBAM IDs) onto `PlayerRates`, so downstream stages read
`player.player.mlbam_id` directly instead of calling `id_mapper.fangraphs_to_mlbam()`.

**Completed:**

1. Added `player: Player | None = None` to `PlayerRates` (pipeline/types.py)
2. Created `PlayerIdentityEnricher` adjuster (pipeline/stages/identity_enricher.py)
3. Propagated `player=` field through all adjuster construction sites (~15 files)
4. Removed `id_mapper` from pipeline adjuster stages:
   - [x] `pipeline/stages/statcast_adjuster.py`
   - [x] `pipeline/stages/pitcher_statcast_adjuster.py`
   - [x] `pipeline/stages/batter_babip_adjuster.py`
   - [x] `pipeline/stages/gb_residual_adjuster.py`
   - [x] `pipeline/stages/mtl_blender.py`
5. Changed remaining `PlayerIdMapper` type annotations to `SfbbMapper`:
   - [x] `pipeline/stages/mtl_rate_computer.py`
   - [x] `minors/rate_computer.py`
   - [x] `pipeline/skill_data.py`
   - [x] `ml/training.py`
   - [x] `ml/mtl/dataset.py`
   - [x] `projections/adapter.py`
   - [x] `minors/training_data.py`
   - [x] `draft/positions.py`
   - [x] `evaluation/cli.py`
   - [x] `pipeline/presets.py`
   - [x] `minors/evaluation.py`
   - [x] `minors/training.py`
   - [x] `ml/cli.py`
6. Updated `PipelineBuilder` to insert `PlayerIdentityEnricher` as first adjuster
   and removed `id_mapper` from adjuster construction calls
7. Deleted `PlayerIdMapper` protocol, `fangraphs_to_yahoo()`, and `fg_to_yahoo_map`
   from `SfbbMapper`
8. Updated all tests to use `Player` objects on `PlayerRates` instead of `FakeIdMapper`

#### 7d. `ADPSource` / `ProjectionSource` — Remove Legacy Side of Dual Registries

These protocols are already bridged — new `DataSource[T]` wrappers exist alongside legacy
classes. Migration is about removing the legacy side:

- [ ] **ADP registry:** Remove `get_source()` / `ADPSourceFactory` from `adp/registry.py`,
  keep only `get_datasource()` / `ADPDataSourceFactory`. Update any remaining `ADPSource`
  imports.
- [ ] **ADP composite:** Remove `CompositeADPSource`, keep `CompositeADPDataSource`.
- [ ] **Projection wrappers:** Remove `FanGraphsProjectionSource` / `CSVProjectionSource` if
  all consumers use the `DataSource[T]` factories. Or keep as inner implementation detail
  wrapped by `BattingProjectionDataSource` / `PitchingProjectionDataSource`.

#### 7e. `valuation/ProjectionSource` — Separate Concern

This is a *different* protocol from `projections/ProjectionSource`:
```python
# valuation/projection_source.py
class ProjectionSource(Protocol):
    def batting_projections(self) -> list[BattingProjection]: ...
    def pitching_projections(self) -> list[PitchingProjection]: ...
```
Used by the evaluation harness (`evaluation/harness.py`, `evaluation/cli.py`) and implemented
by `PipelineProjectionSource` (pipeline/source.py). This is not part of the DataSource
migration — it models "a source of projection results" for evaluation, not raw data fetching.
Keep as-is; optionally rename to `ProjectionProvider` to avoid confusion with
`projections/ProjectionSource`.

---

## Resolved Questions

1. **MinorLeagueDataSource level parameter**: No consumer uses level-specific queries.
   Level-specific methods only exist in `CachedMinorLeagueDataSource`. The new DataSource
   implementations return all levels; consumers filter as needed.
2. **Yahoo sources**: Do not fit `DataSource[T]`. Keep `RosterSource`, `DraftResultsSource`,
   and `PositionSource` as-is (see Phase 5).
3. **DraftResultsSource**: Keep as multi-method protocol — splitting into 3 separate
   DataSources would add complexity with no benefit for a single-consumer protocol.
4. **Backward compatibility**: Remove old protocols once all consumers are migrated within
   each phase (7a–7d). No deprecation period needed — this is internal code.

---

## Dependencies

```
Phase 7a (StatsDataSource)
  └─► pipeline/engine.py, pipeline/protocols.py (signature change)
      └─► evaluation/*, ros/*, ml/* (consumer updates)
          └─► services/container.py (wiring)
              └─► Remove StatsDataSource, PybaseballDataSource, CachedStatsDataSource

Phase 7b (MinorLeagueDataSource)
  └─► minors/rate_computer.py (field type change)
      └─► minors/training_data.py, training.py, evaluation.py (field type change)
          └─► pipeline/builder.py (wiring)
              └─► Remove MinorLeagueDataSource, CachedMinorLeagueDataSource

Phase 7c (PlayerIdMapper) ✅
  └─► PlayerRates.player field + PlayerIdentityEnricher adjuster
      └─► Remove id_mapper from adjuster stages
          └─► Change PlayerIdMapper refs to SfbbMapper
              └─► Delete PlayerIdMapper protocol + unused SfbbMapper methods

Phase 7d (ADPSource / ProjectionSource)
  └─► Remove legacy registry functions
      └─► Remove legacy composite/scraper classes
          └─► Remove ADPSource, ProjectionSource protocols
```
