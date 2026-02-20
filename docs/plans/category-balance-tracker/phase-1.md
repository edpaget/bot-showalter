# Phase 1 Implementation Plan: Roster Category Projection

## Overview

Implement the core roster category projection service that aggregates player projections to compute team-level category totals and rates. This phase creates the foundation for category balance tracking without league rank estimation (deferred to keep scope minimal).

## Implementation Order

### Step 1: Define domain models (TDD: test first)

**File**: `/Users/edward/Projects/fbm/tests/services/test_category_tracker.py`

Create test file with initial failing tests:
- `test_counting_stat_aggregation`: verify HR, R, RBI summed correctly across a 3-player roster
- `test_rate_stat_weighted_average`: verify AVG computed as total H / total AB (not simple average)
- `test_mixed_batters_and_pitchers`: verify batting and pitching categories computed separately
- `test_empty_roster_returns_zeros`: verify empty roster produces 0 for all categories
- `test_player_without_projection_ignored`: verify players without matching projection are skipped

Use fake repos from `/Users/edward/Projects/fbm/tests/fakes/repos.py` (FakePlayerRepo, FakeProjectionRepo).

**File**: `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/category_tracker.py`

Create frozen dataclasses:
```python
@dataclass(frozen=True)
class TeamCategoryProjection:
    category: str  # matches CategoryConfig.key
    projected_value: float
    
@dataclass(frozen=True)
class RosterAnalysis:
    projections: list[TeamCategoryProjection]
```

Note: `league_rank_estimate` and `strength` fields from roadmap are deferred for now to minimize scope. Focus on accurate stat aggregation first.

### Step 2: Implement category aggregation service (TDD: make tests pass)

**File**: `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/category_tracker.py`

Create service with dependency injection pattern (matches existing service structure like `ProjectionLookupService` and `ValuationLookupService`):

```python
class CategoryTrackerService:
    def __init__(self, projection_repo: ProjectionRepo, player_repo: PlayerRepo) -> None:
        self._projection_repo = projection_repo
        self._player_repo = player_repo
    
    def analyze_roster(
        self,
        player_ids: list[int],
        season: int,
        system: str,
        league_settings: LeagueSettings,
    ) -> RosterAnalysis:
        """Compute team category projections for a roster.
        
        For counting stats: sum across all players.
        For rate stats: compute weighted average using denominator 
        (e.g., AVG = sum(H) / sum(AB)).
        """
        ...
```

Implementation details:
- Fetch projections for all player_ids for given season/system
- Group by player_type (batter vs pitcher)
- For each category in league_settings.batting_categories and pitching_categories:
  - If stat_type == COUNTING: sum the stat across roster
  - If stat_type == RATE: parse numerator/denominator, compute weighted average
- Handle expression parsing for numerator/denominator (e.g., "h+bb+hbp" → sum of h, bb, hbp)
- Return TeamCategoryProjection list with one entry per category

Follow patterns from:
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/projection_lookup.py` for repo usage
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/league_settings.py` for CategoryConfig structure
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/config_league.py` for league loading patterns

### Step 3: Add integration tests with real repos

**File**: `/Users/edward/Projects/fbm/tests/services/test_category_tracker.py`

Add integration test class using real SqliteProjectionRepo and SqlitePlayerRepo:
- `test_integration_with_h2h_league`: load the "h2h" league from fbm.toml, seed 5 batters + 3 pitchers with known projections, verify category totals match manual calculation

Use helper patterns from `/Users/edward/Projects/fbm/tests/services/test_projection_lookup.py`:
- `seed_player()` from `/Users/edward/Projects/fbm/tests/helpers.py`
- Custom `_seed_projection()` helper
- Load league with `load_league("h2h", Path.cwd())` from config_league

### Step 4: Verify tests pass

Run:
```bash
uv run pytest tests/services/test_category_tracker.py -v
```

Fix any failures. Ensure all assertions pass and coverage includes both counting and rate stat paths.

### Step 5: Update factory for CLI integration (prepare for future CLI)

**File**: `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/factory.py`

Add context builder function (following pattern from `build_projections_context`, `build_valuations_context`):

```python
@dataclass(frozen=True)
class CategoryTrackerContext:
    conn: sqlite3.Connection
    tracker_service: CategoryTrackerService

@contextmanager
def build_category_tracker_context(data_dir: str) -> Iterator[CategoryTrackerContext]:
    """Composition-root context manager for category tracker commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        tracker_service = CategoryTrackerService(
            SqliteProjectionRepo(conn),
            SqlitePlayerRepo(conn),
        )
        yield CategoryTrackerContext(conn=conn, tracker_service=tracker_service)
    finally:
        conn.close()
```

Do NOT add CLI commands yet — phase 2 will add the `draft needs` command. This step just wires the service into the factory for future use.

### Step 6: Final verification and commit

Run full test suite:
```bash
uv run pytest
```

Run linter and type checker:
```bash
uv run ruff check src tests
uv run ty check src tests
```

Fix any issues. When all pass, commit:
```bash
git add src/fantasy_baseball_manager/domain/category_tracker.py \
        src/fantasy_baseball_manager/services/category_tracker.py \
        src/fantasy_baseball_manager/cli/factory.py \
        tests/services/test_category_tracker.py && \
git commit -m "$(cat <<'EOF'
feat: add roster category projection service (phase 1)

Implement CategoryTrackerService.analyze_roster() to aggregate player
projections into team-level category totals. Supports both counting
stats (sum) and rate stats (weighted average by denominator).

- Add TeamCategoryProjection and RosterAnalysis domain models
- Add CategoryTrackerService with projection_repo and player_repo deps
- Add build_category_tracker_context() factory for future CLI integration
- Add comprehensive tests with fake and real repos

Phase 1 of category balance tracker roadmap. Future phases will add
league rank estimation, need identification, and draft integration.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

## Files Summary

### Files to create:
1. `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/category_tracker.py` — Domain models (TeamCategoryProjection, RosterAnalysis)
2. `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/category_tracker.py` — CategoryTrackerService implementation
3. `/Users/edward/Projects/fbm/tests/services/test_category_tracker.py` — Comprehensive test suite

### Files to modify:
1. `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/factory.py` — Add CategoryTrackerContext and builder function

## Key Implementation Notes

### Numerator/Denominator Expression Parsing

For rate stats like OBP with `numerator = "h+bb+hbp"`, implement simple expression parser:
- Split on `+` to get stat names
- Look up each stat in projection.stat_json
- Sum the values
- Divide numerator sum by denominator sum

### Handling Missing Stats

If a projection is missing a required stat (e.g., no "ab" for AVG calculation):
- Skip that player for that specific category
- Log a debug message
- Continue with remaining players

### Player Type Filtering

- Batting categories only consider projections with `player_type == "batter"`
- Pitching categories only consider projections with `player_type == "pitcher"`
- Match pattern from existing code that groups by player_type

### League Settings Integration

Load league settings from fbm.toml using the existing `load_league()` function from config_league.py. The "h2h" league definition shows the exact structure:
- `batting_categories` and `pitching_categories` are lists of CategoryConfig
- Each CategoryConfig has key, name, stat_type, direction, and optional numerator/denominator
- Use StatType.COUNTING vs StatType.RATE to determine aggregation method

### Testing Strategy

Follow TDD strictly:
1. Write failing test with expected behavior
2. Implement minimal code to make test pass
3. Refactor for clarity
4. Repeat

Use both unit tests (with fakes) and integration tests (with real repos + in-memory DB) like existing service tests do.

### Critical Files for Implementation

- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/category_tracker.py` - Core domain models to create
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/category_tracker.py` - Main service logic to implement
- `/Users/edward/Projects/fbm/tests/services/test_category_tracker.py` - Test suite driving implementation
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/league_settings.py` - Pattern for category config structure
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/projection_lookup.py` - Pattern for service structure with repo injection