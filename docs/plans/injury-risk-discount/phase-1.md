# Phase 1 Implementation Plan: Injury History Profile

## Overview

Build an injury history profiling system that aggregates IL stint data into per-player summaries. This phase creates the domain model, service layer, CLI commands, and comprehensive tests to provide standalone value as a risk-assessment tool before integrating into the valuation pipeline in later phases.

## Implementation Order

Following TDD discipline: write tests first, then implement to pass, then verify.

---

## Step 1: Domain Model - InjuryProfile

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/injury_profile.py`

**Action:** Create new file

**Content:**
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class InjuryProfile:
    player_id: int
    seasons_tracked: int
    total_stints: int
    total_days_lost: int
    avg_days_per_season: float
    max_days_in_season: int
    pct_seasons_with_il: float
    injury_locations: dict[str, int]
    recent_stints: list[tuple[int, str, int]]  # (season, il_type, days)
```

**Details:**
- Follows existing domain model pattern (frozen dataclass)
- `injury_locations` maps location string to count (e.g., `{"Right elbow": 2, "Left hamstring": 1}`)
- `recent_stints` holds last 2 seasons of IL placements as tuples
- All fields required (no Optional) - use sensible defaults for clean histories

---

## Step 2: Test - InjuryProfile Domain Model

**File:** `/Users/edward/Projects/fbm/tests/domain/test_injury_profile.py`

**Action:** Create new file

**Content:**
- Test that InjuryProfile is frozen (immutability)
- Test construction with all fields
- Test that injury_locations and recent_stints work as expected

**Pattern:** Follow `/Users/edward/Projects/fbm/tests/domain/` - simple dataclass validation tests

---

## Step 3: Service - InjuryProfiler Tests (TDD)

**File:** `/Users/edward/Projects/fbm/tests/services/test_injury_profiler.py`

**Action:** Create new file

**Content:**
Write comprehensive tests covering:

1. **Healthy player** (no IL stints) → profile with zeros
2. **Chronically injured player** (multiple stints across seasons) → correct aggregation
3. **Single-incident player** (one 60-day IL) → distinct from chronic
4. **Days calculation** - test both `days` field and `end_date - start_date` fallback
5. **Recent stints** - verify last 2 seasons only
6. **Injury location aggregation** - count by location string
7. **Percentage calculations** - `pct_seasons_with_il` and `avg_days_per_season`
8. **Edge cases:**
   - Player with stints in non-consecutive seasons
   - Multiple stints in same season
   - Missing `days` field (calculate from dates)
   - Missing both `days` and `end_date` (use IL type default: 10-day=10, 15-day=15, 60-day=60)

**Pattern:** Follow `/Users/edward/Projects/fbm/tests/services/test_adp_report.py`
- Use `seed_player()` helper from `tests.helpers`
- Create helper `_seed_il_stint(conn, player_id, season, start_date, il_type, **kwargs)` at top of file
- Organize into classes by scenario: `TestHealthyPlayer`, `TestChronicallyInjured`, `TestDaysCalculation`, etc.
- Each test builds stints in database, calls service, asserts profile fields

---

## Step 4: Service - InjuryProfiler Implementation

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/injury_profiler.py`

**Action:** Create new file

**Content:**
```python
import logging
from datetime import datetime

from fantasy_baseball_manager.domain.il_stint import ILStint
from fantasy_baseball_manager.domain.injury_profile import InjuryProfile

logger = logging.getLogger(__name__)

_IL_TYPE_DEFAULTS = {"10": 10, "15": 15, "60": 60}


def build_profiles(il_stints: list[ILStint], seasons: list[int]) -> dict[int, InjuryProfile]:
    """Aggregate IL stint data into per-player injury profiles.
    
    Args:
        il_stints: List of IL stint records
        seasons: Seasons to track (determines seasons_tracked denominator)
    
    Returns:
        Dict mapping player_id to InjuryProfile
    """
    # Implementation logic:
    # 1. Group stints by player_id
    # 2. For each player:
    #    - Count total stints
    #    - Sum days (with fallback logic)
    #    - Track seasons with IL
    #    - Aggregate injury_locations
    #    - Extract recent_stints (last 2 seasons, sorted desc)
    #    - Calculate derived fields
    # 3. For players in tracking pool with no stints, create clean profile
```

**Details:**
- Use `repos.protocols.ILStintRepo` as constructor param (dependency injection)
- Days calculation logic: use `days` if present, else parse `start_date` and `end_date`, else use IL type default
- Date parsing: `datetime.fromisoformat(date_str)` → `(end - start).days`
- Handle None/missing location gracefully (skip or use "Unknown")
- `recent_stints` sorted by season descending, limit 2
- `pct_seasons_with_il = len(unique_seasons_with_il) / seasons_tracked`
- Return dict keyed by player_id for efficient lookup

---

## Step 5: CLI Command - Single Player Profile (Test)

**File:** `/Users/edward/Projects/fbm/tests/cli/test_injury_profile_commands.py`

**Action:** Create new file

**Content:**
- Test `fbm report injury-profile <player-name>` command
- Mock/stub approach: create in-memory DB, seed player + stints, invoke CLI, capture output
- Assert output contains player name, key stats (total_stints, total_days_lost)

**Pattern:** Follow existing CLI test patterns (check `/Users/edward/Projects/fbm/tests/cli/` for examples)
- Use `typer.testing.CliRunner` if available, or subprocess invocation
- May need to add to existing test file or create new one

---

## Step 6: CLI Command - Single Player Profile (Implementation)

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/app.py`

**Action:** Modify existing file

**Changes:**
1. Add import: `from fantasy_baseball_manager.services.injury_profiler import build_profiles`
2. Add new command under `report_app` subcommand group (around line 1200):

```python
@report_app.command("injury-profile")
def report_injury_profile(
    player_name: Annotated[str, typer.Argument(help="Player name ('Last' or 'Last, First')")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show a player's injury history summary."""
    with build_injury_profile_context(data_dir) as ctx:
        # Parse player name, lookup player_id
        # Fetch IL stints for player
        # Build profile
        # Print via new output function
        profile = ctx.service.lookup_profile(player_name)
        print_injury_profile(profile)
```

**Details:**
- Follow pattern from `report_value_over_adp` command (line 1371)
- Reuse player name parsing logic (see `projections_lookup` for reference, line 715)
- Delegate to new factory function `build_injury_profile_context`

---

## Step 7: CLI Factory - Injury Profile Context

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/factory.py`

**Action:** Modify existing file

**Changes:**
1. Add import: `from fantasy_baseball_manager.services.injury_profiler import InjuryProfilerService`
2. Add context dataclass (around line 200):

```python
@dataclass(frozen=True)
class InjuryProfileContext:
    conn: sqlite3.Connection
    service: InjuryProfilerService
```

3. Add factory function (around line 250):

```python
@contextmanager
def build_injury_profile_context(data_dir: str) -> Iterator[InjuryProfileContext]:
    """Composition-root context manager for injury profile commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        service = InjuryProfilerService(
            player_repo=SqlitePlayerRepo(conn),
            il_stint_repo=SqliteILStintRepo(conn),
        )
        yield InjuryProfileContext(conn=conn, service=service)
    finally:
        conn.close()
```

**Pattern:** Follow `build_adp_report_context` pattern (line ~240 in factory.py)

---

## Step 8: Service - InjuryProfilerService (Wrapper)

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/injury_profiler.py`

**Action:** Modify existing file (add service class)

**Changes:**
Add service class that wraps `build_profiles` function:

```python
class InjuryProfilerService:
    def __init__(self, player_repo: PlayerRepo, il_stint_repo: ILStintRepo) -> None:
        self._player_repo = player_repo
        self._il_stint_repo = il_stint_repo
    
    def lookup_profile(self, player_name: str) -> InjuryProfile | None:
        """Look up injury profile for a single player by name."""
        # Parse name (Last or Last, First)
        # Query player_repo
        # If not found, return None
        # Fetch IL stints
        # Build profile
        # Return profile or clean profile if no stints
    
    def list_high_risk(
        self,
        season: int,
        min_stints: int = 1,
        top_n: int | None = None,
    ) -> list[tuple[InjuryProfile, str]]:
        """List most injury-prone players in projectable pool."""
        # Fetch all IL stints
        # Build all profiles
        # Filter by min_stints
        # Sort by total_days_lost descending
        # Limit to top_n
        # Return list of (profile, player_name) tuples
```

**Pattern:** Follow `ADPReportService` (uses repos to fetch, delegates to pure function for logic)

---

## Step 9: CLI Output - Print Injury Profile

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/_output.py`

**Action:** Modify existing file

**Changes:**
1. Add import: `from fantasy_baseball_manager.domain.injury_profile import InjuryProfile`
2. Add function (around line 800):

```python
def print_injury_profile(profile: InjuryProfile, player_name: str) -> None:
    """Print a single player's injury profile."""
    console.print(f"[bold]Injury Profile:[/bold] {player_name}")
    console.print(f"  Seasons tracked: {profile.seasons_tracked}")
    console.print(f"  Total IL stints: {profile.total_stints}")
    console.print(f"  Total days lost: {profile.total_days_lost}")
    console.print(f"  Avg days/season: {profile.avg_days_per_season:.1f}")
    console.print(f"  Max days in season: {profile.max_days_in_season}")
    console.print(f"  % seasons w/ IL: {profile.pct_seasons_with_il * 100:.1f}%")
    
    if profile.injury_locations:
        console.print(f"  Injury locations:")
        for loc, count in sorted(profile.injury_locations.items(), key=lambda x: -x[1]):
            console.print(f"    - {loc}: {count}x")
    
    if profile.recent_stints:
        console.print(f"  Recent stints (last 2 seasons):")
        for season, il_type, days in profile.recent_stints:
            console.print(f"    - {season}: {il_type}-day IL ({days} days)")
```

**Pattern:** Follow `print_value_over_adp` for rich console formatting

---

## Step 10: CLI Command - Injury Risk Leaderboard (Test)

**File:** `/Users/edward/Projects/fbm/tests/cli/test_injury_profile_commands.py`

**Action:** Modify existing file

**Content:**
- Test `fbm report injury-risks --season 2025 --min-stints 2` command
- Seed multiple players with varying IL histories
- Assert output lists highest-risk players first
- Assert filtering by min-stints works

---

## Step 11: CLI Command - Injury Risk Leaderboard (Implementation)

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/app.py`

**Action:** Modify existing file

**Changes:**
Add command under `report_app`:

```python
@report_app.command("injury-risks")
def report_injury_risks(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    min_stints: Annotated[int, typer.Option("--min-stints", help="Min IL stints")] = 1,
    top: Annotated[int | None, typer.Option("--top", help="Show top N players")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """List the most injury-prone players in the projectable pool."""
    with build_injury_profile_context(data_dir) as ctx:
        profiles = ctx.service.list_high_risk(season, min_stints=min_stints, top_n=top)
        print_injury_risk_leaderboard(profiles, season)
```

---

## Step 12: CLI Output - Print Injury Risk Leaderboard

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/_output.py`

**Action:** Modify existing file

**Changes:**
Add function:

```python
def print_injury_risk_leaderboard(
    profiles: list[tuple[InjuryProfile, str]], season: int
) -> None:
    """Print injury risk leaderboard as a table."""
    console.print(f"[bold]Injury Risk Leaderboard[/bold] — Season {season}")
    
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Rank", justify="right")
    table.add_column("Player")
    table.add_column("Stints", justify="right")
    table.add_column("Days Lost", justify="right")
    table.add_column("Avg/Season", justify="right")
    table.add_column("% Seasons", justify="right")
    table.add_column("Top Location")
    
    for rank, (profile, player_name) in enumerate(profiles, start=1):
        top_loc = max(profile.injury_locations.items(), key=lambda x: x[1])[0] if profile.injury_locations else "—"
        table.add_row(
            str(rank),
            player_name,
            str(profile.total_stints),
            str(profile.total_days_lost),
            f"{profile.avg_days_per_season:.1f}",
            f"{profile.pct_seasons_with_il * 100:.0f}%",
            top_loc,
        )
    
    console.print(table)
```

---

## Step 13: Integration Test - End-to-End

**File:** `/Users/edward/Projects/fbm/tests/services/test_injury_profiler.py`

**Action:** Modify existing file

**Content:**
Add integration test that:
1. Seeds real-world-like IL stint data (mix of healthy, injured, chronic)
2. Calls `build_profiles` with all stints
3. Verifies profiles for multiple players
4. Checks edge cases (no stints, missing data fields)

---

## Step 14: Verification & Cleanup

1. Run full test suite: `uv run pytest tests/services/test_injury_profiler.py -v`
2. Run full test suite: `uv run pytest tests/cli/test_injury_profile_commands.py -v`
3. Run full test suite: `uv run pytest -v` (all tests pass)
4. Run type checker: `uv run ty check src tests`
5. Run linter: `uv run ruff check src tests`
6. Run formatter: `uv run ruff format src tests`
7. Manual smoke test:
   - `uv run fbm report injury-profile "Trout, Mike"`
   - `uv run fbm report injury-risks --season 2024 --min-stints 2 --top 20`

---

## Step 15: Commit

**Command:**
```bash
git add \
  src/fantasy_baseball_manager/domain/injury_profile.py \
  src/fantasy_baseball_manager/services/injury_profiler.py \
  src/fantasy_baseball_manager/cli/app.py \
  src/fantasy_baseball_manager/cli/factory.py \
  src/fantasy_baseball_manager/cli/_output.py \
  tests/domain/test_injury_profile.py \
  tests/services/test_injury_profiler.py \
  tests/cli/test_injury_profile_commands.py && \
git commit -m "$(cat <<'EOF'
feat: add injury history profiling (phase 1)

Create injury profile domain model and service to aggregate IL stint
data into per-player summaries. Add CLI commands for single-player
lookups and injury risk leaderboards.

- InjuryProfile domain model with aggregated stats
- InjuryProfilerService with lookup and leaderboard methods
- CLI: `fbm report injury-profile <player-name>`
- CLI: `fbm report injury-risks --season <year> --min-stints <n>`
- Comprehensive test coverage for all scenarios

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Key Design Decisions

1. **Pure function + service wrapper pattern:** `build_profiles()` is a pure function that takes IL stints and returns profiles. `InjuryProfilerService` wraps it with repo access for CLI convenience. This enables easy testing and reuse.

2. **Days calculation fallback:** Use `days` field if present, else calculate from dates, else use IL type default. Handles incomplete data gracefully.

3. **Recent stints as list of tuples:** Simple, immutable structure. Alternative would be nested dataclass, but tuples are sufficient for this use case.

4. **Dependency injection:** Service takes repos as constructor params (not imported directly). Enables testing with fakes and follows project convention.

5. **CLI organization:** Commands live under `report_app` subcommand group, following existing pattern for analysis/reporting tools.

---

### Critical Files for Implementation

- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/injury_profile.py` - New domain model to create
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/injury_profiler.py` - Core aggregation logic to implement
- `/Users/edward/Projects/fbm/tests/services/test_injury_profiler.py` - Comprehensive test suite (TDD starting point)
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/app.py` - Add two new report commands (lines ~1200-1400)
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/factory.py` - Add context builder for DI wiring