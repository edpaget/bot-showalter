# Phase 1: Scarcity Metrics and Report — Implementation Plan

## Overview

Implement positional scarcity analysis that computes dropoff metrics per position and exposes them through a CLI report. The key metrics are based on league roster settings and measure how quickly value disappears at each position.

## Domain Models

### File: `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/positional_scarcity.py`

Create a new domain module with frozen dataclasses:

1. **`PositionScarcity`** — frozen dataclass with fields:
   - `position: str` — position abbreviation (C, 1B, 2B, 3B, SS, OF, SP, RP, etc.)
   - `tier_1_value: float` — average value of top N players (where N = league teams × roster slots for position)
   - `replacement_value: float` — value at the replacement threshold (N+1)
   - `total_surplus: float` — sum of (value - replacement_value) for all starters
   - `dropoff_slope: float` — linear regression slope of value vs rank
   - `steep_rank: int | None` — rank where dropoff accelerates (elbow detection), None if no clear elbow

## Service Layer

### File: `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/positional_scarcity.py`

Create service with functions that accept dependencies via parameters (not class-based):

1. **`compute_scarcity(valuations: list[Valuation], league_settings: LeagueSettings) -> list[PositionScarcity]`**
   - Group valuations by position
   - For each position:
     - Determine N (starters) = `league_settings.teams * slots_for_position`
     - Calculate `tier_1_value` = mean of top N players' values
     - Calculate `replacement_value` = value at rank N (or N+1)
     - Calculate `total_surplus` = sum of (value - replacement_value) for ranks 1..N
     - Calculate `dropoff_slope` using linear regression (numpy or scipy) on rank vs value for ranks 1..N
     - Detect `steep_rank` using simple elbow detection (e.g., find largest second derivative or use percentage drop threshold)
   - Return list sorted by scarcity severity (steepest dropoff first)

2. **Helper: `_get_roster_slots(position: str, league_settings: LeagueSettings) -> int`**
   - Map position to roster slot count from `league_settings.positions`
   - Handle multi-position slots (e.g., OF might be 3)
   - Handle special cases: UTIL, bench slots
   - Return 0 if position not rostered

3. **Helper: `_detect_elbow(values: list[float]) -> int | None`**
   - Simple elbow detection algorithm
   - Calculate second derivative or percentage drops
   - Return rank where acceleration occurs, or None if no clear elbow

## CLI Integration

### File: `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/app.py`

Add a new subcommand group `draft_app` and a `scarcity` command:

1. Create `draft_app = typer.Typer(name="draft", help="Draft strategy tools")`
2. Add to main app: `app.add_typer(draft_app, name="draft")`
3. Implement `@draft_app.command("scarcity")` with parameters:
   - `season: Annotated[int, typer.Option("--season", help="Season year")]`
   - `system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar"`
   - `version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0"`
   - `league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "h2h"`
   - `data_dir: _DataDirOpt = "./data"`
4. Command implementation:
   - Load league settings via `load_league(league_name, Path.cwd())`
   - Build context with valuations repo
   - Fetch valuations via repo
   - Call `compute_scarcity(valuations, league)`
   - Call output function `print_scarcity_report(scarcities, league)`

### File: `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/factory.py`

Add context manager for draft commands:

1. **`@dataclass(frozen=True) class DraftContext`**:
   - `conn: sqlite3.Connection`
   - `player_repo: SqlitePlayerRepo`
   - `valuation_repo: SqliteValuationRepo`

2. **`@contextmanager def build_draft_context(data_dir: str) -> Iterator[DraftContext]`**:
   - Open DB connection
   - Construct repos
   - Yield context
   - Close connection

### File: `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/_output.py`

Add output formatting function:

1. **`def print_scarcity_report(scarcities: list[PositionScarcity], league: LeagueSettings) -> None`**
   - Print header with league name and settings summary
   - Create Rich table with columns:
     - Position
     - Slots (roster slots per team)
     - Tier1$ (tier 1 avg value)
     - Repl$ (replacement value)
     - Surplus (total surplus)
     - Slope (dropoff slope, formatted with sign and color)
     - SteepRk (elbow rank if detected)
   - Print each position's row
   - Use color coding: red for steep slopes, green for shallow

## Tests

### File: `/Users/edward/Projects/fbm/tests/domain/test_positional_scarcity.py`

Test domain models (minimal, mostly ensure frozen dataclass construction works):

1. `test_position_scarcity_frozen()` — verify immutability
2. `test_position_scarcity_all_fields()` — verify all fields set correctly

### File: `/Users/edward/Projects/fbm/tests/services/test_positional_scarcity.py`

Comprehensive service tests with synthetic data:

1. **`test_compute_scarcity_basic(conn)`**:
   - Seed 24 batters with C position, declining values: $40, $38, $36, ... $-6
   - Seed 60 batters with OF position, gradually declining values
   - Create league with 12 teams, 1 C, 3 OF
   - Compute scarcity
   - Assert C has steeper slope than OF
   - Assert C ranks higher in scarcity order

2. **`test_tier_1_value_calculation(conn)`**:
   - Seed 12 players at SS with known values
   - League has 12 teams, 1 SS slot → N=12
   - Assert `tier_1_value` equals mean of top 12 values

3. **`test_replacement_value(conn)`**:
   - Seed 13 players at 1B
   - League has 12 teams, 1 1B slot → N=12
   - Assert `replacement_value` equals value of rank 12 or 13 player

4. **`test_total_surplus(conn)`**:
   - Seed starters with known values above replacement
   - Manually calculate expected surplus
   - Assert computed surplus matches

5. **`test_dropoff_slope_steep_vs_flat(conn)`**:
   - Seed two positions with same tier 1 avg but different slopes
   - Assert steep position has more negative slope

6. **`test_elbow_detection(conn)`**:
   - Seed position with clear elbow (e.g., $30-$28 for ranks 1-5, then $15-$5 for ranks 6-12)
   - Assert `steep_rank` is detected near rank 6

7. **`test_no_elbow_returns_none(conn)`**:
   - Seed position with linear decline
   - Assert `steep_rank` is None

8. **`test_scarcity_sorting(conn)`**:
   - Seed multiple positions with different slopes
   - Assert result is sorted steepest first

9. **`test_position_not_rostered(conn)`**:
   - Seed players at position not in league settings
   - Assert position excluded or handled gracefully

10. **`test_empty_valuations(conn)`**:
    - Call with empty list
    - Assert returns empty list

### File: `/Users/edward/Projects/fbm/tests/cli/test_draft_commands.py` (new file)

CLI integration tests (optional but recommended):

1. `test_draft_scarcity_command_runs()` — smoke test that command doesn't crash

## Implementation Order (TDD)

Follow strict TDD discipline:

### Step 1: Domain models
1. Write `/Users/edward/Projects/fbm/tests/domain/test_positional_scarcity.py`
2. Implement `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/positional_scarcity.py`
3. Run `uv run pytest tests/domain/test_positional_scarcity.py` — should pass

### Step 2: Service — tier_1_value calculation
1. Write `test_tier_1_value_calculation` in `/Users/edward/Projects/fbm/tests/services/test_positional_scarcity.py`
2. Implement skeleton `compute_scarcity()` and `_get_roster_slots()` in `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/positional_scarcity.py` to pass this test
3. Run `uv run pytest tests/services/test_positional_scarcity.py::test_tier_1_value_calculation` — should pass

### Step 3: Service — replacement_value
1. Write `test_replacement_value`
2. Extend `compute_scarcity()` to calculate replacement value
3. Run test — should pass

### Step 4: Service — total_surplus
1. Write `test_total_surplus`
2. Extend `compute_scarcity()` to calculate surplus
3. Run test — should pass

### Step 5: Service — dropoff_slope
1. Write `test_dropoff_slope_steep_vs_flat`
2. Implement slope calculation using numpy's `polyfit` or scipy's `linregress`
3. Run test — should pass

### Step 6: Service — elbow detection
1. Write `test_elbow_detection` and `test_no_elbow_returns_none`
2. Implement `_detect_elbow()` helper
3. Integrate elbow detection into `compute_scarcity()`
4. Run tests — should pass

### Step 7: Service — scarcity sorting
1. Write `test_scarcity_sorting`
2. Add sorting logic to `compute_scarcity()`
3. Run test — should pass

### Step 8: Service — edge cases
1. Write `test_position_not_rostered`, `test_empty_valuations`
2. Add defensive checks
3. Run tests — should pass

### Step 9: Service — integration test
1. Write `test_compute_scarcity_basic` (full end-to-end service test)
2. Should already pass if previous tests pass
3. Run full service test suite — should pass

### Step 10: CLI factory
1. Write test stub in `/Users/edward/Projects/fbm/tests/cli/test_draft_commands.py` (or skip if not doing CLI tests)
2. Implement `DraftContext` and `build_draft_context()` in `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/factory.py`

### Step 11: CLI output
1. Implement `print_scarcity_report()` in `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/_output.py`
2. Manually verify table formatting looks good (or write output test if desired)

### Step 12: CLI command
1. Implement `draft_app` and `scarcity` command in `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/app.py`
2. Import output function at top of file
3. Wire up context, service call, and output

### Step 13: Manual verification
1. Run `uv run fbm draft scarcity --season 2025 --system zar --version 1.0 --league h2h` (requires existing valuations in DB)
2. Verify table output makes sense

### Step 14: Full test suite
1. Run `uv run pytest` — all tests should pass
2. Run `uv run pytest --cov` — coverage should meet threshold
3. Fix any linting/type errors: `uv run ruff check src tests`, `uv run ty check src tests`

### Step 15: Commit
1. `git add src/fantasy_baseball_manager/domain/positional_scarcity.py src/fantasy_baseball_manager/services/positional_scarcity.py src/fantasy_baseball_manager/cli/app.py src/fantasy_baseball_manager/cli/factory.py src/fantasy_baseball_manager/cli/_output.py tests/domain/test_positional_scarcity.py tests/services/test_positional_scarcity.py && git commit -m "$(cat <<'EOF'`
2. Commit message:
```
feat: add positional scarcity metrics and CLI report

Compute per-position scarcity metrics (tier 1 avg, replacement value,
surplus, dropoff slope, elbow rank) and expose via `fbm draft scarcity`
command. Positions are ranked by dropoff severity to guide draft timing
decisions.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

## Dependencies

- **Existing domain models**: `Valuation`, `LeagueSettings` (already exist)
- **Existing repos**: `ValuationRepo` (already exists)
- **Existing CLI infrastructure**: `typer`, `rich.Table`, factory pattern (all exist)
- **New dependencies**: None (use existing numpy/scipy already in project for linear regression)

## Key Design Decisions

1. **Service is function-based, not class-based** — follows the pattern in `adp_report.py` which uses a class but accepts repos as constructor deps. However, for a pure computation with no state, a function is simpler and matches the project's DI philosophy.

2. **League settings drive roster slots** — use `league_settings.positions` dict to map position to slot count. Default league `h2h` in `fbm.toml` has positions defined.

3. **Scarcity sorting** — steeper (more negative) slope = more scarce = higher in list. This matches the roadmap's intent.

4. **Elbow detection** — simple heuristic (e.g., find max second derivative, or rank where drop exceeds threshold). Can iterate if too noisy.

5. **CLI command structure** — new `draft` subcommand group aligns with future draft-related commands (tiers, draft board export).

6. **Position grouping** — players in valuations already have a `position` field from ZAR valuation system. Group by this field directly.

## Test Data Strategy

- Use `seed_player()` helper from `tests/helpers.py`
- Seed `Valuation` records via `SqliteValuationRepo(conn).upsert()`
- Create synthetic declining values that exhibit clear scarcity differences
- Use small league (e.g., 12 teams) to keep test data manageable

## Acceptance Criteria Verification

- **Scarcity report ranks positions by dropoff severity**: verified by `test_scarcity_sorting`
- **Metrics computed relative to league settings**: verified by `test_tier_1_value_calculation`, `test_replacement_value` using league teams × slots
- **Positions with steep dropoffs rank higher**: verified by `test_compute_scarcity_basic`
- **Report includes all batting and pitching positions**: verified by CLI manual testing with real data

---

### Critical Files for Implementation

- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/positional_scarcity.py` — Core domain model: PositionScarcity dataclass
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/positional_scarcity.py` — Core logic: compute_scarcity function and helpers
- `/Users/edward/Projects/fbm/tests/services/test_positional_scarcity.py` — Primary test suite covering all service logic
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/app.py` — CLI command registration and wiring
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/_output.py` — Output formatting for scarcity table