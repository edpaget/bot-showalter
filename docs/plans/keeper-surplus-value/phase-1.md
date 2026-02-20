# Phase 1: Keeper Cost Storage and Import — Implementation Plan

## Overview

This phase implements the foundation of keeper/dynasty value analysis by creating the data model for keeper costs and providing import mechanisms. We'll build domain models, database persistence, CLI commands, and comprehensive tests following TDD principles.

## Implementation Order

Following TDD, each step implements tests first, then production code to pass those tests.

---

### Step 1: Domain Model — `KeeperCost`

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/keeper.py`

**Test file:** `/Users/edward/Projects/fbm/tests/domain/test_keeper.py`

**Order:**
1. Write test for `KeeperCost` dataclass instantiation and immutability
2. Implement `KeeperCost` frozen dataclass with fields:
   - `player_id: int`
   - `season: int`
   - `league: str` — league identifier matching fbm.toml league names
   - `cost: float` — cost in auction dollars
   - `years_remaining: int = 1` — contract length remaining
   - `source: str` — one of "auction", "draft_round", "contract", "free_agent"
   - `id: int | None = None` — database primary key
   - `loaded_at: str | None = None` — timestamp

**Tests to write:**
- `test_keeper_cost_creation()` — verify all required fields
- `test_keeper_cost_defaults()` — verify `years_remaining=1` default
- `test_keeper_cost_immutable()` — verify frozen dataclass

**Pattern reference:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/adp.py`

---

### Step 2: Database Migration

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/db/migrations/015_keeper_cost.sql`

**No test file** (migrations are validated by connection tests)

**Content:**
```sql
CREATE TABLE IF NOT EXISTS keeper_cost (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id       INTEGER NOT NULL REFERENCES player(id),
    season          INTEGER NOT NULL,
    league          TEXT NOT NULL,
    cost            REAL NOT NULL,
    years_remaining INTEGER NOT NULL DEFAULT 1,
    source          TEXT NOT NULL,
    loaded_at       TEXT,
    UNIQUE(player_id, season, league)
);
```

**Validation:** Migration runs automatically when connection opens. Verify by running tests that create connections.

**Pattern reference:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/db/migrations/014_adp.sql`

---

### Step 3: Repository — `SqliteKeeperCostRepo`

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/repos/keeper_repo.py`

**Test file:** `/Users/edward/Projects/fbm/tests/repos/test_keeper_repo.py`

**Order:**
1. Write tests for `upsert_batch()` method
2. Implement `upsert_batch(costs: list[KeeperCost]) -> int` — batch upsert, returns count
3. Write tests for `find_by_season_league()` method
4. Implement `find_by_season_league(season: int, league: str) -> list[KeeperCost]`
5. Write tests for `find_by_player()` method
6. Implement `find_by_player(player_id: int) -> list[KeeperCost]`
7. Write tests for idempotent upsert (re-importing same data updates existing records)

**Repository methods:**
```python
class SqliteKeeperCostRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert_batch(self, costs: list[KeeperCost]) -> int:
        """Upsert multiple keeper costs. Returns count of records upserted."""
        pass

    def find_by_season_league(self, season: int, league: str) -> list[KeeperCost]:
        """Find all keeper costs for a season and league."""
        pass

    def find_by_player(self, player_id: int) -> list[KeeperCost]:
        """Find all keeper cost records for a player across seasons/leagues."""
        pass

    @staticmethod
    def _select_sql() -> str:
        return "SELECT id, player_id, season, league, cost, years_remaining, source, loaded_at FROM keeper_cost"

    @staticmethod
    def _row_to_keeper_cost(row: sqlite3.Row) -> KeeperCost:
        return KeeperCost(...)
```

**Tests to write:**
- `test_upsert_batch_creates_new()` — batch insert new records
- `test_upsert_batch_updates_existing()` — re-upsert same player/season/league updates cost
- `test_find_by_season_league()` — query by season and league
- `test_find_by_season_league_empty()` — returns empty list when no matches
- `test_find_by_player()` — query all costs for a player
- `test_find_by_player_empty()` — returns empty list for unknown player
- `test_upsert_batch_idempotent()` — calling twice with same data doesn't duplicate

**Pattern reference:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/repos/adp_repo.py`

---

### Step 4: Keeper Cost Import Mapper

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/ingest/keeper_mapper.py`

**Test file:** `/Users/edward/Projects/fbm/tests/ingest/test_keeper_mapper.py`

**Order:**
1. Write tests for player name resolution
2. Implement `_build_player_lookup()` function (reuses name normalization from ADP mapper)
3. Write tests for CSV row parsing
4. Implement `import_keeper_costs()` function
5. Write tests for unmatched players tracking

**Function signature:**
```python
@dataclass(frozen=True)
class KeeperImportResult:
    loaded: int
    skipped: int
    unmatched: list[str]

def import_keeper_costs(
    rows: list[dict[str, Any]],
    repo: KeeperCostRepo,
    players: list[Player],
    season: int,
    league: str,
    default_source: str = "auction",
) -> KeeperImportResult:
    """Import keeper costs from CSV rows.
    
    Expected CSV columns:
    - Player (or Name): player name
    - Cost: auction dollar cost
    - Years (optional): years remaining, defaults to 1
    - Source (optional): auction/draft_round/contract/free_agent
    """
    pass
```

**Tests to write:**
- `test_import_keeper_costs_success()` — successful import with matched players
- `test_import_keeper_costs_unmatched()` — tracks unmatched player names
- `test_import_keeper_costs_optional_fields()` — handles missing Years/Source columns
- `test_import_keeper_costs_empty()` — handles empty input
- `test_import_keeper_costs_name_normalization()` — handles accented names, suffixes

**Pattern reference:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/ingest/adp_mapper.py` (especially `_normalize_name`, `_build_player_lookups`, and `ingest_fantasypros_adp`)

---

### Step 5: CLI Factory — Keeper Context

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/factory.py`

**Test file:** `/Users/edward/Projects/fbm/tests/cli/test_factory.py` (add new test)

**Order:**
1. Write test for `build_keeper_context()` context manager
2. Implement `KeeperContext` dataclass
3. Implement `build_keeper_context()` context manager

**Add to factory.py:**
```python
@dataclass(frozen=True)
class KeeperContext:
    conn: sqlite3.Connection
    keeper_repo: SqliteKeeperCostRepo
    player_repo: SqlitePlayerRepo

@contextmanager
def build_keeper_context(data_dir: str) -> Iterator[KeeperContext]:
    """Composition-root context manager for keeper commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        yield KeeperContext(
            conn=conn,
            keeper_repo=SqliteKeeperCostRepo(conn),
            player_repo=SqlitePlayerRepo(conn),
        )
    finally:
        conn.close()
```

**Tests to write:**
- `test_build_keeper_context()` — verify context manager wires dependencies correctly

**Pattern reference:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/factory.py` lines 159-173 (`RunsContext`)

---

### Step 6: CLI Commands — Keeper Subcommand Group

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/app.py`

**Test file:** `/Users/edward/Projects/fbm/tests/cli/test_keeper.py`

**Order:**
1. Create `keeper_app` Typer subcommand group
2. Write test for `import` command
3. Implement `fbm keeper import` command
4. Write test for `set` command
5. Implement `fbm keeper set` command
6. Add keeper_app to main app

**Add to app.py:**
```python
# Near top with other imports
from fantasy_baseball_manager.repos.keeper_repo import SqliteKeeperCostRepo
from fantasy_baseball_manager.cli.factory import build_keeper_context
from fantasy_baseball_manager.ingest.keeper_mapper import import_keeper_costs

# After other subcommand groups (around line 1200+)
keeper_app = typer.Typer(name="keeper", help="Manage keeper costs and decisions")
app.add_typer(keeper_app, name="keeper")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data"

@keeper_app.command("import")
def keeper_import_cmd(
    csv_path: Annotated[Path, typer.Argument(help="Path to CSV file with keeper costs")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option("--league", help="League name (from fbm.toml)")],
    source: Annotated[str, typer.Option("--source", help="Cost source type")] = "auction",
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Import keeper costs from a CSV file.
    
    CSV should have columns: Player, Cost, Years (optional), Source (optional).
    """
    if not csv_path.exists():
        print_error(f"file not found: {csv_path}")
        raise typer.Exit(code=1)

    with build_keeper_context(data_dir) as ctx:
        # Read CSV
        rows = []
        with csv_path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        players = ctx.player_repo.all()
        result = import_keeper_costs(rows, ctx.keeper_repo, players, season, league, default_source=source)
        ctx.conn.commit()
        
        console.print(f"  Loaded {result.loaded} keeper costs, skipped {result.skipped}")
        if result.unmatched:
            console.print(
                f"  [yellow]Unmatched players ({len(result.unmatched)}):[/yellow] {', '.join(result.unmatched[:20])}"
            )
            if len(result.unmatched) > 20:
                console.print(f"  ... and {len(result.unmatched) - 20} more")

@keeper_app.command("set")
def keeper_set_cmd(
    player_name: Annotated[str, typer.Argument(help="Player name")],
    cost: Annotated[float, typer.Option("--cost", help="Keeper cost in auction dollars")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option("--league", help="League name (from fbm.toml)")],
    years: Annotated[int, typer.Option("--years", help="Years remaining")] = 1,
    source: Annotated[str, typer.Option("--source", help="Cost source type")] = "auction",
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Manually set keeper cost for a single player."""
    with build_keeper_context(data_dir) as ctx:
        # Search for player by name
        matches = ctx.player_repo.search_by_name(player_name)
        if not matches:
            print_error(f"no player found matching '{player_name}'")
            raise typer.Exit(code=1)
        if len(matches) > 1:
            console.print(f"[yellow]Multiple players match '{player_name}':[/yellow]")
            for p in matches:
                console.print(f"  {p.name_first} {p.name_last} (ID: {p.id})")
            print_error("please be more specific")
            raise typer.Exit(code=1)
        
        player = matches[0]
        keeper_cost = KeeperCost(
            player_id=player.id,
            season=season,
            league=league,
            cost=cost,
            years_remaining=years,
            source=source,
        )
        ctx.keeper_repo.upsert_batch([keeper_cost])
        ctx.conn.commit()
        
        console.print(f"[green]Set keeper cost:[/green] {player.name_first} {player.name_last} = ${cost:.0f} ({league}, {season})")
```

**Tests to write:**
- `test_keeper_import_command()` — integration test with temp CSV
- `test_keeper_import_command_file_not_found()` — error handling
- `test_keeper_set_command()` — single player manual entry
- `test_keeper_set_command_player_not_found()` — error when player doesn't exist
- `test_keeper_set_command_ambiguous_player()` — error when multiple matches

**Pattern reference:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/app.py` lines 1110-1143 (ingest ADP command)

---

### Step 7: Update CLI Factory Exports

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/factory.py`

**Add to exports section:**
Ensure `build_keeper_context` and `KeeperContext` are available for import in app.py (they're defined in the module, no explicit export needed in Python).

---

### Step 8: Add Missing Import in app.py

**File:** `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/app.py`

**Add near top:**
```python
import csv  # Add if not already imported
from fantasy_baseball_manager.domain.keeper import KeeperCost
from fantasy_baseball_manager.ingest.keeper_mapper import import_keeper_costs
```

---

### Step 9: Integration Tests

**File:** `/Users/edward/Projects/fbm/tests/cli/test_keeper.py`

**Order:**
1. Write end-to-end test importing CSV and verifying database state
2. Write end-to-end test for manual set command
3. Write test for idempotent import (re-importing same CSV updates costs)

**Tests to write:**
- `test_keeper_import_e2e()` — full workflow: create CSV, import, query results
- `test_keeper_set_e2e()` — full workflow: set cost, verify in DB
- `test_keeper_import_idempotent()` — import twice, verify no duplicates

**Pattern reference:** `/Users/edward/Projects/fbm/tests/cli/test_ingest.py`

---

## File Summary

**New files to create:**

1. `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/keeper.py` — `KeeperCost` dataclass
2. `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/db/migrations/015_keeper_cost.sql` — database schema
3. `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/repos/keeper_repo.py` — `SqliteKeeperCostRepo`
4. `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/ingest/keeper_mapper.py` — `import_keeper_costs()` function
5. `/Users/edward/Projects/fbm/tests/domain/test_keeper.py` — domain model tests
6. `/Users/edward/Projects/fbm/tests/repos/test_keeper_repo.py` — repository tests
7. `/Users/edward/Projects/fbm/tests/ingest/test_keeper_mapper.py` — mapper tests
8. `/Users/edward/Projects/fbm/tests/cli/test_keeper.py` — CLI integration tests

**Files to modify:**

1. `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/factory.py` — add `KeeperContext` and `build_keeper_context()`
2. `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/app.py` — add `keeper_app` subcommand group with `import` and `set` commands
3. `/Users/edward/Projects/fbm/tests/cli/test_factory.py` — add test for `build_keeper_context()`

---

## Test Execution After Each Step

After implementing each step:
1. Run `uv run pytest tests/domain/test_keeper.py` (after step 1)
2. Run `uv run pytest tests/repos/test_keeper_repo.py` (after step 3)
3. Run `uv run pytest tests/ingest/test_keeper_mapper.py` (after step 4)
4. Run `uv run pytest tests/cli/test_factory.py -k keeper` (after step 5)
5. Run `uv run pytest tests/cli/test_keeper.py` (after step 6)
6. Run full suite: `uv run pytest`

---

## Acceptance Criteria Validation

After all steps complete, verify:

1. **Keeper costs are stored and queryable by season and league**
   - Test: `test_find_by_season_league()` passes
   - Manual: `fbm keeper set "Juan Soto" --cost 30 --season 2026 --league h2h`, then query DB

2. **CSV import resolves player names to internal IDs**
   - Test: `test_import_keeper_costs_success()` passes
   - Manual: Create CSV with known players, import, verify player_id is populated

3. **Manual set command works for individual players**
   - Test: `test_keeper_set_e2e()` passes
   - Manual: Run `fbm keeper set` and verify DB record

4. **Re-importing is idempotent (upsert)**
   - Test: `test_upsert_batch_idempotent()` and `test_keeper_import_idempotent()` pass
   - Manual: Import same CSV twice, verify only one record per player/season/league

---

## Implementation Notes

- **All imports must be top-level** (enforced by ruff rule `PLC0415`)
- **Follow TDD**: write test first, implement to pass, refactor
- **Use dependency injection**: repos passed to functions/constructors
- **Type annotations everywhere**: all function signatures, return types
- **Run pytest after each step** to ensure no regressions
- **Commit when all tests pass** with message: `feat: add keeper cost storage and import (phase 1)`

---

### Critical Files for Implementation

- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/adp.py` — Pattern for frozen dataclass with optional fields
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/repos/adp_repo.py` — Pattern for repo with upsert, query methods
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/ingest/adp_mapper.py` — Pattern for player name resolution and CSV import
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/cli/app.py` — Pattern for Typer subcommand groups and CLI commands
- `/Users/edward/Projects/fbm/tests/repos/test_adp_repo.py` — Pattern for repo tests with fixtures