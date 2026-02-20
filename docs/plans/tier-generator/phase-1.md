# Phase 1: Tier Clustering Engine — Implementation Plan

## Overview

Implement the core tier assignment algorithm as a standalone service that groups players by position and assigns them to tiers based on value dropoffs. This phase delivers the foundational clustering logic with two methods (gap-based and Jenks natural breaks) that will be exposed via CLI in Phase 2.

## Prerequisites

- Existing valuation data in the database (from ZAR system)
- Player domain model and repo
- Valuation domain model and repo

## Implementation Order

Follow TDD: write the failing test first, then implement the minimum code to pass, then refactor.

---

## Step 1: Create domain model for tier assignments

### File: `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/tier.py` (NEW)

Create a frozen dataclass to represent a single tier assignment:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TierAssignment:
    player_id: int
    player_name: str
    position: str
    tier: int
    value: float
    rank: int
```

This model captures all the information needed for draft-day tier display: who the player is, which position group they belong to, which tier within that position, and their underlying value/rank from the valuation system.

**Why this structure:**
- Follows existing domain model patterns (frozen dataclass, explicit types)
- Includes both `player_id` and `player_name` so consumers don't need to join back to the player table
- `rank` is within-position rank (1 = best at that position)
- `tier` is a positive integer (1 = tier 1, 2 = tier 2, etc.)

---

## Step 2: Write tests for gap-based tier generation (test-first)

### File: `/Users/edward/Projects/fbm/tests/services/test_tier_generator.py` (NEW)

Start with a test that validates gap-based clustering on synthetic data with obvious gaps.

```python
import sqlite3

from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.services.tier_generator import generate_tiers
from tests.helpers import seed_player


def _seed_valuation(
    conn: sqlite3.Connection,
    player_id: int,
    position: str,
    value: float,
    rank: int,
    season: int = 2026,
    system: str = "zar",
    version: str = "1.0",
    player_type: str = "batter",
) -> None:
    repo = SqliteValuationRepo(conn)
    repo.upsert(
        Valuation(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            projection_system="steamer",
            projection_version="2026.1",
            player_type=player_type,
            position=position,
            value=value,
            rank=rank,
            category_scores={"hr": 1.0},
        )
    )


class TestGapBasedTiers:
    def test_single_tier_no_gaps(self, conn: sqlite3.Connection) -> None:
        """When values are close together, all players go in tier 1."""
        pid1 = seed_player(conn, name_first="Player", name_last="One", mlbam_id=1)
        pid2 = seed_player(conn, name_first="Player", name_last="Two", mlbam_id=2)
        pid3 = seed_player(conn, name_first="Player", name_last="Three", mlbam_id=3)
        
        # Values: 30, 29, 28 — small gaps, should all be tier 1
        _seed_valuation(conn, pid1, position="SS", value=30.0, rank=1)
        _seed_valuation(conn, pid2, position="SS", value=29.0, rank=2)
        _seed_valuation(conn, pid3, position="SS", value=28.0, rank=3)
        
        player_repo = SqlitePlayerRepo(conn)
        valuation_repo = SqliteValuationRepo(conn)
        valuations = valuation_repo.get_by_season(2026, system="zar")
        
        tiers = generate_tiers(valuations, player_repo, method="gap", max_tiers=5)
        
        assert len(tiers) == 3
        assert all(t.tier == 1 for t in tiers)
        assert tiers[0].rank == 1
        assert tiers[1].rank == 2
        assert tiers[2].rank == 3

    def test_two_tiers_obvious_gap(self, conn: sqlite3.Connection) -> None:
        """When there's a large gap, split into two tiers."""
        pid1 = seed_player(conn, name_first="Elite", name_last="Player", mlbam_id=1)
        pid2 = seed_player(conn, name_first="Good", name_last="Player", mlbam_id=2)
        pid3 = seed_player(conn, name_first="Average", name_last="Player", mlbam_id=3)
        
        # Values: 50, 48, 20 — big gap between 48 and 20
        _seed_valuation(conn, pid1, position="OF", value=50.0, rank=1)
        _seed_valuation(conn, pid2, position="OF", value=48.0, rank=2)
        _seed_valuation(conn, pid3, position="OF", value=20.0, rank=3)
        
        player_repo = SqlitePlayerRepo(conn)
        valuation_repo = SqliteValuationRepo(conn)
        valuations = valuation_repo.get_by_season(2026, system="zar")
        
        tiers = generate_tiers(valuations, player_repo, method="gap", max_tiers=5)
        
        of_tiers = [t for t in tiers if t.position == "OF"]
        assert len(of_tiers) == 3
        assert of_tiers[0].tier == 1  # Elite
        assert of_tiers[1].tier == 1  # Good (close to elite)
        assert of_tiers[2].tier == 2  # Average (big gap down)

    def test_multiple_positions_separated(self, conn: sqlite3.Connection) -> None:
        """Tiers are computed independently per position."""
        pid_ss = seed_player(conn, name_first="SS", name_last="One", mlbam_id=1)
        pid_of = seed_player(conn, name_first="OF", name_last="One", mlbam_id=2)
        
        _seed_valuation(conn, pid_ss, position="SS", value=40.0, rank=1)
        _seed_valuation(conn, pid_of, position="OF", value=40.0, rank=2)
        
        player_repo = SqlitePlayerRepo(conn)
        valuation_repo = SqliteValuationRepo(conn)
        valuations = valuation_repo.get_by_season(2026, system="zar")
        
        tiers = generate_tiers(valuations, player_repo, method="gap", max_tiers=5)
        
        # Both should be tier 1 within their own position
        ss_tiers = [t for t in tiers if t.position == "SS"]
        of_tiers = [t for t in tiers if t.position == "OF"]
        assert len(ss_tiers) == 1
        assert len(of_tiers) == 1
        assert ss_tiers[0].tier == 1
        assert ss_tiers[0].rank == 1
        assert of_tiers[0].tier == 1
        assert of_tiers[0].rank == 1

    def test_equal_values_same_tier(self, conn: sqlite3.Connection) -> None:
        """Players with equal value must be in the same tier."""
        pid1 = seed_player(conn, name_first="Player", name_last="One", mlbam_id=1)
        pid2 = seed_player(conn, name_first="Player", name_last="Two", mlbam_id=2)
        
        _seed_valuation(conn, pid1, position="1B", value=25.0, rank=1)
        _seed_valuation(conn, pid2, position="1B", value=25.0, rank=2)
        
        player_repo = SqlitePlayerRepo(conn)
        valuation_repo = SqliteValuationRepo(conn)
        valuations = valuation_repo.get_by_season(2026, system="zar")
        
        tiers = generate_tiers(valuations, player_repo, method="gap", max_tiers=5)
        
        assert len(tiers) == 2
        assert tiers[0].tier == tiers[1].tier == 1

    def test_max_tiers_caps_output(self, conn: sqlite3.Connection) -> None:
        """max_tiers parameter limits the number of tiers per position."""
        players = []
        for i in range(10):
            pid = seed_player(conn, name_first=f"Player", name_last=f"{i}", mlbam_id=i + 1)
            # Create big gaps: 100, 80, 60, 40, 20, 10, 9, 8, 7, 6
            value = 100.0 - i * 20.0 if i < 5 else 10.0 - (i - 5)
            _seed_valuation(conn, pid, position="C", value=value, rank=i + 1)
            players.append(pid)
        
        player_repo = SqlitePlayerRepo(conn)
        valuation_repo = SqliteValuationRepo(conn)
        valuations = valuation_repo.get_by_season(2026, system="zar")
        
        tiers = generate_tiers(valuations, player_repo, method="gap", max_tiers=3)
        
        tier_nums = {t.tier for t in tiers}
        assert len(tier_nums) <= 3
```

**Test Strategy:**
- Start with simple cases (single tier, obvious two-tier split)
- Verify position separation
- Verify edge cases (equal values, max_tiers cap)
- Use synthetic data with known gaps to make assertions clear

---

## Step 3: Implement gap-based tier generation service

### File: `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/tier_generator.py` (NEW)

Implement the core function with gap-based clustering logic:

```python
import logging
from collections import defaultdict

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.tier import TierAssignment
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.protocols import PlayerRepo

logger = logging.getLogger(__name__)


def generate_tiers(
    valuations: list[Valuation],
    player_repo: PlayerRepo,
    method: str = "gap",
    max_tiers: int = 5,
) -> list[TierAssignment]:
    """Generate tier assignments from valuations.
    
    Args:
        valuations: List of player valuations
        player_repo: Repository to resolve player names
        method: Clustering method ("gap" or "jenks")
        max_tiers: Maximum number of tiers per position
        
    Returns:
        List of tier assignments, sorted by position and rank
    """
    # Group valuations by position
    by_position: dict[str, list[Valuation]] = defaultdict(list)
    for val in valuations:
        by_position[val.position].append(val)
    
    # Sort each position group by value descending
    for pos in by_position:
        by_position[pos].sort(key=lambda v: v.value, reverse=True)
    
    # Load player names
    player_ids = [v.player_id for v in valuations]
    players = player_repo.get_by_ids(player_ids)
    player_map: dict[int, Player] = {p.id: p for p in players if p.id is not None}
    
    # Apply clustering method per position
    all_tiers: list[TierAssignment] = []
    for pos, pos_vals in by_position.items():
        if method == "gap":
            tier_assignments = _assign_tiers_gap(pos_vals, player_map, max_tiers)
        elif method == "jenks":
            tier_assignments = _assign_tiers_jenks(pos_vals, player_map, max_tiers)
        else:
            msg = f"Unknown tier method: {method}"
            raise ValueError(msg)
        
        all_tiers.extend(tier_assignments)
    
    # Sort by position, then rank
    all_tiers.sort(key=lambda t: (t.position, t.rank))
    return all_tiers


def _assign_tiers_gap(
    valuations: list[Valuation],
    player_map: dict[int, Player],
    max_tiers: int,
) -> list[TierAssignment]:
    """Assign tiers using gap detection (1.5× median gap threshold)."""
    if not valuations:
        return []
    
    # Already sorted descending by value
    values = [v.value for v in valuations]
    
    # Compute gaps between consecutive players
    gaps = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    
    if not gaps:
        # Single player → tier 1
        return [_make_tier_assignment(valuations[0], player_map, tier=1, rank=1)]
    
    # Threshold: 1.5× the median gap
    sorted_gaps = sorted(gaps)
    median_gap = sorted_gaps[len(sorted_gaps) // 2]
    threshold = 1.5 * median_gap
    
    # Find tier boundaries (indices where gap exceeds threshold)
    tier_boundaries = [0]  # Always start with index 0
    for i, gap in enumerate(gaps):
        if gap > threshold:
            tier_boundaries.append(i + 1)
    
    # Cap at max_tiers
    if len(tier_boundaries) >= max_tiers:
        tier_boundaries = tier_boundaries[:max_tiers]
    
    # Assign tier numbers
    assignments: list[TierAssignment] = []
    tier_num = 1
    boundary_idx = 0
    
    for rank, val in enumerate(valuations, start=1):
        # Check if we've crossed into the next tier
        if boundary_idx + 1 < len(tier_boundaries) and rank > tier_boundaries[boundary_idx + 1]:
            tier_num += 1
            boundary_idx += 1
        
        assignments.append(_make_tier_assignment(val, player_map, tier=tier_num, rank=rank))
    
    return assignments


def _assign_tiers_jenks(
    valuations: list[Valuation],
    player_map: dict[int, Player],
    max_tiers: int,
) -> list[TierAssignment]:
    """Assign tiers using Jenks natural breaks (placeholder for now)."""
    # For Phase 1, implement a simple placeholder that returns tier 1 for all
    # This will be replaced with actual Jenks algorithm in a follow-up step
    assignments: list[TierAssignment] = []
    for rank, val in enumerate(valuations, start=1):
        assignments.append(_make_tier_assignment(val, player_map, tier=1, rank=rank))
    return assignments


def _make_tier_assignment(
    valuation: Valuation,
    player_map: dict[int, Player],
    tier: int,
    rank: int,
) -> TierAssignment:
    """Helper to construct a TierAssignment from a valuation."""
    player = player_map.get(valuation.player_id)
    player_name = f"{player.name_first} {player.name_last}" if player else f"Unknown ({valuation.player_id})"
    
    return TierAssignment(
        player_id=valuation.player_id,
        player_name=player_name,
        position=valuation.position,
        tier=tier,
        value=valuation.value,
        rank=rank,
    )
```

**Implementation notes:**
- Follow dependency injection pattern: accept `player_repo` as parameter
- Use `defaultdict` to group by position
- Gap threshold = 1.5× median gap (tunable, but roadmap doesn't specify making it a param yet)
- Return sorted list for consistent ordering
- Log at DEBUG level for troubleshooting

**Run tests:** `uv run pytest tests/services/test_tier_generator.py::TestGapBasedTiers -v`

---

## Step 4: Add tests for Jenks natural breaks method

### File: `/Users/edward/Projects/fbm/tests/services/test_tier_generator.py` (EXTEND)

Add a test class for Jenks method:

```python
class TestJenksTiers:
    def test_jenks_basic(self, conn: sqlite3.Connection) -> None:
        """Jenks method produces reasonable tiers on test data."""
        # Create players with three clear clusters: [45, 44], [30, 29], [10, 9]
        pids = []
        for i, value in enumerate([45.0, 44.0, 30.0, 29.0, 10.0, 9.0]):
            pid = seed_player(conn, name_first=f"P{i}", name_last="Test", mlbam_id=i + 1)
            _seed_valuation(conn, pid, position="2B", value=value, rank=i + 1)
            pids.append(pid)
        
        player_repo = SqlitePlayerRepo(conn)
        valuation_repo = SqliteValuationRepo(conn)
        valuations = valuation_repo.get_by_season(2026, system="zar")
        
        tiers = generate_tiers(valuations, player_repo, method="jenks", max_tiers=5)
        
        # Should produce 3 tiers
        tier_nums = {t.tier for t in tiers}
        assert len(tier_nums) == 3
        assert tiers[0].tier == tiers[1].tier == 1
        assert tiers[2].tier == tiers[3].tier == 2
        assert tiers[4].tier == tiers[5].tier == 3

    def test_jenks_respects_max_tiers(self, conn: sqlite3.Connection) -> None:
        """Jenks respects max_tiers cap."""
        for i in range(10):
            pid = seed_player(conn, name_first=f"P{i}", name_last="Test", mlbam_id=i + 1)
            # Values that would naturally create many tiers
            value = 100.0 - i * 10.0
            _seed_valuation(conn, pid, position="3B", value=value, rank=i + 1)
        
        player_repo = SqlitePlayerRepo(conn)
        valuation_repo = SqliteValuationRepo(conn)
        valuations = valuation_repo.get_by_season(2026, system="zar")
        
        tiers = generate_tiers(valuations, player_repo, method="jenks", max_tiers=3)
        
        tier_nums = {t.tier for t in tiers}
        assert len(tier_nums) <= 3
```

---

## Step 5: Implement Jenks natural breaks algorithm

### File: `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/tier_generator.py` (EXTEND)

Replace the `_assign_tiers_jenks` placeholder with a real implementation. Use a lightweight pure-Python Jenks algorithm to avoid adding a new dependency.

Add this implementation (adapted from open-source Jenks implementations):

```python
def _assign_tiers_jenks(
    valuations: list[Valuation],
    player_map: dict[int, Player],
    max_tiers: int,
) -> list[TierAssignment]:
    """Assign tiers using Jenks natural breaks optimization."""
    if not valuations:
        return []
    
    values = [v.value for v in valuations]
    
    # Jenks needs at least as many values as tiers
    n_tiers = min(max_tiers, len(values))
    
    if n_tiers == 1:
        # Single tier
        return [_make_tier_assignment(val, player_map, tier=1, rank=i + 1) for i, val in enumerate(valuations)]
    
    # Compute Jenks breaks
    breaks = _jenks_breaks(values, n_tiers)
    
    # Assign each player to a tier based on which break interval they fall into
    assignments: list[TierAssignment] = []
    for rank, val in enumerate(valuations, start=1):
        value = val.value
        tier = 1
        for i in range(1, len(breaks)):
            if value <= breaks[i]:
                tier = i
                break
        assignments.append(_make_tier_assignment(val, player_map, tier=tier, rank=rank))
    
    return assignments


def _jenks_breaks(values: list[float], n_classes: int) -> list[float]:
    """Compute Jenks natural breaks for a list of values.
    
    Returns a list of break points (including min and max).
    """
    # Sort descending to match our value ordering (high = good)
    sorted_values = sorted(values, reverse=True)
    n = len(sorted_values)
    
    # Edge cases
    if n == 0:
        return []
    if n_classes >= n:
        return sorted_values
    
    # Dynamic programming matrices
    # mat1[i][j] = variance for values[0..i] split into j classes
    # mat2[i][j] = index of lower bound of last class
    mat1 = [[float('inf')] * (n_classes + 1) for _ in range(n)]
    mat2 = [[0] * (n_classes + 1) for _ in range(n)]
    
    # Initialize: one class containing first i values
    for i in range(n):
        mat1[i][1] = _variance(sorted_values[0:i + 1])
        mat2[i][1] = 0
    
    # Fill the DP table
    for j in range(2, n_classes + 1):
        for i in range(j - 1, n):
            min_var = float('inf')
            min_k = 0
            for k in range(j - 2, i):
                var = mat1[k][j - 1] + _variance(sorted_values[k + 1:i + 1])
                if var < min_var:
                    min_var = var
                    min_k = k
            mat1[i][j] = min_var
            mat2[i][j] = min_k + 1
    
    # Backtrack to find break points
    breaks = [sorted_values[0]]  # max value
    k = n - 1
    for j in range(n_classes, 1, -1):
        break_idx = mat2[k][j]
        breaks.append(sorted_values[break_idx])
        k = break_idx - 1
    breaks.append(sorted_values[-1])  # min value
    
    return breaks


def _variance(values: list[float]) -> float:
    """Compute variance of a list of values."""
    if not values:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / n
```

**Implementation notes:**
- Pure Python implementation (no external dependency)
- Uses dynamic programming for optimal break points
- Sorts values descending to match our "high value = good" convention
- Returns break points that define tier boundaries

**Run tests:** `uv run pytest tests/services/test_tier_generator.py::TestJenksTiers -v`

---

## Step 6: Add edge case tests

### File: `/Users/edward/Projects/fbm/tests/services/test_tier_generator.py` (EXTEND)

Add tests for edge cases and error conditions:

```python
class TestEdgeCases:
    def test_empty_valuations(self, conn: sqlite3.Connection) -> None:
        """Empty valuations list returns empty tiers."""
        player_repo = SqlitePlayerRepo(conn)
        tiers = generate_tiers([], player_repo, method="gap", max_tiers=5)
        assert tiers == []

    def test_single_player(self, conn: sqlite3.Connection) -> None:
        """Single player gets tier 1."""
        pid = seed_player(conn, name_first="Only", name_last="Player", mlbam_id=1)
        _seed_valuation(conn, pid, position="DH", value=25.0, rank=1)
        
        player_repo = SqlitePlayerRepo(conn)
        valuation_repo = SqliteValuationRepo(conn)
        valuations = valuation_repo.get_by_season(2026, system="zar")
        
        tiers = generate_tiers(valuations, player_repo, method="gap", max_tiers=5)
        
        assert len(tiers) == 1
        assert tiers[0].tier == 1
        assert tiers[0].rank == 1

    def test_unknown_method_raises(self, conn: sqlite3.Connection) -> None:
        """Invalid method raises ValueError."""
        pid = seed_player(conn, name_first="Test", name_last="Player", mlbam_id=1)
        _seed_valuation(conn, pid, position="OF", value=25.0, rank=1)
        
        player_repo = SqlitePlayerRepo(conn)
        valuation_repo = SqliteValuationRepo(conn)
        valuations = valuation_repo.get_by_season(2026, system="zar")
        
        import pytest
        with pytest.raises(ValueError, match="Unknown tier method"):
            generate_tiers(valuations, player_repo, method="invalid", max_tiers=5)

    def test_missing_player_name_shows_unknown(self, conn: sqlite3.Connection) -> None:
        """If player not found in repo, show 'Unknown (id)'."""
        # Manually insert a valuation without a matching player
        repo = SqliteValuationRepo(conn)
        repo.upsert(
            Valuation(
                player_id=99999,  # Non-existent
                season=2026,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="2026.1",
                player_type="batter",
                position="OF",
                value=25.0,
                rank=1,
                category_scores={"hr": 1.0},
            )
        )
        
        player_repo = SqlitePlayerRepo(conn)
        valuations = repo.get_by_season(2026, system="zar")
        
        tiers = generate_tiers(valuations, player_repo, method="gap", max_tiers=5)
        
        assert len(tiers) == 1
        assert "Unknown (99999)" in tiers[0].player_name
```

**Run all tests:** `uv run pytest tests/services/test_tier_generator.py -v`

---

## Step 7: Run full test suite and commit

### Commands:

```bash
# Run tier generator tests
uv run pytest tests/services/test_tier_generator.py -v

# Run full test suite to ensure no regressions
uv run pytest

# If all pass, stage and commit
git add src/fantasy_baseball_manager/domain/tier.py \
        src/fantasy_baseball_manager/services/tier_generator.py \
        tests/services/test_tier_generator.py && \
git commit -m "feat: add tier clustering engine with gap and Jenks methods

- Add TierAssignment domain model
- Implement generate_tiers service function with gap-based and Jenks natural breaks clustering
- Group players by position and assign tiers based on value dropoffs
- Support max_tiers parameter to cap tier count per position
- Add comprehensive test coverage for both methods and edge cases

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Verification Checklist

Before marking Phase 1 complete, verify:

- [ ] `TierAssignment` domain model exists in `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/tier.py`
- [ ] `generate_tiers` function exists in `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/tier_generator.py`
- [ ] Gap-based method produces reasonable tiers on test data with obvious clusters
- [ ] Jenks method produces reasonable tiers on test data
- [ ] `max_tiers` parameter correctly caps tier count per position
- [ ] Players with equal value are placed in the same tier
- [ ] Tiers are computed independently per position
- [ ] Edge cases handled (empty input, single player, unknown method)
- [ ] All tests pass: `uv run pytest tests/services/test_tier_generator.py -v`
- [ ] No regressions: `uv run pytest`
- [ ] Code formatted: `uv run ruff format src tests`
- [ ] No lint errors: `uv run ruff check src tests`
- [ ] Type checks pass: `uv run ty check src tests`
- [ ] Changes committed with conventional commit message

---

## Dependencies for Phase 2

Phase 2 (CLI command and formatted output) will depend on:

1. **Domain model:** `TierAssignment` (created in this phase)
2. **Service function:** `generate_tiers` (created in this phase)
3. **Repos:** `ValuationRepo`, `PlayerRepo` (already exist)
4. **CLI patterns:** Factory context manager, typer command groups, rich Table output (already exist)

Phase 2 will NOT modify the service layer — it will only add:
- CLI command under a new `draft` group
- Factory context builder for tier generation
- Output formatter using `rich.table.Table`

---

### Critical Files for Implementation

- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/tier.py` - New domain model to create
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/tier_generator.py` - Core clustering logic to implement
- `/Users/edward/Projects/fbm/tests/services/test_tier_generator.py` - Test suite to write first (TDD)
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/domain/valuation.py` - Existing pattern to follow for domain models
- `/Users/edward/Projects/fbm/src/fantasy_baseball_manager/services/adp_report.py` - Existing pattern to follow for service structure