# Draft Board Service — Phase 1 Implementation Plan

## Context

Build a stateless draft board service that combines valuations + league settings (and optionally tiers/ADP) into a ranked board of players. This is the core data structure that all draft-day consumers (CLI, HTML, TUI, agents, live draft tracker) will build on. No CLI, no file I/O — just the service layer and tests.

Key design goals:
- **Stateless & re-callable**: accepts the player pool as input. During a live draft, pass a shrinking pool and get updated ranks.
- **League-aware**: uses LeagueSettings to determine which stat categories appear as z-score columns.
- **Optional enrichment**: tiers and ADP are joined when provided, None when absent.

## New files

| File | Purpose |
|------|---------|
| `src/fantasy_baseball_manager/domain/draft_board.py` | `DraftBoardRow`, `DraftBoard`, `TierAssignment` frozen dataclasses |
| `src/fantasy_baseball_manager/services/draft_board.py` | `build_draft_board()` pure function |
| `tests/services/test_draft_board.py` | Comprehensive tests |

## Domain models — `domain/draft_board.py`

```python
TierAssignment(frozen):
    player_id: int
    tier: int

DraftBoardRow(frozen):
    player_id: int
    player_name: str
    rank: int                          # 1-indexed, assigned after sorting by value
    player_type: str                   # "batter" | "pitcher"
    position: str                      # from Valuation.position
    value: float                       # ZAR dollar value
    category_z_scores: dict[str, float]  # filtered to league categories for this player_type
    tier: int | None = None
    adp_overall: float | None = None   # ADP pick number
    adp_rank: int | None = None
    adp_delta: int | None = None       # adp_rank - board_rank (positive = market undervalues)

DraftBoard(frozen):
    rows: list[DraftBoardRow]
    batting_categories: tuple[str, ...]   # category keys, from LeagueSettings
    pitching_categories: tuple[str, ...]  # category keys, from LeagueSettings
```

## Builder function — `services/draft_board.py`

```python
def build_draft_board(
    valuations: list[Valuation],
    league: LeagueSettings,
    player_names: dict[int, str],
    *,
    tiers: list[TierAssignment] | None = None,
    adp: list[ADP] | None = None,
) -> DraftBoard
```

### Logic

1. Extract `batting_categories` and `pitching_categories` key tuples from `league`.
2. Build tier lookup: `{player_id: tier}` from `tiers` list (empty dict if None).
3. Build ADP lookup: `{player_id: ADP}` from `adp` list (empty dict if None).
   - Group by player_id. When duplicates exist (two-way players), resolve per-valuation based on player_type.
   - Reuse the `_is_pitcher_adp()` helper pattern from `services/adp_report.py`.
4. Sort valuations by `value` descending.
5. For each valuation at rank `i`:
   - Filter `category_scores` to only keys matching the league's categories for this `player_type`.
   - Look up tier, ADP, player name.
   - Compute `adp_delta = adp_rank - board_rank` when ADP present.
   - Build `DraftBoardRow`.
6. Return `DraftBoard(rows, batting_categories, pitching_categories)`.

## TDD steps

1. Create `domain/draft_board.py` with the three frozen dataclasses.
2. **Test: rows ranked by value descending** — 3 valuations, verify rank order and rank values.
3. **Test: category z-scores filtered to league categories** — valuation with extra categories, verify only league ones appear.
4. **Test: pitcher uses pitching categories** — pitcher valuation filtered to pitching cats only.
5. Implement `build_draft_board()` — minimal version (no enrichment yet). Run tests.
6. **Test: board metadata** — verify `batting_categories` and `pitching_categories` tuples. Empty input returns empty board.
7. **Test: tier enrichment** — tiers joined by player_id; None when tiers not provided; None for unmatched players.
8. **Test: ADP enrichment** — ADP joined by player_id; delta computed; None when absent; lowest pick wins for duplicates.
9. **Test: two-way ADP** — pitcher valuation prefers SP ADP entry over DH entry.
10. Enhance ADP resolution to be player-type-aware. Run tests.
11. **Test: mixed pool** — batters and pitchers interleaved by value, each with correct category sets.
12. **Test: re-callability** — build board with 3 players, then with 2 (simulating a drafted player removed). Verify re-ranking.
13. **Test: player names** — verify names from the `player_names` map appear on rows; unknown IDs get a fallback.
14. Run full suite, verify no regressions.
15. Commit.
