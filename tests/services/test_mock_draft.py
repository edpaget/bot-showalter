"""Tests for the mock draft simulation engine."""

import random

from fantasy_baseball_manager.domain.draft_board import DraftBoard, DraftBoardRow
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.services.draft_state import build_draft_roster_slots
from fantasy_baseball_manager.services.mock_draft import run_mock_draft
from fantasy_baseball_manager.services.mock_draft_bots import (
    ADPBot,
    BestValueBot,
    PositionalNeedBot,
    RandomBot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BATTING_CAT = CategoryConfig(key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
_PITCHING_CAT = CategoryConfig(key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER)


def _make_row(
    player_id: int,
    name: str,
    position: str,
    value: float,
    *,
    adp: float | None = None,
) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type="B" if position not in ("SP", "RP") else "P",
        position=position,
        value=value,
        category_z_scores={},
        adp_overall=adp if adp is not None else float(player_id),
    )


def _small_league() -> LeagueSettings:
    """4-team league with minimal roster: C, 1B, OF, UTIL, P."""
    return LeagueSettings(
        name="Small Test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=4,
        budget=260,
        roster_batters=3,
        roster_pitchers=1,
        batting_categories=(_BATTING_CAT,),
        pitching_categories=(_PITCHING_CAT,),
        positions={"C": 1, "1B": 1, "OF": 1},
        roster_util=1,
    )


def _small_board() -> DraftBoard:
    """Board with plenty of players for 4 teams × 5 slots = 20 picks."""
    positions = ["C", "1B", "OF", "SP"]
    rows: list[DraftBoardRow] = []
    for i in range(40):
        pos = positions[i % len(positions)]
        rows.append(_make_row(i + 1, f"Player {i + 1}", pos, 40.0 - i, adp=float(i + 1)))
    return DraftBoard(rows=rows, batting_categories=("HR",), pitching_categories=("K",))


def _12_team_league() -> LeagueSettings:
    """Realistic 12-team league."""
    return LeagueSettings(
        name="Standard 12-Team",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=10,
        roster_pitchers=8,
        batting_categories=(_BATTING_CAT,),
        pitching_categories=(_PITCHING_CAT,),
        positions={"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3, "MI": 1, "CI": 1},
        roster_util=1,
    )


def _12_team_board() -> DraftBoard:
    """Board with enough players for 12 teams × 20 slots = 240 picks.

    Position cycle mirrors a realistic draft pool: ~40% pitchers, ~60% batters
    with good coverage at each position including composite-eligible ones.
    """
    rows: list[DraftBoardRow] = []
    # 10 batters + 8 pitchers per cycle = 18 positions
    positions = [
        "C",
        "1B",
        "2B",
        "3B",
        "SS",
        "OF",
        "OF",
        "OF",
        "1B",
        "3B",
        "SP",
        "SP",
        "SP",
        "SP",
        "RP",
        "RP",
        "RP",
        "RP",
    ]
    pid = 1
    for i in range(360):
        pos = positions[i % len(positions)]
        ptype = "B" if pos not in ("SP", "RP") else "P"
        rows.append(
            DraftBoardRow(
                player_id=pid,
                player_name=f"Player {pid}",
                rank=pid,
                player_type=ptype,
                position=pos,
                value=360.0 - i,
                category_z_scores={},
                adp_overall=float(pid),
            )
        )
        pid += 1
    return DraftBoard(rows=rows, batting_categories=("HR",), pitching_categories=("K",))


def _make_bots(n: int, seed: int = 42) -> list[ADPBot | BestValueBot | PositionalNeedBot | RandomBot]:
    """Create a mix of bot strategies for n teams."""
    rng = random.Random(seed)
    bot_classes = [ADPBot, BestValueBot, PositionalNeedBot, RandomBot]
    bots: list[ADPBot | BestValueBot | PositionalNeedBot | RandomBot] = []
    for i in range(n):
        cls = bot_classes[i % len(bot_classes)]
        bots.append(cls(rng=random.Random(rng.randint(0, 2**32))))
    return bots


# ---------------------------------------------------------------------------
# Snake ordering tests
# ---------------------------------------------------------------------------


class TestSnakeOrdering:
    def test_round_1_ascending(self) -> None:
        """Round 1: team 0, 1, 2, 3 (ascending)."""
        league = _small_league()
        board = _small_board()
        bots = [BestValueBot(rng=random.Random(i)) for i in range(4)]
        result = run_mock_draft(board, league, bots, snake=True, seed=42)

        # First 4 picks should be teams 0, 1, 2, 3
        round_1 = [p for p in result.picks if p.round == 1]
        assert [p.team_idx for p in round_1] == [0, 1, 2, 3]

    def test_round_2_descending(self) -> None:
        """Round 2: team 3, 2, 1, 0 (descending)."""
        league = _small_league()
        board = _small_board()
        bots = [BestValueBot(rng=random.Random(i)) for i in range(4)]
        result = run_mock_draft(board, league, bots, snake=True, seed=42)

        round_2 = [p for p in result.picks if p.round == 2]
        assert [p.team_idx for p in round_2] == [3, 2, 1, 0]

    def test_non_snake_always_ascending(self) -> None:
        """Non-snake: every round is 0, 1, 2, 3."""
        league = _small_league()
        board = _small_board()
        bots = [BestValueBot(rng=random.Random(i)) for i in range(4)]
        result = run_mock_draft(board, league, bots, snake=False, seed=42)

        for rnd in range(1, 5):
            round_picks = [p for p in result.picks if p.round == rnd]
            assert [p.team_idx for p in round_picks] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Positional enforcement tests
# ---------------------------------------------------------------------------


class TestPositionalEnforcement:
    def test_no_roster_exceeds_slots(self) -> None:
        """No team's roster exceeds slot counts defined by the league."""
        league = _small_league()
        board = _small_board()
        bots = _make_bots(4)
        result = run_mock_draft(board, league, bots, seed=42)

        slots = build_draft_roster_slots(league)
        for team_idx, roster in result.rosters.items():
            filled: dict[str, int] = {}
            for pick in roster:
                filled[pick.position] = filled.get(pick.position, 0) + 1
            for pos, count in filled.items():
                assert count <= slots[pos], f"Team {team_idx} has {count} players at {pos}, max is {slots[pos]}"

    def test_full_draft_fills_all_slots(self) -> None:
        """Each team should have exactly total_slots picks."""
        league = _small_league()
        board = _small_board()
        bots = _make_bots(4)
        result = run_mock_draft(board, league, bots, seed=42)

        slots = build_draft_roster_slots(league)
        total_slots = sum(slots.values())
        for team_idx, roster in result.rosters.items():
            assert len(roster) == total_slots, f"Team {team_idx} has {len(roster)} picks, expected {total_slots}"


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        league = _small_league()
        board = _small_board()

        bots1 = _make_bots(4, seed=99)
        result1 = run_mock_draft(board, league, bots1, seed=42)

        bots2 = _make_bots(4, seed=99)
        result2 = run_mock_draft(board, league, bots2, seed=42)

        assert len(result1.picks) == len(result2.picks)
        for p1, p2 in zip(result1.picks, result2.picks, strict=True):
            assert p1 == p2


# ---------------------------------------------------------------------------
# 12-team integration test
# ---------------------------------------------------------------------------


class TestFullDraft:
    def test_12_team_mixed_strategies(self) -> None:
        """AC: run_mock_draft() completes a full draft for a 12-team league with mixed bot strategies."""
        league = _12_team_league()
        board = _12_team_board()
        bots = _make_bots(12, seed=42)
        result = run_mock_draft(board, league, bots, seed=123)

        slots = build_draft_roster_slots(league)
        total_slots = sum(slots.values())

        # All 12 teams filled
        assert len(result.rosters) == 12
        for team_idx in range(12):
            assert len(result.rosters[team_idx]) == total_slots

        # Total picks
        assert len(result.picks) == 12 * total_slots

        # Snake ordering check for round 1 and 2
        round_1 = [p for p in result.picks if p.round == 1]
        assert [p.team_idx for p in round_1] == list(range(12))
        round_2 = [p for p in result.picks if p.round == 2]
        assert [p.team_idx for p in round_2] == list(range(11, -1, -1))

        # Positional limits
        for team_idx, roster in result.rosters.items():
            filled: dict[str, int] = {}
            for pick in roster:
                filled[pick.position] = filled.get(pick.position, 0) + 1
            for pos, count in filled.items():
                assert count <= slots[pos], f"Team {team_idx} has {count} at {pos}, max {slots[pos]}"

    def test_12_team_deterministic(self) -> None:
        """AC: Each bot strategy produces deterministic results given a fixed random seed."""
        league = _12_team_league()
        board = _12_team_board()

        bots1 = _make_bots(12, seed=42)
        result1 = run_mock_draft(board, league, bots1, seed=123)

        bots2 = _make_bots(12, seed=42)
        result2 = run_mock_draft(board, league, bots2, seed=123)

        for p1, p2 in zip(result1.picks, result2.picks, strict=True):
            assert p1 == p2


# ---------------------------------------------------------------------------
# DraftResult structure tests
# ---------------------------------------------------------------------------


class TestDraftResult:
    def test_snake_flag(self) -> None:
        league = _small_league()
        board = _small_board()
        bots = _make_bots(4)

        result_snake = run_mock_draft(board, league, bots, snake=True, seed=42)
        assert result_snake.snake is True

        bots2 = _make_bots(4)
        result_linear = run_mock_draft(board, league, bots2, snake=False, seed=42)
        assert result_linear.snake is False

    def test_picks_have_correct_round_numbers(self) -> None:
        league = _small_league()
        board = _small_board()
        bots = _make_bots(4)
        result = run_mock_draft(board, league, bots, seed=42)

        slots = build_draft_roster_slots(league)
        total_rounds = sum(slots.values())

        for pick in result.picks:
            assert 1 <= pick.round <= total_rounds
            assert 1 <= pick.pick <= len(result.picks)

    def test_no_duplicate_players(self) -> None:
        league = _small_league()
        board = _small_board()
        bots = _make_bots(4)
        result = run_mock_draft(board, league, bots, seed=42)

        player_ids = [p.player_id for p in result.picks]
        assert len(player_ids) == len(set(player_ids))
