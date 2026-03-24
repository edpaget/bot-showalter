import pytest

from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.draft_report import DraftReport
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.services.draft_report import (
    draft_report,
)
from fantasy_baseball_manager.services.draft_state import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
    DraftState,
)


def _player(
    player_id: int,
    name: str,
    position: str,
    value: float,
    *,
    player_type: PlayerType = PlayerType.BATTER,
    z_scores: dict[str, float] | None = None,
    adp_rank: int | None = None,
    adp_overall: float | None = None,
    rank: int | None = None,
) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=rank if rank is not None else player_id,
        player_type=player_type,
        position=position,
        value=value,
        category_z_scores=z_scores or {},
        adp_rank=adp_rank,
        adp_overall=adp_overall,
    )


PLAYERS = [
    _player(1, "Trout", "OF", 50.0, z_scores={"HR": 3.0, "AVG": 1.0}),
    _player(2, "Betts", "SS", 45.0, z_scores={"HR": 2.0, "AVG": 2.0}),
    _player(3, "Judge", "OF", 40.0, z_scores={"HR": 4.0, "AVG": -1.0}),
    _player(4, "Acuna", "OF", 35.0, z_scores={"HR": 1.5, "AVG": 1.5}),
    _player(5, "Soto", "OF", 30.0, z_scores={"HR": 2.5, "AVG": 0.5}),
    _player(6, "Cole", "SP", 42.0, z_scores={"ERA": 3.0, "K": 2.0}, player_type=PlayerType.PITCHER),
    _player(7, "Burnes", "SP", 38.0, z_scores={"ERA": 2.5, "K": 1.5}, player_type=PlayerType.PITCHER),
    _player(8, "Alcantara", "SP", 20.0, z_scores={"ERA": 1.0, "K": 0.5}, player_type=PlayerType.PITCHER),
]


def _snake_config(teams: int = 2, user_team: int = 1) -> DraftConfig:
    return DraftConfig(
        teams=teams,
        roster_slots={"OF": 2, "SS": 1, "UTIL": 1, "SP": 1, "P": 1},
        format=DraftFormat.SNAKE,
        user_team=user_team,
        season=2026,
    )


def _auction_config(teams: int = 2, user_team: int = 1, budget: int = 100) -> DraftConfig:
    return DraftConfig(
        teams=teams,
        roster_slots={"OF": 2, "SS": 1, "SP": 1},
        format=DraftFormat.AUCTION,
        user_team=user_team,
        season=2026,
        budget=budget,
    )


def _run_snake_draft(
    players: list[DraftBoardRow],
    picks: list[tuple[int, int, str]],
    config: DraftConfig | None = None,
) -> DraftState:
    """Run a snake draft. picks = [(player_id, team, position), ...]."""
    cfg = config or _snake_config()
    engine = DraftEngine()
    engine.start(players, cfg)
    for pid, team, pos in picks:
        engine.pick(pid, team, pos)
    return engine.state


def _run_auction_draft(
    players: list[DraftBoardRow],
    picks: list[tuple[int, int, str, int]],
    config: DraftConfig | None = None,
) -> DraftState:
    """Run an auction draft. picks = [(player_id, team, position, price), ...]."""
    cfg = config or _auction_config()
    engine = DraftEngine()
    engine.start(players, cfg)
    for pid, team, pos, price in picks:
        engine.pick(pid, team, pos, price=price)
    return engine.state


# --- Optimal value ---


class TestComputeOptimalValue:
    def test_simple_optimal(self) -> None:
        """Best possible roster from a sorted pool."""
        # 2 OF + 1 SS + 1 UTIL + 1 SP + 1 P = 6 slots
        # Best: Trout(OF,50) + Judge(OF,40) + Betts(SS,45) + Cole(SP,42) +
        #        Acuna(UTIL,35) + Burnes(P,38)  = 250
        state = _run_snake_draft(PLAYERS, [])
        report = draft_report(state, PLAYERS)
        assert report.optimal_value == pytest.approx(250.0)

    def test_position_constraints_respected(self) -> None:
        """Can't fill all slots with OF players."""
        pool = [
            _player(1, "A", "OF", 100.0),
            _player(2, "B", "OF", 90.0),
            _player(3, "C", "OF", 80.0),
            _player(4, "D", "OF", 70.0),
            _player(5, "E", "SS", 10.0),
            _player(6, "F", "SP", 5.0),
        ]
        config = DraftConfig(
            teams=2,
            roster_slots={"OF": 2, "SS": 1, "UTIL": 1, "SP": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        state = _run_snake_draft(pool, [], config)
        # OF:100+90, UTIL:80 (OF flex), SS:10, SP:5 = 285
        report = draft_report(state, pool)
        assert report.optimal_value == pytest.approx(285.0)

    def test_pitcher_flex(self) -> None:
        """Pitcher can fill P flex slot."""
        pool = [
            _player(1, "A", "SP", 50.0, player_type=PlayerType.PITCHER),
            _player(2, "B", "SP", 40.0, player_type=PlayerType.PITCHER),
            _player(3, "C", "RP", 30.0, player_type=PlayerType.PITCHER),
        ]
        config = DraftConfig(
            teams=2,
            roster_slots={"SP": 1, "P": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        state = _run_snake_draft(pool, [], config)
        # SP:50, P:40 (SP flex into P) = 90
        report = draft_report(state, pool)
        assert report.optimal_value == pytest.approx(90.0)


# --- Category standings ---


class TestComputeCategoryStandings:
    def test_single_category_ranking(self) -> None:
        """User team ranked among all teams for a single category."""
        # Team 1 picks Trout (HR: 3.0), Team 2 picks Judge (HR: 4.0)
        state = _run_snake_draft(
            PLAYERS,
            [(1, 1, "OF"), (3, 2, "OF")],
        )
        report = draft_report(state, PLAYERS, batting_categories=("HR",))
        hr = next(s for s in report.category_standings if s.category == "HR")
        assert hr.total_z == pytest.approx(3.0)
        assert hr.rank == 2  # Judge has 4.0 HR z

    def test_multi_category(self) -> None:
        """Multiple categories each get a standing."""
        state = _run_snake_draft(
            PLAYERS,
            [(1, 1, "OF"), (3, 2, "OF")],
        )
        report = draft_report(state, PLAYERS, batting_categories=("HR", "AVG"))
        cats = {s.category for s in report.category_standings}
        assert cats == {"HR", "AVG"}

    def test_missing_z_scores_treated_as_zero(self) -> None:
        """Players without z-score for a category contribute 0."""
        pool = [
            _player(1, "A", "OF", 50.0, z_scores={"HR": 3.0}),
            _player(2, "B", "OF", 45.0, z_scores={}),  # No HR z-score
        ]
        state = _run_snake_draft(
            pool,
            [(1, 1, "OF"), (2, 2, "OF")],
            DraftConfig(
                teams=2,
                roster_slots={"OF": 1},
                format=DraftFormat.SNAKE,
                user_team=1,
                season=2026,
            ),
        )
        report = draft_report(state, pool, batting_categories=("HR",))
        hr = next(s for s in report.category_standings if s.category == "HR")
        assert hr.total_z == pytest.approx(3.0)
        assert hr.rank == 1

    def test_pitching_categories(self) -> None:
        """Pitching categories rank by pitcher picks."""
        state = _run_snake_draft(
            PLAYERS,
            [(6, 1, "SP"), (7, 2, "SP")],
            DraftConfig(
                teams=2,
                roster_slots={"SP": 1},
                format=DraftFormat.SNAKE,
                user_team=1,
                season=2026,
            ),
        )
        report = draft_report(state, PLAYERS, pitching_categories=("ERA",))
        era = next(s for s in report.category_standings if s.category == "ERA")
        assert era.total_z == pytest.approx(3.0)  # Cole ERA z
        assert era.rank == 1


# --- Pick grades ---


class TestComputePickGrades:
    def test_perfect_pick(self) -> None:
        """Picking the best available gives grade 1.0."""
        # Team 1 picks Trout first (best OF available) -> perfect
        state = _run_snake_draft(
            PLAYERS,
            [(1, 1, "OF"), (2, 2, "SS")],
        )
        report = draft_report(state, PLAYERS)
        user_grades = report.pick_grades
        assert len(user_grades) == 1
        assert user_grades[0].player_name == "Trout"
        assert user_grades[0].grade == pytest.approx(1.0)

    def test_suboptimal_pick(self) -> None:
        """Picking a non-best player gives grade < 1.0."""
        # Team 1 picks Judge(OF, 40) instead of Trout(OF, 50)
        state = _run_snake_draft(
            PLAYERS,
            [(3, 1, "OF"), (2, 2, "SS")],
        )
        report = draft_report(state, PLAYERS)
        user_grades = report.pick_grades
        assert len(user_grades) == 1
        assert user_grades[0].player_name == "Judge"
        # Best available at OF was Trout (50), got Judge (40)
        assert user_grades[0].grade == pytest.approx(40.0 / 50.0)

    def test_only_user_picks_graded(self) -> None:
        """Only user team picks appear in grades."""
        state = _run_snake_draft(
            PLAYERS,
            [
                (1, 1, "OF"),  # user
                (2, 2, "SS"),  # opponent
                (3, 2, "OF"),  # opponent
                (4, 1, "OF"),  # user
            ],
        )
        report = draft_report(state, PLAYERS)
        user_grades = report.pick_grades
        assert len(user_grades) == 2
        names = [g.player_name for g in user_grades]
        assert names == ["Trout", "Acuna"]

    def test_considers_draft_order(self) -> None:
        """Best available accounts for players already drafted."""
        # Round 1: Team 1 picks Trout(OF,50), Team 2 picks Betts(SS,45)
        # Round 2: Team 2 picks Judge(OF,40), Team 1 picks Acuna(OF,35)
        # For user's 2nd pick, best available OF is Acuna (Judge taken by T2)
        state = _run_snake_draft(
            PLAYERS,
            [
                (1, 1, "OF"),  # user: Trout
                (2, 2, "SS"),  # opponent: Betts
                (3, 2, "OF"),  # opponent: Judge
                (4, 1, "OF"),  # user: Acuna
            ],
        )
        report = draft_report(state, PLAYERS)
        second_grade = report.pick_grades[1]
        assert second_grade.player_name == "Acuna"
        # Best available OF after Trout/Betts/Judge gone is Acuna (35)
        assert second_grade.best_available_value == pytest.approx(35.0)
        assert second_grade.grade == pytest.approx(1.0)

    def test_flex_slot_best_available(self) -> None:
        """When user picks into UTIL, best available considers all batters."""
        state = _run_snake_draft(
            PLAYERS,
            [
                (4, 1, "UTIL"),  # user picks Acuna(OF,35) into UTIL
                (2, 2, "SS"),  # opponent
            ],
        )
        report = draft_report(state, PLAYERS)
        grade = report.pick_grades[0]
        # Best available for UTIL is any batter: Trout(50)
        assert grade.best_available_value == pytest.approx(50.0)
        assert grade.grade == pytest.approx(35.0 / 50.0)


# --- Steals and reaches ---


class TestComputeStealsReaches:
    def test_steal_identified(self) -> None:
        """Player picked much later than ADP = steal."""
        # rank-1 player with adp_rank 20 picked at pick 2
        pool = [
            _player(1, "Steal", "OF", 50.0, adp_rank=20, rank=1),
            _player(2, "Other", "OF", 40.0, adp_rank=2, rank=2),
        ]
        config = DraftConfig(
            teams=2,
            roster_slots={"OF": 1},
            format=DraftFormat.SNAKE,
            user_team=2,
            season=2026,
        )
        state = _run_snake_draft(
            pool,
            [
                (2, 1, "OF"),
                (1, 2, "OF"),  # user picks rank-1 player at pick 2, adp_rank 20
            ],
            config,
        )
        report = draft_report(state, pool, steal_threshold=5)
        # pick_delta = adp_rank - pick_number = 20 - 2 = 18 -> steal
        assert len(report.steals) == 1
        assert report.steals[0].player_name == "Steal"
        assert report.steals[0].pick_delta == 18

    def test_reach_identified(self) -> None:
        """Player picked later than ADP expected = reach (adp_rank < pick_number)."""
        pool = [
            _player(1, "Reach", "OF", 30.0, adp_rank=1, rank=3),
            _player(2, "Other", "OF", 40.0, adp_rank=2, rank=2),
            _player(3, "Third", "SS", 25.0, adp_rank=3, rank=4),
        ]
        config = DraftConfig(
            teams=2,
            roster_slots={"OF": 1, "SS": 1},
            format=DraftFormat.SNAKE,
            user_team=2,
            season=2026,
        )
        state = _run_snake_draft(
            pool,
            [
                (2, 1, "OF"),  # opponent
                (1, 2, "OF"),  # user picks Reach at pick 2, adp_rank 1
                (3, 2, "SS"),  # opponent in round 2
            ],
            config,
        )
        report = draft_report(state, pool, steal_threshold=0)
        # pick_delta = adp_rank - pick_number = 1 - 2 = -1 -> reach
        assert len(report.reaches) >= 1
        reach = report.reaches[0]
        assert reach.player_name == "Reach"
        assert reach.pick_delta == -1

    def test_within_threshold_not_flagged(self) -> None:
        """Player within threshold of ADP is neither steal nor reach."""
        pool = [
            _player(1, "Normal", "OF", 50.0, adp_rank=2, rank=1),
            _player(2, "Other", "OF", 40.0, adp_rank=1, rank=2),
        ]
        config = DraftConfig(
            teams=2,
            roster_slots={"OF": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        state = _run_snake_draft(
            pool,
            [(1, 1, "OF"), (2, 2, "OF")],
            config,
        )
        report = draft_report(state, pool, steal_threshold=5)
        # pick_delta = adp_rank - pick_number = 2 - 1 = 1, within threshold 5
        assert len(report.steals) == 0
        assert len(report.reaches) == 0

    def test_no_adp_uses_value_rank(self) -> None:
        """When no ADP, use value rank for comparison."""
        pool = [
            _player(1, "Good", "OF", 50.0, rank=1),  # No ADP
            _player(2, "Other", "OF", 40.0, rank=2),
        ]
        config = DraftConfig(
            teams=2,
            roster_slots={"OF": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        state = _run_snake_draft(
            pool,
            [(1, 1, "OF"), (2, 2, "OF")],
            config,
        )
        report = draft_report(state, pool, steal_threshold=5)
        # No ADP, rank 1 picked at pick 1 -> delta 0, within threshold
        assert len(report.steals) == 0
        assert len(report.reaches) == 0


# --- Integration: draft_report() ---


class TestDraftReportIntegration:
    def test_completed_snake_draft(self) -> None:
        """Full report for a completed snake draft."""
        # 2-team, 6 slots each (OF:2, SS:1, UTIL:1, SP:1, P:1)
        # 12 picks total for a full draft
        state = _run_snake_draft(
            PLAYERS,
            [
                (1, 1, "OF"),  # Trout -> T1
                (6, 2, "SP"),  # Cole -> T2
                (7, 2, "P"),  # Burnes -> T2
                (2, 1, "SS"),  # Betts -> T1
                (3, 1, "OF"),  # Judge -> T1
                (4, 2, "OF"),  # Acuna -> T2
                (5, 2, "OF"),  # Soto -> T2
                (8, 1, "SP"),  # Alcantara -> T1
            ],
        )
        report = draft_report(
            state,
            PLAYERS,
            batting_categories=("HR",),
            pitching_categories=("ERA",),
        )
        assert isinstance(report, DraftReport)
        assert report.total_value == pytest.approx(50.0 + 45.0 + 40.0 + 20.0)
        assert report.budget is None
        assert report.total_spent is None
        assert len(report.pick_grades) == 4  # 4 user picks
        assert report.mean_grade > 0.0
        assert len(report.category_standings) == 2  # HR + ERA

    def test_auction_draft_budget(self) -> None:
        """Auction draft includes budget info."""
        config = _auction_config(budget=100)
        state = _run_auction_draft(
            PLAYERS,
            [
                (1, 1, "OF", 30),  # user
                (6, 2, "SP", 25),  # opponent
                (2, 1, "SS", 20),  # user
                (7, 2, "OF", 15),  # opponent
                (3, 1, "OF", 25),  # user
                (4, 2, "OF", 20),  # opponent
            ],
            config,
        )
        report = draft_report(state, PLAYERS)
        assert report.budget == 100
        assert report.total_spent == 75  # 30 + 20 + 25
        assert report.total_value == pytest.approx(50.0 + 45.0 + 40.0)

    def test_value_efficiency(self) -> None:
        """Value efficiency = total_value / optimal_value."""
        state = _run_snake_draft(
            PLAYERS,
            [
                (1, 1, "OF"),  # Trout(50)
                (6, 2, "SP"),  # Cole
            ],
        )
        report = draft_report(state, PLAYERS)
        assert report.value_efficiency == pytest.approx(report.total_value / report.optimal_value)

    def test_mean_grade(self) -> None:
        """Mean grade is average of all pick grades."""
        state = _run_snake_draft(
            PLAYERS,
            [
                (1, 1, "OF"),  # Trout(50) - best OF
                (2, 2, "SS"),  # Betts
                (6, 2, "SP"),  # Cole
                (3, 1, "OF"),  # Judge(40) - best remaining OF
            ],
        )
        report = draft_report(state, PLAYERS)
        grades = [g.grade for g in report.pick_grades]
        assert report.mean_grade == pytest.approx(sum(grades) / len(grades))

    def test_empty_draft(self) -> None:
        """Report on a draft with no picks."""
        state = _run_snake_draft(PLAYERS, [])
        report = draft_report(state, PLAYERS)
        assert report.total_value == 0.0
        assert report.pick_grades == []
        assert report.mean_grade == 0.0
        assert report.steals == []
        assert report.reaches == []
