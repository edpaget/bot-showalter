import pytest

from fantasy_baseball_manager.domain.draft_report import (
    CategoryStanding,
    DraftReport,
    PickGrade,
    StealOrReach,
)


class TestPickGrade:
    def test_frozen(self) -> None:
        grade = PickGrade(
            pick_number=1,
            player_id=100,
            player_name="Mike Trout",
            position="OF",
            value=45.0,
            best_available_value=45.0,
            grade=1.0,
        )
        with pytest.raises(AttributeError):
            grade.grade = 0.5  # type: ignore[misc]

    def test_fields(self) -> None:
        grade = PickGrade(
            pick_number=3,
            player_id=200,
            player_name="Mookie Betts",
            position="SS",
            value=30.0,
            best_available_value=40.0,
            grade=0.75,
        )
        assert grade.pick_number == 3
        assert grade.player_id == 200
        assert grade.player_name == "Mookie Betts"
        assert grade.position == "SS"
        assert grade.value == 30.0
        assert grade.best_available_value == 40.0
        assert grade.grade == 0.75


class TestStealOrReach:
    def test_frozen(self) -> None:
        steal = StealOrReach(
            pick_number=5,
            player_id=100,
            player_name="Mike Trout",
            position="OF",
            value=45.0,
            pick_delta=10,
        )
        with pytest.raises(AttributeError):
            steal.pick_delta = 0  # type: ignore[misc]

    def test_steal_positive_delta(self) -> None:
        steal = StealOrReach(
            pick_number=15,
            player_id=100,
            player_name="Mike Trout",
            position="OF",
            value=45.0,
            pick_delta=10,
        )
        assert steal.pick_delta == 10

    def test_reach_negative_delta(self) -> None:
        reach = StealOrReach(
            pick_number=5,
            player_id=200,
            player_name="Mookie Betts",
            position="SS",
            value=30.0,
            pick_delta=-8,
        )
        assert reach.pick_delta == -8


class TestCategoryStanding:
    def test_frozen(self) -> None:
        standing = CategoryStanding(
            category="HR",
            total_z=2.5,
            rank=3,
            teams=12,
        )
        with pytest.raises(AttributeError):
            standing.rank = 1  # type: ignore[misc]

    def test_fields(self) -> None:
        standing = CategoryStanding(
            category="ERA",
            total_z=-1.2,
            rank=8,
            teams=10,
        )
        assert standing.category == "ERA"
        assert standing.total_z == pytest.approx(-1.2)
        assert standing.rank == 8
        assert standing.teams == 10


class TestDraftReport:
    def test_frozen(self) -> None:
        report = DraftReport(
            total_value=100.0,
            optimal_value=120.0,
            value_efficiency=0.833,
            budget=None,
            total_spent=None,
            category_standings=[],
            pick_grades=[],
            mean_grade=0.0,
            steals=[],
            reaches=[],
        )
        with pytest.raises(AttributeError):
            report.total_value = 50.0  # type: ignore[misc]

    def test_snake_report_no_budget(self) -> None:
        report = DraftReport(
            total_value=100.0,
            optimal_value=120.0,
            value_efficiency=0.833,
            budget=None,
            total_spent=None,
            category_standings=[],
            pick_grades=[],
            mean_grade=0.0,
            steals=[],
            reaches=[],
        )
        assert report.budget is None
        assert report.total_spent is None

    def test_auction_report_with_budget(self) -> None:
        report = DraftReport(
            total_value=100.0,
            optimal_value=120.0,
            value_efficiency=0.833,
            budget=260,
            total_spent=245,
            category_standings=[],
            pick_grades=[],
            mean_grade=0.85,
            steals=[],
            reaches=[],
        )
        assert report.budget == 260
        assert report.total_spent == 245

    def test_with_nested_types(self) -> None:
        grade = PickGrade(
            pick_number=1,
            player_id=100,
            player_name="Mike Trout",
            position="OF",
            value=45.0,
            best_available_value=45.0,
            grade=1.0,
        )
        steal = StealOrReach(
            pick_number=1,
            player_id=100,
            player_name="Mike Trout",
            position="OF",
            value=45.0,
            pick_delta=10,
        )
        standing = CategoryStanding(
            category="HR",
            total_z=2.5,
            rank=3,
            teams=12,
        )
        report = DraftReport(
            total_value=45.0,
            optimal_value=50.0,
            value_efficiency=0.9,
            budget=None,
            total_spent=None,
            category_standings=[standing],
            pick_grades=[grade],
            mean_grade=1.0,
            steals=[steal],
            reaches=[],
        )
        assert len(report.pick_grades) == 1
        assert len(report.steals) == 1
        assert len(report.reaches) == 0
        assert len(report.category_standings) == 1
