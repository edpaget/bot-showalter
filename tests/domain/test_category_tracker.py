import pytest

from fantasy_baseball_manager.domain.category_tracker import (
    RosterAnalysis,
    TeamCategoryProjection,
)


class TestTeamCategoryProjection:
    def test_frozen(self) -> None:
        proj = TeamCategoryProjection(
            category="hr",
            projected_value=120.0,
            league_rank_estimate=3,
            strength="strong",
        )
        with pytest.raises(AttributeError):
            proj.category = "rbi"  # type: ignore[misc]

    def test_fields(self) -> None:
        proj = TeamCategoryProjection(
            category="avg",
            projected_value=0.275,
            league_rank_estimate=6,
            strength="average",
        )
        assert proj.category == "avg"
        assert proj.projected_value == pytest.approx(0.275)
        assert proj.league_rank_estimate == 6
        assert proj.strength == "average"


class TestRosterAnalysis:
    def test_frozen(self) -> None:
        analysis = RosterAnalysis(
            projections=[],
            strongest_categories=[],
            weakest_categories=[],
        )
        with pytest.raises(AttributeError):
            analysis.projections = []  # type: ignore[misc]

    def test_fields(self) -> None:
        proj = TeamCategoryProjection(
            category="hr",
            projected_value=100.0,
            league_rank_estimate=2,
            strength="strong",
        )
        analysis = RosterAnalysis(
            projections=[proj],
            strongest_categories=["hr"],
            weakest_categories=["sb"],
        )
        assert len(analysis.projections) == 1
        assert analysis.strongest_categories == ["hr"]
        assert analysis.weakest_categories == ["sb"]
