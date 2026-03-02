import pytest

from fantasy_baseball_manager.domain.category_tracker import (
    CategoryNeed,
    PlayerRecommendation,
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


class TestPlayerRecommendation:
    def test_frozen(self) -> None:
        rec = PlayerRecommendation(
            player_id=1,
            player_name="Mike Trout",
            category_impact=5.0,
            tradeoff_categories=(),
        )
        with pytest.raises(AttributeError):
            rec.player_id = 2  # type: ignore[misc]

    def test_fields(self) -> None:
        rec = PlayerRecommendation(
            player_id=42,
            player_name="Juan Soto",
            category_impact=3.5,
            tradeoff_categories=("avg", "obp"),
        )
        assert rec.player_id == 42
        assert rec.player_name == "Juan Soto"
        assert rec.category_impact == pytest.approx(3.5)
        assert rec.tradeoff_categories == ("avg", "obp")


class TestCategoryNeed:
    def test_frozen(self) -> None:
        need = CategoryNeed(
            category="sb",
            current_rank=10,
            target_rank=8,
            best_available=(),
        )
        with pytest.raises(AttributeError):
            need.category = "hr"  # type: ignore[misc]

    def test_fields(self) -> None:
        rec = PlayerRecommendation(
            player_id=1,
            player_name="Test Player",
            category_impact=2.0,
            tradeoff_categories=(),
        )
        need = CategoryNeed(
            category="hr",
            current_rank=11,
            target_rank=8,
            best_available=(rec,),
        )
        assert need.category == "hr"
        assert need.current_rank == 11
        assert need.target_rank == 8
        assert len(need.best_available) == 1
        assert need.best_available[0].player_name == "Test Player"
