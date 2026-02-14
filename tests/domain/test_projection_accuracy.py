import pytest

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.projection_accuracy import (
    ProjectionComparison,
    compare_to_batting_actuals,
    compare_to_pitching_actuals,
)


def _batting_projection(*, player_id: int = 1, **stat_overrides: object) -> Projection:
    stats = {"hr": 30, "avg": 0.280, "pa": 600, "sb": 15}
    stats.update(stat_overrides)
    return Projection(
        player_id=player_id,
        season=2025,
        system="steamer",
        version="2025.1",
        player_type="batter",
        stat_json=stats,
    )


def _batting_actuals(*, player_id: int = 1, **overrides: object) -> BattingStats:
    defaults = {
        "player_id": player_id,
        "season": 2025,
        "source": "fangraphs",
        "hr": 28,
        "avg": 0.265,
        "pa": 580,
        "sb": 12,
    }
    defaults.update(overrides)
    return BattingStats(**defaults)  # type: ignore[arg-type]


class TestProjectionComparison:
    def test_frozen(self) -> None:
        comp = ProjectionComparison(stat_name="hr", projected=30.0, actual=28.0, error=2.0)
        with pytest.raises(AttributeError):
            comp.stat_name = "avg"  # type: ignore[misc]

    def test_fields(self) -> None:
        comp = ProjectionComparison(stat_name="hr", projected=30.0, actual=28.0, error=2.0)
        assert comp.stat_name == "hr"
        assert comp.projected == 30.0
        assert comp.actual == 28.0
        assert comp.error == 2.0


class TestCompareToBattingActuals:
    def test_matching_stats_compared(self) -> None:
        proj = _batting_projection()
        actual = _batting_actuals()
        comparisons = compare_to_batting_actuals(proj, actual)

        by_stat = {c.stat_name: c for c in comparisons}
        assert "hr" in by_stat
        assert by_stat["hr"].projected == 30.0
        assert by_stat["hr"].actual == 28.0
        assert by_stat["hr"].error == 2.0

    def test_multiple_stats(self) -> None:
        proj = _batting_projection()
        actual = _batting_actuals()
        comparisons = compare_to_batting_actuals(proj, actual)

        stat_names = {c.stat_name for c in comparisons}
        assert "hr" in stat_names
        assert "avg" in stat_names
        assert "pa" in stat_names
        assert "sb" in stat_names

    def test_stat_only_in_projection_skipped(self) -> None:
        proj = _batting_projection(war=5.0)
        actual = _batting_actuals()  # war defaults to None
        comparisons = compare_to_batting_actuals(proj, actual)

        stat_names = {c.stat_name for c in comparisons}
        assert "war" not in stat_names

    def test_stat_only_in_actuals_skipped(self) -> None:
        proj = _batting_projection()
        actual = _batting_actuals(bb=60)
        comparisons = compare_to_batting_actuals(proj, actual)

        stat_names = {c.stat_name for c in comparisons}
        # bb is not in the projection stat_json
        assert "bb" not in stat_names

    def test_empty_projection_returns_empty(self) -> None:
        proj = Projection(
            player_id=1,
            season=2025,
            system="steamer",
            version="2025.1",
            player_type="batter",
            stat_json={},
        )
        actual = _batting_actuals()
        comparisons = compare_to_batting_actuals(proj, actual)
        assert comparisons == []

    def test_negative_error(self) -> None:
        proj = _batting_projection(hr=25)
        actual = _batting_actuals(hr=30)
        comparisons = compare_to_batting_actuals(proj, actual)

        by_stat = {c.stat_name: c for c in comparisons}
        assert by_stat["hr"].error == -5.0


class TestCompareToPitchingActuals:
    def test_matching_stats_compared(self) -> None:
        proj = Projection(
            player_id=1,
            season=2025,
            system="steamer",
            version="2025.1",
            player_type="pitcher",
            stat_json={"era": 3.20, "so": 200, "w": 12},
        )
        actual = PitchingStats(
            player_id=1,
            season=2025,
            source="fangraphs",
            era=3.50,
            so=190,
            w=10,
        )
        comparisons = compare_to_pitching_actuals(proj, actual)

        by_stat = {c.stat_name: c for c in comparisons}
        assert "era" in by_stat
        assert by_stat["era"].projected == 3.20
        assert by_stat["era"].actual == 3.50
        assert by_stat["so"].error == 10.0
        assert by_stat["w"].error == 2.0
