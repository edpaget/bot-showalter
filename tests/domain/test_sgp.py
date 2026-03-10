import pytest

from fantasy_baseball_manager.domain.sgp import SgpDenominators, SgpSeasonDenominator


class TestSgpSeasonDenominator:
    def test_construction(self) -> None:
        d = SgpSeasonDenominator(category="HR", season=2024, denominator=8.5, num_teams=12)
        assert d.category == "HR"
        assert d.season == 2024
        assert d.denominator == 8.5
        assert d.num_teams == 12

    def test_frozen(self) -> None:
        d = SgpSeasonDenominator(category="HR", season=2024, denominator=8.5, num_teams=12)
        with pytest.raises(AttributeError):
            d.denominator = 10.0  # type: ignore[misc]


class TestSgpDenominators:
    def test_construction(self) -> None:
        season_denoms = (
            SgpSeasonDenominator(category="HR", season=2024, denominator=8.0, num_teams=12),
            SgpSeasonDenominator(category="HR", season=2023, denominator=9.0, num_teams=12),
        )
        result = SgpDenominators(per_season=season_denoms, averages={"HR": 8.5})
        assert len(result.per_season) == 2
        assert result.averages["HR"] == 8.5

    def test_frozen(self) -> None:
        result = SgpDenominators(per_season=(), averages={})
        with pytest.raises(AttributeError):
            result.averages = {}  # type: ignore[misc]
