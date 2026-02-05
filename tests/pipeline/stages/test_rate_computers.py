from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.pipeline.stages.rate_computers import MarcelRateComputer
from fantasy_baseball_manager.result import Ok

if TYPE_CHECKING:
    from fantasy_baseball_manager.context import Context


def _make_player(
    player_id: str = "p1",
    name: str = "Test Hitter",
    year: int = 2024,
    age: int = 28,
    pa: int = 600,
    ab: int = 540,
    h: int = 160,
    singles: int = 100,
    doubles: int = 30,
    triples: int = 5,
    hr: int = 25,
    bb: int = 50,
    so: int = 120,
    hbp: int = 5,
    sf: int = 3,
    sh: int = 2,
    sb: int = 10,
    cs: int = 3,
    r: int = 80,
    rbi: int = 90,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        pa=pa,
        ab=ab,
        h=h,
        singles=singles,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=hbp,
        sf=sf,
        sh=sh,
        sb=sb,
        cs=cs,
        r=r,
        rbi=rbi,
    )


def _make_league(year: int = 2024) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        pa=6000,
        ab=5400,
        h=1500,
        singles=900,
        doubles=300,
        triples=30,
        hr=200,
        bb=500,
        so=1400,
        hbp=50,
        sf=30,
        sh=20,
        sb=100,
        cs=30,
        r=800,
        rbi=750,
    )


def _make_pitcher(
    player_id: str = "sp1",
    name: str = "Test Pitcher",
    year: int = 2024,
    age: int = 28,
    ip: float = 180.0,
    g: int = 32,
    gs: int = 32,
    er: int = 70,
    h: int = 150,
    bb: int = 50,
    so: int = 200,
    hr: int = 20,
    hbp: int = 5,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        ip=ip,
        g=g,
        gs=gs,
        er=er,
        h=h,
        bb=bb,
        so=so,
        hr=hr,
        hbp=hbp,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


def _make_league_pitching(year: int = 2024) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        ip=1450.0,
        g=500,
        gs=162,
        er=650,
        h=1350,
        bb=500,
        so=1400,
        hr=180,
        hbp=60,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


class FakeDataSource:
    def __init__(
        self,
        player_batting: dict[int, list[BattingSeasonStats]] | None = None,
        team_batting: dict[int, list[BattingSeasonStats]] | None = None,
        player_pitching: dict[int, list[PitchingSeasonStats]] | None = None,
        team_pitching: dict[int, list[PitchingSeasonStats]] | None = None,
    ) -> None:
        self._player_batting = player_batting or {}
        self._team_batting = team_batting or {}
        self._player_pitching = player_pitching or {}
        self._team_pitching = team_pitching or {}

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._player_batting.get(year, [])

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._player_pitching.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return self._team_batting.get(year, [])

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return self._team_pitching.get(year, [])


class TestMarcelRateComputerBatting:
    def test_returns_player_rates(self) -> None:
        league = _make_league()
        ds = FakeDataSource(
            player_batting={2024: [_make_player()]},
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_batting_rates(ds, 2025, 3)
        assert len(rates) == 1
        assert rates[0].player_id == "p1"
        assert rates[0].year == 2025
        assert rates[0].age == 29

    def test_hr_rate_hand_calculated(self) -> None:
        """Verify HR rate matches hand calculation.

        Player: 25 HR in 600 PA (y1), 20 HR in 550 PA (y2), 18 HR in 500 PA (y3)
        League HR rate: 200/6000 = 0.03333...
        num = 5*25 + 4*20 + 3*18 + 1200*0.03333 = 125+80+54+40 = 299
        den = 5*600 + 4*550 + 3*500 + 1200 = 7900
        rate = 299/7900 = 0.037848...
        """
        league = _make_league()
        ds = FakeDataSource(
            player_batting={
                2024: [_make_player(year=2024, age=28, pa=600, hr=25)],
                2023: [_make_player(year=2023, age=27, pa=550, hr=20)],
                2022: [_make_player(year=2022, age=26, pa=500, hr=18)],
            },
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_batting_rates(ds, 2025, 3)
        assert rates[0].rates["hr"] == pytest.approx(299 / 7900, abs=1e-6)

    def test_metadata_contains_league_rates(self) -> None:
        league = _make_league()
        ds = FakeDataSource(
            player_batting={2024: [_make_player()]},
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_batting_rates(ds, 2025, 3)
        assert "avg_league_rates" in rates[0].metadata
        assert "target_rates" in rates[0].metadata
        assert "pa_per_year" in rates[0].metadata


class TestMarcelRateComputerPitching:
    def test_returns_player_rates(self) -> None:
        league = _make_league_pitching()
        ds = FakeDataSource(
            player_pitching={2024: [_make_pitcher()]},
            team_pitching={2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_pitching_rates(ds, 2025, 3)
        assert len(rates) == 1
        assert rates[0].player_id == "sp1"
        assert rates[0].year == 2025

    def test_starter_detected(self) -> None:
        league = _make_league_pitching()
        ds = FakeDataSource(
            player_pitching={2024: [_make_pitcher(gs=32, g=32)]},
            team_pitching={2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_pitching_rates(ds, 2025, 3)
        assert rates[0].metadata["is_starter"] is True

    def test_reliever_detected(self) -> None:
        league = _make_league_pitching()
        ds = FakeDataSource(
            player_pitching={2024: [_make_pitcher(gs=0, g=65, ip=70.0)]},
            team_pitching={2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_pitching_rates(ds, 2025, 3)
        assert rates[0].metadata["is_starter"] is False


class TestMarcelRateComputerNewStyleDataSources:
    """Tests for new DataSource[T] pattern integration."""

    def test_compute_batting_rates_with_new_style_source(self, test_context: "Context") -> None:
        """MarcelRateComputer works with new-style DataSource callables."""
        player_batting = {
            2024: [_make_player(year=2024, age=28)],
            2023: [_make_player(year=2023, age=27)],
            2022: [_make_player(year=2022, age=26)],
        }
        league = _make_league()
        team_batting = {2024: [league], 2023: [league], 2022: [league]}

        def batting_source(query: object) -> Ok[list[BattingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            return Ok(player_batting.get(year, []))

        def team_batting_source(query: object) -> Ok[list[BattingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            return Ok(team_batting.get(year, []))

        computer = MarcelRateComputer()
        rates = computer.compute_batting_rates_v2(
            batting_source=batting_source,  # type: ignore[arg-type]
            team_batting_source=team_batting_source,  # type: ignore[arg-type]
            year=2025,
            years_back=3,
        )

        assert len(rates) == 1
        assert rates[0].player_id == "p1"
        assert rates[0].year == 2025
        assert rates[0].age == 29

    def test_new_style_source_uses_context_for_years(self, test_context: "Context") -> None:
        """Verifies that multi-year queries use context switching."""
        years_queried: list[int] = []
        league = _make_league()

        def batting_source(query: object) -> Ok[list[BattingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            years_queried.append(year)
            return Ok([_make_player(year=year, age=25 + (2024 - year))])

        def team_batting_source(query: object) -> Ok[list[BattingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            _ = get_context().year  # Access context to ensure it's set
            return Ok([league])

        computer = MarcelRateComputer()
        computer.compute_batting_rates_v2(
            batting_source=batting_source,  # type: ignore[arg-type]
            team_batting_source=team_batting_source,  # type: ignore[arg-type]
            year=2025,
            years_back=3,
        )

        # Should query years 2024, 2023, 2022
        assert set(years_queried) == {2024, 2023, 2022}

    def test_compute_pitching_rates_with_new_style_source(self, test_context: "Context") -> None:
        """MarcelRateComputer works with new-style DataSource callables for pitching."""
        player_pitching = {
            2024: [_make_pitcher(year=2024, age=28)],
            2023: [_make_pitcher(year=2023, age=27)],
            2022: [_make_pitcher(year=2022, age=26)],
        }
        league = _make_league_pitching()
        team_pitching = {2024: [league], 2023: [league], 2022: [league]}

        def pitching_source(query: object) -> Ok[list[PitchingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            return Ok(player_pitching.get(year, []))

        def team_pitching_source(query: object) -> Ok[list[PitchingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            return Ok(team_pitching.get(year, []))

        computer = MarcelRateComputer()
        rates = computer.compute_pitching_rates_v2(
            pitching_source=pitching_source,  # type: ignore[arg-type]
            team_pitching_source=team_pitching_source,  # type: ignore[arg-type]
            year=2025,
            years_back=3,
        )

        assert len(rates) == 1
        assert rates[0].player_id == "sp1"
        assert rates[0].year == 2025
        assert rates[0].age == 29

    def test_pitching_v2_detects_starter(self, test_context: "Context") -> None:
        """Verifies starter detection in new-style pitching method."""
        player_pitching = {
            2024: [_make_pitcher(year=2024, gs=32, g=32)],
            2023: [_make_pitcher(year=2023, gs=30, g=30)],
            2022: [_make_pitcher(year=2022, gs=28, g=28)],
        }
        league = _make_league_pitching()
        team_pitching = {2024: [league], 2023: [league], 2022: [league]}

        def pitching_source(query: object) -> Ok[list[PitchingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            return Ok(player_pitching.get(year, []))

        def team_pitching_source(query: object) -> Ok[list[PitchingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            return Ok(team_pitching.get(year, []))

        computer = MarcelRateComputer()
        rates = computer.compute_pitching_rates_v2(
            pitching_source=pitching_source,  # type: ignore[arg-type]
            team_pitching_source=team_pitching_source,  # type: ignore[arg-type]
            year=2025,
            years_back=3,
        )

        assert rates[0].metadata["is_starter"] is True

    def test_pitching_v2_detects_reliever(self, test_context: "Context") -> None:
        """Verifies reliever detection in new-style pitching method."""
        player_pitching = {
            2024: [_make_pitcher(year=2024, gs=0, g=65, ip=70.0)],
            2023: [_make_pitcher(year=2023, gs=0, g=60, ip=65.0)],
            2022: [_make_pitcher(year=2022, gs=0, g=55, ip=60.0)],
        }
        league = _make_league_pitching()
        team_pitching = {2024: [league], 2023: [league], 2022: [league]}

        def pitching_source(query: object) -> Ok[list[PitchingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            return Ok(player_pitching.get(year, []))

        def team_pitching_source(query: object) -> Ok[list[PitchingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            return Ok(team_pitching.get(year, []))

        computer = MarcelRateComputer()
        rates = computer.compute_pitching_rates_v2(
            pitching_source=pitching_source,  # type: ignore[arg-type]
            team_pitching_source=team_pitching_source,  # type: ignore[arg-type]
            year=2025,
            years_back=3,
        )

        assert rates[0].metadata["is_starter"] is False

    def test_pitching_v2_uses_context_for_years(self, test_context: "Context") -> None:
        """Verifies that multi-year queries use context switching."""
        years_queried: list[int] = []
        league = _make_league_pitching()

        def pitching_source(query: object) -> Ok[list[PitchingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            years_queried.append(year)
            return Ok([_make_pitcher(year=year, age=25 + (2024 - year))])

        def team_pitching_source(query: object) -> Ok[list[PitchingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            _ = get_context().year  # Access context to ensure it's set
            return Ok([league])

        computer = MarcelRateComputer()
        computer.compute_pitching_rates_v2(
            pitching_source=pitching_source,  # type: ignore[arg-type]
            team_pitching_source=team_pitching_source,  # type: ignore[arg-type]
            year=2025,
            years_back=3,
        )

        # Should query years 2024, 2023, 2022
        assert set(years_queried) == {2024, 2023, 2022}
