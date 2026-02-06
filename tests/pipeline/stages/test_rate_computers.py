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


def _make_batting_source(
    player_batting: dict[int, list[BattingSeasonStats]],
) -> object:
    """Create a fake DataSource[BattingSeasonStats] that reads year from context."""

    def source(_query: object) -> Ok[list[BattingSeasonStats]]:
        from fantasy_baseball_manager.context import get_context

        year = get_context().year
        return Ok(player_batting.get(year, []))

    return source


def _make_pitching_source(
    player_pitching: dict[int, list[PitchingSeasonStats]],
) -> object:
    """Create a fake DataSource[PitchingSeasonStats] that reads year from context."""

    def source(_query: object) -> Ok[list[PitchingSeasonStats]]:
        from fantasy_baseball_manager.context import get_context

        year = get_context().year
        return Ok(player_pitching.get(year, []))

    return source


class TestMarcelRateComputerBatting:
    def test_returns_player_rates(self, test_context: "Context") -> None:
        league = _make_league()
        batting_source = _make_batting_source({2024: [_make_player()]})
        team_batting_source = _make_batting_source(
            {2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_batting_rates(
            batting_source,  # type: ignore[arg-type]
            team_batting_source,  # type: ignore[arg-type]
            2025,
            3,
        )
        assert len(rates) == 1
        assert rates[0].player_id == "p1"
        assert rates[0].year == 2025
        assert rates[0].age == 29

    def test_hr_rate_hand_calculated(self, test_context: "Context") -> None:
        """Verify HR rate matches hand calculation.

        Player: 25 HR in 600 PA (y1), 20 HR in 550 PA (y2), 18 HR in 500 PA (y3)
        League HR rate: 200/6000 = 0.03333...
        num = 5*25 + 4*20 + 3*18 + 1200*0.03333 = 125+80+54+40 = 299
        den = 5*600 + 4*550 + 3*500 + 1200 = 7900
        rate = 299/7900 = 0.037848...
        """
        league = _make_league()
        batting_source = _make_batting_source(
            {
                2024: [_make_player(year=2024, age=28, pa=600, hr=25)],
                2023: [_make_player(year=2023, age=27, pa=550, hr=20)],
                2022: [_make_player(year=2022, age=26, pa=500, hr=18)],
            }
        )
        team_batting_source = _make_batting_source(
            {2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_batting_rates(
            batting_source,  # type: ignore[arg-type]
            team_batting_source,  # type: ignore[arg-type]
            2025,
            3,
        )
        assert rates[0].rates["hr"] == pytest.approx(299 / 7900, abs=1e-6)

    def test_metadata_contains_league_rates(self, test_context: "Context") -> None:
        league = _make_league()
        batting_source = _make_batting_source({2024: [_make_player()]})
        team_batting_source = _make_batting_source(
            {2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_batting_rates(
            batting_source,  # type: ignore[arg-type]
            team_batting_source,  # type: ignore[arg-type]
            2025,
            3,
        )
        assert "avg_league_rates" in rates[0].metadata
        assert "target_rates" in rates[0].metadata
        assert "pa_per_year" in rates[0].metadata

    def test_uses_context_for_years(self, test_context: "Context") -> None:
        """Verifies that multi-year queries use context switching."""
        years_queried: list[int] = []
        league = _make_league()

        def batting_source(_query: object) -> Ok[list[BattingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            years_queried.append(year)
            return Ok([_make_player(year=year, age=25 + (2024 - year))])

        def team_batting_source(_query: object) -> Ok[list[BattingSeasonStats]]:
            return Ok([league])

        computer = MarcelRateComputer()
        computer.compute_batting_rates(
            batting_source,  # type: ignore[arg-type]
            team_batting_source,  # type: ignore[arg-type]
            2025,
            3,
        )

        # Should query years 2024, 2023, 2022
        assert set(years_queried) == {2024, 2023, 2022}


class TestMarcelRateComputerPitching:
    def test_returns_player_rates(self, test_context: "Context") -> None:
        league = _make_league_pitching()
        pitching_source = _make_pitching_source({2024: [_make_pitcher()]})
        team_pitching_source = _make_pitching_source(
            {2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_pitching_rates(
            pitching_source,  # type: ignore[arg-type]
            team_pitching_source,  # type: ignore[arg-type]
            2025,
            3,
        )
        assert len(rates) == 1
        assert rates[0].player_id == "sp1"
        assert rates[0].year == 2025

    def test_starter_detected(self, test_context: "Context") -> None:
        league = _make_league_pitching()
        pitching_source = _make_pitching_source({2024: [_make_pitcher(gs=32, g=32)]})
        team_pitching_source = _make_pitching_source(
            {2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_pitching_rates(
            pitching_source,  # type: ignore[arg-type]
            team_pitching_source,  # type: ignore[arg-type]
            2025,
            3,
        )
        assert rates[0].metadata["is_starter"] is True

    def test_reliever_detected(self, test_context: "Context") -> None:
        league = _make_league_pitching()
        pitching_source = _make_pitching_source(
            {2024: [_make_pitcher(gs=0, g=65, ip=70.0)]},
        )
        team_pitching_source = _make_pitching_source(
            {2024: [league], 2023: [league], 2022: [league]},
        )
        computer = MarcelRateComputer()
        rates = computer.compute_pitching_rates(
            pitching_source,  # type: ignore[arg-type]
            team_pitching_source,  # type: ignore[arg-type]
            2025,
            3,
        )
        assert rates[0].metadata["is_starter"] is False

    def test_uses_context_for_years(self, test_context: "Context") -> None:
        """Verifies that multi-year queries use context switching."""
        years_queried: list[int] = []
        league = _make_league_pitching()

        def pitching_source(_query: object) -> Ok[list[PitchingSeasonStats]]:
            from fantasy_baseball_manager.context import get_context

            year = get_context().year
            years_queried.append(year)
            return Ok([_make_pitcher(year=year, age=25 + (2024 - year))])

        def team_pitching_source(_query: object) -> Ok[list[PitchingSeasonStats]]:
            return Ok([league])

        computer = MarcelRateComputer()
        computer.compute_pitching_rates(
            pitching_source,  # type: ignore[arg-type]
            team_pitching_source,  # type: ignore[arg-type]
            2025,
            3,
        )

        # Should query years 2024, 2023, 2022
        assert set(years_queried) == {2024, 2023, 2022}
