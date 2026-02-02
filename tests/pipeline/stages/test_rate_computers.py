import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.pipeline.stages.rate_computers import MarcelRateComputer


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
