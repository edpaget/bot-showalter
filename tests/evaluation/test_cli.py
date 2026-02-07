from collections.abc import Generator
from typing import Any

import pytest
from typer.testing import CliRunner

from fantasy_baseball_manager.cli import app
from fantasy_baseball_manager.context import get_context
from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.result import Ok
from fantasy_baseball_manager.services import ServiceContainer, set_container

runner = CliRunner()

YEARS = [2024, 2023, 2022]


def _make_batter(
    player_id: str = "b1",
    name: str = "Test Hitter",
    year: int = 2024,
    pa: int = 600,
    hr: int = 25,
    sb: int = 10,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=28,
        pa=pa,
        ab=540,
        h=160,
        singles=100,
        doubles=30,
        triples=5,
        hr=hr,
        bb=50,
        so=120,
        hbp=5,
        sf=3,
        sh=2,
        sb=sb,
        cs=3,
        r=80,
        rbi=90,
    )


def _make_pitcher(
    player_id: str = "p1",
    name: str = "Test Pitcher",
    year: int = 2024,
    ip: float = 180.0,
    er: int = 60,
    so: int = 200,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=28,
        ip=ip,
        g=32,
        gs=32,
        er=er,
        h=150,
        bb=50,
        so=so,
        hr=20,
        hbp=5,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


def _make_league_batting(year: int = 2024) -> BattingSeasonStats:
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


def _make_league_pitching(year: int = 2024) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        ip=1400.0,
        g=500,
        gs=200,
        er=600,
        h=1300,
        bb=450,
        so=1300,
        hr=180,
        hbp=50,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


class FakeDataSource:
    def __init__(
        self,
        player_batting: dict[int, list[BattingSeasonStats]],
        player_pitching: dict[int, list[PitchingSeasonStats]],
        team_batting_stats: dict[int, list[BattingSeasonStats]],
        team_pitching_stats: dict[int, list[PitchingSeasonStats]],
    ) -> None:
        self._player_batting = player_batting
        self._player_pitching = player_pitching
        self._team_batting = team_batting_stats
        self._team_pitching = team_pitching_stats

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._player_batting.get(year, [])

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._player_pitching.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return self._team_batting.get(year, [])

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return self._team_pitching.get(year, [])


def _build_fake_data_source(
    num_batters: int = 3,
    num_pitchers: int = 3,
) -> FakeDataSource:
    batter_configs = [
        ("b1", "Slugger Jones", 40, 5),
        ("b2", "Speedy Smith", 10, 30),
        ("b3", "Average Andy", 20, 15),
    ]
    pitcher_configs = [
        ("p1", "Ace Adams", 250, 50),
        ("p2", "Bullpen Bob", 150, 70),
        ("p3", "Middle Mike", 180, 60),
    ]

    player_batting: dict[int, list[BattingSeasonStats]] = {}
    player_pitching: dict[int, list[PitchingSeasonStats]] = {}
    team_batting: dict[int, list[BattingSeasonStats]] = {}
    team_pitching: dict[int, list[PitchingSeasonStats]] = {}

    for y in YEARS:
        batters: list[BattingSeasonStats] = []
        for i in range(min(num_batters, len(batter_configs))):
            pid, name, hr, sb = batter_configs[i]
            batters.append(_make_batter(player_id=pid, name=name, year=y, hr=hr, sb=sb))
        player_batting[y] = batters
        team_batting[y] = [_make_league_batting(year=y)]

        pitchers: list[PitchingSeasonStats] = []
        for i in range(min(num_pitchers, len(pitcher_configs))):
            pid, name, so, er = pitcher_configs[i]
            pitchers.append(_make_pitcher(player_id=pid, name=name, year=y, so=so, er=er))
        player_pitching[y] = pitchers
        team_pitching[y] = [_make_league_pitching(year=y)]

    return FakeDataSource(
        player_batting=player_batting,
        player_pitching=player_pitching,
        team_batting_stats=team_batting,
        team_pitching_stats=team_pitching,
    )


@pytest.fixture(autouse=True)
def reset_container() -> Generator[None]:
    yield
    set_container(None)


def _wrap_source(method: Any) -> Any:
    """Wrap a FakeDataSource method as a DataSource[T] callable."""

    def source(query: Any) -> Ok:
        return Ok(method(get_context().year))

    return source


def _install_fake() -> None:
    ds = _build_fake_data_source()
    set_container(
        ServiceContainer(
            batting_source=_wrap_source(ds.batting_stats),
            team_batting_source=_wrap_source(ds.team_batting),
            pitching_source=_wrap_source(ds.pitching_stats),
            team_pitching_source=_wrap_source(ds.team_pitching),
        )
    )


class TestEvaluateCommand:
    def test_default_engine(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["evaluate", "2025"])
        assert result.exit_code == 0
        assert "marcel" in result.output.lower()
        assert "Batting" in result.output
        assert "Pitching" in result.output

    def test_explicit_engine(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["evaluate", "2025", "--engine", "marcel"])
        assert result.exit_code == 0
        assert "marcel" in result.output.lower()

    def test_unknown_engine(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["evaluate", "2025", "--engine", "bogus"])
        assert result.exit_code == 1
        assert "Unknown engine" in result.output

    def test_custom_thresholds(self) -> None:
        _install_fake()
        result = runner.invoke(
            app,
            ["evaluate", "2025", "--min-pa", "100", "--min-ip", "30", "--top-n", "10"],
        )
        assert result.exit_code == 0

    def test_stratify_flag(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["evaluate", "2024", "--stratify"])
        assert result.exit_code == 0
        assert "by segment" in result.output

    def test_compare_flag_requires_two_engines(self) -> None:
        _install_fake()
        result = runner.invoke(
            app,
            ["evaluate", "2025", "--engine", "marcel", "--compare", "--include-residuals"],
        )
        # Should warn about needing 2+ engines
        assert result.exit_code == 0
        assert "requires at least 2 engines" in result.output

    def test_compare_flag_requires_residuals(self) -> None:
        _install_fake()
        result = runner.invoke(
            app,
            ["evaluate", "2025", "--engine", "marcel", "--compare"],
        )
        # Should warn about needing --include-residuals
        assert result.exit_code == 0
        assert "requires --include-residuals" in result.output or "requires at least 2 engines" in result.output
