from collections.abc import Generator

import pytest
from typer.testing import CliRunner

from fantasy_baseball_manager.cli import app
from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.services import ServiceContainer, set_container

runner = CliRunner()


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


def _make_batter(
    player_id: str = "b1",
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


def _make_pitcher(
    player_id: str = "p1",
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


def _build_fake_data_source(
    batting_years: list[int] | None = None,
    pitching_years: list[int] | None = None,
) -> FakeDataSource:
    player_batting: dict[int, list[BattingSeasonStats]] = {}
    team_batting: dict[int, list[BattingSeasonStats]] = {}
    player_pitching: dict[int, list[PitchingSeasonStats]] = {}
    team_pitching: dict[int, list[PitchingSeasonStats]] = {}

    for y in batting_years or []:
        player_batting[y] = [_make_batter(year=y, age=28 - (2024 - y))]
        team_batting[y] = [_make_league_batting(year=y)]

    for y in pitching_years or []:
        player_pitching[y] = [_make_pitcher(year=y, age=28 - (2024 - y))]
        team_pitching[y] = [_make_league_pitching(year=y)]

    return FakeDataSource(
        player_batting=player_batting,
        player_pitching=player_pitching,
        team_batting_stats=team_batting,
        team_pitching_stats=team_pitching,
    )


YEARS = [2024, 2023, 2022]


@pytest.fixture(autouse=True)
def reset_container() -> Generator[None]:
    yield
    set_container(None)


def _install_fake(batting: bool = True, pitching: bool = True) -> None:
    ds = _build_fake_data_source(
        batting_years=YEARS if batting else None,
        pitching_years=YEARS if pitching else None,
    )
    set_container(ServiceContainer(data_source=ds))


class TestMarcelCommand:
    def test_default_shows_both(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025"])
        assert result.exit_code == 0
        assert "projections for 2025" in result.output
        assert "projected batters" in result.output
        assert "projected pitchers" in result.output

    def test_batting_only(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--batting"])
        assert result.exit_code == 0
        assert "projected batters" in result.output
        assert "projected pitchers" not in result.output

    def test_pitching_only(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--pitching"])
        assert result.exit_code == 0
        assert "projected pitchers" in result.output
        assert "projected batters" not in result.output

    def test_both_flags(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--batting", "--pitching"])
        assert result.exit_code == 0
        assert "projected batters" in result.output
        assert "projected pitchers" in result.output

    def test_top_flag(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--batting", "--top", "5"])
        assert result.exit_code == 0
        assert "Top 5 projected batters" in result.output

    def test_sort_by_batting(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--batting", "--sort-by", "sb"])
        assert result.exit_code == 0
        assert "projected batters" in result.output

    def test_sort_by_pitching(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--pitching", "--sort-by", "era"])
        assert result.exit_code == 0
        assert "projected pitchers" in result.output

    def test_invalid_batting_sort(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--batting", "--sort-by", "xyz"])
        assert result.exit_code == 1

    def test_invalid_pitching_sort(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--pitching", "--sort-by", "xyz"])
        assert result.exit_code == 1

    def test_output_contains_player_name(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--batting"])
        assert result.exit_code == 0
        assert "Test Hitter" in result.output

    def test_pitching_output_contains_player_name(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--pitching"])
        assert result.exit_code == 0
        assert "Test Pitcher" in result.output

    def test_batting_output_contains_r_and_rbi(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--batting"])
        assert result.exit_code == 0
        assert "  R " in result.output
        assert "RBI" in result.output

    def test_sort_by_r(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--batting", "--sort-by", "r"])
        assert result.exit_code == 0
        assert "projected batters" in result.output

    def test_sort_by_rbi(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--batting", "--sort-by", "rbi"])
        assert result.exit_code == 0
        assert "projected batters" in result.output

    def test_engine_marcel_accepted(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--engine", "marcel"])
        assert result.exit_code == 0

    def test_engine_unknown_rejected(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "project", "2025", "--engine", "not_a_real_engine"])
        assert result.exit_code == 1
        assert "Unknown engine" in result.output
