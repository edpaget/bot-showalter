from typer.testing import CliRunner

from fantasy_baseball_manager.cli import app
from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.valuation.cli import set_data_source_factory

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
        r=80,
        rbi=90,
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
    )


YEARS = [2024, 2023, 2022]


def _build_fake_data_source(
    batting_years: list[int] | None = None,
    pitching_years: list[int] | None = None,
    num_batters: int = 3,
    num_pitchers: int = 3,
) -> FakeDataSource:
    player_batting: dict[int, list[BattingSeasonStats]] = {}
    team_batting: dict[int, list[BattingSeasonStats]] = {}
    player_pitching: dict[int, list[PitchingSeasonStats]] = {}
    team_pitching: dict[int, list[PitchingSeasonStats]] = {}

    batter_configs = [
        ("b1", "Slugger Jones", 40, 5, 600),
        ("b2", "Speedy Smith", 10, 30, 600),
        ("b3", "Average Andy", 20, 15, 600),
    ]
    pitcher_configs = [
        ("p1", "Ace Adams", 250, 50, 130),
        ("p2", "Bullpen Bob", 150, 70, 180),
        ("p3", "Middle Mike", 180, 60, 155),
    ]

    for y in batting_years or []:
        batters: list[BattingSeasonStats] = []
        for i in range(min(num_batters, len(batter_configs))):
            pid, name, hr, sb, pa = batter_configs[i]
            batters.append(
                _make_batter(
                    player_id=pid,
                    name=name,
                    year=y,
                    age=28 - (2024 - y),
                    hr=hr,
                    sb=sb,
                    pa=pa,
                )
            )
        player_batting[y] = batters
        team_batting[y] = [_make_league_batting(year=y)]

    for y in pitching_years or []:
        pitchers: list[PitchingSeasonStats] = []
        for i in range(min(num_pitchers, len(pitcher_configs))):
            pid, name, so, er, h = pitcher_configs[i]
            pitchers.append(
                _make_pitcher(
                    player_id=pid,
                    name=name,
                    year=y,
                    age=28 - (2024 - y),
                    so=so,
                    er=er,
                    h=h,
                )
            )
        player_pitching[y] = pitchers
        team_pitching[y] = [_make_league_pitching(year=y)]

    return FakeDataSource(
        player_batting=player_batting,
        player_pitching=player_pitching,
        team_batting_stats=team_batting,
        team_pitching_stats=team_pitching,
    )


def _install_fake(batting: bool = True, pitching: bool = True) -> None:
    ds = _build_fake_data_source(
        batting_years=YEARS if batting else None,
        pitching_years=YEARS if pitching else None,
    )
    set_data_source_factory(lambda: ds)


class TestValuateCommand:
    def test_default_shows_both(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025"])
        assert result.exit_code == 0
        assert "batting" in result.output.lower()
        assert "pitching" in result.output.lower()

    def test_batting_only(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--batting"])
        assert result.exit_code == 0
        assert "batting" in result.output.lower()
        assert "pitching" not in result.output.lower()

    def test_pitching_only(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--pitching"])
        assert result.exit_code == 0
        assert "pitching" in result.output.lower()
        assert "batting" not in result.output.lower()

    def test_categories_override(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--batting", "--categories", "hr,sb"])
        assert result.exit_code == 0
        assert "HR" in result.output
        assert "SB" in result.output
        assert "OBP" not in result.output

    def test_invalid_category(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--batting", "--categories", "xyz"])
        assert result.exit_code == 1

    def test_rbi_category(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--batting", "--categories", "rbi"])
        assert result.exit_code == 0
        assert "RBI" in result.output

    def test_output_sorted_by_total_descending(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--batting"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        # Find lines with player data (contain Total values)
        value_lines = [line for line in lines if any(name in line for name in ["Slugger", "Speedy", "Average"])]
        assert len(value_lines) >= 2
        # Extract total values (last number on each line)
        totals = []
        for line in value_lines:
            parts = line.split()
            totals.append(float(parts[-1]))
        assert totals == sorted(totals, reverse=True)

    def test_output_contains_player_name(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--batting"])
        assert result.exit_code == 0
        assert "Slugger Jones" in result.output

    def test_top_limits_rows(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--batting", "--top", "2"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        player_lines = [line for line in lines if any(name in line for name in ["Slugger", "Speedy", "Average"])]
        assert len(player_lines) == 2

    def test_engine_marcel_accepted(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--engine", "marcel"])
        assert result.exit_code == 0

    def test_engine_unknown_rejected(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--engine", "steamer"])
        assert result.exit_code == 1
        assert "Unknown engine" in result.output

    def test_method_zscore_accepted(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--method", "zscore"])
        assert result.exit_code == 0

    def test_method_unknown_rejected(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "valuate", "2025", "--method", "sgp"])
        assert result.exit_code == 1
        assert "Unknown method" in result.output
