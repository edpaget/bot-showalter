from typer.testing import CliRunner

from fantasy_baseball_manager.cli import app
from fantasy_baseball_manager.league.cli import (
    set_data_source_factory,
    set_id_mapper_factory,
    set_roster_source_factory,
)
from fantasy_baseball_manager.league.models import LeagueRosters, RosterPlayer, TeamRoster
from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats

runner = CliRunner()


class FakeIdMapper:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._yahoo_to_fg = mapping

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return self._yahoo_to_fg.get(yahoo_id)

    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None:
        return None


class FakeRosterSource:
    def __init__(self, rosters: LeagueRosters) -> None:
        self._rosters = rosters

    def fetch_rosters(self) -> LeagueRosters:
        return self._rosters


class FakeDataSource:
    def __init__(
        self,
        batting: dict[int, list[BattingSeasonStats]],
        pitching: dict[int, list[PitchingSeasonStats]],
        team_batting: dict[int, list[BattingSeasonStats]],
        team_pitching: dict[int, list[PitchingSeasonStats]],
    ) -> None:
        self._batting = batting
        self._pitching = pitching
        self._team_batting = team_batting
        self._team_pitching = team_pitching

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._batting.get(year, [])

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._pitching.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return self._team_batting.get(year, [])

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return self._team_pitching.get(year, [])


def _make_batter(
    player_id: str = "fg1",
    name: str = "Test Hitter",
    year: int = 2024,
    age: int = 28,
    pa: int = 600,
    ab: int = 540,
    h: int = 160,
    hr: int = 25,
    bb: int = 50,
    so: int = 120,
    hbp: int = 5,
    sb: int = 10,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        pa=pa,
        ab=ab,
        h=h,
        singles=h - hr - 30 - 5,
        doubles=30,
        triples=5,
        hr=hr,
        bb=bb,
        so=so,
        hbp=hbp,
        sf=3,
        sh=2,
        sb=sb,
        cs=3,
        r=80,
        rbi=90,
    )


def _make_pitcher(
    player_id: str = "fgp1",
    name: str = "Test Pitcher",
    year: int = 2024,
    age: int = 28,
    ip: float = 180.0,
    er: int = 70,
    h: int = 150,
    bb: int = 50,
    so: int = 200,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        ip=ip,
        g=32,
        gs=32,
        er=er,
        h=h,
        bb=bb,
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


YEARS = [2024, 2023, 2022]


def _install_fakes(
    rosters: LeagueRosters | None = None,
    id_mapping: dict[str, str] | None = None,
) -> None:
    if rosters is None:
        rosters = LeagueRosters(
            league_key="lg1",
            teams=(
                TeamRoster(
                    team_key="t1",
                    team_name="Alpha Squad",
                    players=(
                        RosterPlayer(yahoo_id="y1", name="Test Hitter", position_type="B", eligible_positions=("1B",)),
                        RosterPlayer(
                            yahoo_id="yp1", name="Test Pitcher", position_type="P", eligible_positions=("SP",)
                        ),
                    ),
                ),
            ),
        )
    if id_mapping is None:
        id_mapping = {"y1": "fg1", "yp1": "fgp1"}

    player_batting: dict[int, list[BattingSeasonStats]] = {}
    team_batting: dict[int, list[BattingSeasonStats]] = {}
    player_pitching: dict[int, list[PitchingSeasonStats]] = {}
    team_pitching: dict[int, list[PitchingSeasonStats]] = {}

    for y in YEARS:
        player_batting[y] = [_make_batter(year=y, age=28 - (2024 - y))]
        team_batting[y] = [_make_league_batting(year=y)]
        player_pitching[y] = [_make_pitcher(year=y, age=28 - (2024 - y))]
        team_pitching[y] = [_make_league_pitching(year=y)]

    ds = FakeDataSource(
        batting=player_batting,
        pitching=player_pitching,
        team_batting=team_batting,
        team_pitching=team_pitching,
    )

    set_roster_source_factory(lambda: FakeRosterSource(rosters))
    set_id_mapper_factory(lambda: FakeIdMapper(id_mapping))
    set_data_source_factory(lambda: ds)


class TestLeagueProjectionsCommand:
    def test_shows_team_name(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "roster", "2025"])
        assert result.exit_code == 0
        assert "Alpha Squad" in result.output

    def test_shows_player_names(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "roster", "2025"])
        assert result.exit_code == 0
        assert "Test Hitter" in result.output
        assert "Test Pitcher" in result.output

    def test_shows_header(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "roster", "2025"])
        assert result.exit_code == 0
        assert "League projections for 2025" in result.output

    def test_invalid_sort_field(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "roster", "2025", "--sort-by", "xyz"])
        assert result.exit_code == 1

    def test_unmatched_player_warning(self) -> None:
        rosters = LeagueRosters(
            league_key="lg1",
            teams=(
                TeamRoster(
                    team_key="t1",
                    team_name="Team X",
                    players=(
                        RosterPlayer(
                            yahoo_id="y_unknown", name="Mystery Player", position_type="B", eligible_positions=("Util",)
                        ),
                    ),
                ),
            ),
        )
        _install_fakes(rosters=rosters, id_mapping={})
        result = runner.invoke(app, ["teams", "roster", "2025"])
        assert result.exit_code == 0
        assert "could not be matched" in result.output

    def test_engine_marcel_accepted(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "roster", "2025", "--engine", "marcel"])
        assert result.exit_code == 0

    def test_engine_unknown_rejected(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "roster", "2025", "--engine", "steamer"])
        assert result.exit_code == 1
        assert "Unknown engine" in result.output


class TestLeagueCompareCommand:
    def test_shows_team_name(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "compare", "2025"])
        assert result.exit_code == 0
        assert "Alpha Squad" in result.output

    def test_shows_header(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "compare", "2025"])
        assert result.exit_code == 0
        assert "League comparison for 2025" in result.output

    def test_invalid_sort_field(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "compare", "2025", "--sort-by", "xyz"])
        assert result.exit_code == 1

    def test_compare_table_has_stats(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "compare", "2025"])
        assert result.exit_code == 0
        # Table header should show stat columns
        assert "HR" in result.output
        assert "ERA" in result.output
        assert "WHIP" in result.output

    def test_multiple_teams(self) -> None:
        rosters = LeagueRosters(
            league_key="lg1",
            teams=(
                TeamRoster(
                    team_key="t1",
                    team_name="Alpha Squad",
                    players=(
                        RosterPlayer(yahoo_id="y1", name="Test Hitter", position_type="B", eligible_positions=("1B",)),
                    ),
                ),
                TeamRoster(
                    team_key="t2",
                    team_name="Beta Team",
                    players=(
                        RosterPlayer(
                            yahoo_id="yp1", name="Test Pitcher", position_type="P", eligible_positions=("SP",)
                        ),
                    ),
                ),
            ),
        )
        _install_fakes(rosters=rosters)
        result = runner.invoke(app, ["teams", "compare", "2025"])
        assert result.exit_code == 0
        assert "Alpha Squad" in result.output
        assert "Beta Team" in result.output

    def test_engine_marcel_accepted(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "compare", "2025", "--engine", "marcel"])
        assert result.exit_code == 0

    def test_engine_unknown_rejected(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "compare", "2025", "--engine", "steamer"])
        assert result.exit_code == 1
        assert "Unknown engine" in result.output

    def test_method_zscore_accepted(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "compare", "2025", "--method", "zscore"])
        assert result.exit_code == 0

    def test_method_unknown_rejected(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "compare", "2025", "--method", "sgp"])
        assert result.exit_code == 1
        assert "Unknown method" in result.output
