from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from fantasy_baseball_manager.cli import app
from fantasy_baseball_manager.config import clear_cli_overrides, create_config
from fantasy_baseball_manager.league.cli import (
    _get_roster_source,
    format_compare_table,
    format_team_projections,
)
from fantasy_baseball_manager.league.models import LeagueRosters, RosterPlayer, TeamProjection, TeamRoster
from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
from fantasy_baseball_manager.services import ServiceContainer, set_container
from fantasy_baseball_manager.valuation.models import LeagueSettings, StatCategory

runner = CliRunner()


class FakeIdMapper:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._yahoo_to_fg = mapping

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return self._yahoo_to_fg.get(yahoo_id)

    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None:
        return None

    def fangraphs_to_mlbam(self, fangraphs_id: str) -> str | None:
        return None

    def mlbam_to_fangraphs(self, mlbam_id: str) -> str | None:
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


@pytest.fixture(autouse=True)
def reset_container() -> Generator[None, None, None]:
    yield
    set_container(None)


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

    set_container(
        ServiceContainer(
            data_source=ds,
            id_mapper=FakeIdMapper(id_mapping),
            roster_source=FakeRosterSource(rosters),
        )
    )


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


class TestLeagueIdAndSeasonOptions:
    def teardown_method(self) -> None:
        clear_cli_overrides()

    def test_projections_accepts_league_id(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "roster", "2025", "--league-id", "99999"])
        assert result.exit_code == 0

    def test_projections_accepts_season(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "roster", "2025", "--season", "2024"])
        assert result.exit_code == 0

    def test_compare_accepts_league_id(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "compare", "2025", "--league-id", "99999"])
        assert result.exit_code == 0

    def test_compare_accepts_season(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "compare", "2025", "--season", "2024"])
        assert result.exit_code == 0

    def test_overrides_cleared_after_projections(self) -> None:
        _install_fakes()
        runner.invoke(app, ["teams", "roster", "2025", "--league-id", "99999"])
        cfg = create_config(yaml_path="/nonexistent/config.yaml")
        assert cfg["league.id"] == ""

    def test_overrides_cleared_after_compare(self) -> None:
        _install_fakes()
        runner.invoke(app, ["teams", "compare", "2025", "--league-id", "99999"])
        cfg = create_config(yaml_path="/nonexistent/config.yaml")
        assert cfg["league.id"] == ""


class TestKeeperPredraftAutoDetection:
    def setup_method(self) -> None:
        set_container(None)

    def teardown_method(self) -> None:
        clear_cli_overrides()
        set_container(None)

    def test_keeper_predraft_uses_previous_season(self) -> None:
        """When is_keeper is true and draft_status is predraft,
        _get_roster_source should call get_league_for_season(current - 1)."""
        mock_league = MagicMock()
        mock_league.settings.return_value = {"draft_status": "predraft"}

        mock_client = MagicMock()
        mock_client.get_league.return_value = mock_league
        mock_client.get_league_for_season.return_value = mock_league

        with (
            patch("fantasy_baseball_manager.league.cli.create_config") as mock_config,
            patch("fantasy_baseball_manager.league.cli.YahooFantasyClient", return_value=mock_client),
            patch("fantasy_baseball_manager.league.cli.YahooRosterSource"),
        ):
            cfg = {"league.is_keeper": True, "league.season": 2025, "cache.rosters_ttl": 3600}
            mock_config.return_value = cfg
            _get_roster_source(no_cache=True)
            mock_client.get_league_for_season.assert_called_once_with(2024)

    def test_keeper_postdraft_uses_current_season(self) -> None:
        """When is_keeper is true but draft_status is not predraft,
        _get_roster_source should call get_league() (no season walk)."""
        mock_league = MagicMock()
        mock_league.settings.return_value = {"draft_status": "postdraft"}

        mock_client = MagicMock()
        mock_client.get_league.return_value = mock_league

        with (
            patch("fantasy_baseball_manager.league.cli.create_config") as mock_config,
            patch("fantasy_baseball_manager.league.cli.YahooFantasyClient", return_value=mock_client),
            patch("fantasy_baseball_manager.league.cli.YahooRosterSource"),
        ):
            cfg = {"league.is_keeper": True, "league.season": 2025, "cache.rosters_ttl": 3600}
            mock_config.return_value = cfg
            _get_roster_source(no_cache=True)
            mock_client.get_league_for_season.assert_not_called()

    def test_non_keeper_uses_current_season(self) -> None:
        """When is_keeper is false, _get_roster_source should use get_league()
        regardless of draft status."""
        mock_client = MagicMock()

        with (
            patch("fantasy_baseball_manager.league.cli.create_config") as mock_config,
            patch("fantasy_baseball_manager.league.cli.YahooFantasyClient", return_value=mock_client),
            patch("fantasy_baseball_manager.league.cli.YahooRosterSource"),
        ):
            cfg = {"league.is_keeper": False, "league.season": 2025, "cache.rosters_ttl": 3600}
            mock_config.return_value = cfg
            _get_roster_source(no_cache=True)
            mock_client.get_league.assert_called_once()
            mock_client.get_league_for_season.assert_not_called()

    def test_explicit_target_season_skips_keeper_detection(self) -> None:
        """When target_season is explicitly provided, keeper detection is skipped."""
        mock_client = MagicMock()

        with (
            patch("fantasy_baseball_manager.league.cli.create_config") as mock_config,
            patch("fantasy_baseball_manager.league.cli.YahooFantasyClient", return_value=mock_client),
            patch("fantasy_baseball_manager.league.cli.YahooRosterSource"),
        ):
            cfg = {"league.is_keeper": True, "league.season": 2025, "cache.rosters_ttl": 3600}
            mock_config.return_value = cfg
            _get_roster_source(no_cache=True, target_season=2023)
            mock_client.get_league_for_season.assert_called_once_with(2023)
            mock_client.get_league.assert_not_called()


def _make_team_projection(**overrides: float | int | str) -> TeamProjection:
    from fantasy_baseball_manager.league.models import PlayerMatchResult

    defaults: dict[str, object] = {
        "team_name": "Test Team",
        "team_key": "t1",
        "players": (
            PlayerMatchResult(
                roster_player=RosterPlayer(yahoo_id="y1", name="Hitter", position_type="B", eligible_positions=("1B",)),
                batting_projection=None,
                pitching_projection=None,
                matched=False,
            ),
        ),
        "total_hr": 100.0,
        "total_sb": 50.0,
        "total_h": 500.0,
        "total_pa": 2000.0,
        "team_avg": 0.260,
        "team_obp": 0.330,
        "total_r": 300.0,
        "total_rbi": 280.0,
        "total_ip": 600.0,
        "total_so": 500.0,
        "total_w": 40.0,
        "total_nsvh": 30.0,
        "team_era": 3.80,
        "team_whip": 1.200,
        "unmatched_count": 0,
    }
    defaults.update(overrides)
    return TeamProjection(**defaults)  # type: ignore[arg-type]


class TestLeagueSettingsColumns:
    def test_roster_shows_configured_batting_categories(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "roster", "2025"])
        assert result.exit_code == 0
        # Default config has HR, SB, OBP — verify they appear in the header
        assert "HR" in result.output
        assert "SB" in result.output
        assert "OBP" in result.output

    def test_roster_shows_configured_pitching_categories(self) -> None:
        _install_fakes()
        result = runner.invoke(app, ["teams", "roster", "2025"])
        assert result.exit_code == 0
        # Default config has K, ERA, WHIP
        assert "ERA" in result.output
        assert "WHIP" in result.output

    def test_compare_table_shows_only_configured_categories(self) -> None:
        settings = LeagueSettings(
            team_count=12,
            batting_categories=(StatCategory.HR, StatCategory.OBP),
            pitching_categories=(StatCategory.ERA,),
        )
        tp = [_make_team_projection()]
        output = format_compare_table(tp, settings)
        assert "HR" in output
        assert "OBP" in output
        assert "ERA" in output
        # Categories NOT configured should not appear
        assert " SB" not in output
        assert " R " not in output
        assert "WHIP" not in output
        assert "NSVH" not in output

    def test_compare_table_shows_all_league_categories(self) -> None:
        settings = LeagueSettings(
            team_count=12,
            batting_categories=(StatCategory.HR, StatCategory.R, StatCategory.RBI, StatCategory.SB, StatCategory.OBP),
            pitching_categories=(
                StatCategory.W,
                StatCategory.K,
                StatCategory.ERA,
                StatCategory.WHIP,
                StatCategory.NSVH,
            ),
        )
        tp = [_make_team_projection()]
        output = format_compare_table(tp, settings)
        for cat in ("HR", "R", "RBI", "SB", "OBP", "W", "K", "ERA", "WHIP", "NSVH"):
            assert cat in output

    def test_roster_format_shows_only_configured_batting(self) -> None:
        from fantasy_baseball_manager.league.models import PlayerMatchResult
        from fantasy_baseball_manager.marcel.models import BattingProjection

        bp = BattingProjection(
            player_id="fg1",
            name="Hitter",
            year=2025,
            age=28,
            pa=600,
            ab=540,
            h=160,
            singles=100,
            doubles=30,
            triples=5,
            hr=25,
            bb=50,
            so=120,
            hbp=5,
            sf=3,
            sh=2,
            sb=10,
            cs=3,
            r=80,
            rbi=90,
        )
        player = PlayerMatchResult(
            roster_player=RosterPlayer(yahoo_id="y1", name="Hitter", position_type="B", eligible_positions=("1B",)),
            batting_projection=bp,
            pitching_projection=None,
            matched=True,
        )
        tp = [
            TeamProjection(
                team_name="Test",
                team_key="t1",
                players=(player,),
                total_hr=25,
                total_sb=10,
                total_h=160,
                total_pa=600,
                team_avg=0.296,
                team_obp=0.358,
                total_r=80,
                total_rbi=90,
                total_ip=0,
                total_so=0,
                total_w=0,
                total_nsvh=0,
                team_era=0,
                team_whip=0,
                unmatched_count=0,
            )
        ]
        # Only HR and OBP configured — SB should not appear as a column header
        settings = LeagueSettings(
            team_count=12,
            batting_categories=(StatCategory.HR, StatCategory.OBP),
            pitching_categories=(StatCategory.ERA,),
        )
        output = format_team_projections(tp, settings)
        lines = output.split("\n")
        batter_header = next(line for line in lines if "Batters" in line)
        assert "HR" in batter_header
        assert "OBP" in batter_header
        assert "SB" not in batter_header
