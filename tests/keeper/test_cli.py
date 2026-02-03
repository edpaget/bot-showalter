from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli import app
from fantasy_baseball_manager.draft.cli import set_data_source_factory
from fantasy_baseball_manager.keeper.cli import (
    set_id_mapper_factory as set_keeper_id_mapper_factory,
)
from fantasy_baseball_manager.keeper.cli import (
    set_roster_source_factory as set_keeper_roster_source_factory,
)
from fantasy_baseball_manager.keeper.cli import (
    set_yahoo_league_factory as set_keeper_yahoo_league_factory,
)
from fantasy_baseball_manager.league.models import LeagueRosters, RosterPlayer, TeamRoster
from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

YEARS = [2024, 2023, 2022]


def _make_batter(
    player_id: str = "b1",
    name: str = "Test Hitter",
    year: int = 2024,
    hr: int = 25,
    sb: int = 10,
    pa: int = 600,
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
    so: int = 200,
    er: int = 70,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=28,
        ip=180.0,
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


def _build_fake(num_batters: int = 8, num_pitchers: int = 8) -> FakeDataSource:
    batter_configs = [
        ("b1", "Slugger Jones", 40, 5),
        ("b2", "Speedy Smith", 10, 30),
        ("b3", "Average Andy", 20, 15),
        ("b4", "Power Pete", 35, 3),
        ("b5", "Contact Carl", 15, 20),
        ("b6", "Bench Bob", 8, 5),
        ("b7", "Backup Bill", 5, 3),
        ("b8", "Filler Fred", 3, 2),
    ]
    pitcher_configs = [
        ("p1", "Ace Adams", 250, 50),
        ("p2", "Bullpen Bob", 150, 70),
        ("p3", "Middle Mike", 180, 60),
        ("p4", "Starter Stan", 220, 55),
        ("p5", "Setup Sam", 130, 75),
        ("p6", "Long Larry", 100, 80),
        ("p7", "Spot Steve", 90, 85),
        ("p8", "Mop Mo", 70, 90),
    ]

    player_batting: dict[int, list[BattingSeasonStats]] = {}
    team_batting: dict[int, list[BattingSeasonStats]] = {}
    player_pitching: dict[int, list[PitchingSeasonStats]] = {}
    team_pitching: dict[int, list[PitchingSeasonStats]] = {}

    for y in YEARS:
        batters = [
            _make_batter(player_id=pid, name=name, year=y, hr=hr, sb=sb)
            for pid, name, hr, sb in batter_configs[:num_batters]
        ]
        player_batting[y] = batters
        team_batting[y] = [_make_league_batting(year=y)]

        pitchers = [
            _make_pitcher(player_id=pid, name=name, year=y, so=so, er=er)
            for pid, name, so, er in pitcher_configs[:num_pitchers]
        ]
        player_pitching[y] = pitchers
        team_pitching[y] = [_make_league_pitching(year=y)]

    return FakeDataSource(
        player_batting=player_batting,
        player_pitching=player_pitching,
        team_batting_stats=team_batting,
        team_pitching_stats=team_pitching,
    )


def _install_fake() -> None:
    ds = _build_fake()
    set_data_source_factory(lambda: ds)


class TestKeeperRankCommand:
    def test_rank_shows_candidates(self) -> None:
        _install_fake()
        result = runner.invoke(
            app, ["keeper", "rank", "2025", "--candidates", "b1,b2,b3", "--user-pick", "1"]
        )
        assert result.exit_code == 0, result.output
        assert "Surplus Value" in result.output or "Surplus" in result.output
        assert "Slugger Jones" in result.output
        assert "Speedy Smith" in result.output
        assert "Average Andy" in result.output

    def test_rank_no_candidates_errors(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["keeper", "rank", "2025"])
        assert result.exit_code == 1

    def test_rank_unknown_candidate_errors(self) -> None:
        _install_fake()
        result = runner.invoke(
            app, ["keeper", "rank", "2025", "--candidates", "nonexistent"]
        )
        assert result.exit_code == 1
        assert "Unknown candidate ID" in result.output

    def test_rank_with_keepers_file(self, tmp_path: Path) -> None:
        _install_fake()
        keepers_file = tmp_path / "keepers.yaml"
        keepers_file.write_text(
            "teams:\n"
            "  1:\n"
            "    keepers: [b4, b5]\n"
            "  2:\n"
            "    keepers: [p1, p2]\n"
        )
        result = runner.invoke(
            app,
            [
                "keeper", "rank", "2025",
                "--candidates", "b1,b2,b3",
                "--keepers", str(keepers_file),
                "--user-pick", "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Slugger Jones" in result.output


class TestKeeperOptimizeCommand:
    def test_optimize_shows_recommendations(self) -> None:
        _install_fake()
        result = runner.invoke(
            app,
            [
                "keeper", "optimize", "2025",
                "--candidates", "b1,b2,b3,b4,b5,b6",
                "--user-pick", "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Optimal Keepers" in result.output
        assert "Total Surplus" in result.output
        assert "All Candidates" in result.output

    def test_optimize_no_candidates_errors(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["keeper", "optimize", "2025"])
        assert result.exit_code == 1

    def test_optimize_with_keepers_file(self, tmp_path: Path) -> None:
        _install_fake()
        keepers_file = tmp_path / "keepers.yaml"
        keepers_file.write_text(
            "teams:\n"
            "  1:\n"
            "    keepers: [p1, p2]\n"
        )
        result = runner.invoke(
            app,
            [
                "keeper", "optimize", "2025",
                "--candidates", "b1,b2,b3,b4,b5",
                "--keepers", str(keepers_file),
                "--user-pick", "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Optimal Keepers" in result.output


# --- Yahoo integration fakes and helpers ---


class FakeKeeperRosterSource:
    def __init__(self, rosters: LeagueRosters) -> None:
        self._rosters = rosters

    def fetch_rosters(self) -> LeagueRosters:
        return self._rosters


class FakeKeeperIdMapper:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._yahoo_to_fg = mapping
        self._fg_to_yahoo = {v: k for k, v in mapping.items()}

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return self._yahoo_to_fg.get(yahoo_id)

    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None:
        return self._fg_to_yahoo.get(fangraphs_id)

    def fangraphs_to_mlbam(self, fangraphs_id: str) -> str | None:
        return None

    def mlbam_to_fangraphs(self, mlbam_id: str) -> str | None:
        return None


class FakeYahooLeague:
    """Mimics yahoo_fantasy_api.League enough for team_key()."""

    def __init__(self, team_key: str) -> None:
        self._team_key = team_key

    def team_key(self) -> str:
        return self._team_key


def _make_yahoo_rosters() -> LeagueRosters:
    """Build rosters where Yahoo IDs map to the FakeDataSource's player IDs (b1, b2, p1, etc.)."""
    user_team = TeamRoster(
        team_key="422.l.123.t.1",
        team_name="My Team",
        players=(
            RosterPlayer(yahoo_id="Y100", name="Slugger Jones", position_type="B", eligible_positions=("1B", "DH")),
            RosterPlayer(yahoo_id="Y101", name="Speedy Smith", position_type="B", eligible_positions=("OF",)),
            RosterPlayer(yahoo_id="Y102", name="Average Andy", position_type="B", eligible_positions=("2B", "SS")),
        ),
    )
    other_team = TeamRoster(
        team_key="422.l.123.t.2",
        team_name="Other Team",
        players=(
            RosterPlayer(yahoo_id="Y200", name="Power Pete", position_type="B", eligible_positions=("OF",)),
            RosterPlayer(yahoo_id="Y201", name="Ace Adams", position_type="P", eligible_positions=("SP",)),
        ),
    )
    return LeagueRosters(league_key="422.l.123", teams=(user_team, other_team))


def _yahoo_id_mapping() -> dict[str, str]:
    return {
        "Y100": "b1",  # Slugger Jones
        "Y101": "b2",  # Speedy Smith
        "Y102": "b3",  # Average Andy
        "Y200": "b4",  # Power Pete
        "Y201": "p1",  # Ace Adams
    }


def _install_yahoo_fakes() -> None:
    _install_fake()  # data source
    rosters = _make_yahoo_rosters()
    mapping = _yahoo_id_mapping()
    set_keeper_roster_source_factory(lambda: FakeKeeperRosterSource(rosters))
    set_keeper_id_mapper_factory(lambda: FakeKeeperIdMapper(mapping))
    set_keeper_yahoo_league_factory(lambda: FakeYahooLeague("422.l.123.t.1"))


class TestKeeperRankYahoo:
    def test_yahoo_auto_populates_candidates(self) -> None:
        _install_yahoo_fakes()
        result = runner.invoke(
            app,
            ["keeper", "rank", "2025", "--yahoo", "--user-pick", "1", "--teams", "2"],
        )
        assert result.exit_code == 0, result.output
        assert "Slugger Jones" in result.output
        assert "Speedy Smith" in result.output
        assert "Average Andy" in result.output

    def test_yahoo_with_candidates_filter(self) -> None:
        _install_yahoo_fakes()
        result = runner.invoke(
            app,
            ["keeper", "rank", "2025", "--yahoo", "--candidates", "b1,b3", "--user-pick", "1", "--teams", "2"],
        )
        assert result.exit_code == 0, result.output
        assert "Slugger Jones" in result.output
        assert "Average Andy" in result.output
        # b2 (Speedy Smith) should be filtered out
        assert "Speedy Smith" not in result.output

    def test_yahoo_with_keepers_file(self, tmp_path: Path) -> None:
        _install_yahoo_fakes()
        keepers_file = tmp_path / "keepers.yaml"
        keepers_file.write_text(
            "teams:\n"
            "  rival:\n"
            "    keepers: [b4, p1]\n"
        )
        result = runner.invoke(
            app,
            [
                "keeper", "rank", "2025",
                "--yahoo",
                "--keepers", str(keepers_file),
                "--user-pick", "1",
                "--teams", "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Slugger Jones" in result.output

    def test_yahoo_unmapped_players_warn(self) -> None:
        _install_fake()
        # Add an unmappable player
        rosters = LeagueRosters(
            league_key="422.l.123",
            teams=(
                TeamRoster(
                    team_key="422.l.123.t.1",
                    team_name="My Team",
                    players=(
                        RosterPlayer(
                            yahoo_id="Y100", name="Slugger Jones",
                            position_type="B", eligible_positions=("1B",),
                        ),
                        RosterPlayer(
                            yahoo_id="Y999", name="Unknown Guy",
                            position_type="B", eligible_positions=("OF",),
                        ),
                    ),
                ),
            ),
        )
        # Y999 is not in the mapping
        mapping = {"Y100": "b1"}
        set_keeper_roster_source_factory(lambda: FakeKeeperRosterSource(rosters))
        set_keeper_id_mapper_factory(lambda: FakeKeeperIdMapper(mapping))
        set_keeper_yahoo_league_factory(lambda: FakeYahooLeague("422.l.123.t.1"))
        result = runner.invoke(
            app,
            ["keeper", "rank", "2025", "--yahoo", "--user-pick", "1", "--teams", "1"],
        )
        assert result.exit_code == 0, result.output
        assert "unmapped" in result.output.lower() or "Warning" in result.output


class TestKeeperOptimizeYahoo:
    def test_optimize_yahoo_auto_populates(self) -> None:
        _install_yahoo_fakes()
        result = runner.invoke(
            app,
            ["keeper", "optimize", "2025", "--yahoo", "--user-pick", "1", "--teams", "2"],
        )
        assert result.exit_code == 0, result.output
        assert "Optimal Keepers" in result.output
        assert "Total Surplus" in result.output

    def test_optimize_yahoo_with_candidates_filter(self) -> None:
        _install_yahoo_fakes()
        result = runner.invoke(
            app,
            [
                "keeper", "optimize", "2025",
                "--yahoo", "--candidates", "b1,b3",
                "--user-pick", "1",
                "--teams", "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Optimal Keepers" in result.output
        # Only b1 and b3 should be candidates
        assert "Speedy Smith" not in result.output or "All Candidates" in result.output


# --- League command tests ---


def _make_league_rosters() -> LeagueRosters:
    """Build a 3-team league where Yahoo IDs map to FakeDataSource player IDs."""
    team_1 = TeamRoster(
        team_key="422.l.123.t.1",
        team_name="Alpha Team",
        players=(
            RosterPlayer(yahoo_id="Y100", name="Slugger Jones", position_type="B", eligible_positions=("1B", "DH")),
            RosterPlayer(yahoo_id="Y101", name="Speedy Smith", position_type="B", eligible_positions=("OF",)),
        ),
    )
    team_2 = TeamRoster(
        team_key="422.l.123.t.2",
        team_name="Beta Team",
        players=(
            RosterPlayer(yahoo_id="Y102", name="Average Andy", position_type="B", eligible_positions=("2B", "SS")),
            RosterPlayer(yahoo_id="Y103", name="Power Pete", position_type="B", eligible_positions=("OF",)),
        ),
    )
    team_3 = TeamRoster(
        team_key="422.l.123.t.3",
        team_name="Gamma Team",
        players=(
            RosterPlayer(yahoo_id="Y104", name="Contact Carl", position_type="B", eligible_positions=("3B",)),
            RosterPlayer(yahoo_id="Y105", name="Ace Adams", position_type="P", eligible_positions=("SP",)),
        ),
    )
    return LeagueRosters(league_key="422.l.123", teams=(team_1, team_2, team_3))


def _league_id_mapping() -> dict[str, str]:
    return {
        "Y100": "b1",  # Slugger Jones
        "Y101": "b2",  # Speedy Smith
        "Y102": "b3",  # Average Andy
        "Y103": "b4",  # Power Pete
        "Y104": "b5",  # Contact Carl
        "Y105": "p1",  # Ace Adams
    }


def _install_league_fakes() -> None:
    _install_fake()
    rosters = _make_league_rosters()
    mapping = _league_id_mapping()
    set_keeper_roster_source_factory(lambda: FakeKeeperRosterSource(rosters))
    set_keeper_id_mapper_factory(lambda: FakeKeeperIdMapper(mapping))
    set_keeper_yahoo_league_factory(lambda: FakeYahooLeague("422.l.123.t.1"))


class TestKeeperLeagueCommand:
    def test_league_shows_all_teams(self) -> None:
        _install_league_fakes()
        result = runner.invoke(
            app,
            ["keeper", "league", "2025", "--keeper-slots", "1"],
        )
        assert result.exit_code == 0, result.output
        assert "Alpha Team" in result.output
        assert "Beta Team" in result.output
        assert "Gamma Team" in result.output

    def test_league_shows_surplus_per_team(self) -> None:
        _install_league_fakes()
        result = runner.invoke(
            app,
            ["keeper", "league", "2025", "--keeper-slots", "1"],
        )
        assert result.exit_code == 0, result.output
        assert "Total Surplus" in result.output

    def test_league_shows_keeper_table(self) -> None:
        _install_league_fakes()
        result = runner.invoke(
            app,
            ["keeper", "league", "2025", "--keeper-slots", "1"],
        )
        assert result.exit_code == 0, result.output
        # Should have table headers
        assert "Rk" in result.output
        assert "Surplus" in result.output

    def test_league_with_draft_order(self) -> None:
        _install_league_fakes()
        result = runner.invoke(
            app,
            [
                "keeper", "league", "2025",
                "--keeper-slots", "1",
                "--draft-order", "422.l.123.t.3,422.l.123.t.1,422.l.123.t.2",
            ],
        )
        assert result.exit_code == 0, result.output
        # Gamma Team should be pick #1
        assert "Pick #1" in result.output
        # Alpha Team should be pick #2
        assert "Pick #2" in result.output

    def test_league_with_unmapped_players(self) -> None:
        _install_fake()
        rosters = LeagueRosters(
            league_key="422.l.123",
            teams=(
                TeamRoster(
                    team_key="422.l.123.t.1",
                    team_name="Team A",
                    players=(
                        RosterPlayer(yahoo_id="Y100", name="Slugger Jones",
                                     position_type="B", eligible_positions=("1B",)),
                        RosterPlayer(yahoo_id="Y999", name="Unknown Guy",
                                     position_type="B", eligible_positions=("OF",)),
                    ),
                ),
            ),
        )
        mapping = {"Y100": "b1"}
        set_keeper_roster_source_factory(lambda: FakeKeeperRosterSource(rosters))
        set_keeper_id_mapper_factory(lambda: FakeKeeperIdMapper(mapping))
        set_keeper_yahoo_league_factory(lambda: FakeYahooLeague("422.l.123.t.1"))
        result = runner.invoke(
            app,
            ["keeper", "league", "2025", "--keeper-slots", "1", "--teams", "1"],
        )
        assert result.exit_code == 0, result.output
        assert "unmapped" in result.output.lower() or "Warning" in result.output

    def test_league_skips_players_without_projections(self) -> None:
        """Players mapped to FanGraphs IDs but lacking projections should be skipped."""
        _install_fake()
        rosters = LeagueRosters(
            league_key="422.l.123",
            teams=(
                TeamRoster(
                    team_key="422.l.123.t.1",
                    team_name="Team A",
                    players=(
                        RosterPlayer(yahoo_id="Y100", name="Slugger Jones",
                                     position_type="B", eligible_positions=("1B",)),
                        RosterPlayer(yahoo_id="Y900", name="Minor Leaguer",
                                     position_type="B", eligible_positions=("OF",)),
                    ),
                ),
            ),
        )
        # Y900 maps to a FanGraphs ID that has no projections
        mapping = {"Y100": "b1", "Y900": "no_projection"}
        set_keeper_roster_source_factory(lambda: FakeKeeperRosterSource(rosters))
        set_keeper_id_mapper_factory(lambda: FakeKeeperIdMapper(mapping))
        set_keeper_yahoo_league_factory(lambda: FakeYahooLeague("422.l.123.t.1"))
        result = runner.invoke(
            app,
            ["keeper", "league", "2025", "--keeper-slots", "1", "--teams", "1"],
        )
        assert result.exit_code == 0, result.output
        assert "Slugger Jones" in result.output
