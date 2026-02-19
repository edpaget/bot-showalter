import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from typer.testing import CliRunner

from fantasy_baseball_manager.cli._output import print_ingest_result
from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.cli.factory import IngestContainer
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.load_log import LoadLog
from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.minor_league_batting_stats_repo import SqliteMinorLeagueBattingStatsRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo

runner = CliRunner()


# --- Fake data source for testing ---


class _FakeSource:
    def __init__(self, rows: list[dict[str, Any]], source_type: str, source_detail: str) -> None:
        self._rows = rows
        self._source_type = source_type
        self._source_detail = source_detail

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def source_detail(self) -> str:
        return self._source_detail

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        return self._rows


# --- Test container subclass ---


class _TestIngestContainer(IngestContainer):
    def __init__(
        self,
        conn: sqlite3.Connection,
        fake_source: DataSource,
        fake_bio_source: DataSource | None = None,
        fake_appearances_source: DataSource | None = None,
        fake_teams_source: DataSource | None = None,
        fake_milb_batting_source: DataSource | None = None,
    ) -> None:
        super().__init__(conn)
        self._fake_source = fake_source
        self._fake_bio_source = fake_bio_source
        self._fake_appearances_source = fake_appearances_source
        self._fake_teams_source = fake_teams_source
        self._fake_milb_batting_source = fake_milb_batting_source

    def player_source(self) -> DataSource:
        return self._fake_source

    def batting_source(self, name: str) -> DataSource:
        return self._fake_source

    def pitching_source(self, name: str) -> DataSource:
        return self._fake_source

    def bio_source(self) -> DataSource:
        if self._fake_bio_source is not None:
            return self._fake_bio_source
        return self._fake_source

    def appearances_source(self) -> DataSource:
        if self._fake_appearances_source is not None:
            return self._fake_appearances_source
        return self._fake_source

    def teams_source(self) -> DataSource:
        if self._fake_teams_source is not None:
            return self._fake_teams_source
        return self._fake_source

    def milb_batting_source(self) -> DataSource:
        if self._fake_milb_batting_source is not None:
            return self._fake_milb_batting_source
        return self._fake_source


def _make_player_rows() -> list[dict[str, Any]]:
    return [
        {
            "key_mlbam": 545361,
            "name_first": "Mike",
            "name_last": "Trout",
            "key_fangraphs": 10155,
            "key_bbref": "troutmi01",
            "key_retro": "troum001",
        },
        {
            "key_mlbam": 660271,
            "name_first": "Shohei",
            "name_last": "Ohtani",
            "key_fangraphs": 19755,
            "key_bbref": "ohtansh01",
            "key_retro": "ohtas001",
        },
    ]


@contextmanager
def _build_test_container(
    conn: sqlite3.Connection,
    fake_source: DataSource,
    fake_bio_source: DataSource | None = None,
    fake_appearances_source: DataSource | None = None,
    fake_teams_source: DataSource | None = None,
    fake_milb_batting_source: DataSource | None = None,
) -> Iterator[IngestContainer]:
    yield _TestIngestContainer(
        conn,
        fake_source,
        fake_bio_source,
        fake_appearances_source=fake_appearances_source,
        fake_teams_source=fake_teams_source,
        fake_milb_batting_source=fake_milb_batting_source,
    )


# --- Tests ---


class TestPrintIngestResult:
    def test_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        log = LoadLog(
            source_type="pybaseball",
            source_detail="chadwick_register",
            target_table="player",
            rows_loaded=42,
            started_at="2024-01-01T00:00:00",
            finished_at="2024-01-01T00:01:00",
            status="success",
        )
        print_ingest_result(log)
        captured = capsys.readouterr()
        assert "42" in captured.out
        assert "player" in captured.out
        assert "chadwick_register" in captured.out
        assert "success" in captured.out

    def test_error_shows_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        log = LoadLog(
            source_type="pybaseball",
            source_detail="fg_batting_data",
            target_table="batting_stats",
            rows_loaded=0,
            started_at="2024-01-01T00:00:00",
            finished_at="2024-01-01T00:01:00",
            status="error",
            error_message="connection timeout",
        )
        print_ingest_result(log)
        captured = capsys.readouterr()
        assert "error" in captured.out
        assert "connection timeout" in captured.out


class TestIngestPlayers:
    def test_ingest_players_loads_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        player_rows = _make_player_rows()
        fake_source = _FakeSource(player_rows, "chadwick_bureau", "chadwick_register")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            lambda data_dir: _build_test_container(conn, fake_source),
        )

        result = runner.invoke(app, ["ingest", "players"])
        assert result.exit_code == 0, result.output
        assert "2" in result.output
        assert "player" in result.output
        assert "success" in result.output

        # Verify data actually in DB
        repo = SqlitePlayerRepo(conn)
        players = repo.all()
        assert len(players) == 2
        names = {p.name_last for p in players}
        assert "Trout" in names
        assert "Ohtani" in names
        conn.close()

    def test_ingest_players_custom_data_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        player_rows = _make_player_rows()
        fake_source = _FakeSource(player_rows, "chadwick_bureau", "chadwick_register")

        captured_data_dir: list[str] = []

        @contextmanager
        def _capturing_container(data_dir: str) -> Iterator[IngestContainer]:
            captured_data_dir.append(data_dir)
            yield _TestIngestContainer(conn, fake_source)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            _capturing_container,
        )

        result = runner.invoke(app, ["ingest", "players", "--data-dir", "/custom/path"])
        assert result.exit_code == 0, result.output
        assert captured_data_dir == ["/custom/path"]
        conn.close()


def _seed_players(conn: sqlite3.Connection) -> None:
    """Seed player data needed for batting/pitching/bio mapper lookups."""
    repo = SqlitePlayerRepo(conn)
    repo.upsert(
        Player(
            name_first="Mike",
            name_last="Trout",
            mlbam_id=545361,
            fangraphs_id=10155,
            bbref_id="troutmi01",
            retro_id="troum001",
        )
    )
    repo.upsert(
        Player(
            name_first="Shohei",
            name_last="Ohtani",
            mlbam_id=660271,
            fangraphs_id=19755,
            bbref_id="ohtansh01",
            retro_id="ohtas001",
        )
    )
    conn.commit()


def _make_fg_batting_rows() -> list[dict[str, Any]]:
    return [
        {
            "IDfg": 10155,
            "Season": 2023,
            "PA": 600,
            "AB": 500,
            "H": 150,
            "2B": 30,
            "3B": 5,
            "HR": 35,
            "RBI": 90,
            "R": 100,
            "SB": 10,
            "CS": 3,
            "BB": 80,
            "SO": 120,
            "HBP": 5,
            "SF": 4,
            "SH": 0,
            "GDP": 8,
            "IBB": 10,
            "AVG": 0.300,
            "OBP": 0.400,
            "SLG": 0.600,
            "OPS": 1.000,
            "wOBA": 0.420,
            "wRC+": 180.0,
            "WAR": 8.5,
        },
    ]


class TestIngestBatting:
    def test_ingest_batting_loads_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_players(conn)

        batting_rows = _make_fg_batting_rows()
        fake_source = _FakeSource(batting_rows, "pybaseball", "fg_batting_data")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            lambda data_dir: _build_test_container(conn, fake_source),
        )

        result = runner.invoke(app, ["ingest", "batting", "--season", "2023"])
        assert result.exit_code == 0, result.output
        assert "1" in result.output
        assert "batting_stats" in result.output
        assert "success" in result.output

        # Verify data in DB
        repo = SqliteBattingStatsRepo(conn)
        stats = repo.get_by_season(2023, source="fangraphs")
        assert len(stats) == 1
        assert stats[0].hr == 35
        assert stats[0].pa == 600
        conn.close()

    def test_ingest_batting_multiple_seasons(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_players(conn)

        batting_rows = _make_fg_batting_rows()
        fake_source = _FakeSource(batting_rows, "pybaseball", "fg_batting_data")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            lambda data_dir: _build_test_container(conn, fake_source),
        )

        result = runner.invoke(app, ["ingest", "batting", "--season", "2022", "--season", "2023"])
        assert result.exit_code == 0, result.output
        # Should see two result outputs (one per season)
        assert result.output.count("success") == 2

    def test_ingest_batting_requires_season(self) -> None:
        result = runner.invoke(app, ["ingest", "batting"])
        assert result.exit_code != 0


def _make_fg_pitching_rows() -> list[dict[str, Any]]:
    return [
        {
            "IDfg": 19755,
            "Season": 2024,
            "W": 15,
            "L": 5,
            "G": 30,
            "GS": 30,
            "SV": 0,
            "H": 120,
            "ER": 50,
            "HR": 15,
            "BB": 40,
            "SO": 200,
            "ERA": 2.80,
            "IP": 180.0,
            "WHIP": 0.95,
            "K/9": 10.0,
            "BB/9": 2.0,
            "FIP": 2.90,
            "xFIP": 3.00,
            "WAR": 6.0,
        },
    ]


class TestIngestPitching:
    def test_ingest_pitching_loads_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_players(conn)

        pitching_rows = _make_fg_pitching_rows()
        fake_source = _FakeSource(pitching_rows, "pybaseball", "fg_pitching_data")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            lambda data_dir: _build_test_container(conn, fake_source),
        )

        result = runner.invoke(app, ["ingest", "pitching", "--season", "2024"])
        assert result.exit_code == 0, result.output
        assert "1" in result.output
        assert "pitching_stats" in result.output
        assert "success" in result.output

        # Verify data in DB
        repo = SqlitePitchingStatsRepo(conn)
        stats = repo.get_by_season(2024, source="fangraphs")
        assert len(stats) == 1
        assert stats[0].w == 15
        assert stats[0].so == 200
        conn.close()

    def test_ingest_pitching_multiple_seasons(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_players(conn)

        pitching_rows = _make_fg_pitching_rows()
        fake_source = _FakeSource(pitching_rows, "pybaseball", "fg_pitching_data")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            lambda data_dir: _build_test_container(conn, fake_source),
        )

        result = runner.invoke(app, ["ingest", "pitching", "--season", "2023", "--season", "2024"])
        assert result.exit_code == 0, result.output
        assert result.output.count("success") == 2

    def test_ingest_pitching_requires_season(self) -> None:
        result = runner.invoke(app, ["ingest", "pitching"])
        assert result.exit_code != 0


def _make_lahman_people_rows() -> list[dict[str, Any]]:
    return [
        {
            "retroID": "troum001",
            "bbrefID": "troutmi01",
            "birthYear": 1991,
            "birthMonth": 8,
            "birthDay": 7,
            "bats": "R",
            "throws": "R",
            "eligible_positions": "CF,RF",
        },
        {
            "retroID": "ohtas001",
            "bbrefID": "ohtansh01",
            "birthYear": 1994,
            "birthMonth": 7,
            "birthDay": 5,
            "bats": "L",
            "throws": "R",
            "eligible_positions": "DH,P",
        },
        {
            "retroID": "noone999",
            "bbrefID": "nobody01",
            "birthYear": 1985,
            "birthMonth": 1,
            "birthDay": 1,
            "bats": "R",
            "throws": "R",
            "eligible_positions": "1B",
        },
    ]


class TestIngestBio:
    def test_ingest_bio_enriches_existing_players(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_players(conn)

        lahman_rows = _make_lahman_people_rows()
        fake_bio_source = _FakeSource(lahman_rows, "lahman", "people")
        # Player source not used here, but container needs one
        fake_player_source = _FakeSource([], "chadwick_bureau", "chadwick_register")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            lambda data_dir: _build_test_container(conn, fake_player_source, fake_bio_source),
        )

        result = runner.invoke(app, ["ingest", "bio"])
        assert result.exit_code == 0, result.output
        assert "2" in result.output
        assert "player" in result.output
        assert "success" in result.output

        # Verify bio data actually in DB
        repo = SqlitePlayerRepo(conn)
        trout = repo.get_by_mlbam_id(545361)
        assert trout is not None
        assert trout.birth_date == "1991-08-07"
        assert trout.bats == "R"
        assert trout.throws == "R"

        ohtani = repo.get_by_mlbam_id(660271)
        assert ohtani is not None
        assert ohtani.birth_date == "1994-07-05"
        assert ohtani.bats == "L"
        assert ohtani.throws == "R"
        conn.close()

    def test_ingest_bio_skips_unmatched_players(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_players(conn)

        # Only unmatched player in Lahman data
        lahman_rows = [
            {
                "retroID": "noone999",
                "bbrefID": "nobody01",
                "birthYear": 1985,
                "birthMonth": 1,
                "birthDay": 1,
                "bats": "R",
                "throws": "R",
            },
        ]
        fake_bio_source = _FakeSource(lahman_rows, "lahman", "people")
        fake_player_source = _FakeSource([], "chadwick_bureau", "chadwick_register")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            lambda data_dir: _build_test_container(conn, fake_player_source, fake_bio_source),
        )

        result = runner.invoke(app, ["ingest", "bio"])
        assert result.exit_code == 0, result.output
        # 0 rows loaded since no players matched
        assert "0" in result.output
        conn.close()


def _make_appearances_rows() -> list[dict[str, Any]]:
    return [
        {"playerID": "troutmi01", "yearID": 2023, "teamID": "LAA", "position": "CF", "games": 82},
        {"playerID": "troutmi01", "yearID": 2023, "teamID": "LAA", "position": "DH", "games": 25},
        {"playerID": "ohtansh01", "yearID": 2023, "teamID": "LAA", "position": "DH", "games": 135},
    ]


def _make_teams_rows() -> list[dict[str, Any]]:
    return [
        {"teamID": "LAA", "name": "Los Angeles Angels", "lgID": "AL", "divID": "W", "yearID": 2023},
    ]


def _make_roster_appearances_rows() -> list[dict[str, Any]]:
    """Lahman Appearances rows (pre-exploded format used by the roster mapper)."""
    return [
        {"playerID": "troutmi01", "yearID": 2023, "teamID": "LAA"},
        {"playerID": "ohtansh01", "yearID": 2023, "teamID": "LAA"},
    ]


class TestIngestAppearances:
    def test_ingest_appearances_loads_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_players(conn)

        appearances_data = _make_appearances_rows()
        fake_appearances = _FakeSource(appearances_data, "lahman", "appearances")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            lambda data_dir: _build_test_container(
                conn,
                _FakeSource([], "chadwick_bureau", "chadwick_register"),
                fake_appearances_source=fake_appearances,
            ),
        )

        result = runner.invoke(app, ["ingest", "appearances", "--season", "2023"])
        assert result.exit_code == 0, result.output
        assert "position_appearance" in result.output
        assert "success" in result.output

        repo = SqlitePositionAppearanceRepo(conn)
        appearances = repo.get_by_season(2023)
        assert len(appearances) == 3
        conn.close()

    def test_ingest_appearances_requires_season(self) -> None:
        result = runner.invoke(app, ["ingest", "appearances"])
        assert result.exit_code != 0


class TestIngestRoster:
    def test_ingest_roster_loads_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_players(conn)

        roster_data = _make_roster_appearances_rows()
        teams_data = _make_teams_rows()
        fake_appearances = _FakeSource(roster_data, "lahman", "appearances")
        fake_teams = _FakeSource(teams_data, "lahman", "teams")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            lambda data_dir: _build_test_container(
                conn,
                _FakeSource([], "chadwick_bureau", "chadwick_register"),
                fake_appearances_source=fake_appearances,
                fake_teams_source=fake_teams,
            ),
        )

        result = runner.invoke(app, ["ingest", "roster", "--season", "2023"])
        assert result.exit_code == 0, result.output
        assert "roster_stint" in result.output
        assert "success" in result.output

        # Verify teams were auto-upserted
        team_repo = SqliteTeamRepo(conn)
        teams = team_repo.all()
        assert len(teams) == 1
        assert teams[0].abbreviation == "LAA"

        # Verify roster stints
        stint_repo = SqliteRosterStintRepo(conn)
        stints = stint_repo.get_by_season(2023)
        assert len(stints) == 2
        conn.close()

    def test_ingest_roster_requires_season(self) -> None:
        result = runner.invoke(app, ["ingest", "roster"])
        assert result.exit_code != 0


def _make_milb_batting_rows(**overrides: object) -> list[dict[str, Any]]:
    defaults: dict[str, object] = {
        "mlbam_id": 545361,
        "season": 2024,
        "level": "AAA",
        "league": "International League",
        "team": "Syracuse Mets",
        "g": 120,
        "pa": 500,
        "ab": 450,
        "h": 130,
        "doubles": 25,
        "triples": 3,
        "hr": 18,
        "r": 70,
        "rbi": 65,
        "bb": 40,
        "so": 100,
        "sb": 15,
        "cs": 5,
        "avg": 0.289,
        "obp": 0.350,
        "slg": 0.480,
        "age": 24.5,
        "hbp": 8,
        "sf": 4,
        "sh": 1,
        "first_name": "Mike",
        "last_name": "Trout",
    }
    defaults.update(overrides)
    return [defaults]


class TestIngestMilbBatting:
    def test_ingest_milb_batting_loads_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        _seed_players(conn)

        milb_rows = _make_milb_batting_rows()
        fake_milb = _FakeSource(milb_rows, "mlb_api", "milb_batting")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            lambda data_dir: _build_test_container(
                conn,
                _FakeSource([], "chadwick_bureau", "chadwick_register"),
                fake_milb_batting_source=fake_milb,
            ),
        )

        result = runner.invoke(app, ["ingest", "milb-batting", "--season", "2024", "--level", "AAA"])
        assert result.exit_code == 0, result.output
        assert "minor_league_batting_stats" in result.output
        assert "success" in result.output

        repo = SqliteMinorLeagueBattingStatsRepo(conn)
        stats = repo.get_by_season_level(2024, "AAA")
        assert len(stats) == 1
        assert stats[0].hr == 18
        conn.close()

    def test_ingest_milb_batting_auto_registers_unknown_players(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        # No players seeded — mlbam_id 777777 is unknown
        milb_rows = _make_milb_batting_rows(
            mlbam_id=777777,
            first_name="Prospect",
            last_name="Jones",
        )
        fake_milb = _FakeSource(milb_rows, "mlb_api", "milb_batting")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.build_ingest_container",
            lambda data_dir: _build_test_container(
                conn,
                _FakeSource([], "chadwick_bureau", "chadwick_register"),
                fake_milb_batting_source=fake_milb,
            ),
        )

        result = runner.invoke(app, ["ingest", "milb-batting", "--season", "2024", "--level", "AAA"])
        assert result.exit_code == 0, result.output
        assert "1" in result.output

        # Player should have been auto-registered
        player_repo = SqlitePlayerRepo(conn)
        player = player_repo.get_by_mlbam_id(777777)
        assert player is not None
        assert player.name_first == "Prospect"
        assert player.name_last == "Jones"

        # Stats should be loaded
        repo = SqliteMinorLeagueBattingStatsRepo(conn)
        stats = repo.get_by_season_level(2024, "AAA")
        assert len(stats) == 1
        conn.close()

    def test_ingest_milb_batting_requires_season(self) -> None:
        result = runner.invoke(app, ["ingest", "milb-batting"])
        assert result.exit_code != 0
