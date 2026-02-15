import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.cli.factory import ComputeContainer
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.minor_league_batting_stats import MinorLeagueBattingStats
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos.league_environment_repo import SqliteLeagueEnvironmentRepo
from fantasy_baseball_manager.repos.minor_league_batting_stats_repo import SqliteMinorLeagueBattingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo

runner = CliRunner()


def _seed_milb_stats(conn: sqlite3.Connection) -> None:
    """Seed a player and minor league stats for testing."""
    player_repo = SqlitePlayerRepo(conn)
    pid = player_repo.upsert(Player(name_first="Test", name_last="Player", mlbam_id=12345))

    stats_repo = SqliteMinorLeagueBattingStatsRepo(conn)
    stats_repo.upsert(
        MinorLeagueBattingStats(
            player_id=pid,
            season=2024,
            level="AAA",
            league="International League",
            team="Syracuse Mets",
            g=100,
            pa=400,
            ab=350,
            h=91,
            doubles=20,
            triples=2,
            hr=12,
            r=50,
            rbi=45,
            bb=35,
            so=80,
            sb=10,
            cs=3,
            avg=0.260,
            obp=0.330,
            slg=0.420,
            age=24.5,
            hbp=5,
            sf=4,
            sh=1,
        )
    )
    conn.commit()


@contextmanager
def _build_test_compute_container(conn: sqlite3.Connection) -> Iterator[ComputeContainer]:
    yield ComputeContainer(conn)


class TestComputeLeagueEnv:
    def test_compute_league_env_loads_data(self, monkeypatch: object) -> None:
        conn = create_connection(":memory:")
        _seed_milb_stats(conn)

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.app.build_compute_container",
            lambda data_dir: _build_test_compute_container(conn),
        )

        result = runner.invoke(app, ["compute", "league-env", "--season", "2024", "--level", "AAA"])
        assert result.exit_code == 0, result.output
        assert "1 league(s) computed" in result.output
        assert "Done" in result.output

        env_repo = SqliteLeagueEnvironmentRepo(conn)
        envs = env_repo.get_by_season_level(2024, "AAA")
        assert len(envs) == 1
        assert envs[0].league == "International League"
        assert envs[0].avg > 0
        conn.close()

    def test_compute_league_env_no_data(self, monkeypatch: object) -> None:
        conn = create_connection(":memory:")

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.app.build_compute_container",
            lambda data_dir: _build_test_compute_container(conn),
        )

        result = runner.invoke(app, ["compute", "league-env", "--season", "2024", "--level", "AAA"])
        assert result.exit_code == 0, result.output
        assert "0 league(s) computed" in result.output
        conn.close()

    def test_compute_league_env_requires_season(self) -> None:
        result = runner.invoke(app, ["compute", "league-env"])
        assert result.exit_code != 0
