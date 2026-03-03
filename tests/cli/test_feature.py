from contextlib import contextmanager
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.cli.factory import FeatureContext
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.repos.feature_candidate_repo import (
    SqliteFeatureCandidateRepo,
)
from fantasy_baseball_manager.services.data_profiler import CorrelationScanner

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator

runner = CliRunner()


def _seed_statcast(conn: sqlite3.Connection) -> None:
    """Insert minimal statcast data for testing."""
    rows = [
        (1, "2023-06-01", 100, 200, 1, 1, 90.0, 1, "single", "hit_into_play"),
        (2, "2023-06-02", 100, 200, 1, 1, 95.0, 0, "field_out", "hit_into_play"),
        (3, "2023-07-01", 300, 200, 1, 1, 85.0, 0, "field_out", "hit_into_play"),
    ]
    for row in rows:
        conn.execute(
            """INSERT INTO statcast_pitch
               (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number,
                launch_speed, barrel, events, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            row,
        )
    conn.commit()


@contextmanager
def _build_test_feature_context(
    statcast_conn: sqlite3.Connection,
    fbm_conn: sqlite3.Connection | None = None,
) -> Iterator[FeatureContext]:
    if fbm_conn is None:
        fbm_conn = create_connection(":memory:")
    yield FeatureContext(
        statcast_conn=statcast_conn,
        fbm_conn=fbm_conn,
        candidate_repo=SqliteFeatureCandidateRepo(fbm_conn),
        scanner=CorrelationScanner(statcast_conn, fbm_conn),
    )


class TestFeatureCandidateCommand:
    def test_basic_invocation(self, monkeypatch: object) -> None:
        conn = create_statcast_connection(":memory:")
        _seed_statcast(conn)

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.feature.build_feature_context",
            lambda data_dir: _build_test_feature_context(conn),
        )

        result = runner.invoke(
            app,
            ["feature", "candidate", "AVG(launch_speed)", "--season", "2023", "--player-type", "batter"],
        )
        assert result.exit_code == 0, result.output
        assert "player-seasons" in result.output

    def test_correlate_flag(self, monkeypatch: object) -> None:
        statcast_conn = create_statcast_connection(":memory:")
        _seed_statcast(statcast_conn)
        fbm_conn = create_connection(":memory:")
        # Insert a player so correlation scanner can map mlbam->player id
        fbm_conn.execute("INSERT INTO player (id, mlbam_id, name_first, name_last) VALUES (1, 100, 'Test', 'Player')")
        # Insert batting stats so correlations can compute
        fbm_conn.execute(
            """INSERT INTO batting_stats (player_id, season, source, pa, ab, h, hr, rbi, sb, r, bb, so, hbp, sf, ibb)
               VALUES (1, 2023, 'actual', 500, 450, 130, 25, 80, 10, 70, 50, 100, 5, 3, 2)"""
        )
        fbm_conn.commit()

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.feature.build_feature_context",
            lambda data_dir: _build_test_feature_context(statcast_conn, fbm_conn),
        )

        result = runner.invoke(
            app,
            [
                "feature",
                "candidate",
                "AVG(launch_speed)",
                "--season",
                "2023",
                "--player-type",
                "batter",
                "--correlate",
            ],
        )
        assert result.exit_code == 0, result.output
        # Should show correlation output (season header or pooled header)
        assert "Season" in result.output or "Pooled" in result.output

    def test_save_name(self, monkeypatch: object) -> None:
        statcast_conn = create_statcast_connection(":memory:")
        _seed_statcast(statcast_conn)
        fbm_conn = create_connection(":memory:")

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.feature.build_feature_context",
            lambda data_dir: _build_test_feature_context(statcast_conn, fbm_conn),
        )

        result = runner.invoke(
            app,
            [
                "feature",
                "candidate",
                "AVG(launch_speed)",
                "--season",
                "2023",
                "--player-type",
                "batter",
                "--name",
                "avg_ev",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "avg_ev" in result.output

        # Verify it was saved
        repo = SqliteFeatureCandidateRepo(fbm_conn)
        saved = repo.get_by_name("avg_ev")
        assert saved is not None
        assert saved.expression == "AVG(launch_speed)"

    def test_invalid_expression(self, monkeypatch: object) -> None:
        conn = create_statcast_connection(":memory:")

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.feature.build_feature_context",
            lambda data_dir: _build_test_feature_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "feature",
                "candidate",
                "DROP TABLE statcast_pitch",
                "--season",
                "2023",
                "--player-type",
                "batter",
            ],
        )
        assert result.exit_code != 0

    def test_min_pa_filtering(self, monkeypatch: object) -> None:
        conn = create_statcast_connection(":memory:")
        _seed_statcast(conn)

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.feature.build_feature_context",
            lambda data_dir: _build_test_feature_context(conn),
        )

        result = runner.invoke(
            app,
            [
                "feature",
                "candidate",
                "AVG(launch_speed)",
                "--season",
                "2023",
                "--player-type",
                "batter",
                "--min-pa",
                "100",
            ],
        )
        assert result.exit_code == 0, result.output
        # With min_pa=100, no players should qualify (only 2-3 pitches each)
        assert "0 player-seasons" in result.output
