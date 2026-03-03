from contextlib import contextmanager
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.cli.factory import ProfileContext
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.services.data_profiler import (
    CorrelationScanner,
    StatcastColumnProfiler,
    TemporalStabilityChecker,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator

runner = CliRunner()


@contextmanager
def _build_test_profile_context(
    conn: sqlite3.Connection, stats_conn: sqlite3.Connection | None = None
) -> Iterator[ProfileContext]:
    if stats_conn is None:
        stats_conn = create_connection(":memory:")
    scanner = CorrelationScanner(conn, stats_conn)
    yield ProfileContext(
        profiler=StatcastColumnProfiler(conn),
        scanner=scanner,
        stability_checker=TemporalStabilityChecker(scanner),
    )


class TestProfileColumnsCommand:
    def test_basic_invocation(self, monkeypatch: object) -> None:
        conn = create_statcast_connection(":memory:")
        # Insert test data
        conn.execute(
            """INSERT INTO statcast_pitch
               (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number, launch_speed)
               VALUES (1, '2023-06-01', 1, 100, 1, 1, 90.0)"""
        )
        conn.execute(
            """INSERT INTO statcast_pitch
               (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number, launch_speed)
               VALUES (2, '2023-06-01', 2, 100, 1, 1, 95.0)"""
        )
        conn.commit()

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.profile.build_profile_context",
            lambda data_dir: _build_test_profile_context(conn),
        )

        result = runner.invoke(
            app,
            ["profile", "columns", "launch_speed", "--season", "2023", "--player-type", "batter"],
        )
        assert result.exit_code == 0, result.output
        # Rich table may truncate column names in narrow terminals
        assert "lau" in result.output
        assert "20" in result.output
        conn.close()

    def test_all_flag(self, monkeypatch: object) -> None:
        conn = create_statcast_connection(":memory:")
        # Insert minimal data
        conn.execute(
            """INSERT INTO statcast_pitch
               (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number,
                launch_speed, release_speed)
               VALUES (1, '2023-06-01', 1, 100, 1, 1, 90.0, 95.0)"""
        )
        conn.commit()

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.profile.build_profile_context",
            lambda data_dir: _build_test_profile_context(conn),
        )

        result = runner.invoke(
            app,
            ["profile", "columns", "--all", "--season", "2023", "--player-type", "batter"],
        )
        assert result.exit_code == 0, result.output
        # Should see multiple columns profiled (Rich may truncate names)
        assert "lau" in result.output
        assert "rel" in result.output
        conn.close()

    def test_no_columns_and_no_all_flag(self) -> None:
        result = runner.invoke(
            app,
            ["profile", "columns", "--season", "2023", "--player-type", "batter"],
        )
        assert result.exit_code != 0

    def test_missing_season_option(self) -> None:
        result = runner.invoke(
            app,
            ["profile", "columns", "launch_speed", "--player-type", "batter"],
        )
        assert result.exit_code != 0

    def test_missing_player_type_option(self) -> None:
        result = runner.invoke(
            app,
            ["profile", "columns", "launch_speed", "--season", "2023"],
        )
        assert result.exit_code != 0

    def test_invalid_column_shows_error(self, monkeypatch: object) -> None:
        conn = create_statcast_connection(":memory:")

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.profile.build_profile_context",
            lambda data_dir: _build_test_profile_context(conn),
        )

        result = runner.invoke(
            app,
            ["profile", "columns", "fake_column", "--season", "2023", "--player-type", "batter"],
        )
        assert result.exit_code != 0
        conn.close()

    def test_pitcher_player_type(self, monkeypatch: object) -> None:
        conn = create_statcast_connection(":memory:")
        conn.execute(
            """INSERT INTO statcast_pitch
               (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number, release_speed)
               VALUES (1, '2023-06-01', 1, 101, 1, 1, 93.0)"""
        )
        conn.execute(
            """INSERT INTO statcast_pitch
               (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number, release_speed)
               VALUES (2, '2023-06-01', 2, 102, 1, 1, 96.0)"""
        )
        conn.commit()

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.profile.build_profile_context",
            lambda data_dir: _build_test_profile_context(conn),
        )

        result = runner.invoke(
            app,
            ["profile", "columns", "release_speed", "--season", "2023", "--player-type", "pitcher"],
        )
        assert result.exit_code == 0, result.output
        # Rich table may truncate column names
        assert "rel" in result.output
        conn.close()


def _setup_correlate_test_data(statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
    """Set up minimal test data for correlate command tests."""
    for i in range(1, 6):
        mlbam_id = 1000 + i
        statcast_conn.execute(
            """INSERT INTO statcast_pitch
               (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number, launch_speed)
               VALUES (?, '2023-06-01', ?, 9999, 1, 1, ?)""",
            (i, mlbam_id, 80.0 + i),
        )
        stats_conn.execute(
            "INSERT INTO player (id, name_first, name_last, mlbam_id) VALUES (?, 'Test', 'Player', ?)",
            (i, mlbam_id),
        )
        stats_conn.execute(
            """INSERT INTO batting_stats
               (player_id, season, source, avg, obp, slg, woba, h, hr, ab, so, sf)
               VALUES (?, 2023, 'fangraphs', ?, ?, ?, 0.320, 150, 20, 500, 100, 5)""",
            (i, 0.250 + i * 0.01, 0.330 + i * 0.01, 0.400 + i * 0.01),
        )
    statcast_conn.commit()
    stats_conn.commit()


class TestCorrelateCommand:
    def test_basic_invocation(self, monkeypatch: object) -> None:
        statcast_conn = create_statcast_connection(":memory:")
        stats_conn = create_connection(":memory:")
        _setup_correlate_test_data(statcast_conn, stats_conn)

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.profile.build_profile_context",
            lambda data_dir: _build_test_profile_context(statcast_conn, stats_conn),
        )

        result = runner.invoke(
            app,
            ["profile", "correlate", "launch_speed", "--season", "2023", "--player-type", "batter"],
        )
        assert result.exit_code == 0, result.output
        # Should show target names in output
        assert "avg" in result.output
        assert "slg" in result.output
        assert "Pooled" in result.output
        statcast_conn.close()
        stats_conn.close()

    def test_missing_season(self) -> None:
        result = runner.invoke(
            app,
            ["profile", "correlate", "launch_speed", "--player-type", "batter"],
        )
        assert result.exit_code != 0

    def test_missing_player_type(self) -> None:
        result = runner.invoke(
            app,
            ["profile", "correlate", "launch_speed", "--season", "2023"],
        )
        assert result.exit_code != 0

    def test_multiple_columns_shows_ranking(self, monkeypatch: object) -> None:
        statcast_conn = create_statcast_connection(":memory:")
        stats_conn = create_connection(":memory:")
        _setup_correlate_test_data(statcast_conn, stats_conn)

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.profile.build_profile_context",
            lambda data_dir: _build_test_profile_context(statcast_conn, stats_conn),
        )

        result = runner.invoke(
            app,
            [
                "profile",
                "correlate",
                "launch_speed",
                "release_speed",
                "--season",
                "2023",
                "--player-type",
                "batter",
            ],
        )
        assert result.exit_code == 0, result.output
        # Should show ranking table for multiple columns
        assert "Avg" in result.output
        statcast_conn.close()
        stats_conn.close()


def _setup_stability_test_data(statcast_conn: sqlite3.Connection, stats_conn: sqlite3.Connection) -> None:
    """Set up test data with two seasons for stability tests."""
    for season in [2022, 2023]:
        for i in range(1, 11):
            mlbam_id = 1000 + i
            statcast_conn.execute(
                """INSERT INTO statcast_pitch
                   (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number, launch_speed)
                   VALUES (?, ?, ?, 9999, 1, 1, ?)""",
                (season * 100 + i, f"{season}-06-01", mlbam_id, 80.0 + i),
            )
            if season == 2022:
                stats_conn.execute(
                    "INSERT INTO player (id, name_first, name_last, mlbam_id) VALUES (?, 'Test', 'Player', ?)",
                    (i, mlbam_id),
                )
            stats_conn.execute(
                """INSERT INTO batting_stats
                   (player_id, season, source, avg, obp, slg, woba, h, hr, ab, so, sf)
                   VALUES (?, ?, 'fangraphs', ?, ?, ?, 0.320, 150, 20, 500, 100, 5)""",
                (i, season, 0.250 + i * 0.01, 0.330 + i * 0.01, 0.400 + i * 0.01),
            )
    statcast_conn.commit()
    stats_conn.commit()


class TestStabilityCommand:
    def test_basic_single_target(self, monkeypatch: object) -> None:
        statcast_conn = create_statcast_connection(":memory:")
        stats_conn = create_connection(":memory:")
        _setup_stability_test_data(statcast_conn, stats_conn)

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.profile.build_profile_context",
            lambda data_dir: _build_test_profile_context(statcast_conn, stats_conn),
        )

        result = runner.invoke(
            app,
            [
                "profile",
                "stability",
                "launch_speed",
                "--season",
                "2022",
                "--season",
                "2023",
                "--player-type",
                "batter",
                "--target",
                "woba",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "woba" in result.output
        statcast_conn.close()
        stats_conn.close()

    def test_all_targets(self, monkeypatch: object) -> None:
        statcast_conn = create_statcast_connection(":memory:")
        stats_conn = create_connection(":memory:")
        _setup_stability_test_data(statcast_conn, stats_conn)

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.profile.build_profile_context",
            lambda data_dir: _build_test_profile_context(statcast_conn, stats_conn),
        )

        result = runner.invoke(
            app,
            [
                "profile",
                "stability",
                "launch_speed",
                "--season",
                "2022",
                "--season",
                "2023",
                "--player-type",
                "batter",
                "--all-targets",
            ],
        )
        assert result.exit_code == 0, result.output
        # Should see multiple targets in the matrix
        assert "slg" in result.output
        assert "avg" in result.output
        statcast_conn.close()
        stats_conn.close()

    def test_exclude_season(self, monkeypatch: object) -> None:
        statcast_conn = create_statcast_connection(":memory:")
        stats_conn = create_connection(":memory:")
        _setup_stability_test_data(statcast_conn, stats_conn)

        monkeypatch.setattr(  # type: ignore[union-attr]
            "fantasy_baseball_manager.cli.commands.profile.build_profile_context",
            lambda data_dir: _build_test_profile_context(statcast_conn, stats_conn),
        )

        result = runner.invoke(
            app,
            [
                "profile",
                "stability",
                "launch_speed",
                "--season",
                "2022",
                "--season",
                "2023",
                "--player-type",
                "batter",
                "--target",
                "slg",
                "--exclude-season",
                "2022",
            ],
        )
        assert result.exit_code == 0, result.output
        # Only 2023 should appear, 2022 excluded
        assert "2023" in result.output
        statcast_conn.close()
        stats_conn.close()

    def test_missing_target_and_no_all_targets(self) -> None:
        result = runner.invoke(
            app,
            [
                "profile",
                "stability",
                "launch_speed",
                "--season",
                "2023",
                "--player-type",
                "batter",
            ],
        )
        assert result.exit_code != 0
