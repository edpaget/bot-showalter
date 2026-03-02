from contextlib import contextmanager
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.cli.factory import ProfileContext
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.services.data_profiler import StatcastColumnProfiler

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator

runner = CliRunner()


@contextmanager
def _build_test_profile_context(conn: sqlite3.Connection) -> Iterator[ProfileContext]:
    yield ProfileContext(profiler=StatcastColumnProfiler(conn))


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
