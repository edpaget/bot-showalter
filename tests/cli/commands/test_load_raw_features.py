"""Tests for _load_raw_features in cli/commands/residuals.py."""

import sqlite3

import pytest

from fantasy_baseball_manager.cli.commands.residuals import _load_raw_features


def _make_statcast_conn() -> sqlite3.Connection:
    """Create an in-memory statcast_pitch table with the columns the query expects."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE statcast_pitch (
            batter INTEGER,
            pitcher INTEGER,
            game_year INTEGER,
            release_speed REAL,
            release_spin_rate REAL,
            launch_speed REAL,
            launch_angle REAL,
            hit_distance_sc REAL,
            barrel INTEGER,
            estimated_ba_using_speedangle REAL,
            estimated_woba_using_speedangle REAL,
            estimated_slg_using_speedangle REAL,
            release_extension REAL
        )
    """)
    return conn


def _seed_pitches(conn: sqlite3.Connection, *, count: int = 60, batter: int = 100, pitcher: int = 200) -> None:
    """Seed pitch rows for a single batter/pitcher pair."""
    for _i in range(count):
        conn.execute(
            """INSERT INTO statcast_pitch
               (batter, pitcher, game_year, release_speed, release_spin_rate,
                launch_speed, launch_angle, hit_distance_sc, barrel,
                estimated_ba_using_speedangle, estimated_woba_using_speedangle,
                estimated_slg_using_speedangle, release_extension)
               VALUES (?, ?, 2024, 93.0, 2400.0, 95.0, 15.0, 350.0, 1, 0.280, 0.350, 0.450, 6.2)""",
            (batter, pitcher),
        )
    conn.commit()


class TestLoadRawFeatures:
    def test_batter_features_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _make_statcast_conn()
        _seed_pitches(conn, count=60, batter=100)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.residuals.create_statcast_connection",
            lambda _path: conn,
        )

        result = _load_raw_features("./data", 2024, "batter")

        assert 100 in result
        features = result[100]
        assert "release_speed" in features
        assert "launch_speed" in features
        assert features["release_speed"] == pytest.approx(93.0)

    def test_pitcher_uses_pitcher_column(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _make_statcast_conn()
        _seed_pitches(conn, count=60, batter=100, pitcher=200)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.residuals.create_statcast_connection",
            lambda _path: conn,
        )

        result = _load_raw_features("./data", 2024, "pitcher")

        # Pitcher query groups by pitcher column
        assert 200 in result
        assert 100 not in result

    def test_null_values_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _make_statcast_conn()
        # Seed rows where some values are NULL
        for _i in range(60):
            conn.execute(
                """INSERT INTO statcast_pitch
                   (batter, pitcher, game_year, release_speed, release_spin_rate,
                    launch_speed, launch_angle, hit_distance_sc, barrel,
                    estimated_ba_using_speedangle, estimated_woba_using_speedangle,
                    estimated_slg_using_speedangle, release_extension)
                   VALUES (100, 200, 2024, 93.0, NULL, 95.0, NULL, NULL, NULL, NULL, NULL, NULL, NULL)""",
            )
        conn.commit()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.residuals.create_statcast_connection",
            lambda _path: conn,
        )

        result = _load_raw_features("./data", 2024, "batter")

        features = result[100]
        assert "release_speed" in features
        assert "launch_speed" in features
        # NULL columns should not appear in the features dict
        assert "release_spin_rate" not in features
        assert "launch_angle" not in features
