import statistics
from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.services.data_profiler import NUMERIC_COLUMNS, StatcastColumnProfiler

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Generator

_next_game_pk = 1000


@pytest.fixture
def statcast_conn() -> Generator[sqlite3.Connection]:
    global _next_game_pk  # noqa: PLW0603
    _next_game_pk = 1000
    connection = create_statcast_connection(":memory:")
    yield connection
    connection.close()


def _insert_pitches(
    conn: sqlite3.Connection,
    *,
    batter_id: int,
    pitcher_id: int,
    game_date: str,
    launch_speeds: list[float | None],
    release_speeds: list[float | None] | None = None,
) -> None:
    """Insert synthetic statcast_pitch rows for testing."""
    global _next_game_pk  # noqa: PLW0603
    for i, ls in enumerate(launch_speeds):
        rs = release_speeds[i] if release_speeds else None
        conn.execute(
            """INSERT INTO statcast_pitch
               (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number,
                launch_speed, release_speed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (_next_game_pk, game_date, batter_id, pitcher_id, i + 1, 1, ls, rs),
        )
        _next_game_pk += 1
    conn.commit()


class TestProfileColumnsBatter:
    """Test profiling with batter aggregation."""

    def test_basic_distribution_stats(self, statcast_conn: sqlite3.Connection) -> None:
        # 5 batters, each with known launch_speed values in 2023
        # Batter 1: pitches at 80, 82 -> avg 81
        # Batter 2: pitches at 85, 87 -> avg 86
        # Batter 3: pitches at 90, 90 -> avg 90
        # Batter 4: pitches at 95, 93 -> avg 94
        # Batter 5: pitches at 100, 98 -> avg 99
        _insert_pitches(statcast_conn, batter_id=1, pitcher_id=100, game_date="2023-06-01", launch_speeds=[80.0, 82.0])
        _insert_pitches(statcast_conn, batter_id=2, pitcher_id=100, game_date="2023-06-01", launch_speeds=[85.0, 87.0])
        _insert_pitches(statcast_conn, batter_id=3, pitcher_id=100, game_date="2023-06-01", launch_speeds=[90.0, 90.0])
        _insert_pitches(statcast_conn, batter_id=4, pitcher_id=100, game_date="2023-06-01", launch_speeds=[95.0, 93.0])
        _insert_pitches(statcast_conn, batter_id=5, pitcher_id=100, game_date="2023-06-01", launch_speeds=[100.0, 98.0])

        profiler = StatcastColumnProfiler(statcast_conn)
        results = profiler.profile_columns(["launch_speed"], [2023], "batter")

        assert len(results) == 1
        profile = results[0]
        assert profile.column == "launch_speed"
        assert profile.season == 2023
        assert profile.player_type == "batter"
        assert profile.count == 5

        # Player averages: 81, 86, 90, 94, 99
        values = [81.0, 86.0, 90.0, 94.0, 99.0]
        assert profile.mean == pytest.approx(statistics.mean(values), abs=0.01)
        assert profile.median == pytest.approx(statistics.median(values), abs=0.01)
        assert profile.std == pytest.approx(statistics.stdev(values), abs=0.01)
        assert profile.min == pytest.approx(81.0, abs=0.01)
        assert profile.max == pytest.approx(99.0, abs=0.01)
        assert profile.null_count == 0
        assert profile.null_pct == 0.0

    def test_null_values(self, statcast_conn: sqlite3.Connection) -> None:
        # Batter 1: all non-null launch_speed
        _insert_pitches(statcast_conn, batter_id=1, pitcher_id=100, game_date="2023-06-01", launch_speeds=[90.0, 92.0])
        # Batter 2: all null launch_speed
        _insert_pitches(statcast_conn, batter_id=2, pitcher_id=100, game_date="2023-06-01", launch_speeds=[None, None])
        # Batter 3: non-null
        _insert_pitches(statcast_conn, batter_id=3, pitcher_id=100, game_date="2023-06-01", launch_speeds=[88.0, 86.0])

        profiler = StatcastColumnProfiler(statcast_conn)
        results = profiler.profile_columns(["launch_speed"], [2023], "batter")

        assert len(results) == 1
        profile = results[0]
        assert profile.count == 2
        assert profile.null_count == 1
        assert profile.null_pct == pytest.approx(100.0 / 3, abs=0.01)


class TestProfileColumnsPitcher:
    """Test profiling with pitcher aggregation."""

    def test_uses_pitcher_id(self, statcast_conn: sqlite3.Connection) -> None:
        # 3 pitchers, each with known release_speed values
        # Pitcher 101: avg 92
        # Pitcher 102: avg 95
        # Pitcher 103: avg 88
        _insert_pitches(
            statcast_conn,
            batter_id=1,
            pitcher_id=101,
            game_date="2023-06-01",
            launch_speeds=[None],
            release_speeds=[91.0],
        )
        _insert_pitches(
            statcast_conn,
            batter_id=2,
            pitcher_id=101,
            game_date="2023-06-01",
            launch_speeds=[None],
            release_speeds=[93.0],
        )
        _insert_pitches(
            statcast_conn,
            batter_id=1,
            pitcher_id=102,
            game_date="2023-06-01",
            launch_speeds=[None],
            release_speeds=[94.0],
        )
        _insert_pitches(
            statcast_conn,
            batter_id=2,
            pitcher_id=102,
            game_date="2023-06-01",
            launch_speeds=[None],
            release_speeds=[96.0],
        )
        _insert_pitches(
            statcast_conn,
            batter_id=1,
            pitcher_id=103,
            game_date="2023-06-01",
            launch_speeds=[None],
            release_speeds=[87.0],
        )
        _insert_pitches(
            statcast_conn,
            batter_id=2,
            pitcher_id=103,
            game_date="2023-06-01",
            launch_speeds=[None],
            release_speeds=[89.0],
        )

        profiler = StatcastColumnProfiler(statcast_conn)
        results = profiler.profile_columns(["release_speed"], [2023], "pitcher")

        assert len(results) == 1
        profile = results[0]
        assert profile.player_type == "pitcher"
        assert profile.count == 3
        # Pitcher averages: 92, 95, 88
        values = [92.0, 95.0, 88.0]
        assert profile.mean == pytest.approx(statistics.mean(values), abs=0.01)
        assert profile.median == pytest.approx(statistics.median(values), abs=0.01)


class TestProfileColumnsMultiSeason:
    """Test that results are grouped by season."""

    def test_multi_season_returns_separate_profiles(self, statcast_conn: sqlite3.Connection) -> None:
        # 2023 data
        _insert_pitches(statcast_conn, batter_id=1, pitcher_id=100, game_date="2023-06-01", launch_speeds=[90.0])
        _insert_pitches(statcast_conn, batter_id=2, pitcher_id=100, game_date="2023-06-01", launch_speeds=[95.0])
        # 2024 data
        _insert_pitches(statcast_conn, batter_id=1, pitcher_id=100, game_date="2024-06-01", launch_speeds=[88.0])
        _insert_pitches(statcast_conn, batter_id=2, pitcher_id=100, game_date="2024-06-01", launch_speeds=[92.0])

        profiler = StatcastColumnProfiler(statcast_conn)
        results = profiler.profile_columns(["launch_speed"], [2023, 2024], "batter")

        assert len(results) == 2
        by_season = {r.season: r for r in results}
        assert 2023 in by_season
        assert 2024 in by_season
        # 2023: avg values 90, 95 -> mean 92.5
        assert by_season[2023].mean == pytest.approx(92.5, abs=0.01)
        # 2024: avg values 88, 92 -> mean 90
        assert by_season[2024].mean == pytest.approx(90.0, abs=0.01)


class TestProfileColumnsMultiColumn:
    """Test profiling multiple columns at once."""

    def test_returns_profile_per_column_per_season(self, statcast_conn: sqlite3.Connection) -> None:
        _insert_pitches(
            statcast_conn,
            batter_id=1,
            pitcher_id=100,
            game_date="2023-06-01",
            launch_speeds=[90.0],
            release_speeds=[95.0],
        )
        _insert_pitches(
            statcast_conn,
            batter_id=2,
            pitcher_id=100,
            game_date="2023-06-01",
            launch_speeds=[85.0],
            release_speeds=[92.0],
        )

        profiler = StatcastColumnProfiler(statcast_conn)
        results = profiler.profile_columns(["launch_speed", "release_speed"], [2023], "batter")

        assert len(results) == 2
        by_col = {r.column: r for r in results}
        assert "launch_speed" in by_col
        assert "release_speed" in by_col
        assert by_col["launch_speed"].mean == pytest.approx(87.5, abs=0.01)
        assert by_col["release_speed"].mean == pytest.approx(93.5, abs=0.01)


class TestProfileColumnsValidation:
    """Test column name validation."""

    def test_invalid_column_raises_value_error(self, statcast_conn: sqlite3.Connection) -> None:
        profiler = StatcastColumnProfiler(statcast_conn)
        with pytest.raises(ValueError, match="not_a_column"):
            profiler.profile_columns(["not_a_column"], [2023], "batter")

    def test_invalid_player_type_raises_value_error(self, statcast_conn: sqlite3.Connection) -> None:
        profiler = StatcastColumnProfiler(statcast_conn)
        with pytest.raises(ValueError, match="player_type"):
            profiler.profile_columns(["launch_speed"], [2023], "catcher")


class TestProfileColumnsSkewness:
    """Test that skewness is computed correctly."""

    def test_symmetric_data_has_near_zero_skewness(self, statcast_conn: sqlite3.Connection) -> None:
        # Symmetric data: 80, 85, 90, 95, 100
        for i, batter_id in enumerate([1, 2, 3, 4, 5]):
            _insert_pitches(
                statcast_conn,
                batter_id=batter_id,
                pitcher_id=100,
                game_date="2023-06-01",
                launch_speeds=[80.0 + i * 5.0],
            )

        profiler = StatcastColumnProfiler(statcast_conn)
        results = profiler.profile_columns(["launch_speed"], [2023], "batter")

        profile = results[0]
        assert abs(profile.skewness) < 0.01

    def test_right_skewed_data(self, statcast_conn: sqlite3.Connection) -> None:
        # Right-skewed: most values low, one high outlier
        values = [80.0, 81.0, 82.0, 83.0, 120.0]
        for i, val in enumerate(values):
            _insert_pitches(
                statcast_conn,
                batter_id=i + 1,
                pitcher_id=100,
                game_date="2023-06-01",
                launch_speeds=[val],
            )

        profiler = StatcastColumnProfiler(statcast_conn)
        results = profiler.profile_columns(["launch_speed"], [2023], "batter")

        profile = results[0]
        assert profile.skewness > 0  # positive skew

    def test_single_value_has_zero_skewness(self, statcast_conn: sqlite3.Connection) -> None:
        """When there's only one player-season, std=0, skewness should be 0."""
        _insert_pitches(
            statcast_conn,
            batter_id=1,
            pitcher_id=100,
            game_date="2023-06-01",
            launch_speeds=[90.0],
        )

        profiler = StatcastColumnProfiler(statcast_conn)
        results = profiler.profile_columns(["launch_speed"], [2023], "batter")

        profile = results[0]
        assert profile.skewness == 0.0
        assert profile.std == 0.0


class TestProfileColumnsPercentiles:
    """Test that percentiles are computed correctly."""

    def test_percentiles_with_enough_data(self, statcast_conn: sqlite3.Connection) -> None:
        # 10 batters with values 10, 20, 30, ..., 100
        for i in range(10):
            _insert_pitches(
                statcast_conn,
                batter_id=i + 1,
                pitcher_id=100,
                game_date="2023-06-01",
                launch_speeds=[float((i + 1) * 10)],
            )

        profiler = StatcastColumnProfiler(statcast_conn)
        results = profiler.profile_columns(["launch_speed"], [2023], "batter")

        profile = results[0]
        values = [float(i * 10) for i in range(1, 11)]
        quantiles_10 = statistics.quantiles(values, n=10)
        # p10 is quantiles[0], p25 is from quartiles, etc.
        quartiles = statistics.quantiles(values, n=4)
        assert profile.p25 == pytest.approx(quartiles[0], abs=0.01)
        assert profile.p75 == pytest.approx(quartiles[2], abs=0.01)
        assert profile.p10 == pytest.approx(quantiles_10[0], abs=0.01)
        assert profile.p90 == pytest.approx(quantiles_10[8], abs=0.01)


class TestNumericColumns:
    """Test the NUMERIC_COLUMNS constant."""

    def test_contains_expected_columns(self) -> None:
        expected = {
            "release_speed",
            "release_spin_rate",
            "pfx_x",
            "pfx_z",
            "plate_x",
            "plate_z",
            "launch_speed",
            "launch_angle",
            "hit_distance_sc",
            "barrel",
            "estimated_ba_using_speedangle",
            "estimated_woba_using_speedangle",
            "estimated_slg_using_speedangle",
            "hc_x",
            "hc_y",
            "release_extension",
        }
        assert set(NUMERIC_COLUMNS) == expected

    def test_is_tuple(self) -> None:
        assert isinstance(NUMERIC_COLUMNS, tuple)
