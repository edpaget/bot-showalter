from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from fantasy_baseball_manager.db.connection import create_connection

_STATCAST_MIGRATIONS_DIR = (
    Path(__file__).parent.parent.parent / "src" / "fantasy_baseball_manager" / "db" / "statcast_migrations"
)


@pytest.fixture
def conn() -> Generator[sqlite3.Connection]:
    connection = create_connection(":memory:")
    yield connection
    connection.close()


def seed_batting_data(conn: sqlite3.Connection) -> None:
    """Insert 2 players x 4 seasons of batting stats for integration tests."""
    # Player 1: Mike Trout, born 1991-08-07, mlbam_id 545361
    conn.execute(
        "INSERT INTO player (id, name_first, name_last, birth_date, bats, mlbam_id) "
        "VALUES (1, 'Mike', 'Trout', '1991-08-07', 'R', 545361)"
    )
    # Player 2: Mookie Betts, born 1992-10-07, mlbam_id 605141
    conn.execute(
        "INSERT INTO player (id, name_first, name_last, birth_date, bats, mlbam_id) "
        "VALUES (2, 'Mookie', 'Betts', '1992-10-07', 'R', 605141)"
    )

    batting_rows = [
        # (player_id, season, source, pa, hr, bb, ab, h)
        (1, 2020, "fangraphs", 250, 17, 35, 200, 56),
        (1, 2021, "fangraphs", 500, 30, 60, 420, 120),
        (1, 2022, "fangraphs", 550, 40, 70, 460, 140),
        (1, 2023, "fangraphs", 600, 35, 65, 500, 150),
        (2, 2020, "fangraphs", 200, 10, 25, 170, 45),
        (2, 2021, "fangraphs", 480, 22, 50, 400, 110),
        (2, 2022, "fangraphs", 520, 28, 55, 440, 130),
        (2, 2023, "fangraphs", 580, 32, 60, 480, 145),
    ]
    conn.executemany(
        "INSERT INTO batting_stats (player_id, season, source, pa, hr, bb, ab, h) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        batting_rows,
    )
    conn.commit()


def seed_projection_data(conn: sqlite3.Connection) -> None:
    """Insert projection data for 2 players x 1 season x 2 systems."""
    projection_rows = [
        # (player_id, season, system, version, player_type, pa, hr, bb, avg, war)
        (1, 2023, "steamer", "2023.1", "batter", 620, 38, 68, 0.285, 6.0),
        (1, 2023, "zips", "2023.1", "batter", 580, 33, 62, 0.275, 5.2),
        (2, 2023, "steamer", "2023.1", "batter", 610, 30, 58, 0.290, 5.5),
        (2, 2023, "zips", "2023.1", "batter", 570, 27, 52, 0.280, 4.8),
    ]
    for pid, season, system, version, ptype, pa, hr, bb, avg, war in projection_rows:
        conn.execute(
            "INSERT INTO projection (player_id, season, system, version, player_type, pa, hr, bb, avg, war)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, season, system, version, ptype, pa, hr, bb, avg, war),
        )
    conn.commit()


def seed_projection_v2_data(conn: sqlite3.Connection) -> None:
    """Insert version='2023.2' projection rows with different stats."""
    projection_rows = [
        # (player_id, season, system, version, player_type, pa, hr, bb, avg, war)
        (1, 2023, "steamer", "2023.2", "batter", 630, 42, 72, 0.295, 7.0),
        (2, 2023, "steamer", "2023.2", "batter", 620, 34, 62, 0.300, 6.0),
    ]
    for pid, season, system, version, ptype, pa, hr, bb, avg, war in projection_rows:
        conn.execute(
            "INSERT INTO projection (player_id, season, system, version, player_type, pa, hr, bb, avg, war)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, season, system, version, ptype, pa, hr, bb, avg, war),
        )
    conn.commit()


def seed_projection_pitcher_data(conn: sqlite3.Connection) -> None:
    """Insert player_type='pitcher' projection rows with different stats."""
    projection_rows = [
        # (player_id, season, system, version, player_type, pa, hr, bb, avg, war)
        (1, 2023, "steamer", "2023.1", "pitcher", 50, 2, 80, 0.150, 3.0),
        (2, 2023, "steamer", "2023.1", "pitcher", 45, 1, 75, 0.140, 2.5),
    ]
    for pid, season, system, version, ptype, pa, hr, bb, avg, war in projection_rows:
        conn.execute(
            "INSERT INTO projection (player_id, season, system, version, player_type, pa, hr, bb, avg, war)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, season, system, version, ptype, pa, hr, bb, avg, war),
        )
    conn.commit()


def seed_distribution_data(conn: sqlite3.Connection) -> None:
    """Insert projection_distribution rows for steamer/zips projections.

    Requires seed_projection_data to have been called first (uses projection IDs).
    """
    # Look up projection IDs
    rows = conn.execute("SELECT id, player_id, system FROM projection WHERE season = 2023").fetchall()
    proj_map = {(r[1], r[2]): r[0] for r in rows}  # (player_id, system) -> id

    dist_rows = [
        # (projection_id, stat, p10, p25, p50, p75, p90, mean, std)
        (proj_map[(1, "steamer")], "hr", 25.0, 30.0, 37.0, 42.0, 48.0, 37.5, 7.0),
        (proj_map[(1, "steamer")], "bb", 45.0, 55.0, 67.0, 75.0, 85.0, 66.5, 12.0),
        (proj_map[(2, "steamer")], "hr", 18.0, 23.0, 29.0, 35.0, 40.0, 29.5, 6.5),
        (proj_map[(1, "zips")], "hr", 20.0, 26.0, 32.0, 38.0, 44.0, 32.5, 7.5),
    ]
    conn.executemany(
        "INSERT INTO projection_distribution (projection_id, stat, p10, p25, p50, p75, p90, mean, std)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        dist_rows,
    )
    conn.commit()


def seed_il_stint_data(conn: sqlite3.Connection) -> None:
    """Insert IL stint data for 2 players across 2021-2022."""
    il_rows = [
        # (player_id, season, start_date, il_type, days, injury_location, transaction_type)
        (1, 2021, "2021-05-18", "10", 15, "Right calf strain", "placement"),
        (1, 2022, "2022-07-12", "60", 65, "Back surgery", "placement"),
        (1, 2022, "2022-04-01", "10", 12, "Left hamstring", "placement"),
        (2, 2021, "2021-06-15", "15", 20, "Right hip", "placement"),
    ]
    conn.executemany(
        "INSERT INTO il_stint (player_id, season, start_date, il_type, days, injury_location, transaction_type)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        il_rows,
    )
    conn.commit()


@pytest.fixture
def seeded_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Connection with 2 players x 4 seasons of batting data pre-loaded."""
    seed_batting_data(conn)
    return conn


def seed_statcast_data(sc_conn: sqlite3.Connection) -> None:
    """Insert statcast pitch data for 2 test players across 2022-2023."""
    pitches = [
        # (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number,
        #  pitch_type, release_speed, spin, pfx_x, pfx_z, plate_x,
        #  launch_speed, launch_angle, hit_distance, barrel)
        # Player 1 (mlbam_id 545361), 2022 season
        (1001, "2022-06-15", 545361, 100001, 1, 1, "FF", 95.0, None, None, None, None, 100.0, 25.0, None, 1),
        (1001, "2022-06-15", 545361, 100001, 1, 2, "FF", 97.0, None, None, None, None, None, None, None, None),
        (1001, "2022-06-15", 545361, 100001, 2, 1, "SL", 85.0, None, None, None, None, 90.0, 10.0, None, 0),
        (1002, "2022-07-20", 545361, 100002, 1, 1, "CH", 82.0, None, None, None, None, None, None, None, None),
        (1002, "2022-07-20", 545361, 100002, 1, 2, "FF", 96.0, None, None, None, None, 105.0, 30.0, None, 1),
        (1002, "2022-07-20", 545361, 100002, 2, 1, "CU", 78.0, None, None, None, None, 80.0, -5.0, None, 0),
        # Player 1 (mlbam_id 545361), 2023 season
        (1003, "2023-05-10", 545361, 100003, 1, 1, "FF", 94.0, None, None, None, None, 95.0, 20.0, None, 0),
        (1003, "2023-05-10", 545361, 100003, 1, 2, "SL", 84.0, None, None, None, None, None, None, None, None),
        (1003, "2023-05-10", 545361, 100003, 2, 1, "FF", 96.0, None, None, None, None, 102.0, 28.0, None, 1),
        # Player 2 (mlbam_id 605141), 2022 season
        (1004, "2022-08-01", 605141, 100004, 1, 1, "FF", 92.0, None, None, None, None, 88.0, 12.0, None, 0),
        (1004, "2022-08-01", 605141, 100004, 1, 2, "SL", 83.0, None, None, None, None, None, None, None, None),
        (1004, "2022-08-01", 605141, 100004, 2, 1, "FF", 93.0, None, None, None, None, 98.0, 22.0, None, 1),
        (1004, "2022-08-01", 605141, 100004, 2, 2, "CH", 80.0, None, None, None, None, 85.0, 8.0, None, 0),
        # Player 2 (mlbam_id 605141), 2023 season
        (1005, "2023-06-15", 605141, 100005, 1, 1, "FF", 91.0, None, None, None, None, 96.0, 18.0, None, 0),
        (1005, "2023-06-15", 605141, 100005, 1, 2, "FF", 94.0, None, None, None, None, 101.0, 26.0, None, 1),
    ]
    sc_conn.executemany(
        "INSERT INTO statcast_pitch "
        "(game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number, "
        "pitch_type, release_speed, release_spin_rate, pfx_x, pfx_z, plate_x, "
        "launch_speed, launch_angle, hit_distance_sc, barrel) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        pitches,
    )
    sc_conn.commit()


@pytest.fixture
def statcast_db_path() -> Generator[Path]:
    """Create a temporary file-based statcast DB with schema and seed data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    sc_conn = create_connection(path, migrations_dir=_STATCAST_MIGRATIONS_DIR)
    seed_statcast_data(sc_conn)
    sc_conn.close()
    yield path
    path.unlink(missing_ok=True)
