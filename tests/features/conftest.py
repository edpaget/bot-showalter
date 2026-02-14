from __future__ import annotations

import sqlite3
from collections.abc import Generator

import pytest

from fantasy_baseball_manager.db.connection import create_connection


@pytest.fixture
def conn() -> Generator[sqlite3.Connection]:
    connection = create_connection(":memory:")
    yield connection
    connection.close()


def seed_batting_data(conn: sqlite3.Connection) -> None:
    """Insert 2 players x 4 seasons of batting stats for integration tests."""
    # Player 1: Mike Trout, born 1991-08-07
    conn.execute(
        "INSERT INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (1, 'Mike', 'Trout', '1991-08-07', 'R')"
    )
    # Player 2: Mookie Betts, born 1992-10-07
    conn.execute(
        "INSERT INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (2, 'Mookie', 'Betts', '1992-10-07', 'R')"
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
        "INSERT INTO batting_stats (player_id, season, source, pa, hr, bb, ab, h) " "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        batting_rows,
    )
    conn.commit()


@pytest.fixture
def seeded_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Connection with 2 players x 4 seasons of batting data pre-loaded."""
    seed_batting_data(conn)
    return conn
