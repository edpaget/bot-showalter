import sqlite3

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo


def seed_player(
    conn: sqlite3.Connection,
    *,
    player_id: int | None = None,
    name_first: str = "Test",
    name_last: str = "Player",
    mlbam_id: int | None = None,
    fangraphs_id: int | None = None,
    bbref_id: str | None = None,
    retro_id: str | None = None,
    bats: str = "R",
    birth_date: str = "1990-01-01",
) -> int:
    """Seed a player row for testing.

    When ``player_id`` is provided, inserts via raw SQL with an explicit ID.
    When omitted, delegates to ``SqlitePlayerRepo.upsert`` for auto-generated ID.
    """
    if player_id is not None:
        conn.execute(
            "INSERT OR IGNORE INTO player (id, name_first, name_last, mlbam_id, "
            "fangraphs_id, bbref_id, retro_id, bats, birth_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (player_id, name_first, name_last, mlbam_id, fangraphs_id, bbref_id, retro_id, bats, birth_date),
        )
        conn.commit()
        return player_id

    repo = SqlitePlayerRepo(conn)
    return repo.upsert(
        Player(
            name_first=name_first,
            name_last=name_last,
            mlbam_id=mlbam_id,
            fangraphs_id=fangraphs_id,
            bbref_id=bbref_id,
            retro_id=retro_id,
            bats=bats,
            birth_date=birth_date,
        )
    )
