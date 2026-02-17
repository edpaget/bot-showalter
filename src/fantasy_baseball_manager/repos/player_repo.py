import sqlite3
from typing import Never

from fantasy_baseball_manager.domain.player import Player, Team
from fantasy_baseball_manager.repos.errors import PlayerConflictError

_SECONDARY_KEYS = ("fangraphs_id", "bbref_id", "retro_id")


class SqlitePlayerRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, player: Player) -> int:
        existing = self._conn.execute("SELECT id FROM player WHERE mlbam_id = ?", (player.mlbam_id,)).fetchone()

        if existing:
            try:
                self._conn.execute(
                    """UPDATE player SET
                           name_first=?, name_last=?, fangraphs_id=?, bbref_id=?,
                           retro_id=?, bats=?, throws=?, birth_date=?
                       WHERE mlbam_id=?""",
                    (
                        player.name_first,
                        player.name_last,
                        player.fangraphs_id,
                        player.bbref_id,
                        player.retro_id,
                        player.bats,
                        player.throws,
                        player.birth_date,
                        player.mlbam_id,
                    ),
                )
                return existing["id"]
            except sqlite3.IntegrityError:
                self._raise_conflict(player)

        try:
            cursor = self._conn.execute(
                """INSERT INTO player (name_first, name_last, mlbam_id, fangraphs_id,
                                       bbref_id, retro_id, bats, throws, birth_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    player.name_first,
                    player.name_last,
                    player.mlbam_id,
                    player.fangraphs_id,
                    player.bbref_id,
                    player.retro_id,
                    player.bats,
                    player.throws,
                    player.birth_date,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.IntegrityError:
            self._raise_conflict(player)

    def _raise_conflict(self, player: Player) -> Never:
        for col in _SECONDARY_KEYS:
            value = getattr(player, col)
            if value is None:
                continue
            row = self._conn.execute(f"SELECT * FROM player WHERE {col} = ?", (value,)).fetchone()
            if row is not None:
                raise PlayerConflictError(player, self._row_to_player(row), col)
        raise  # re-raise original IntegrityError if we can't identify the conflict

    def get_by_id(self, player_id: int) -> Player | None:
        row = self._conn.execute("SELECT * FROM player WHERE id = ?", (player_id,)).fetchone()
        return self._row_to_player(row) if row else None

    def get_by_ids(self, player_ids: list[int]) -> list[Player]:
        if not player_ids:
            return []
        placeholders = ",".join("?" * len(player_ids))
        rows = self._conn.execute(
            f"SELECT * FROM player WHERE id IN ({placeholders})",
            player_ids,
        ).fetchall()
        return [self._row_to_player(row) for row in rows]

    def get_by_mlbam_id(self, mlbam_id: int) -> Player | None:
        row = self._conn.execute("SELECT * FROM player WHERE mlbam_id = ?", (mlbam_id,)).fetchone()
        return self._row_to_player(row) if row else None

    def get_by_bbref_id(self, bbref_id: str) -> Player | None:
        row = self._conn.execute("SELECT * FROM player WHERE bbref_id = ?", (bbref_id,)).fetchone()
        return self._row_to_player(row) if row else None

    def search_by_name(self, name: str) -> list[Player]:
        pattern = f"%{name}%"
        rows = self._conn.execute(
            "SELECT * FROM player WHERE name_first LIKE ? OR name_last LIKE ?",
            (pattern, pattern),
        ).fetchall()
        return [self._row_to_player(row) for row in rows]

    def get_by_last_name(self, last_name: str) -> list[Player]:
        rows = self._conn.execute(
            "SELECT * FROM player WHERE name_last = ? COLLATE NOCASE",
            (last_name,),
        ).fetchall()
        return [self._row_to_player(row) for row in rows]

    def all(self) -> list[Player]:
        rows = self._conn.execute("SELECT * FROM player").fetchall()
        return [self._row_to_player(row) for row in rows]

    @staticmethod
    def _row_to_player(row: sqlite3.Row) -> Player:
        return Player(
            id=row["id"],
            name_first=row["name_first"],
            name_last=row["name_last"],
            mlbam_id=row["mlbam_id"],
            fangraphs_id=row["fangraphs_id"],
            bbref_id=row["bbref_id"],
            retro_id=row["retro_id"],
            bats=row["bats"],
            throws=row["throws"],
            birth_date=row["birth_date"],
        )


class SqliteTeamRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, team: Team) -> int:
        cursor = self._conn.execute(
            """INSERT INTO team (abbreviation, name, league, division)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(abbreviation) DO UPDATE SET
                   name=excluded.name, league=excluded.league, division=excluded.division""",
            (team.abbreviation, team.name, team.league, team.division),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_abbreviation(self, abbreviation: str) -> Team | None:
        row = self._conn.execute("SELECT * FROM team WHERE abbreviation = ?", (abbreviation,)).fetchone()
        return self._row_to_team(row) if row else None

    def all(self) -> list[Team]:
        rows = self._conn.execute("SELECT * FROM team").fetchall()
        return [self._row_to_team(row) for row in rows]

    @staticmethod
    def _row_to_team(row: sqlite3.Row) -> Team:
        return Team(
            id=row["id"],
            abbreviation=row["abbreviation"],
            name=row["name"],
            league=row["league"],
            division=row["division"],
        )
