import sqlite3
from typing import TYPE_CHECKING, Never

from fantasy_baseball_manager.domain import Player, Team
from fantasy_baseball_manager.repos.errors import PlayerConflictError

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos.protocols import ConnectionProvider

_SECONDARY_KEYS = ("fangraphs_id", "bbref_id", "retro_id")


class SqlitePlayerRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def upsert(self, player: Player) -> int:
        with self._provider.connection() as conn:
            existing = conn.execute("SELECT id FROM player WHERE mlbam_id = ?", (player.mlbam_id,)).fetchone()

            if existing:
                try:
                    conn.execute(
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
                cursor = conn.execute(
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
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                self._raise_conflict(player)

    def _raise_conflict(self, player: Player) -> Never:
        with self._provider.connection() as conn:
            for col in _SECONDARY_KEYS:
                value = getattr(player, col)
                if value is None:
                    continue
                row = conn.execute(f"SELECT * FROM player WHERE {col} = ?", (value,)).fetchone()
                if row is not None:
                    raise PlayerConflictError(player, self._row_to_player(row), col)
            raise  # re-raise original IntegrityError if we can't identify the conflict

    def get_by_id(self, player_id: int) -> Player | None:
        with self._provider.connection() as conn:
            row = conn.execute("SELECT * FROM player WHERE id = ?", (player_id,)).fetchone()
            return self._row_to_player(row) if row else None

    def get_by_ids(self, player_ids: list[int]) -> list[Player]:
        with self._provider.connection() as conn:
            if not player_ids:
                return []
            placeholders = ",".join("?" * len(player_ids))
            rows = conn.execute(
                f"SELECT * FROM player WHERE id IN ({placeholders})",
                player_ids,
            ).fetchall()
            return [self._row_to_player(row) for row in rows]

    def get_by_mlbam_id(self, mlbam_id: int) -> Player | None:
        with self._provider.connection() as conn:
            row = conn.execute("SELECT * FROM player WHERE mlbam_id = ?", (mlbam_id,)).fetchone()
            return self._row_to_player(row) if row else None

    def get_by_bbref_id(self, bbref_id: str) -> Player | None:
        with self._provider.connection() as conn:
            row = conn.execute("SELECT * FROM player WHERE bbref_id = ?", (bbref_id,)).fetchone()
            return self._row_to_player(row) if row else None

    def search_by_name(self, name: str) -> list[Player]:
        with self._provider.connection() as conn:
            pattern = f"%{name}%"
            rows = conn.execute(
                "SELECT * FROM player WHERE name_first LIKE ? OR name_last LIKE ?",
                (pattern, pattern),
            ).fetchall()
            return [self._row_to_player(row) for row in rows]

    def get_by_last_name(self, last_name: str) -> list[Player]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM player WHERE name_last = ? COLLATE NOCASE",
                (last_name,),
            ).fetchall()
            return [self._row_to_player(row) for row in rows]

    def search_by_last_name_normalized(self, last_name: str) -> list[Player]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM player WHERE strip_accents(name_last) = ? COLLATE NOCASE",
                (last_name,),
            ).fetchall()
            return [self._row_to_player(row) for row in rows]

    def all(self) -> list[Player]:
        with self._provider.connection() as conn:
            rows = conn.execute("SELECT * FROM player").fetchall()
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
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def upsert(self, team: Team) -> int:
        with self._provider.connection() as conn:
            cursor = conn.execute(
                """INSERT INTO team (abbreviation, name, league, division)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(abbreviation) DO UPDATE SET
                       name=excluded.name, league=excluded.league, division=excluded.division""",
                (team.abbreviation, team.name, team.league, team.division),
            )
            return cursor.lastrowid

    def get_by_abbreviation(self, abbreviation: str) -> Team | None:
        with self._provider.connection() as conn:
            row = conn.execute("SELECT * FROM team WHERE abbreviation = ?", (abbreviation,)).fetchone()
            return self._row_to_team(row) if row else None

    def all(self) -> list[Team]:
        with self._provider.connection() as conn:
            rows = conn.execute("SELECT * FROM team").fetchall()
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
