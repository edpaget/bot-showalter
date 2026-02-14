import sqlite3

from fantasy_baseball_manager.domain.player import Player, Team


class SqlitePlayerRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, player: Player) -> int:
        cursor = self._conn.execute(
            """INSERT INTO player (name_first, name_last, mlbam_id, fangraphs_id, bbref_id,
                                   retro_id, bats, throws, birth_date, position)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(mlbam_id) DO UPDATE SET
                   name_first=excluded.name_first, name_last=excluded.name_last,
                   fangraphs_id=excluded.fangraphs_id, bbref_id=excluded.bbref_id,
                   retro_id=excluded.retro_id, bats=excluded.bats, throws=excluded.throws,
                   birth_date=excluded.birth_date, position=excluded.position""",
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
                player.position,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_id(self, player_id: int) -> Player | None:
        row = self._conn.execute("SELECT * FROM player WHERE id = ?", (player_id,)).fetchone()
        return self._row_to_player(row) if row else None

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

    def all(self) -> list[Player]:
        rows = self._conn.execute("SELECT * FROM player").fetchall()
        return [self._row_to_player(row) for row in rows]

    @staticmethod
    def _row_to_player(row: tuple) -> Player:
        return Player(
            id=row[0],
            name_first=row[1],
            name_last=row[2],
            mlbam_id=row[3],
            fangraphs_id=row[4],
            bbref_id=row[5],
            retro_id=row[6],
            bats=row[7],
            throws=row[8],
            birth_date=row[9],
            position=row[10],
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
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_abbreviation(self, abbreviation: str) -> Team | None:
        row = self._conn.execute("SELECT * FROM team WHERE abbreviation = ?", (abbreviation,)).fetchone()
        return self._row_to_team(row) if row else None

    def all(self) -> list[Team]:
        rows = self._conn.execute("SELECT * FROM team").fetchall()
        return [self._row_to_team(row) for row in rows]

    @staticmethod
    def _row_to_team(row: tuple) -> Team:
        return Team(
            id=row[0],
            abbreviation=row[1],
            name=row[2],
            league=row[3],
            division=row[4],
        )
