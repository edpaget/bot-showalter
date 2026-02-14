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
            position=row["position"],
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
