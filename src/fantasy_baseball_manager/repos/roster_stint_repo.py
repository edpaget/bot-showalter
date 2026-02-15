import sqlite3

from fantasy_baseball_manager.domain.roster_stint import RosterStint


class SqliteRosterStintRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, stint: RosterStint) -> int:
        cursor = self._conn.execute(
            """INSERT INTO roster_stint
                   (player_id, team_id, season, start_date, end_date, loaded_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(player_id, team_id, start_date) DO UPDATE SET
                   season=excluded.season,
                   end_date=excluded.end_date,
                   loaded_at=excluded.loaded_at""",
            (
                stint.player_id,
                stint.team_id,
                stint.season,
                stint.start_date,
                stint.end_date,
                stint.loaded_at,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_player(self, player_id: int) -> list[RosterStint]:
        rows = self._conn.execute(
            "SELECT * FROM roster_stint WHERE player_id = ?",
            (player_id,),
        ).fetchall()
        return [self._row_to_stint(row) for row in rows]

    def get_by_player_season(self, player_id: int, season: int) -> list[RosterStint]:
        rows = self._conn.execute(
            "SELECT * FROM roster_stint WHERE player_id = ? AND season = ?",
            (player_id, season),
        ).fetchall()
        return [self._row_to_stint(row) for row in rows]

    def get_by_team_season(self, team_id: int, season: int) -> list[RosterStint]:
        rows = self._conn.execute(
            "SELECT * FROM roster_stint WHERE team_id = ? AND season = ?",
            (team_id, season),
        ).fetchall()
        return [self._row_to_stint(row) for row in rows]

    def get_by_season(self, season: int) -> list[RosterStint]:
        rows = self._conn.execute(
            "SELECT * FROM roster_stint WHERE season = ?",
            (season,),
        ).fetchall()
        return [self._row_to_stint(row) for row in rows]

    @staticmethod
    def _row_to_stint(row: sqlite3.Row) -> RosterStint:
        return RosterStint(
            id=row["id"],
            player_id=row["player_id"],
            team_id=row["team_id"],
            season=row["season"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            loaded_at=row["loaded_at"],
        )
