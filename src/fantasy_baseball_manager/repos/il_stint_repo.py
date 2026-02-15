import sqlite3

from fantasy_baseball_manager.domain.il_stint import ILStint


class SqliteILStintRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, stint: ILStint) -> int:
        cursor = self._conn.execute(
            """INSERT INTO il_stint
                   (player_id, season, start_date, il_type,
                    end_date, days, injury_location, transaction_type, loaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(player_id, start_date, il_type) DO UPDATE SET
                   season=excluded.season,
                   end_date=excluded.end_date,
                   days=excluded.days,
                   injury_location=excluded.injury_location,
                   transaction_type=excluded.transaction_type,
                   loaded_at=excluded.loaded_at""",
            (
                stint.player_id,
                stint.season,
                stint.start_date,
                stint.il_type,
                stint.end_date,
                stint.days,
                stint.injury_location,
                stint.transaction_type,
                stint.loaded_at,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_player(self, player_id: int) -> list[ILStint]:
        rows = self._conn.execute(
            "SELECT * FROM il_stint WHERE player_id = ?",
            (player_id,),
        ).fetchall()
        return [self._row_to_stint(row) for row in rows]

    def get_by_player_season(self, player_id: int, season: int) -> list[ILStint]:
        rows = self._conn.execute(
            "SELECT * FROM il_stint WHERE player_id = ? AND season = ?",
            (player_id, season),
        ).fetchall()
        return [self._row_to_stint(row) for row in rows]

    def get_by_season(self, season: int) -> list[ILStint]:
        rows = self._conn.execute(
            "SELECT * FROM il_stint WHERE season = ?",
            (season,),
        ).fetchall()
        return [self._row_to_stint(row) for row in rows]

    @staticmethod
    def _row_to_stint(row: sqlite3.Row) -> ILStint:
        return ILStint(
            id=row["id"],
            player_id=row["player_id"],
            season=row["season"],
            start_date=row["start_date"],
            il_type=row["il_type"],
            end_date=row["end_date"],
            days=row["days"],
            injury_location=row["injury_location"],
            transaction_type=row["transaction_type"],
            loaded_at=row["loaded_at"],
        )
