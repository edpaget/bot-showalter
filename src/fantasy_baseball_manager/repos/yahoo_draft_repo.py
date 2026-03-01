from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import YahooDraftPick

if TYPE_CHECKING:
    import sqlite3


class SqliteYahooDraftRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, pick: YahooDraftPick) -> None:
        self._conn.execute(
            "INSERT INTO yahoo_draft_pick"
            "    (league_key, season, round, pick, team_key,"
            "     yahoo_player_key, player_id, player_name, position, cost)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(league_key, season, round, pick) DO UPDATE SET"
            "    team_key=excluded.team_key,"
            "    yahoo_player_key=excluded.yahoo_player_key,"
            "    player_id=excluded.player_id,"
            "    player_name=excluded.player_name,"
            "    position=excluded.position,"
            "    cost=excluded.cost",
            (
                pick.league_key,
                pick.season,
                pick.round,
                pick.pick,
                pick.team_key,
                pick.yahoo_player_key,
                pick.player_id,
                pick.player_name,
                pick.position,
                pick.cost,
            ),
        )

    def get_by_league_season(self, league_key: str, season: int) -> list[YahooDraftPick]:
        rows = self._conn.execute(
            "SELECT id, league_key, season, round, pick, team_key,"
            "    yahoo_player_key, player_id, player_name, position, cost"
            " FROM yahoo_draft_pick"
            " WHERE league_key = ? AND season = ?"
            " ORDER BY round, pick",
            (league_key, season),
        ).fetchall()
        return [self._load_pick(row) for row in rows]

    def get_pick_count(self, league_key: str, season: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM yahoo_draft_pick WHERE league_key = ? AND season = ?",
            (league_key, season),
        ).fetchone()
        assert row is not None  # noqa: S101 - type narrowing
        return row["cnt"]

    @staticmethod
    def _load_pick(row: sqlite3.Row) -> YahooDraftPick:
        return YahooDraftPick(
            id=row["id"],
            league_key=row["league_key"],
            season=row["season"],
            round=row["round"],
            pick=row["pick"],
            team_key=row["team_key"],
            yahoo_player_key=row["yahoo_player_key"],
            player_id=row["player_id"],
            player_name=row["player_name"],
            position=row["position"],
            cost=row["cost"],
        )
