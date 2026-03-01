from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import KeeperCost

if TYPE_CHECKING:
    import sqlite3


class SqliteKeeperCostRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert_batch(self, costs: list[KeeperCost]) -> int:
        count = 0
        for cost in costs:
            self._conn.execute(
                "INSERT INTO keeper_cost"
                "    (player_id, season, league, cost, years_remaining, source, loaded_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(player_id, season, league) DO UPDATE SET"
                "    cost=excluded.cost,"
                "    years_remaining=excluded.years_remaining,"
                "    source=excluded.source,"
                "    loaded_at=excluded.loaded_at",
                (
                    cost.player_id,
                    cost.season,
                    cost.league,
                    cost.cost,
                    cost.years_remaining,
                    cost.source,
                    cost.loaded_at,
                ),
            )
            count += 1
        return count

    def find_by_season_league(self, season: int, league: str) -> list[KeeperCost]:
        rows = self._conn.execute(
            self._select_sql() + " WHERE season = ? AND league = ?",
            (season, league),
        ).fetchall()
        return [self._row_to_keeper_cost(row) for row in rows]

    def find_by_player(self, player_id: int) -> list[KeeperCost]:
        rows = self._conn.execute(
            self._select_sql() + " WHERE player_id = ?",
            (player_id,),
        ).fetchall()
        return [self._row_to_keeper_cost(row) for row in rows]

    @staticmethod
    def _select_sql() -> str:
        return "SELECT id, player_id, season, league, cost, years_remaining, source, loaded_at FROM keeper_cost"

    @staticmethod
    def _row_to_keeper_cost(row: sqlite3.Row) -> KeeperCost:
        return KeeperCost(
            id=row["id"],
            player_id=row["player_id"],
            season=row["season"],
            league=row["league"],
            cost=row["cost"],
            years_remaining=row["years_remaining"],
            source=row["source"],
            loaded_at=row["loaded_at"],
        )
