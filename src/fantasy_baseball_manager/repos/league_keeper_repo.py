from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import LeagueKeeper, PlayerType

if TYPE_CHECKING:
    import sqlite3

    from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class SqliteLeagueKeeperRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def upsert_batch(self, keepers: list[LeagueKeeper]) -> int:
        with self._provider.connection() as conn:
            count = 0
            for keeper in keepers:
                conn.execute(
                    "INSERT INTO league_keeper"
                    "    (player_id, season, league, team_name, cost, source, player_type)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(player_id, season, league, player_type) DO UPDATE SET"
                    "    team_name=excluded.team_name,"
                    "    cost=excluded.cost,"
                    "    source=excluded.source",
                    (
                        keeper.player_id,
                        keeper.season,
                        keeper.league,
                        keeper.team_name,
                        keeper.cost,
                        keeper.source,
                        keeper.player_type or "",
                    ),
                )
                count += 1
            conn.commit()
            return count

    def find_by_season_league(self, season: int, league: str) -> list[LeagueKeeper]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                self._select_sql() + " WHERE season = ? AND league = ?",
                (season, league),
            ).fetchall()
            return [self._row_to_league_keeper(row) for row in rows]

    def find_by_team(self, season: int, league: str, team_name: str) -> list[LeagueKeeper]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                self._select_sql() + " WHERE season = ? AND league = ? AND team_name = ?",
                (season, league, team_name),
            ).fetchall()
            return [self._row_to_league_keeper(row) for row in rows]

    def delete_by_season_league(self, season: int, league: str) -> int:
        with self._provider.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM league_keeper WHERE season = ? AND league = ?",
                (season, league),
            )
            conn.commit()
            return cursor.rowcount

    @staticmethod
    def _select_sql() -> str:
        return "SELECT id, player_id, season, league, team_name, cost, source, player_type FROM league_keeper"

    @staticmethod
    def _row_to_league_keeper(row: sqlite3.Row) -> LeagueKeeper:
        return LeagueKeeper(
            id=row["id"],
            player_id=row["player_id"],
            season=row["season"],
            league=row["league"],
            team_name=row["team_name"],
            cost=row["cost"],
            source=row["source"],
            player_type=PlayerType(row["player_type"]) if row["player_type"] else None,
        )
