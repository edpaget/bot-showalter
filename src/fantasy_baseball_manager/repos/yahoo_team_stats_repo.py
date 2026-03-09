import json
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import TeamSeasonStats

if TYPE_CHECKING:
    import sqlite3

    from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class SqliteYahooTeamStatsRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def upsert(self, stats: TeamSeasonStats) -> int:
        with self._provider.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO yahoo_team_season_stats"
                "    (team_key, league_key, season, team_name, final_rank, stat_values_json)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(team_key, league_key, season) DO UPDATE SET"
                "    team_name=excluded.team_name,"
                "    final_rank=excluded.final_rank,"
                "    stat_values_json=excluded.stat_values_json",
                (
                    stats.team_key,
                    stats.league_key,
                    stats.season,
                    stats.team_name,
                    stats.final_rank,
                    json.dumps(stats.stat_values),
                ),
            )
            return cursor.lastrowid

    def get_by_league_season(self, league_key: str, season: int) -> list[TeamSeasonStats]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                self._select_sql() + " WHERE league_key = ? AND season = ? ORDER BY final_rank",
                (league_key, season),
            ).fetchall()
            return [self._row_to_stats(row) for row in rows]

    def get_all_seasons(self, league_key: str) -> list[TeamSeasonStats]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                self._select_sql() + " WHERE league_key = ? ORDER BY season, final_rank",
                (league_key,),
            ).fetchall()
            return [self._row_to_stats(row) for row in rows]

    @staticmethod
    def _select_sql() -> str:
        return (
            "SELECT id, team_key, league_key, season, team_name, final_rank, stat_values_json"
            " FROM yahoo_team_season_stats"
        )

    @staticmethod
    def _row_to_stats(row: sqlite3.Row) -> TeamSeasonStats:
        return TeamSeasonStats(
            id=row["id"],
            team_key=row["team_key"],
            league_key=row["league_key"],
            season=row["season"],
            team_name=row["team_name"],
            final_rank=row["final_rank"],
            stat_values=json.loads(row["stat_values_json"]),
        )
