import sqlite3

from fantasy_baseball_manager.domain.league_environment import LeagueEnvironment


class SqliteLeagueEnvironmentRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, env: LeagueEnvironment) -> int:
        cursor = self._conn.execute(
            """INSERT INTO league_environment
                   (league, season, level, runs_per_game, avg, obp, slg,
                    k_pct, bb_pct, hr_per_pa, babip, loaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(league, season, level) DO UPDATE SET
                   runs_per_game=excluded.runs_per_game,
                   avg=excluded.avg,
                   obp=excluded.obp,
                   slg=excluded.slg,
                   k_pct=excluded.k_pct,
                   bb_pct=excluded.bb_pct,
                   hr_per_pa=excluded.hr_per_pa,
                   babip=excluded.babip,
                   loaded_at=excluded.loaded_at""",
            (
                env.league,
                env.season,
                env.level,
                env.runs_per_game,
                env.avg,
                env.obp,
                env.slg,
                env.k_pct,
                env.bb_pct,
                env.hr_per_pa,
                env.babip,
                env.loaded_at,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_league_season_level(self, league: str, season: int, level: str) -> LeagueEnvironment | None:
        row = self._conn.execute(
            "SELECT * FROM league_environment WHERE league = ? AND season = ? AND level = ?",
            (league, season, level),
        ).fetchone()
        return self._row_to_env(row) if row else None

    def get_by_season_level(self, season: int, level: str) -> list[LeagueEnvironment]:
        rows = self._conn.execute(
            "SELECT * FROM league_environment WHERE season = ? AND level = ?",
            (season, level),
        ).fetchall()
        return [self._row_to_env(row) for row in rows]

    def get_by_season(self, season: int) -> list[LeagueEnvironment]:
        rows = self._conn.execute(
            "SELECT * FROM league_environment WHERE season = ?",
            (season,),
        ).fetchall()
        return [self._row_to_env(row) for row in rows]

    @staticmethod
    def _row_to_env(row: sqlite3.Row) -> LeagueEnvironment:
        return LeagueEnvironment(
            id=row["id"],
            league=row["league"],
            season=row["season"],
            level=row["level"],
            runs_per_game=row["runs_per_game"],
            avg=row["avg"],
            obp=row["obp"],
            slg=row["slg"],
            k_pct=row["k_pct"],
            bb_pct=row["bb_pct"],
            hr_per_pa=row["hr_per_pa"],
            babip=row["babip"],
            loaded_at=row["loaded_at"],
        )
