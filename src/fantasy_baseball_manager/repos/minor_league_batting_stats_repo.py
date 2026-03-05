from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import MinorLeagueBattingStats

if TYPE_CHECKING:
    import sqlite3

    from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class SqliteMinorLeagueBattingStatsRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def upsert(self, stats: MinorLeagueBattingStats) -> int:
        with self._provider.connection() as conn:
            cursor = conn.execute(
                """INSERT INTO minor_league_batting_stats
                       (player_id, season, level, league, team,
                        g, pa, ab, h, doubles, triples, hr, r, rbi,
                        bb, so, sb, cs, avg, obp, slg, age,
                        hbp, sf, sh, loaded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(player_id, season, level, team) DO UPDATE SET
                       league=excluded.league,
                       g=excluded.g,
                       pa=excluded.pa,
                       ab=excluded.ab,
                       h=excluded.h,
                       doubles=excluded.doubles,
                       triples=excluded.triples,
                       hr=excluded.hr,
                       r=excluded.r,
                       rbi=excluded.rbi,
                       bb=excluded.bb,
                       so=excluded.so,
                       sb=excluded.sb,
                       cs=excluded.cs,
                       avg=excluded.avg,
                       obp=excluded.obp,
                       slg=excluded.slg,
                       age=excluded.age,
                       hbp=excluded.hbp,
                       sf=excluded.sf,
                       sh=excluded.sh,
                       loaded_at=excluded.loaded_at""",
                (
                    stats.player_id,
                    stats.season,
                    stats.level,
                    stats.league,
                    stats.team,
                    stats.g,
                    stats.pa,
                    stats.ab,
                    stats.h,
                    stats.doubles,
                    stats.triples,
                    stats.hr,
                    stats.r,
                    stats.rbi,
                    stats.bb,
                    stats.so,
                    stats.sb,
                    stats.cs,
                    stats.avg,
                    stats.obp,
                    stats.slg,
                    stats.age,
                    stats.hbp,
                    stats.sf,
                    stats.sh,
                    stats.loaded_at,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_by_player(self, player_id: int) -> list[MinorLeagueBattingStats]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM minor_league_batting_stats WHERE player_id = ?",
                (player_id,),
            ).fetchall()
            return [self._row_to_stats(row) for row in rows]

    def get_by_player_season(self, player_id: int, season: int) -> list[MinorLeagueBattingStats]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM minor_league_batting_stats WHERE player_id = ? AND season = ?",
                (player_id, season),
            ).fetchall()
            return [self._row_to_stats(row) for row in rows]

    def get_by_season_level(self, season: int, level: str) -> list[MinorLeagueBattingStats]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM minor_league_batting_stats WHERE season = ? AND level = ?",
                (season, level),
            ).fetchall()
            return [self._row_to_stats(row) for row in rows]

    @staticmethod
    def _row_to_stats(row: sqlite3.Row) -> MinorLeagueBattingStats:
        return MinorLeagueBattingStats(
            id=row["id"],
            player_id=row["player_id"],
            season=row["season"],
            level=row["level"],
            league=row["league"],
            team=row["team"],
            g=row["g"],
            pa=row["pa"],
            ab=row["ab"],
            h=row["h"],
            doubles=row["doubles"],
            triples=row["triples"],
            hr=row["hr"],
            r=row["r"],
            rbi=row["rbi"],
            bb=row["bb"],
            so=row["so"],
            sb=row["sb"],
            cs=row["cs"],
            avg=row["avg"],
            obp=row["obp"],
            slg=row["slg"],
            age=row["age"],
            hbp=row["hbp"],
            sf=row["sf"],
            sh=row["sh"],
            loaded_at=row["loaded_at"],
        )
