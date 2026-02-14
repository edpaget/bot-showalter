import sqlite3

from fantasy_baseball_manager.domain.batting_stats import BattingStats


class SqliteBattingStatsRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, stats: BattingStats) -> int:
        cursor = self._conn.execute(
            """INSERT INTO batting_stats
                   (player_id, season, team_id, source,
                    pa, ab, h, doubles, triples, hr, rbi, r,
                    sb, cs, bb, so, hbp, sf, sh, gdp, ibb,
                    avg, obp, slg, ops, woba, wrc_plus, war, loaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(player_id, season, source) DO UPDATE SET
                   team_id=excluded.team_id,
                   pa=excluded.pa, ab=excluded.ab, h=excluded.h,
                   doubles=excluded.doubles, triples=excluded.triples,
                   hr=excluded.hr, rbi=excluded.rbi, r=excluded.r,
                   sb=excluded.sb, cs=excluded.cs, bb=excluded.bb,
                   so=excluded.so, hbp=excluded.hbp, sf=excluded.sf,
                   sh=excluded.sh, gdp=excluded.gdp, ibb=excluded.ibb,
                   avg=excluded.avg, obp=excluded.obp, slg=excluded.slg,
                   ops=excluded.ops, woba=excluded.woba, wrc_plus=excluded.wrc_plus,
                   war=excluded.war, loaded_at=excluded.loaded_at""",
            (
                stats.player_id,
                stats.season,
                stats.team_id,
                stats.source,
                stats.pa,
                stats.ab,
                stats.h,
                stats.doubles,
                stats.triples,
                stats.hr,
                stats.rbi,
                stats.r,
                stats.sb,
                stats.cs,
                stats.bb,
                stats.so,
                stats.hbp,
                stats.sf,
                stats.sh,
                stats.gdp,
                stats.ibb,
                stats.avg,
                stats.obp,
                stats.slg,
                stats.ops,
                stats.woba,
                stats.wrc_plus,
                stats.war,
                stats.loaded_at,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_player_season(self, player_id: int, season: int, source: str | None = None) -> list[BattingStats]:
        if source is not None:
            rows = self._conn.execute(
                "SELECT * FROM batting_stats WHERE player_id = ? AND season = ? AND source = ?",
                (player_id, season, source),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM batting_stats WHERE player_id = ? AND season = ?",
                (player_id, season),
            ).fetchall()
        return [self._row_to_stats(row) for row in rows]

    def get_by_season(self, season: int, source: str | None = None) -> list[BattingStats]:
        if source is not None:
            rows = self._conn.execute(
                "SELECT * FROM batting_stats WHERE season = ? AND source = ?",
                (season, source),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM batting_stats WHERE season = ?",
                (season,),
            ).fetchall()
        return [self._row_to_stats(row) for row in rows]

    @staticmethod
    def _row_to_stats(row: sqlite3.Row) -> BattingStats:
        return BattingStats(
            id=row["id"],
            player_id=row["player_id"],
            season=row["season"],
            team_id=row["team_id"],
            source=row["source"],
            pa=row["pa"],
            ab=row["ab"],
            h=row["h"],
            doubles=row["doubles"],
            triples=row["triples"],
            hr=row["hr"],
            rbi=row["rbi"],
            r=row["r"],
            sb=row["sb"],
            cs=row["cs"],
            bb=row["bb"],
            so=row["so"],
            hbp=row["hbp"],
            sf=row["sf"],
            sh=row["sh"],
            gdp=row["gdp"],
            ibb=row["ibb"],
            avg=row["avg"],
            obp=row["obp"],
            slg=row["slg"],
            ops=row["ops"],
            woba=row["woba"],
            wrc_plus=row["wrc_plus"],
            war=row["war"],
            loaded_at=row["loaded_at"],
        )
