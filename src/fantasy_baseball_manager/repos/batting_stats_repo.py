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
        self._conn.commit()
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
    def _row_to_stats(row: tuple) -> BattingStats:
        return BattingStats(
            id=row[0],
            player_id=row[1],
            season=row[2],
            team_id=row[3],
            source=row[4],
            pa=row[5],
            ab=row[6],
            h=row[7],
            doubles=row[8],
            triples=row[9],
            hr=row[10],
            rbi=row[11],
            r=row[12],
            sb=row[13],
            cs=row[14],
            bb=row[15],
            so=row[16],
            hbp=row[17],
            sf=row[18],
            sh=row[19],
            gdp=row[20],
            ibb=row[21],
            avg=row[22],
            obp=row[23],
            slg=row[24],
            ops=row[25],
            woba=row[26],
            wrc_plus=row[27],
            war=row[28],
            loaded_at=row[29],
        )
