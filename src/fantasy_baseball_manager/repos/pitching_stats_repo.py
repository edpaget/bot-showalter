import sqlite3

from fantasy_baseball_manager.domain.pitching_stats import PitchingStats


class SqlitePitchingStatsRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, stats: PitchingStats) -> int:
        cursor = self._conn.execute(
            """INSERT INTO pitching_stats
                   (player_id, season, team_id, source,
                    w, l, era, g, gs, sv, hld, ip,
                    h, er, hr, bb, so, whip,
                    k_per_9, bb_per_9, fip, xfip, war, loaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(player_id, season, source) DO UPDATE SET
                   team_id=excluded.team_id,
                   w=excluded.w, l=excluded.l, era=excluded.era,
                   g=excluded.g, gs=excluded.gs, sv=excluded.sv, hld=excluded.hld,
                   ip=excluded.ip, h=excluded.h, er=excluded.er, hr=excluded.hr,
                   bb=excluded.bb, so=excluded.so, whip=excluded.whip,
                   k_per_9=excluded.k_per_9, bb_per_9=excluded.bb_per_9,
                   fip=excluded.fip, xfip=excluded.xfip,
                   war=excluded.war, loaded_at=excluded.loaded_at""",
            (
                stats.player_id,
                stats.season,
                stats.team_id,
                stats.source,
                stats.w,
                stats.l,
                stats.era,
                stats.g,
                stats.gs,
                stats.sv,
                stats.hld,
                stats.ip,
                stats.h,
                stats.er,
                stats.hr,
                stats.bb,
                stats.so,
                stats.whip,
                stats.k_per_9,
                stats.bb_per_9,
                stats.fip,
                stats.xfip,
                stats.war,
                stats.loaded_at,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_player_season(self, player_id: int, season: int, source: str | None = None) -> list[PitchingStats]:
        if source is not None:
            rows = self._conn.execute(
                "SELECT * FROM pitching_stats WHERE player_id = ? AND season = ? AND source = ?",
                (player_id, season, source),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM pitching_stats WHERE player_id = ? AND season = ?",
                (player_id, season),
            ).fetchall()
        return [self._row_to_stats(row) for row in rows]

    def get_by_season(self, season: int, source: str | None = None) -> list[PitchingStats]:
        if source is not None:
            rows = self._conn.execute(
                "SELECT * FROM pitching_stats WHERE season = ? AND source = ?",
                (season, source),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM pitching_stats WHERE season = ?",
                (season,),
            ).fetchall()
        return [self._row_to_stats(row) for row in rows]

    @staticmethod
    def _row_to_stats(row: tuple) -> PitchingStats:
        return PitchingStats(
            id=row[0],
            player_id=row[1],
            season=row[2],
            team_id=row[3],
            source=row[4],
            w=row[5],
            l=row[6],
            era=row[7],
            g=row[8],
            gs=row[9],
            sv=row[10],
            hld=row[11],
            ip=row[12],
            h=row[13],
            er=row[14],
            hr=row[15],
            bb=row[16],
            so=row[17],
            whip=row[18],
            k_per_9=row[19],
            bb_per_9=row[20],
            fip=row[21],
            xfip=row[22],
            war=row[23],
            loaded_at=row[24],
        )
