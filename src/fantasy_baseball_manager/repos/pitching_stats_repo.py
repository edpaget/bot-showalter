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
    def _row_to_stats(row: sqlite3.Row) -> PitchingStats:
        return PitchingStats(
            id=row["id"],
            player_id=row["player_id"],
            season=row["season"],
            team_id=row["team_id"],
            source=row["source"],
            w=row["w"],
            l=row["l"],
            era=row["era"],
            g=row["g"],
            gs=row["gs"],
            sv=row["sv"],
            hld=row["hld"],
            ip=row["ip"],
            h=row["h"],
            er=row["er"],
            hr=row["hr"],
            bb=row["bb"],
            so=row["so"],
            whip=row["whip"],
            k_per_9=row["k_per_9"],
            bb_per_9=row["bb_per_9"],
            fip=row["fip"],
            xfip=row["xfip"],
            war=row["war"],
            loaded_at=row["loaded_at"],
        )
