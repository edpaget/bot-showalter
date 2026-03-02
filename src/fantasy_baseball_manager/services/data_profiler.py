import statistics
from collections import defaultdict
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import ColumnProfile

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence

NUMERIC_COLUMNS: tuple[str, ...] = (
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "barrel",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "estimated_slg_using_speedangle",
    "hc_x",
    "hc_y",
    "release_extension",
)

_VALID_PLAYER_TYPES = {"batter", "pitcher"}


class StatcastColumnProfiler:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def profile_columns(
        self,
        columns: Sequence[str],
        seasons: Sequence[int],
        player_type: str,
    ) -> list[ColumnProfile]:
        """Profile the given columns, returning one ColumnProfile per (column, season)."""
        if player_type not in _VALID_PLAYER_TYPES:
            msg = f"player_type must be 'batter' or 'pitcher', got '{player_type}'"
            raise ValueError(msg)

        invalid = [c for c in columns if c not in NUMERIC_COLUMNS]
        if invalid:
            msg = f"Invalid column(s): {', '.join(invalid)}. Must be one of {NUMERIC_COLUMNS}"
            raise ValueError(msg)

        player_col = "batter_id" if player_type == "batter" else "pitcher_id"
        results: list[ColumnProfile] = []

        for column in columns:
            rows = self._fetch_aggregated(column, player_col, seasons)
            by_season: dict[int, list[tuple[float | None, int]]] = defaultdict(list)
            for _player_id, season, avg_val, pitch_count in rows:
                by_season[season].append((avg_val, pitch_count))

            for season in sorted(seasons):
                season_data = by_season.get(season, [])
                results.append(self._compute_profile(column, season, player_type, season_data))

        return results

    def _fetch_aggregated(
        self,
        column: str,
        player_col: str,
        seasons: Sequence[int],
    ) -> list[tuple[int, int, float | None, int]]:
        """Fetch per-player-season aggregated values."""
        placeholders = ",".join("?" for _ in seasons)
        sql = f"""
            SELECT {player_col} AS player_id,
                   CAST(SUBSTR(game_date, 1, 4) AS INTEGER) AS season,
                   AVG({column}) AS avg_val,
                   COUNT(*) AS pitch_count
            FROM statcast_pitch
            WHERE CAST(SUBSTR(game_date, 1, 4) AS INTEGER) IN ({placeholders})
            GROUP BY {player_col}, season
        """  # noqa: S608
        cursor = self._conn.execute(sql, list(seasons))
        return cursor.fetchall()

    @staticmethod
    def _compute_profile(
        column: str,
        season: int,
        player_type: str,
        data: list[tuple[float | None, int]],
    ) -> ColumnProfile:
        """Compute distribution statistics from aggregated player-season values."""
        non_null = [val for val, _ in data if val is not None]
        null_count = len(data) - len(non_null)
        total = len(data)
        null_pct = (null_count / total * 100) if total > 0 else 0.0

        if not non_null:
            return ColumnProfile(
                column=column,
                season=season,
                player_type=player_type,
                count=0,
                null_count=null_count,
                null_pct=null_pct,
                mean=0.0,
                median=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                p10=0.0,
                p25=0.0,
                p75=0.0,
                p90=0.0,
                skewness=0.0,
            )

        count = len(non_null)
        mean = statistics.mean(non_null)
        median = statistics.median(non_null)
        std = statistics.stdev(non_null) if count > 1 else 0.0
        min_val = min(non_null)
        max_val = max(non_null)

        if count >= 2:
            quartiles = statistics.quantiles(non_null, n=4)
            deciles = statistics.quantiles(non_null, n=10)
            p10 = deciles[0]
            p25 = quartiles[0]
            p75 = quartiles[2]
            p90 = deciles[8]
        else:
            p10 = p25 = p75 = p90 = non_null[0]

        skewness = _compute_skewness(non_null, mean, std)

        return ColumnProfile(
            column=column,
            season=season,
            player_type=player_type,
            count=count,
            null_count=null_count,
            null_pct=null_pct,
            mean=mean,
            median=median,
            std=std,
            min=min_val,
            max=max_val,
            p10=p10,
            p25=p25,
            p75=p75,
            p90=p90,
            skewness=skewness,
        )


def _compute_skewness(values: list[float], mean: float, std: float) -> float:
    """Compute Fisher's skewness: mean((x - mean)^3) / std^3."""
    if std == 0.0 or len(values) < 3:
        return 0.0
    n = len(values)
    m3 = sum((x - mean) ** 3 for x in values) / n
    return m3 / (std**3)
