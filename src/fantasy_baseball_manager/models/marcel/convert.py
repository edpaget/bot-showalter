from collections.abc import Sequence

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.marcel.types import MarcelProjection, SeasonLine


def rows_to_player_seasons(
    rows: list[dict],
    categories: Sequence[str],
    lags: int,
    pitcher: bool = False,
) -> dict[int, tuple[int, list[SeasonLine], int]]:
    """Convert assembler output rows to per-player SeasonLine lists.

    Returns {player_id: (player_id, [SeasonLine most-recent-first], age)}.
    Uses the row with the highest season per player.
    """
    best_rows: dict[int, dict] = {}
    for row in rows:
        pid = int(row["player_id"])
        if pid not in best_rows or row["season"] > best_rows[pid]["season"]:
            best_rows[pid] = row

    result: dict[int, tuple[int, list[SeasonLine], int]] = {}
    for pid, row in best_rows.items():
        age = int(row.get("age", 0))
        season_lines: list[SeasonLine] = []
        for lag_n in range(1, lags + 1):
            stats = {cat: float(row.get(f"{cat}_{lag_n}", 0.0)) for cat in categories}
            if pitcher:
                ip = float(row.get(f"ip_{lag_n}", 0.0))
                g = int(row.get(f"g_{lag_n}", 0))
                gs = int(row.get(f"gs_{lag_n}", 0))
                season_lines.append(SeasonLine(stats=stats, ip=ip, g=g, gs=gs))
            else:
                pa = int(row.get(f"pa_{lag_n}", 0))
                season_lines.append(SeasonLine(stats=stats, pa=pa))
        result[pid] = (pid, season_lines, age)

    return result


def projection_to_domain(
    proj: MarcelProjection,
    version: str,
    player_type: str,
) -> Projection:
    """Convert a MarcelProjection to a domain Projection for storage."""
    stat_json: dict[str, object] = dict(proj.stats)
    stat_json["rates"] = dict(proj.rates)
    if proj.pa > 0:
        stat_json["pa"] = proj.pa
    if proj.ip > 0:
        stat_json["ip"] = proj.ip

    return Projection(
        player_id=proj.player_id,
        season=proj.projected_season,
        system="marcel",
        version=version,
        player_type=player_type,
        stat_json=stat_json,
    )
