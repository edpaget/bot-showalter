from typing import Any

from fantasy_baseball_manager.domain.projection import Projection


def extract_projected_pt(rows: list[dict[str, Any]], pitcher: bool = False) -> dict[int, float]:
    """Extract projected playing time from assembler rows.

    Reads proj_pa (batters) or proj_ip (pitchers) from the best row per player.
    """
    col = "proj_ip" if pitcher else "proj_pa"
    best_rows: dict[int, dict[str, Any]] = {}
    for row in rows:
        pid = int(row["player_id"])
        if pid not in best_rows or row["season"] > best_rows[pid]["season"]:
            best_rows[pid] = row
    return {pid: float(row.get(col) or 0.0) for pid, row in best_rows.items()}


def composite_projection_to_domain(
    player_id: int,
    projected_season: int,
    stats: dict[str, float],
    rates: dict[str, float],
    pt: float,
    pitcher: bool,
    version: str,
) -> Projection:
    """Convert composite projection data to a domain Projection."""
    stat_json: dict[str, object] = dict(stats)
    stat_json["rates"] = dict(rates)
    stat_json["_pt_system"] = "playing_time"

    if pitcher:
        stat_json["ip"] = pt
        player_type = "pitcher"
    else:
        stat_json["pa"] = int(pt)
        player_type = "batter"

    return Projection(
        player_id=player_id,
        season=projected_season,
        system="composite",
        version=version,
        player_type=player_type,
        stat_json=stat_json,
    )
