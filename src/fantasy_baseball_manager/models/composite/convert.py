from typing import Any

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.stat_utils import (
    best_rows_per_player,
    compute_batter_rates,
    compute_pitcher_rates,
)


def extract_projected_pt(rows: list[dict[str, Any]], pitcher: bool = False) -> dict[int, float]:
    """Extract projected playing time from assembler rows.

    Reads proj_pa (batters) or proj_ip (pitchers) from the best row per player.
    """
    col = "proj_ip" if pitcher else "proj_pa"
    best = best_rows_per_player(rows)
    return {pid: float(row.get(col) or 0.0) for pid, row in best.items()}


def batter_rates_to_counting(rates: dict[str, float], pa: int) -> dict[str, float]:
    """Convert batter rate stats to approximate counting stats given projected PA.

    Uses standard PA-based ratios to estimate AB, then derives counting stats.
    """
    if pa <= 0:
        return {
            "ab": 0.0,
            "h": 0.0,
            "hr": 0.0,
            "doubles": 0.0,
            "triples": 0.0,
            "bb": 0.0,
            "hbp": 0.0,
            "sf": 0.0,
            "pa": 0,
        }

    # Approximate component fractions of PA
    bb_frac = 0.085
    hbp_frac = 0.012
    sf_frac = 0.01

    bb = pa * bb_frac
    hbp = pa * hbp_frac
    sf = pa * sf_frac
    ab = pa - bb - hbp - sf

    avg = rates.get("avg", 0.0)
    slg = rates.get("slg", 0.0)

    h = avg * ab
    # iso = slg - avg; tb = slg * ab; xbh_tb = tb - h = (slg - avg) * ab
    iso = slg - avg
    xbh_tb = iso * ab  # extra-base total bases

    # Approximate XBH split: HR ~ 40%, 2B ~ 50%, 3B ~ 10% of XBH total bases
    # tb from XBH: hr contributes 4 each, 2b contributes 2, 3b contributes 3
    # Using typical ratios: hr_tb_frac=0.55, doubles_tb_frac=0.35, triples_tb_frac=0.10
    hr = xbh_tb * 0.55 / 4 if xbh_tb > 0 else 0.0
    doubles = xbh_tb * 0.35 / 2 if xbh_tb > 0 else 0.0
    triples = xbh_tb * 0.10 / 3 if xbh_tb > 0 else 0.0

    return {
        "ab": ab,
        "h": h,
        "hr": hr,
        "doubles": doubles,
        "triples": triples,
        "bb": bb,
        "hbp": hbp,
        "sf": sf,
        "pa": pa,
    }


def pitcher_rates_to_counting(rates: dict[str, float], ip: float) -> dict[str, float]:
    """Convert pitcher rate stats to counting stats given projected IP."""
    if ip <= 0:
        return {"er": 0.0, "so": 0.0, "bb": 0.0, "hr": 0.0, "h": 0.0}
    er = rates.get("era", 0.0) * ip / 9
    so = rates.get("k_per_9", 0.0) * ip / 9
    bb = rates.get("bb_per_9", 0.0) * ip / 9
    hr = rates.get("hr_per_9", 0.0) * ip / 9
    h = rates.get("whip", 0.0) * ip - bb
    return {"er": er, "so": so, "bb": bb, "hr": hr, "h": h}


def composite_projection_to_domain(
    player_id: int,
    projected_season: int,
    stats: dict[str, float],
    rates: dict[str, float],
    pt: float,
    pitcher: bool,
    version: str,
    system: str = "composite",
) -> Projection:
    """Convert composite projection data to a domain Projection."""
    stat_json: dict[str, object] = dict(stats)
    stat_json["rates"] = dict(rates)
    stat_json["_pt_system"] = "playing_time"

    if pitcher:
        stat_json["ip"] = pt
        stat_json.update(compute_pitcher_rates(stats, pt))
        player_type = "pitcher"
    else:
        stat_json["pa"] = int(pt)
        stat_json.update(compute_batter_rates(stats, int(pt)))
        player_type = "batter"

    return Projection(
        player_id=player_id,
        season=projected_season,
        system=system,
        version=version,
        player_type=player_type,
        stat_json=stat_json,
    )
