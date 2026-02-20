"""Shared stat-computation utilities used across projection models."""

from typing import Any

_WOBA_BB = 0.690
_WOBA_HBP = 0.720
_WOBA_1B = 0.880
_WOBA_2B = 1.240
_WOBA_3B = 1.560
_WOBA_HR = 2.010


def best_rows_per_player(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Deduplicate rows to one per player, keeping the latest season."""
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        pid = int(row["player_id"])
        if pid not in best or row["season"] > best[pid]["season"]:
            best[pid] = row
    return best


def compute_batter_rates(stats: dict[str, float], pa: int) -> dict[str, float]:
    """Derive batting rate stats from counting stats and PA."""
    if pa <= 0:
        return {}
    bb = stats.get("bb", 0.0)
    hbp = stats.get("hbp", 0.0)
    sf = stats.get("sf", 0.0)
    h = stats.get("h", 0.0)
    doubles = stats.get("doubles", 0.0)
    triples = stats.get("triples", 0.0)
    hr = stats.get("hr", 0.0)

    ab = pa - bb - hbp - sf
    if ab <= 0:
        return {}
    singles = h - doubles - triples - hr
    avg = h / ab
    obp = (h + bb + hbp) / (ab + bb + hbp + sf)
    slg = (singles + 2 * doubles + 3 * triples + 4 * hr) / ab

    ibb = stats.get("ibb", 0.0)
    woba_denom = ab + bb - ibb + sf + hbp
    if woba_denom > 0:
        woba = (
            _WOBA_BB * bb
            + _WOBA_HBP * hbp
            + _WOBA_1B * singles
            + _WOBA_2B * doubles
            + _WOBA_3B * triples
            + _WOBA_HR * hr
        ) / woba_denom
    else:
        woba = 0.0

    return {"ab": ab, "avg": avg, "obp": obp, "slg": slg, "ops": obp + slg, "woba": woba}


def compute_pitcher_rates(stats: dict[str, float], ip: float) -> dict[str, float]:
    """Derive pitching rate stats from counting stats and IP."""
    if ip <= 0:
        return {}
    er = stats.get("er", 0.0)
    h = stats.get("h", 0.0)
    bb = stats.get("bb", 0.0)
    so = stats.get("so", 0.0)
    return {
        "era": er * 9 / ip,
        "whip": (h + bb) / ip,
        "k_per_9": so * 9 / ip,
        "bb_per_9": bb * 9 / ip,
    }
