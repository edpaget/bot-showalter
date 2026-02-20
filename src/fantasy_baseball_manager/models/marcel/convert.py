import math
from collections.abc import Sequence

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.marcel.types import MarcelInput, MarcelProjection, SeasonLine


def extract_pt_from_rows(rows: list[dict], col: str) -> dict[int, float]:
    """Extract playing-time values from assembler rows by column name.

    Returns {player_id: pt_value} for players with valid non-zero PT.
    Players with missing, zero, or NaN PT are excluded so the caller
    can fall back to Marcel's native formula.
    """
    best_rows: dict[int, dict] = {}
    for row in rows:
        pid = int(row["player_id"])
        if pid not in best_rows or row["season"] > best_rows[pid]["season"]:
            best_rows[pid] = row
    result: dict[int, float] = {}
    for pid, row in best_rows.items():
        val = row.get(col)
        if val is not None:
            fval = float(val)
            if not math.isnan(fval) and fval > 0:
                result[pid] = fval
    return result


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
        age = int(row.get("age") or 0)
        season_lines: list[SeasonLine] = []
        for lag_n in range(1, lags + 1):
            stats = {cat: float(row.get(f"{cat}_{lag_n}") or 0.0) for cat in categories}
            if pitcher:
                ip = float(row.get(f"ip_{lag_n}") or 0.0)
                g = int(row.get(f"g_{lag_n}") or 0)
                gs = int(row.get(f"gs_{lag_n}") or 0)
                season_lines.append(SeasonLine(stats=stats, ip=ip, g=g, gs=gs))
            else:
                pa = int(row.get(f"pa_{lag_n}") or 0)
                season_lines.append(SeasonLine(stats=stats, pa=pa))
        result[pid] = (pid, season_lines, age)

    return result


def rows_to_marcel_inputs(
    rows: list[dict],
    categories: Sequence[str],
    lags: int,
    pitcher: bool = False,
) -> dict[int, MarcelInput]:
    """Convert assembler output rows to per-player MarcelInput.

    Reads pre-computed derived columns ({cat}_wavg, weighted_pt,
    league_{cat}_rate) plus builds SeasonLine lists for PT projection.
    Returns {player_id: MarcelInput}.
    """
    best_rows: dict[int, dict] = {}
    for row in rows:
        pid = int(row["player_id"])
        if pid not in best_rows or row["season"] > best_rows[pid]["season"]:
            best_rows[pid] = row

    result: dict[int, MarcelInput] = {}
    for pid, row in best_rows.items():
        age = int(row.get("age") or 0)
        weighted_rates = {cat: float(row.get(f"{cat}_wavg") or 0.0) for cat in categories}
        weighted_pt = float(row.get("weighted_pt") or 0.0)
        league_rates = {cat: float(row.get(f"league_{cat}_rate") or 0.0) for cat in categories}

        season_lines: list[SeasonLine] = []
        for lag_n in range(1, lags + 1):
            stats = {cat: float(row.get(f"{cat}_{lag_n}") or 0.0) for cat in categories}
            if pitcher:
                ip = float(row.get(f"ip_{lag_n}") or 0.0)
                g = int(row.get(f"g_{lag_n}") or 0)
                gs = int(row.get(f"gs_{lag_n}") or 0)
                season_lines.append(SeasonLine(stats=stats, ip=ip, g=g, gs=gs))
            else:
                pa = int(row.get(f"pa_{lag_n}") or 0)
                season_lines.append(SeasonLine(stats=stats, pa=pa))

        result[pid] = MarcelInput(
            weighted_rates=weighted_rates,
            weighted_pt=weighted_pt,
            league_rates=league_rates,
            age=age,
            seasons=tuple(season_lines),
        )

    return result


_WOBA_BB = 0.690
_WOBA_HBP = 0.720
_WOBA_1B = 0.880
_WOBA_2B = 1.240
_WOBA_3B = 1.560
_WOBA_HR = 2.010


def _compute_batter_rates(stats: dict[str, float], pa: int) -> dict[str, float]:
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


def _compute_pitcher_rates(stats: dict[str, float], ip: float) -> dict[str, float]:
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


def projection_to_domain(
    proj: MarcelProjection,
    version: str,
    player_type: str,
) -> Projection:
    """Convert a MarcelProjection to a domain Projection for storage."""
    stat_json: dict[str, object] = dict(proj.stats)
    if proj.pa > 0:
        stat_json["pa"] = proj.pa
        stat_json.update(_compute_batter_rates(proj.stats, proj.pa))
    if proj.ip > 0:
        stat_json["ip"] = proj.ip
        stat_json.update(_compute_pitcher_rates(proj.stats, proj.ip))

    return Projection(
        player_id=proj.player_id,
        season=proj.projected_season,
        system="marcel",
        version=version,
        player_type=player_type,
        stat_json=stat_json,
    )
