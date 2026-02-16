"""Augment Marcel inputs with statcast-gbm true-talent rate estimates."""

from collections.abc import Sequence
from typing import Any

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.marcel.types import MarcelInput


def _blend(marcel_rate: float, statcast_rate: float, weight: float) -> float:
    return marcel_rate * (1 - weight) + statcast_rate * weight


def _adjust_batter_rates(
    rates: dict[str, float],
    stat_json: dict[str, Any],
    weight: float,
) -> dict[str, float]:
    """Blend statcast batter rate estimates into Marcel per-PA counting rates.

    Conversions:
    - avg (h/ab) → h per PA, using existing bb/hbp/sf rates for ab fraction
    - obp ((h+bb+hbp)/pa) → bb per PA, after h adjustment
    """
    adjusted = dict(rates)

    bb_rate = rates.get("bb", 0.0)
    hbp_rate = rates.get("hbp", 0.0)
    sf_rate = rates.get("sf", 0.0)
    ab_fraction = 1 - bb_rate - hbp_rate - sf_rate
    if ab_fraction <= 0:
        return adjusted

    if "avg" in stat_json:
        marcel_avg = rates.get("h", 0.0) / ab_fraction
        statcast_avg = float(stat_json["avg"])
        blended_avg = _blend(marcel_avg, statcast_avg, weight)
        adjusted["h"] = blended_avg * ab_fraction

    if "obp" in stat_json:
        # obp ≈ (h + bb + hbp) / pa = h_per_pa + bb_per_pa + hbp_per_pa
        current_h = adjusted.get("h", 0.0)
        marcel_obp = rates.get("h", 0.0) + bb_rate + hbp_rate
        statcast_obp = float(stat_json["obp"])
        blended_obp = _blend(marcel_obp, statcast_obp, weight)
        new_bb = blended_obp - current_h - hbp_rate
        if new_bb >= 0:
            adjusted["bb"] = new_bb

    return adjusted


def _adjust_pitcher_rates(
    rates: dict[str, float],
    stat_json: dict[str, Any],
    weight: float,
) -> dict[str, float]:
    """Blend statcast pitcher rate estimates into Marcel per-IP counting rates.

    Conversions:
    - k_per_9 → so per IP (k_per_9 / 9)
    - bb_per_9 → bb per IP (bb_per_9 / 9)
    - hr_per_9 → hr per IP (hr_per_9 / 9)
    - era → er per IP (era / 9)
    - whip → h per IP (whip - blended bb per IP)
    """
    adjusted = dict(rates)

    if "k_per_9" in stat_json:
        statcast_so = float(stat_json["k_per_9"]) / 9
        adjusted["so"] = _blend(rates.get("so", 0.0), statcast_so, weight)

    if "bb_per_9" in stat_json:
        statcast_bb = float(stat_json["bb_per_9"]) / 9
        adjusted["bb"] = _blend(rates.get("bb", 0.0), statcast_bb, weight)

    if "hr_per_9" in stat_json:
        statcast_hr = float(stat_json["hr_per_9"]) / 9
        adjusted["hr"] = _blend(rates.get("hr", 0.0), statcast_hr, weight)

    if "era" in stat_json:
        statcast_er = float(stat_json["era"]) / 9
        adjusted["er"] = _blend(rates.get("er", 0.0), statcast_er, weight)

    if "whip" in stat_json:
        statcast_h = float(stat_json["whip"]) - adjusted.get("bb", rates.get("bb", 0.0))
        if statcast_h >= 0:
            adjusted["h"] = _blend(rates.get("h", 0.0), statcast_h, weight)

    return adjusted


def augment_inputs_with_statcast(
    *,
    inputs: dict[int, MarcelInput],
    statcast_projections: Sequence[Projection],
    blend_weight: float = 0.3,
) -> dict[int, MarcelInput]:
    """Blend statcast-gbm true-talent rates into Marcel weighted rates.

    Unlike MLE augmentation (which adds sample size), statcast augmentation
    adjusts existing rates without changing weighted_pt. This reflects the
    fact that statcast provides a better estimate of the same data Marcel
    already sees, not additional data.

    Only players already in inputs are affected — statcast projections for
    unknown players are skipped.
    """
    if blend_weight == 0.0:
        return dict(inputs)

    result = dict(inputs)

    batter_projs: dict[int, dict[str, Any]] = {}
    pitcher_projs: dict[int, dict[str, Any]] = {}
    for proj in statcast_projections:
        if proj.player_type == "batter":
            batter_projs[proj.player_id] = proj.stat_json
        elif proj.player_type == "pitcher":
            pitcher_projs[proj.player_id] = proj.stat_json

    for player_id, mi in inputs.items():
        pitcher = any(s.ip > 0 for s in mi.seasons)

        if pitcher and player_id in pitcher_projs:
            new_rates = _adjust_pitcher_rates(mi.weighted_rates, pitcher_projs[player_id], blend_weight)
        elif not pitcher and player_id in batter_projs:
            new_rates = _adjust_batter_rates(mi.weighted_rates, batter_projs[player_id], blend_weight)
        else:
            continue

        result[player_id] = MarcelInput(
            weighted_rates=new_rates,
            weighted_pt=mi.weighted_pt,
            league_rates=mi.league_rates,
            age=mi.age,
            seasons=mi.seasons,
        )

    return result
