from __future__ import annotations

from typing import Any

from fantasy_baseball_manager.features.types import Source, TransformFeature

_SWING_DESCRIPTIONS = frozenset(
    {
        "foul",
        "foul_bunt",
        "foul_tip",
        "hit_into_play",
        "swinging_strike",
        "swinging_strike_blocked",
    }
)

_MISS_DESCRIPTIONS = frozenset(
    {
        "swinging_strike",
        "swinging_strike_blocked",
    }
)

_IN_ZONE = frozenset(range(1, 10))


def plate_discipline_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute plate discipline metrics from statcast pitch data."""
    total_pitches = 0
    out_of_zone = 0
    out_of_zone_swings = 0
    in_zone_swings = 0
    in_zone_contact = 0
    total_swings = 0
    total_misses = 0
    called_strikes = 0

    for row in rows:
        zone = row.get("zone")
        if zone is None:
            continue
        desc = row.get("description")
        total_pitches += 1
        is_in_zone = zone in _IN_ZONE
        is_swing = desc in _SWING_DESCRIPTIONS
        is_miss = desc in _MISS_DESCRIPTIONS

        if not is_in_zone:
            out_of_zone += 1
            if is_swing:
                out_of_zone_swings += 1

        if is_swing:
            total_swings += 1
            if is_miss:
                total_misses += 1
            if is_in_zone:
                in_zone_swings += 1
                if not is_miss:
                    in_zone_contact += 1

        if desc == "called_strike":
            called_strikes += 1

    return {
        "chase_rate": (out_of_zone_swings / out_of_zone * 100.0) if out_of_zone > 0 else float("nan"),
        "zone_contact_pct": (in_zone_contact / in_zone_swings * 100.0) if in_zone_swings > 0 else float("nan"),
        "whiff_rate": (total_misses / total_swings * 100.0) if total_swings > 0 else float("nan"),
        "swinging_strike_pct": (total_misses / total_pitches * 100.0) if total_pitches > 0 else float("nan"),
        "called_strike_pct": (called_strikes / total_pitches * 100.0) if total_pitches > 0 else float("nan"),
    }


PLATE_DISCIPLINE = TransformFeature(
    name="plate_discipline",
    source=Source.STATCAST,
    columns=("zone", "description"),
    group_by=("player_id", "season"),
    transform=plate_discipline_profile,
    outputs=(
        "chase_rate",
        "zone_contact_pct",
        "whiff_rate",
        "swinging_strike_pct",
        "called_strike_pct",
    ),
)
