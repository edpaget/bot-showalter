from __future__ import annotations

from typing import Any

from fantasy_baseball_manager.features.types import Source, TransformFeature

_IN_ZONE = frozenset(range(1, 10))

_STRIKE_DESCRIPTIONS = frozenset(
    {
        "called_strike",
        "swinging_strike",
        "swinging_strike_blocked",
        "foul",
        "foul_bunt",
        "foul_tip",
        "hit_into_play",
    }
)


def command_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute pitcher command metrics from statcast pitch data."""
    pitches_with_zone = 0
    in_zone = 0
    first_pitches = 0
    first_pitch_strikes = 0

    for row in rows:
        zone = row.get("zone")
        desc = row.get("description")
        pitch_number = row.get("pitch_number")

        if zone is not None:
            pitches_with_zone += 1
            if zone in _IN_ZONE:
                in_zone += 1

        if pitch_number == 1:
            first_pitches += 1
            if desc in _STRIKE_DESCRIPTIONS:
                first_pitch_strikes += 1

    return {
        "zone_rate": (in_zone / pitches_with_zone * 100.0) if pitches_with_zone > 0 else float("nan"),
        "first_pitch_strike_pct": (first_pitch_strikes / first_pitches * 100.0) if first_pitches > 0 else float("nan"),
    }


COMMAND = TransformFeature(
    name="command",
    source=Source.STATCAST,
    columns=("zone", "description", "pitch_number"),
    group_by=("player_id", "season"),
    transform=command_profile,
    outputs=("zone_rate", "first_pitch_strike_pct"),
)
