from __future__ import annotations

from typing import Any

from fantasy_baseball_manager.features.types import Source, TransformFeature

_TRACKED_PITCH_TYPES = ("FF", "SL", "CU")


def spin_profile_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute spin rate and movement profile metrics from statcast pitch data."""
    total_spin = 0.0
    spin_count = 0
    type_spin_sums: dict[str, float] = {pt: 0.0 for pt in _TRACKED_PITCH_TYPES}
    type_spin_counts: dict[str, int] = {pt: 0 for pt in _TRACKED_PITCH_TYPES}
    total_h_break = 0.0
    total_v_break = 0.0
    break_count = 0

    for row in rows:
        spin = row.get("release_spin_rate")
        if spin is not None:
            total_spin += spin
            spin_count += 1
            pt = row.get("pitch_type")
            if pt in type_spin_sums:
                type_spin_sums[pt] += spin
                type_spin_counts[pt] += 1

        pfx_x = row.get("pfx_x")
        pfx_z = row.get("pfx_z")
        if pfx_x is not None and pfx_z is not None:
            total_h_break += pfx_x
            total_v_break += pfx_z
            break_count += 1

    result: dict[str, Any] = {
        "avg_spin_rate": (total_spin / spin_count) if spin_count > 0 else 0.0,
    }
    for pt in _TRACKED_PITCH_TYPES:
        key = pt.lower()
        count = type_spin_counts[pt]
        result[f"{key}_spin"] = (type_spin_sums[pt] / count) if count > 0 else 0.0

    result["avg_h_break"] = (total_h_break / break_count) if break_count > 0 else 0.0
    result["avg_v_break"] = (total_v_break / break_count) if break_count > 0 else 0.0

    return result


SPIN_PROFILE = TransformFeature(
    name="spin_profile",
    source=Source.STATCAST,
    columns=("release_spin_rate", "pitch_type", "pfx_x", "pfx_z"),
    group_by=("player_id", "season"),
    transform=spin_profile_metrics,
    outputs=(
        "avg_spin_rate",
        "ff_spin",
        "sl_spin",
        "cu_spin",
        "avg_h_break",
        "avg_v_break",
    ),
)
