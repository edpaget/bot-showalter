from __future__ import annotations

from typing import Any

from fantasy_baseball_manager.features.types import Source, TransformFeature

_PITCH_TYPES = ("FF", "SL", "CH", "CU", "SI", "FC")


def pitch_mix_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute pitch-type usage percentages and average velocities."""
    counts: dict[str, int] = {pt: 0 for pt in _PITCH_TYPES}
    velo_sums: dict[str, float] = {pt: 0.0 for pt in _PITCH_TYPES}
    velo_counts: dict[str, int] = {pt: 0 for pt in _PITCH_TYPES}

    total = 0
    for row in rows:
        pt = row.get("pitch_type")
        if pt not in counts:
            continue
        counts[pt] += 1
        total += 1
        speed = row.get("release_speed")
        if speed is not None:
            velo_sums[pt] += speed
            velo_counts[pt] += 1

    result: dict[str, Any] = {}
    for pt in _PITCH_TYPES:
        key = pt.lower()
        result[f"{key}_pct"] = (counts[pt] / total * 100.0) if total > 0 else float("nan")
        result[f"{key}_velo"] = (velo_sums[pt] / velo_counts[pt]) if velo_counts[pt] > 0 else float("nan")
    return result


PITCH_MIX = TransformFeature(
    name="pitch_mix",
    source=Source.STATCAST,
    columns=("pitch_type", "release_speed"),
    group_by=("player_id", "season"),
    transform=pitch_mix_profile,
    outputs=tuple(f"{pt.lower()}_{suffix}" for pt in _PITCH_TYPES for suffix in ("pct", "velo")),
)
