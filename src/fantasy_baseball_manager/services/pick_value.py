from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PickValue, PickValueCurve

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import ADP, LeagueSettings, Valuation

_SMOOTHING_WINDOW = 5


def compute_pick_value_curve(
    adp: list[ADP],
    valuations: list[Valuation],
    league: LeagueSettings,
    player_names: dict[int, str] | None = None,
) -> PickValueCurve:
    """Build a smooth, monotonically non-increasing pick value curve.

    Joins ADP entries to valuations by player_id, then interpolates gaps,
    applies rolling-average smoothing, and enforces monotonicity.
    """
    total_picks = league.teams * (league.roster_batters + league.roster_pitchers)

    # Extract metadata from the first ADP/valuation entry
    season = adp[0].season if adp else 0
    provider = adp[0].provider if adp else ""
    system = valuations[0].system if valuations else ""

    # Build valuation lookup (use max value if multiple per player)
    val_by_id: dict[int, float] = {}
    for v in valuations:
        if v.player_id not in val_by_id or v.value > val_by_id[v.player_id]:
            val_by_id[v.player_id] = v.value

    # Build raw pick→(values, names) mapping from ADP-valuation join
    raw_values: dict[int, list[float]] = {}
    raw_names: dict[int, list[str]] = {}
    for entry in adp:
        if entry.player_id not in val_by_id:
            continue
        pick_num = round(entry.overall_pick)
        if pick_num < 1:
            continue
        raw_values.setdefault(pick_num, []).append(val_by_id[entry.player_id])
        if player_names and entry.player_id in player_names:
            raw_names.setdefault(pick_num, []).append(player_names[entry.player_id])

    # Average multiple players at the same pick
    averaged: dict[int, float] = {}
    for pick_num, values in raw_values.items():
        averaged[pick_num] = sum(values) / len(values)

    # Track which picks have direct data vs interpolated vs extrapolated
    direct_picks: set[int] = set(averaged.keys())

    # Interpolate gaps between known points
    filled = _interpolate(averaged, total_picks)

    # Track interpolated picks (between known points, not direct)
    interpolated_picks: set[int] = set()
    if direct_picks:
        min_direct = min(direct_picks)
        max_direct = max(direct_picks)
        for pick_num in range(min_direct, min(max_direct + 1, total_picks + 1)):
            if pick_num not in direct_picks:
                interpolated_picks.add(pick_num)

    # Apply rolling-average smoothing
    smoothed = _smooth(filled, total_picks, _SMOOTHING_WINDOW)

    # Enforce monotonicity: walk from pick 1 to N
    mono = _enforce_monotonicity(smoothed, total_picks)

    # Build PickValue list
    picks: list[PickValue] = []
    for pick_num in range(1, total_picks + 1):
        value = mono.get(pick_num, 0.0)

        # Determine confidence
        if pick_num in direct_picks:
            confidence = "high"
        elif pick_num in interpolated_picks:
            confidence = "medium"
        else:
            confidence = "low"

        # Determine player name (use first name from direct data if available)
        name: str | None = None
        if pick_num in raw_names:
            name = raw_names[pick_num][0]

        picks.append(
            PickValue(
                pick=pick_num,
                expected_value=round(value, 2),
                player_name=name,
                confidence=confidence,
            )
        )

    return PickValueCurve(
        season=season,
        provider=provider,
        system=system,
        picks=picks,
        total_picks=total_picks,
    )


def _interpolate(known: dict[int, float], total_picks: int) -> dict[int, float]:
    """Fill gaps between known pick values with linear interpolation."""
    if not known:
        return {i: 0.0 for i in range(1, total_picks + 1)}

    result = dict(known)
    sorted_picks = sorted(known.keys())

    # Fill before the first known pick (extrapolate flat)
    if sorted_picks[0] > 1:
        for pick_num in range(1, sorted_picks[0]):
            result[pick_num] = known[sorted_picks[0]]

    # Interpolate between known points
    for i in range(len(sorted_picks) - 1):
        p1, p2 = sorted_picks[i], sorted_picks[i + 1]
        v1, v2 = known[p1], known[p2]
        for pick_num in range(p1 + 1, p2):
            fraction = (pick_num - p1) / (p2 - p1)
            result[pick_num] = v1 + fraction * (v2 - v1)

    # Extrapolate after the last known pick (linear decay toward 0)
    if sorted_picks[-1] < total_picks:
        last_pick = sorted_picks[-1]
        last_val = known[last_pick]
        remaining = total_picks - last_pick
        for pick_num in range(last_pick + 1, total_picks + 1):
            fraction = (pick_num - last_pick) / (remaining + 1)
            result[pick_num] = max(0.0, last_val * (1 - fraction))

    return result


def _smooth(values: dict[int, float], total_picks: int, window: int) -> dict[int, float]:
    """Apply rolling-average smoothing."""
    result: dict[int, float] = {}
    half = window // 2
    for pick_num in range(1, total_picks + 1):
        start = max(1, pick_num - half)
        end = min(total_picks, pick_num + half)
        window_vals = [values[i] for i in range(start, end + 1) if i in values]
        result[pick_num] = sum(window_vals) / len(window_vals) if window_vals else 0.0
    return result


def _enforce_monotonicity(values: dict[int, float], total_picks: int) -> dict[int, float]:
    """Ensure values are monotonically non-increasing from pick 1 to N."""
    result: dict[int, float] = {}
    prev = float("inf")
    for pick_num in range(1, total_picks + 1):
        val = values.get(pick_num, 0.0)
        val = min(val, prev)
        result[pick_num] = val
        prev = val
    return result
