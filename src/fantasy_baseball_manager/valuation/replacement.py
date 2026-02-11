from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.draft.models import RosterConfig
    from fantasy_baseball_manager.valuation.models import PlayerValue

BATTER_SCARCITY_ORDER = ("C", "SS", "2B", "3B", "1B", "OF")
PITCHER_SCARCITY_ORDER = ("SP", "RP")


@dataclass(frozen=True)
class PositionThreshold:
    position: str
    roster_spots: int
    replacement_rank: int
    replacement_value: float


@dataclass(frozen=True)
class ReplacementConfig:
    team_count: int
    roster_config: RosterConfig
    smoothing_window: int = 5


def _roster_spots_for_position(position: str, config: ReplacementConfig) -> int:
    for slot in config.roster_config.slots:
        if slot.position == position:
            return config.team_count * slot.count
    return 0


def assign_positions(
    players: list[PlayerValue],
    player_positions: dict[str, tuple[str, ...]],
    config: ReplacementConfig,
    scarcity_order: tuple[str, ...],
) -> dict[str, str]:
    """Greedy scarcity-first assignment. Returns player_id -> position.
    After scarcity positions, fills Util from remaining batters if configured.
    Unrosterable players omitted from result."""
    assigned: dict[str, str] = {}
    assigned_ids: set[str] = set()

    for position in scarcity_order:
        spots = _roster_spots_for_position(position, config)
        if spots == 0:
            continue
        eligible = [
            p
            for p in sorted(players, key=lambda p: p.total_value, reverse=True)
            if p.player_id not in assigned_ids
            and p.player_id in player_positions
            and position in player_positions[p.player_id]
        ]
        for p in eligible[:spots]:
            assigned[p.player_id] = position
            assigned_ids.add(p.player_id)

    util_spots = _roster_spots_for_position("Util", config)
    if util_spots > 0:
        remaining = [
            p
            for p in sorted(players, key=lambda p: p.total_value, reverse=True)
            if p.player_id not in assigned_ids and p.player_id in player_positions
        ]
        for p in remaining[:util_spots]:
            assigned[p.player_id] = "Util"
            assigned_ids.add(p.player_id)

    return assigned


def _build_position_pool(
    position: str,
    players: list[PlayerValue],
    player_positions: dict[str, tuple[str, ...]],
    assignments: dict[str, str],
) -> list[PlayerValue]:
    """Build the pool for a position: players assigned here + unassigned players eligible here.
    Players assigned to other positions are excluded."""
    pool: list[PlayerValue] = []
    for p in players:
        if p.player_id not in player_positions:
            continue
        assigned_pos = assignments.get(p.player_id)
        if assigned_pos == position or (
            assigned_pos is None and position in player_positions[p.player_id]
        ):
            pool.append(p)
    pool.sort(key=lambda p: p.total_value, reverse=True)
    return pool


def _compute_smoothed_value(
    pool: list[PlayerValue], threshold_index: int, window: int
) -> float:
    if not pool:
        return 0.0
    if threshold_index >= len(pool):
        return pool[-1].total_value
    half = window // 2
    start = max(0, threshold_index - half)
    end = min(len(pool), threshold_index + half + 1)
    values = [p.total_value for p in pool[start:end]]
    return sum(values) / len(values)


def compute_replacement_levels(
    players: list[PlayerValue],
    player_positions: dict[str, tuple[str, ...]],
    config: ReplacementConfig,
    scarcity_order: tuple[str, ...],
) -> list[PositionThreshold]:
    """Calls assign_positions(), then computes smoothed replacement threshold
    for each position (plus Util if in roster config)."""
    assignments = assign_positions(players, player_positions, config, scarcity_order)

    positions_to_compute = list(scarcity_order)
    if _roster_spots_for_position("Util", config) > 0:
        positions_to_compute.append("Util")

    thresholds: list[PositionThreshold] = []
    for position in positions_to_compute:
        roster_spots = _roster_spots_for_position(position, config)
        if roster_spots == 0:
            continue
        if position == "Util":
            pool = _build_util_pool(players, player_positions, assignments)
        else:
            pool = _build_position_pool(
                position, players, player_positions, assignments
            )
        replacement_value = _compute_smoothed_value(
            pool, roster_spots, config.smoothing_window
        )
        thresholds.append(
            PositionThreshold(
                position=position,
                roster_spots=roster_spots,
                replacement_rank=roster_spots,
                replacement_value=replacement_value,
            )
        )
    return thresholds


def _build_util_pool(
    players: list[PlayerValue],
    player_positions: dict[str, tuple[str, ...]],
    assignments: dict[str, str],
) -> list[PlayerValue]:
    """Build the Util pool: players assigned to Util + unassigned players with any position."""
    pool: list[PlayerValue] = []
    for p in players:
        if p.player_id not in player_positions:
            continue
        assigned_pos = assignments.get(p.player_id)
        if assigned_pos == "Util" or assigned_pos is None:
            pool.append(p)
    pool.sort(key=lambda p: p.total_value, reverse=True)
    return pool
