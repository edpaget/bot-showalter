"""Shared data orchestration for CLI modules."""

from fantasy_baseball_manager.config import load_league_settings
from fantasy_baseball_manager.draft.models import RosterConfig
from fantasy_baseball_manager.draft.positions import DEFAULT_ROSTER_CONFIG, infer_pitcher_role
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.services import get_container
from fantasy_baseball_manager.valuation.models import PlayerValue
from fantasy_baseball_manager.valuation.replacement import (
    BATTER_SCARCITY_ORDER,
    PITCHER_SCARCITY_ORDER,
    ReplacementConfig,
    apply_replacement_adjustment,
    assign_positions,
    compute_replacement_levels,
)
from fantasy_baseball_manager.valuation.zscore import zscore_batting, zscore_pitching


def _apply_pool_replacement(
    players: list[PlayerValue],
    player_positions: dict[str, tuple[str, ...]],
    config: ReplacementConfig,
    scarcity_order: tuple[str, ...],
) -> list[PlayerValue]:
    """Apply replacement-level adjustment to a single player pool."""
    if not players:
        return []
    assignments = assign_positions(players, player_positions, config, scarcity_order)
    thresholds = compute_replacement_levels(players, player_positions, config, scarcity_order)
    return apply_replacement_adjustment(players, assignments, thresholds, player_positions)


def build_projections_and_positions(
    engine: str,
    year: int,
    roster_config: RosterConfig | None = None,
) -> tuple[list[PlayerValue], dict[tuple[str, str], tuple[str, ...]]]:
    """Build player values and composite positions for simulation.

    This function is shared between draft and keeper CLI modules to generate
    projections and position eligibility data.

    Args:
        engine: Projection engine name (e.g., "marcel", "enhanced").
        year: Projection year.

    Returns:
        A tuple of (player_values, composite_positions) where:
        - player_values: List of PlayerValue objects for all batters and pitchers.
        - composite_positions: Dict mapping (player_id, position_type) to eligible positions.
    """
    league_settings = load_league_settings()
    container = get_container()
    pipeline = PIPELINES[engine]()

    batting_projections = pipeline.project_batters(container.batting_source, container.team_batting_source, year)
    batting_values = zscore_batting(batting_projections, league_settings.batting_categories)
    batting_ids = {p.player_id for p in batting_projections}

    pitching_projections = pipeline.project_pitchers(container.pitching_source, container.team_pitching_source, year)
    player_positions: dict[str, tuple[str, ...]] = {}
    for proj in pitching_projections:
        if proj.player_id not in player_positions:
            role = infer_pitcher_role(proj)
            player_positions[proj.player_id] = (role,)
    pitching_values = zscore_pitching(pitching_projections, league_settings.pitching_categories)
    pitching_ids = {p.player_id for p in pitching_projections}

    all_values: list[PlayerValue] = list(batting_values) + list(pitching_values)

    # Apply replacement-level (VORP) adjustment
    _PITCHER_POSITIONS: frozenset[str] = frozenset({"SP", "RP", "P"})
    effective_roster = roster_config if roster_config is not None else DEFAULT_ROSTER_CONFIG
    replacement_config = ReplacementConfig(
        team_count=league_settings.team_count,
        roster_config=effective_roster,
    )

    # Batters: no positions available in orchestration path → pass through unchanged
    batter_values = [v for v in all_values if v.position_type == "B"]
    batter_positions: dict[str, tuple[str, ...]] = {
        pid: tuple(p for p in pos if p not in _PITCHER_POSITIONS)
        for pid, pos in player_positions.items()
        if pid in batting_ids and any(p not in _PITCHER_POSITIONS for p in pos)
    }
    adjusted_batters = _apply_pool_replacement(
        batter_values, batter_positions, replacement_config, BATTER_SCARCITY_ORDER
    )
    adjusted_batter_map = {p.player_id: p for p in adjusted_batters}

    # Pitchers: positions available from infer_pitcher_role
    pitcher_values = [v for v in all_values if v.position_type == "P"]
    pitcher_positions_map: dict[str, tuple[str, ...]] = {
        pid: tuple(p for p in pos if p in _PITCHER_POSITIONS)
        for pid, pos in player_positions.items()
        if pid in pitching_ids and any(p in _PITCHER_POSITIONS for p in pos)
    }
    adjusted_pitchers = _apply_pool_replacement(
        pitcher_values, pitcher_positions_map, replacement_config, PITCHER_SCARCITY_ORDER
    )
    adjusted_pitcher_map = {p.player_id: p for p in adjusted_pitchers}

    all_values = [
        adjusted_batter_map[v.player_id]
        if v.position_type == "B" and v.player_id in adjusted_batter_map
        else adjusted_pitcher_map[v.player_id]
        if v.position_type == "P" and v.player_id in adjusted_pitcher_map
        else v
        for v in all_values
    ]

    # Build composite-keyed positions dict
    two_way_ids = batting_ids & pitching_ids
    composite_positions: dict[tuple[str, str], tuple[str, ...]] = {}
    for pid, positions in player_positions.items():
        if pid in two_way_ids:
            batting_pos = tuple(p for p in positions if p not in _PITCHER_POSITIONS)
            pitching_pos = tuple(p for p in positions if p in _PITCHER_POSITIONS)
            if batting_pos:
                composite_positions[(pid, "B")] = batting_pos
            if pitching_pos:
                composite_positions[(pid, "P")] = pitching_pos
        elif pid in batting_ids:
            composite_positions[(pid, "B")] = positions
        elif pid in pitching_ids:
            composite_positions[(pid, "P")] = positions
        else:
            composite_positions[(pid, "B")] = positions
            composite_positions[(pid, "P")] = positions

    return all_values, composite_positions
