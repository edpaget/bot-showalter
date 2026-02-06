"""Shared data orchestration for CLI modules."""

from fantasy_baseball_manager.config import load_league_settings
from fantasy_baseball_manager.draft.positions import infer_pitcher_role
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.services import get_container
from fantasy_baseball_manager.valuation.models import PlayerValue
from fantasy_baseball_manager.valuation.zscore import zscore_batting, zscore_pitching


def build_projections_and_positions(
    engine: str,
    year: int,
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

    # Build composite-keyed positions dict
    _PITCHER_POSITIONS: frozenset[str] = frozenset({"SP", "RP", "P"})
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
