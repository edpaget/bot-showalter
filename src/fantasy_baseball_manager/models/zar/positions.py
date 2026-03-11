from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import Position, consolidate_outfield, position_from_raw

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import LeagueSettings, PositionAppearance


def _normalize_position(raw: str) -> str | None:
    """Convert a raw position string to a league-slot key (uppercase).

    Returns None for positions that don't map to league slots.
    """
    try:
        pos = position_from_raw(raw)
    except ValueError:
        return None
    pos = consolidate_outfield(pos)
    if pos == Position.DH:
        return Position.UTIL.value
    return pos.value


def build_position_map(
    appearances: list[PositionAppearance],
    league: LeagueSettings,
    min_games: int = 1,
) -> dict[int, list[str]]:
    """Map player IDs to lists of eligible league-settings position keys."""
    valid_positions = set(league.positions.keys())
    if league.roster_util > 0:
        valid_positions.add("UTIL")

    result: dict[int, list[str]] = {}
    for app in appearances:
        if app.games < min_games:
            continue
        league_pos = _normalize_position(app.position)
        if league_pos and league_pos in valid_positions:
            result.setdefault(app.player_id, [])
            if league_pos not in result[app.player_id]:
                result[app.player_id].append(league_pos)

    # Every batter with position data qualifies for util if util slots exist.
    if league.roster_util > 0:
        for positions in result.values():
            if "UTIL" not in positions:
                positions.append("UTIL")

    return result


def build_roster_spots(
    league: LeagueSettings,
    *,
    pitcher_roster_spots: dict[str, int] | None = None,
) -> dict[str, int]:
    """Build the roster-spots dict from league settings, with optional pitcher override."""
    if pitcher_roster_spots is not None:
        return pitcher_roster_spots
    roster_spots = dict(league.positions)
    if league.roster_util > 0:
        roster_spots["UTIL"] = league.roster_util
    return roster_spots


_FLEX_POSITIONS = frozenset({"P", "UTIL"})


def best_position(eligible: list[str], replacement: dict[str, float]) -> str:
    """Pick the best position for display, preferring specific over flex.

    Specific positions (SP, RP, C, 1B, etc.) are preferred over flex slots
    (P, UTIL) when available, since flex slots exist to provide roster
    flexibility, not to describe a player's role.  Dollar values are computed
    independently by the pipeline — this only affects the position label.
    """
    if not eligible:
        return "UTIL"
    specific = [p for p in eligible if p not in _FLEX_POSITIONS]
    candidates = specific if specific else eligible
    return min(candidates, key=lambda p: replacement.get(p, float("inf")))
