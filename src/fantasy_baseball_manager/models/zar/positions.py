from fantasy_baseball_manager.domain.league_settings import LeagueSettings
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance

# Maps PositionAppearance position codes to league-settings keys.
POSITION_ALIASES: dict[str, str] = {
    "C": "c",
    "1B": "first_base",
    "2B": "second_base",
    "3B": "third_base",
    "SS": "ss",
    "LF": "of",
    "CF": "of",
    "RF": "of",
    "OF": "of",
    "DH": "util",
}


def build_position_map(
    appearances: list[PositionAppearance],
    league: LeagueSettings,
) -> dict[int, list[str]]:
    """Map player IDs to lists of eligible league-settings position keys."""
    valid_positions = set(league.positions.keys())
    if league.roster_util > 0:
        valid_positions.add("util")

    result: dict[int, list[str]] = {}
    for app in appearances:
        league_pos = POSITION_ALIASES.get(app.position)
        if league_pos and league_pos in valid_positions:
            result.setdefault(app.player_id, [])
            if league_pos not in result[app.player_id]:
                result[app.player_id].append(league_pos)

    # Every batter with position data qualifies for util if util slots exist.
    if league.roster_util > 0:
        for positions in result.values():
            if "util" not in positions:
                positions.append("util")

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
        roster_spots["util"] = league.roster_util
    return roster_spots


def best_position(eligible: list[str], replacement: dict[str, float]) -> str:
    """Pick the position with the lowest replacement level (highest VAR)."""
    if not eligible:
        return "util"
    return min(eligible, key=lambda p: replacement.get(p, float("inf")))
