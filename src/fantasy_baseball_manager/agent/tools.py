"""LangChain tool definitions for the fantasy baseball agent.

Each tool wraps existing functionality from the projection, valuation, and keeper modules,
adapting them for use by an LLM agent.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime

from langchain_core.tools import tool

from fantasy_baseball_manager.agent.formatters import (
    format_batter_table,
    format_keeper_rankings,
    format_pitcher_table,
    format_player_comparison,
    format_player_lookup,
)
from fantasy_baseball_manager.agent.player_lookup import (
    find_all_matches,
    find_player_by_name,
    find_players_by_names,
)
from fantasy_baseball_manager.config import load_league_settings
from fantasy_baseball_manager.draft.cli import build_projections_and_positions
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, SUPPORTED_ENGINES
from fantasy_baseball_manager.keeper.models import KeeperCandidate
from fantasy_baseball_manager.keeper.replacement import DraftPoolReplacementCalculator
from fantasy_baseball_manager.keeper.surplus import SurplusCalculator


def _get_default_year() -> int:
    """Return the current year as the default projection year."""
    return datetime.now().year


@tool
def project_batters(
    year: int | None = None,
    engine: str = DEFAULT_ENGINE,
    top_n: int = 25,
) -> str:
    """Get projected batter statistics and z-score valuations.

    Args:
        year: The projection year. Defaults to current year.
        engine: Projection engine to use. Options: marcel_classic, marcel, marcel_full, marcel_gb.
            Defaults to 'marcel'.
        top_n: Number of top batters to return. Defaults to 25.

    Returns:
        A formatted table of the top projected batters with their z-score values
        and category stats (HR, RBI, R, SB, OBP, etc. based on league settings).
    """
    if year is None:
        year = _get_default_year()

    if engine not in SUPPORTED_ENGINES:
        return f"Invalid engine '{engine}'. Supported engines: {', '.join(SUPPORTED_ENGINES)}"

    all_values, _ = build_projections_and_positions(engine, year)
    batter_values = [pv for pv in all_values if pv.position_type == "B"]

    return format_batter_table(batter_values, top_n)


@tool
def project_pitchers(
    year: int | None = None,
    engine: str = DEFAULT_ENGINE,
    top_n: int = 25,
) -> str:
    """Get projected pitcher statistics and z-score valuations.

    Args:
        year: The projection year. Defaults to current year.
        engine: Projection engine to use. Options: marcel_classic, marcel, marcel_full, marcel_gb.
            Defaults to 'marcel'.
        top_n: Number of top pitchers to return. Defaults to 25.

    Returns:
        A formatted table of the top projected pitchers with their z-score values
        and category stats (K, ERA, WHIP, W, NSVH, etc. based on league settings).
    """
    if year is None:
        year = _get_default_year()

    if engine not in SUPPORTED_ENGINES:
        return f"Invalid engine '{engine}'. Supported engines: {', '.join(SUPPORTED_ENGINES)}"

    all_values, _ = build_projections_and_positions(engine, year)
    pitcher_values = [pv for pv in all_values if pv.position_type == "P"]

    return format_pitcher_table(pitcher_values, top_n)


@tool
def lookup_player(name: str) -> str:
    """Look up a player's projected stats and valuation by name.

    Args:
        name: The player's name (case-insensitive partial match supported).

    Returns:
        The player's projected stats, z-score value, and category breakdown.
        If multiple players match, returns all matches.
    """
    year = _get_default_year()
    all_values, _ = build_projections_and_positions(DEFAULT_ENGINE, year)

    matches = find_all_matches(name, all_values)

    if not matches:
        return f"No players found matching '{name}'."

    return format_player_lookup(matches)


@tool
def compare_players(names: str) -> str:
    """Compare multiple players side by side.

    Args:
        names: Comma-separated list of player names to compare.

    Returns:
        A side-by-side comparison table showing each player's projected stats
        and z-score values.
    """
    year = _get_default_year()
    all_values, _ = build_projections_and_positions(DEFAULT_ENGINE, year)

    name_list = [n.strip() for n in names.split(",") if n.strip()]
    if len(name_list) < 2:
        return "Please provide at least two player names separated by commas."

    found_players, not_found = find_players_by_names(name_list, all_values)

    if not found_players:
        return f"No players found. Searched for: {', '.join(name_list)}"

    return format_player_comparison(found_players, not_found)


@tool
def rank_keepers(
    candidates: str,
    user_pick: int = 5,
    teams: int = 12,
    keeper_slots: int = 4,
) -> str:
    """Rank keeper candidates by surplus value over draft replacement level.

    Args:
        candidates: Comma-separated list of player names who are keeper candidates.
        user_pick: User's draft position (1-based). Defaults to 5.
        teams: Number of teams in the league. Defaults to 12.
        keeper_slots: Number of keeper slots per team. Defaults to 4.

    Returns:
        A ranked table of keeper candidates showing their projected value,
        replacement value, and surplus value.
    """
    year = _get_default_year()
    all_values, composite_positions = build_projections_and_positions(DEFAULT_ENGINE, year)

    # Find matching players
    name_list = [n.strip() for n in candidates.split(",") if n.strip()]
    if not name_list:
        return "Please provide at least one player name as a keeper candidate."

    candidate_list: list[KeeperCandidate] = []
    not_found: list[str] = []

    for name in name_list:
        player = find_player_by_name(name, all_values)
        if player:
            # Get positions
            positions: list[str] = []
            for (pid, _), pos in composite_positions.items():
                if pid == player.player_id:
                    positions.extend(pos)
            eligible = tuple(dict.fromkeys(positions))

            candidate_list.append(
                KeeperCandidate(
                    player_id=player.player_id,
                    name=player.name,
                    player_value=player,
                    eligible_positions=eligible,
                )
            )
        else:
            not_found.append(name)

    if not candidate_list:
        return f"No players found. Searched for: {', '.join(name_list)}"

    # Calculate surplus values
    calc = DraftPoolReplacementCalculator(user_pick_position=user_pick)
    surplus_calc = SurplusCalculator(calc, num_teams=teams, num_keeper_slots=keeper_slots)
    ranked = surplus_calc.rank_candidates(candidate_list, all_values, set())

    return format_keeper_rankings(ranked, not_found, user_pick, teams, keeper_slots)


@tool
def get_league_settings() -> str:
    """Get the current league settings from config.

    Returns:
        The league configuration including team count, scoring categories,
        and scoring style.
    """
    settings = load_league_settings()

    lines: list[str] = []
    lines.append("League Settings:")
    lines.append(f"  Teams: {settings.team_count}")
    lines.append(f"  Scoring Style: {settings.scoring_style.value}")
    lines.append("")
    lines.append("Batting Categories:")
    for cat in settings.batting_categories:
        lines.append(f"  - {cat.value}")
    lines.append("")
    lines.append("Pitching Categories:")
    for cat in settings.pitching_categories:
        lines.append(f"  - {cat.value}")

    return "\n".join(lines)


_logger = logging.getLogger(__name__)

MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"


def _fetch_mlb_player_info(mlbam_id: str) -> dict | None:
    """Fetch player info from the MLB Stats API."""
    url = f"{MLB_STATS_API_BASE}/people/{mlbam_id}?hydrate=currentTeam,stats(type=season)"
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-baseball-manager"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            people = data.get("people", [])
            return people[0] if people else None
    except Exception as e:
        _logger.warning("Failed to fetch MLB player info for %s: %s", mlbam_id, e)
        return None


@tool
def get_player_info(name: str) -> str:
    """Get current biographical and team info for a player from MLB.com.

    Args:
        name: The player's name (case-insensitive partial match supported).

    Returns:
        Current team, position, age, bats/throws, and other biographical info.
        This data comes from MLB.com and reflects current roster status.
    """
    from fantasy_baseball_manager.services.container import get_container

    year = _get_default_year()
    all_values, _ = build_projections_and_positions(DEFAULT_ENGINE, year)

    player = find_player_by_name(name, all_values)

    if not player:
        return f"No players found matching '{name}'."

    fangraphs_id = player.player_id

    # Map to MLBAM ID
    mapper = get_container().id_mapper
    mlbam_id = mapper.fangraphs_to_mlbam(fangraphs_id)

    if not mlbam_id:
        return (
            f"Found {player.name} in projections but could not find their MLB ID. "
            "This player may be a minor leaguer or international player."
        )

    # Fetch from MLB Stats API
    info = _fetch_mlb_player_info(mlbam_id)

    if not info:
        return f"Could not retrieve current info for {player.name} from MLB.com."

    # Format response
    lines: list[str] = []
    lines.append(f"{info.get('fullName', player.name)}")
    lines.append("")

    # Team info
    team = info.get("currentTeam", {})
    if team:
        lines.append(f"Team: {team.get('name', 'Unknown')}")
    else:
        lines.append("Team: Free Agent / Not on 40-man roster")

    # Position and physical info
    position = info.get("primaryPosition", {})
    if position:
        lines.append(f"Position: {position.get('name', 'Unknown')} ({position.get('abbreviation', '')})")

    birth_date = info.get("birthDate", "")
    age = info.get("currentAge", "")
    if age:
        lines.append(f"Age: {age}")
    elif birth_date:
        lines.append(f"Born: {birth_date}")

    bats = info.get("batSide", {}).get("code", "")
    throws = info.get("pitchHand", {}).get("code", "")
    if bats or throws:
        lines.append(f"Bats/Throws: {bats}/{throws}")

    height = info.get("height", "")
    weight = info.get("weight", "")
    if height or weight:
        lines.append(f"Height/Weight: {height}, {weight} lbs")

    birth_place = []
    if info.get("birthCity"):
        birth_place.append(info["birthCity"])
    if info.get("birthStateProvince"):
        birth_place.append(info["birthStateProvince"])
    if info.get("birthCountry"):
        birth_place.append(info["birthCountry"])
    if birth_place:
        lines.append(f"From: {', '.join(birth_place)}")

    mlb_debut = info.get("mlbDebutDate", "")
    if mlb_debut:
        lines.append(f"MLB Debut: {mlb_debut}")

    return "\n".join(lines)


# Export all tools
ALL_TOOLS = [
    project_batters,
    project_pitchers,
    lookup_player,
    compare_players,
    rank_keepers,
    get_league_settings,
    get_player_info,
]
