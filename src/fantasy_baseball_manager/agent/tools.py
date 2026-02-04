"""LangChain tool definitions for the fantasy baseball agent.

Each tool wraps existing functionality from the projection, valuation, and keeper modules,
adapting them for use by an LLM agent.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from fantasy_baseball_manager.config import load_league_settings
from fantasy_baseball_manager.draft.cli import build_projections_and_positions
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, SUPPORTED_ENGINES
from fantasy_baseball_manager.keeper.models import KeeperCandidate
from fantasy_baseball_manager.keeper.replacement import DraftPoolReplacementCalculator
from fantasy_baseball_manager.keeper.surplus import SurplusCalculator

if TYPE_CHECKING:
    from fantasy_baseball_manager.valuation.models import PlayerValue


def _get_default_year() -> int:
    """Return the current year as the default projection year."""
    return datetime.now().year


def _format_batter_table(values: list[PlayerValue], top_n: int) -> str:
    """Format a list of batter valuations as a text table."""
    sorted_values = sorted(values, key=lambda p: p.total_value, reverse=True)[:top_n]

    lines: list[str] = []
    lines.append(f"Top {len(sorted_values)} Projected Batters:")
    lines.append("")
    header = f"{'Rk':>4} {'Name':<25} {'Value':>7}"
    # Add category columns
    if sorted_values and sorted_values[0].category_values:
        for cv in sorted_values[0].category_values:
            header += f" {cv.category.value:>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, pv in enumerate(sorted_values, start=1):
        row = f"{i:>4} {pv.name:<25} {pv.total_value:>7.1f}"
        for cv in pv.category_values:
            row += f" {cv.raw_stat:>6.1f}"
        lines.append(row)

    return "\n".join(lines)


def _format_pitcher_table(values: list[PlayerValue], top_n: int) -> str:
    """Format a list of pitcher valuations as a text table."""
    sorted_values = sorted(values, key=lambda p: p.total_value, reverse=True)[:top_n]

    lines: list[str] = []
    lines.append(f"Top {len(sorted_values)} Projected Pitchers:")
    lines.append("")
    header = f"{'Rk':>4} {'Name':<25} {'Value':>7}"
    if sorted_values and sorted_values[0].category_values:
        for cv in sorted_values[0].category_values:
            header += f" {cv.category.value:>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, pv in enumerate(sorted_values, start=1):
        row = f"{i:>4} {pv.name:<25} {pv.total_value:>7.1f}"
        for cv in pv.category_values:
            # Format ERA/WHIP with more decimals
            if cv.category.value in ("ERA", "WHIP"):
                row += f" {cv.raw_stat:>6.2f}"
            else:
                row += f" {cv.raw_stat:>6.1f}"
        lines.append(row)

    return "\n".join(lines)


@tool
def project_batters(
    year: int | None = None,
    engine: str = DEFAULT_ENGINE,
    top_n: int = 25,
) -> str:
    """Get projected batter statistics and z-score valuations.

    Args:
        year: The projection year. Defaults to current year.
        engine: Projection engine to use. Options: marcel_classic, marcel, marcel_full.
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

    return _format_batter_table(batter_values, top_n)


@tool
def project_pitchers(
    year: int | None = None,
    engine: str = DEFAULT_ENGINE,
    top_n: int = 25,
) -> str:
    """Get projected pitcher statistics and z-score valuations.

    Args:
        year: The projection year. Defaults to current year.
        engine: Projection engine to use. Options: marcel_classic, marcel, marcel_full.
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

    return _format_pitcher_table(pitcher_values, top_n)


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

    # Case-insensitive partial match
    name_lower = name.lower()
    matches = [pv for pv in all_values if name_lower in pv.name.lower()]

    if not matches:
        return f"No players found matching '{name}'."

    lines: list[str] = []
    for pv in sorted(matches, key=lambda p: p.total_value, reverse=True):
        pos_type = "Batter" if pv.position_type == "B" else "Pitcher"
        lines.append(f"{pv.name} ({pos_type})")
        lines.append(f"  Total Z-Score Value: {pv.total_value:.2f}")
        lines.append("  Category Breakdown:")
        for cv in pv.category_values:
            if cv.category.value in ("ERA", "WHIP"):
                lines.append(f"    {cv.category.value}: {cv.raw_stat:.2f} (z={cv.value:.2f})")
            else:
                lines.append(f"    {cv.category.value}: {cv.raw_stat:.1f} (z={cv.value:.2f})")
        lines.append("")

    return "\n".join(lines)


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

    # Find best match for each name
    found_players: list[PlayerValue] = []
    not_found: list[str] = []

    for name in name_list:
        name_lower = name.lower()
        matches = [pv for pv in all_values if name_lower in pv.name.lower()]
        if matches:
            # Take best match by value if multiple
            best = max(matches, key=lambda p: p.total_value)
            found_players.append(best)
        else:
            not_found.append(name)

    if not found_players:
        return f"No players found. Searched for: {', '.join(name_list)}"

    lines: list[str] = []

    if not_found:
        lines.append(f"Note: Could not find: {', '.join(not_found)}")
        lines.append("")

    # Build comparison table
    lines.append("Player Comparison:")
    lines.append("")

    # Header row
    header = f"{'Stat':<12}"
    for pv in found_players:
        header += f" {pv.name[:15]:<15}"
    lines.append(header)
    lines.append("-" * len(header))

    # Position type
    row = f"{'Type':<12}"
    for pv in found_players:
        pos_type = "Batter" if pv.position_type == "B" else "Pitcher"
        row += f" {pos_type:<15}"
    lines.append(row)

    # Total value
    row = f"{'Total Value':<12}"
    for pv in found_players:
        row += f" {pv.total_value:<15.2f}"
    lines.append(row)

    # Category values - collect all unique categories
    all_cats: dict[str, dict[str, tuple[float, float]]] = {}
    for pv in found_players:
        for cv in pv.category_values:
            cat_name = cv.category.value
            if cat_name not in all_cats:
                all_cats[cat_name] = {}
            all_cats[cat_name][pv.player_id] = (cv.raw_stat, cv.value)

    for cat_name, player_stats in all_cats.items():
        row = f"{cat_name:<12}"
        for pv in found_players:
            if pv.player_id in player_stats:
                raw, z = player_stats[pv.player_id]
                if cat_name in ("ERA", "WHIP"):
                    row += f" {raw:.2f} (z={z:.1f})  "
                else:
                    row += f" {raw:.0f} (z={z:.1f})   "
            else:
                row += f" {'-':<15}"
        lines.append(row)

    return "\n".join(lines)


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

    pv_by_id: dict[str, PlayerValue] = {}
    for pv in all_values:
        if pv.player_id not in pv_by_id or pv.total_value > pv_by_id[pv.player_id].total_value:
            pv_by_id[pv.player_id] = pv

    for name in name_list:
        name_lower = name.lower()
        matches = [pv for pv in all_values if name_lower in pv.name.lower()]
        if matches:
            best = max(matches, key=lambda p: p.total_value)
            # Get positions
            positions: list[str] = []
            for (pid, _), pos in composite_positions.items():
                if pid == best.player_id:
                    positions.extend(pos)
            eligible = tuple(dict.fromkeys(positions))

            candidate_list.append(
                KeeperCandidate(
                    player_id=best.player_id,
                    name=best.name,
                    player_value=best,
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

    # Format output
    lines: list[str] = []

    if not_found:
        lines.append(f"Note: Could not find: {', '.join(not_found)}")
        lines.append("")

    lines.append(f"Keeper Rankings (Pick #{user_pick}, {teams} teams, {keeper_slots} keepers):")
    lines.append("")
    header = f"{'Rk':>4} {'Name':<25} {'Pos':<12} {'Value':>7} {'Repl':>7} {'Surplus':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, ks in enumerate(ranked, start=1):
        pos_str = "/".join(ks.eligible_positions) if ks.eligible_positions else "-"
        lines.append(
            f"{i:>4} {ks.name:<25} {pos_str:<12}"
            f" {ks.player_value:>7.1f} {ks.replacement_value:>7.1f}"
            f" {ks.surplus_value:>8.1f}"
        )

    return "\n".join(lines)


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

    # Find matching player
    name_lower = name.lower()
    matches = [pv for pv in all_values if name_lower in pv.name.lower()]

    if not matches:
        return f"No players found matching '{name}'."

    # Use best match by value
    best = max(matches, key=lambda p: p.total_value)
    fangraphs_id = best.player_id

    # Map to MLBAM ID
    mapper = get_container().id_mapper
    mlbam_id = mapper.fangraphs_to_mlbam(fangraphs_id)

    if not mlbam_id:
        return (
            f"Found {best.name} in projections but could not find their MLB ID. "
            "This player may be a minor leaguer or international player."
        )

    # Fetch from MLB Stats API
    info = _fetch_mlb_player_info(mlbam_id)

    if not info:
        return f"Could not retrieve current info for {best.name} from MLB.com."

    # Format response
    lines: list[str] = []
    lines.append(f"{info.get('fullName', best.name)}")
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
