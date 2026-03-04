"""Stat group presets and route-group expansion for the ensemble model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import LeagueSettings

BUILTIN_GROUPS: dict[str, frozenset[str]] = {
    "batting_counting": frozenset(
        {
            "pa",
            "ab",
            "h",
            "doubles",
            "triples",
            "hr",
            "rbi",
            "r",
            "sb",
            "cs",
            "bb",
            "so",
            "hbp",
            "sf",
            "sh",
            "gdp",
            "ibb",
        }
    ),
    "batting_rate": frozenset(
        {
            "avg",
            "obp",
            "slg",
            "ops",
            "woba",
            "wrc_plus",
            "iso",
            "babip",
            "k_pct",
            "bb_pct",
        }
    ),
    "pitching_counting": frozenset(
        {
            "w",
            "l",
            "g",
            "gs",
            "sv",
            "hld",
            "ip",
            "er",
            "so",
            "bb",
            "h",
            "hr",
        }
    ),
    "pitching_rate": frozenset(
        {
            "era",
            "whip",
            "k_per_9",
            "bb_per_9",
            "fip",
            "xfip",
            "hr_per_9",
            "babip",
        }
    ),
    "war": frozenset({"war"}),
}


def expand_route_groups(
    route_groups: dict[str, str],
    routes: dict[str, str] | None = None,
    custom_groups: dict[str, list[str]] | None = None,
    league: LeagueSettings | None = None,
) -> dict[str, str]:
    """Expand group-level routing into a flat stat→system dict.

    Processing order (later overrides earlier):
    1. ``league_required`` pseudo-group (broadest)
    2. Named groups (builtin or custom)
    3. Per-stat ``routes`` (most specific)
    """
    result: dict[str, str] = {}
    custom = custom_groups or {}

    # Process league_required first if present
    if "league_required" in route_groups:
        if league is None:
            msg = "league_required group requires a LeagueSettings object"
            raise ValueError(msg)
        system = route_groups["league_required"]
        for stat in league_required_stats(league):
            result[stat] = system

    # Process remaining groups
    for group_name, system in route_groups.items():
        if group_name == "league_required":
            continue
        if group_name in custom:
            stats = custom[group_name]
        elif group_name in BUILTIN_GROUPS:
            stats = BUILTIN_GROUPS[group_name]
        else:
            msg = f"Unknown stat group: {group_name!r}"
            raise ValueError(msg)
        for stat in stats:
            result[stat] = system

    # Per-stat routes override everything
    if routes:
        result.update(routes)

    return result


def league_required_stats(league: LeagueSettings) -> frozenset[str]:
    """Return the set of stats required by the league's categories.

    For counting categories, includes the key (which may be compound like
    ``sv+hld``).  For rate categories, includes the key plus numerator and
    denominator components.
    """
    stats: set[str] = set()
    all_categories = [*league.batting_categories, *league.pitching_categories]
    for cat in all_categories:
        # Split compound keys like "sv+hld"
        for part in cat.key.split("+"):
            stats.add(part)
        if cat.numerator is not None:
            for part in cat.numerator.split("+"):
                stats.add(part)
        if cat.denominator is not None:
            for part in cat.denominator.split("+"):
                stats.add(part)
    return frozenset(stats)
