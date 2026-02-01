from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)

BATTING_COMPONENT_STATS = (
    "singles",
    "doubles",
    "triples",
    "hr",
    "bb",
    "so",
    "hbp",
    "sf",
    "sh",
    "sb",
    "cs",
)

PITCHING_COMPONENT_STATS = (
    "h",
    "bb",
    "so",
    "hr",
    "hbp",
    "er",
)


def compute_batting_league_rates(
    team_stats: list[BattingSeasonStats],
) -> dict[str, float]:
    """Compute league-wide rate stats from aggregated team batting data."""
    total_pa = sum(t.pa for t in team_stats)
    rates: dict[str, float] = {}
    for stat in BATTING_COMPONENT_STATS:
        total_stat = sum(getattr(t, stat) for t in team_stats)
        rates[stat] = total_stat / total_pa
    return rates


def compute_pitching_league_rates(
    team_stats: list[PitchingSeasonStats],
) -> dict[str, float]:
    """Compute league-wide rate stats from aggregated team pitching data.

    Rates are per-out (IP * 3) to match the regression units.
    """
    total_outs = sum(t.ip * 3 for t in team_stats)
    rates: dict[str, float] = {}
    for stat in PITCHING_COMPONENT_STATS:
        total_stat = sum(getattr(t, stat) for t in team_stats)
        rates[stat] = total_stat / total_outs
    return rates


def rebaseline(
    projected_rates: dict[str, float],
    source_rates: dict[str, float],
    target_rates: dict[str, float],
) -> dict[str, float]:
    """Scale projected rates to match the target league environment.

    Adjusts each player's rate by the ratio of the target year's league
    rate to the source (averaged) league rate used during regression.
    """
    result: dict[str, float] = {}
    for stat, proj_rate in projected_rates.items():
        if source_rates[stat] == 0.0:
            result[stat] = proj_rate
        else:
            result[stat] = proj_rate * (target_rates[stat] / source_rates[stat])
    return result
