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
    target_rates: dict[str, float],
) -> dict[str, float]:
    """Scale projected rates so league totals match target rates.

    For each stat, multiplies the projected rate by (target / projected)
    so the aggregate matches the target league environment.
    """
    result: dict[str, float] = {}
    for stat, proj_rate in projected_rates.items():
        if proj_rate == 0.0:
            result[stat] = 0.0
        else:
            result[stat] = proj_rate * (target_rates[stat] / proj_rate)
    return result
