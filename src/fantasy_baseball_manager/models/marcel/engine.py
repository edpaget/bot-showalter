from collections.abc import Sequence

from fantasy_baseball_manager.models.marcel.types import (
    LeagueAverages,
    MarcelConfig,
    MarcelProjection,
    SeasonLine,
)


def weighted_average_rates(
    seasons: Sequence[SeasonLine],
    weights: tuple[float, ...],
    categories: Sequence[str],
) -> dict[str, float]:
    """Compute weighted-average rate per PA/IP for each category across seasons.

    Weights are positional: weights[0] for seasons[0] (most recent), etc.
    If fewer seasons than weights, only available seasons are used.
    Zero-PA/IP seasons contribute nothing to numerator or denominator.
    """
    if not seasons:
        return {cat: 0.0 for cat in categories}

    n = min(len(seasons), len(weights))
    weighted_stats: dict[str, float] = {cat: 0.0 for cat in categories}
    weighted_pt = 0.0

    for i in range(n):
        season = seasons[i]
        w = weights[i]
        pt = season.pa if season.pa > 0 else season.ip
        weighted_pt += pt * w
        for cat in categories:
            weighted_stats[cat] += season.stats.get(cat, 0.0) * w

    if weighted_pt == 0.0:
        return {cat: 0.0 for cat in categories}

    return {cat: weighted_stats[cat] / weighted_pt for cat in categories}


def regress_to_mean(
    rates: dict[str, float],
    league: LeagueAverages,
    playing_time: float,
    regression_n: float,
) -> dict[str, float]:
    """Regress player rates toward league averages.

    Formula: (rate * pt + league_rate * n) / (pt + n) per stat.
    """
    denominator = playing_time + regression_n
    return {
        cat: (rate * playing_time + league.rates.get(cat, 0.0) * regression_n) / denominator
        for cat, rate in rates.items()
    }


def _is_pitcher(seasons: Sequence[SeasonLine]) -> bool:
    return any(s.ip > 0 for s in seasons)


def _is_starter(seasons: Sequence[SeasonLine], config: MarcelConfig) -> bool:
    for s in seasons:
        if s.g > 0:
            return s.gs / s.g >= config.reliever_gs_ratio
    return False


def project_playing_time(
    seasons: Sequence[SeasonLine],
    config: MarcelConfig,
) -> float:
    """Project playing time (PA for batters, IP for pitchers).

    Uses weighted sum of recent years plus a baseline.
    """
    pitcher = _is_pitcher(seasons)

    if pitcher:
        pt_weights = config.ip_weights
        starter = _is_starter(seasons, config)
        baseline = config.pitching_starter_baseline_ip if starter else config.pitching_reliever_baseline_ip
    else:
        pt_weights = config.pa_weights
        baseline = config.batting_baseline_pa

    total = baseline
    for i, w in enumerate(pt_weights):
        if i < len(seasons):
            pt = seasons[i].ip if pitcher else seasons[i].pa
            total += w * pt

    return total


def age_adjust(
    rates: dict[str, float],
    age: int,
    config: MarcelConfig,
) -> dict[str, float]:
    """Multiply rates by an aging factor based on distance from peak age."""
    diff = config.age_peak - age
    if diff > 0:
        factor = 1.0 + diff * config.age_improvement_rate
    elif diff < 0:
        factor = 1.0 + diff * config.age_decline_rate  # diff is negative
    else:
        factor = 1.0
    return {cat: rate * factor for cat, rate in rates.items()}


def compute_league_averages(
    all_seasons: dict[int, list[SeasonLine]],
    categories: Sequence[str],
) -> LeagueAverages:
    """Compute league-average rates from most-recent season lines across all players."""
    total_stats: dict[str, float] = {cat: 0.0 for cat in categories}
    total_pt = 0.0

    for seasons in all_seasons.values():
        if not seasons:
            continue
        most_recent = seasons[0]
        pt = most_recent.pa if most_recent.pa > 0 else most_recent.ip
        total_pt += pt
        for cat in categories:
            total_stats[cat] += most_recent.stats.get(cat, 0.0)

    if total_pt == 0.0:
        return LeagueAverages(rates={cat: 0.0 for cat in categories})

    return LeagueAverages(rates={cat: total_stats[cat] / total_pt for cat in categories})


def _weighted_playing_time(seasons: Sequence[SeasonLine], weights: tuple[float, ...]) -> float:
    """Total weighted playing time used for regression denominator."""
    total = 0.0
    pitcher = _is_pitcher(seasons)
    for i, w in enumerate(weights):
        if i < len(seasons):
            pt = seasons[i].ip if pitcher else seasons[i].pa
            total += pt * w
    return total


def project_player(
    player_id: int,
    seasons: list[SeasonLine],
    age: int,
    projected_season: int,
    league_avg: LeagueAverages,
    config: MarcelConfig,
) -> MarcelProjection:
    """Chain weighted averages, regression, age adjustment, and playing time projection."""
    pitcher = _is_pitcher(seasons)

    if pitcher:
        categories = list(config.pitching_categories)
        weights = config.pitching_weights
        regression_n = config.pitching_regression_ip
    else:
        categories = list(config.batting_categories)
        weights = config.batting_weights
        regression_n = config.batting_regression_pa

    rates = weighted_average_rates(seasons, weights, categories)
    weighted_pt = _weighted_playing_time(seasons, weights)
    regressed = regress_to_mean(rates, league_avg, weighted_pt, regression_n)
    adjusted = age_adjust(regressed, age, config)
    projected_pt = project_playing_time(seasons, config)

    projected_stats = {cat: adjusted[cat] * projected_pt for cat in categories}

    if pitcher:
        return MarcelProjection(
            player_id=player_id,
            projected_season=projected_season,
            age=age,
            stats=projected_stats,
            rates=adjusted,
            ip=projected_pt,
        )
    return MarcelProjection(
        player_id=player_id,
        projected_season=projected_season,
        age=age,
        stats=projected_stats,
        rates=adjusted,
        pa=int(projected_pt),
    )


def project_all(
    players: dict[int, tuple[list[SeasonLine], int]],
    projected_season: int,
    league_avg: LeagueAverages,
    config: MarcelConfig,
) -> list[MarcelProjection]:
    """Project all players. players maps player_id → (seasons, age)."""
    return [
        project_player(pid, seasons, age, projected_season, league_avg, config)
        for pid, (seasons, age) in players.items()
    ]
