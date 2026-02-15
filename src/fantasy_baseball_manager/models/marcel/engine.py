from collections.abc import Sequence

from fantasy_baseball_manager.models.marcel.types import (
    LeagueAverages,
    MarcelConfig,
    MarcelInput,
    MarcelProjection,
    SeasonLine,
)


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


def project_player(
    player_id: int,
    marcel_input: MarcelInput,
    projected_season: int,
    config: MarcelConfig,
    projected_pt: float | None = None,
) -> MarcelProjection:
    """Regress, age-adjust, project PT, and multiply to produce a projection."""
    pitcher = _is_pitcher(marcel_input.seasons)

    if pitcher:
        regression_n = config.pitching_regression_ip
    else:
        regression_n = config.batting_regression_pa

    league = LeagueAverages(rates=marcel_input.league_rates)
    regressed = regress_to_mean(marcel_input.weighted_rates, league, marcel_input.weighted_pt, regression_n)
    adjusted = age_adjust(regressed, marcel_input.age, config)

    if projected_pt is not None:
        pt = projected_pt
    else:
        pt = project_playing_time(marcel_input.seasons, config)

    projected_stats = {cat: adjusted[cat] * pt for cat in adjusted}

    if pitcher:
        return MarcelProjection(
            player_id=player_id,
            projected_season=projected_season,
            age=marcel_input.age,
            stats=projected_stats,
            rates=adjusted,
            ip=pt,
        )
    return MarcelProjection(
        player_id=player_id,
        projected_season=projected_season,
        age=marcel_input.age,
        stats=projected_stats,
        rates=adjusted,
        pa=int(pt),
    )


def project_all(
    players: dict[int, MarcelInput],
    projected_season: int,
    config: MarcelConfig,
    projected_pts: dict[int, float] | None = None,
) -> list[MarcelProjection]:
    """Project all players. players maps player_id -> MarcelInput."""
    pts = projected_pts or {}
    return [
        project_player(pid, marcel_input, projected_season, config, projected_pt=pts.get(pid))
        for pid, marcel_input in players.items()
    ]
