from fantasy_baseball_manager.marcel.age_adjustment import age_multiplier
from fantasy_baseball_manager.marcel.data_source import StatsDataSource
from fantasy_baseball_manager.marcel.league_averages import (
    PITCHING_COMPONENT_STATS,
    compute_pitching_league_rates,
    rebaseline,
)
from fantasy_baseball_manager.marcel.models import (
    PitchingProjection,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.marcel.weights import projected_ip, weighted_rate

YEAR_WEIGHTS = [3, 2, 1]
REGRESSION_OUTS = 134
STARTER_GS_RATIO = 0.5


def _is_starter(seasons: dict[int, PitchingSeasonStats]) -> bool:
    """Determine if a pitcher is a starter based on career GS/G ratio."""
    total_gs = sum(s.gs for s in seasons.values())
    total_g = sum(s.g for s in seasons.values())
    if total_g == 0:
        return True
    return (total_gs / total_g) >= STARTER_GS_RATIO


def project_pitchers(
    data_source: StatsDataSource,
    year: int,
) -> list[PitchingProjection]:
    """Generate MARCEL pitching projections for the given year."""
    years = [year - 1, year - 2, year - 3]

    player_seasons: dict[int, list[PitchingSeasonStats]] = {}
    for y in years:
        player_seasons[y] = data_source.pitching_stats(y)

    league_rates: dict[int, dict[str, float]] = {}
    for y in years:
        team_stats = data_source.team_pitching(y)
        if team_stats:
            league_rates[y] = compute_pitching_league_rates(team_stats)

    target_rates = league_rates[years[0]]

    avg_league_rates: dict[str, float] = {}
    for stat in PITCHING_COMPONENT_STATS:
        rates_for_stat = [league_rates[y][stat] for y in years if y in league_rates]
        avg_league_rates[stat] = sum(rates_for_stat) / len(rates_for_stat)

    # Group player data by player_id
    player_data: dict[str, dict[int, PitchingSeasonStats]] = {}
    for y in years:
        for p in player_seasons.get(y, []):
            if p.player_id not in player_data:
                player_data[p.player_id] = {}
            player_data[p.player_id][y] = p

    projections: list[PitchingProjection] = []
    for player_id, seasons in player_data.items():
        most_recent = next(seasons[y] for y in years if y in seasons)
        projection_age = most_recent.age + (year - most_recent.year)

        # Outs per year (IP * 3) for rate denominators
        outs_per_year = [seasons[y].ip * 3 if y in seasons else 0 for y in years]
        ip_per_year = [seasons[y].ip if y in seasons else 0 for y in years]

        # Compute weighted rates for each component stat (per out)
        raw_rates: dict[str, float] = {}
        for stat in PITCHING_COMPONENT_STATS:
            stat_per_year = [getattr(seasons[y], stat) if y in seasons else 0 for y in years]
            raw_rates[stat] = weighted_rate(
                stats=stat_per_year,
                opportunities=outs_per_year,
                weights=YEAR_WEIGHTS[: len(years)],
                league_rate=avg_league_rates[stat],
                regression_pa=REGRESSION_OUTS,
            )

        rates = rebaseline(raw_rates, avg_league_rates, target_rates)

        # Age adjustment
        mult = age_multiplier(projection_age)
        rates = {stat: rate * mult for stat, rate in rates.items()}

        # Project playing time
        starter = _is_starter(seasons)
        proj_ip = projected_ip(
            ip_y1=ip_per_year[0],
            ip_y2=ip_per_year[1] if len(ip_per_year) > 1 else 0,
            is_starter=starter,
        )
        proj_outs = proj_ip * 3

        # Counting stats (rates are per-out)
        projected_stats = {stat: rate * proj_outs for stat, rate in rates.items()}

        # Derive ERA and WHIP
        proj_er = projected_stats["er"]
        proj_h = projected_stats["h"]
        proj_bb = projected_stats["bb"]
        era = (proj_er / proj_ip) * 9 if proj_ip > 0 else 0.0
        whip = (proj_h + proj_bb) / proj_ip if proj_ip > 0 else 0.0

        # Estimate GS and G from starter/reliever status
        if starter:
            proj_gs = proj_ip / 6.0  # ~6 IP per start
            proj_g = proj_gs
        else:
            proj_gs = 0.0
            proj_g = proj_ip  # ~1 IP per relief appearance

        projections.append(
            PitchingProjection(
                player_id=player_id,
                name=most_recent.name,
                year=year,
                age=projection_age,
                ip=proj_ip,
                g=proj_g,
                gs=proj_gs,
                er=proj_er,
                h=proj_h,
                bb=proj_bb,
                so=projected_stats["so"],
                hr=projected_stats["hr"],
                hbp=projected_stats["hbp"],
                era=era,
                whip=whip,
            )
        )

    return projections
