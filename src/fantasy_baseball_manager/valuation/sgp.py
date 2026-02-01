import statistics

from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.valuation.models import CategoryValue, PlayerValue, SGPDenominators, StatCategory
from fantasy_baseball_manager.valuation.stat_extractors import extract_batting_stat, extract_pitching_stat

_BATTING_RATIO_STATS: set[StatCategory] = {StatCategory.OBP}
_PITCHING_RATIO_STATS: set[StatCategory] = {StatCategory.ERA, StatCategory.WHIP}


def _obp_components(proj: BattingProjection) -> tuple[float, float]:
    return (proj.h + proj.bb + proj.hbp, proj.pa)


def _era_components(proj: PitchingProjection) -> tuple[float, float]:
    return (proj.er, proj.ip)


def _whip_components(proj: PitchingProjection) -> tuple[float, float]:
    return (proj.h + proj.bb, proj.ip)


def compute_sgp_denominators(
    historical_standings: dict[StatCategory, list[float]],
    team_count: int,
) -> SGPDenominators:
    if team_count < 2:
        raise ValueError("team_count must be at least 2 to compute SGP denominators")

    denominators: dict[StatCategory, float] = {}
    for cat, values in historical_standings.items():
        best = max(values)
        worst = min(values)
        denominators[cat] = (best - worst) / (team_count - 1)
    return SGPDenominators(denominators=denominators)


def sgp_batting(
    projections: list[BattingProjection],
    categories: tuple[StatCategory, ...],
    denominators: SGPDenominators,
) -> list[PlayerValue]:
    if not projections:
        return []

    category_sgp: dict[StatCategory, list[float]] = {}
    category_raw: dict[StatCategory, list[float]] = {}

    for cat in categories:
        denom = denominators.denominators[cat]

        if cat in _BATTING_RATIO_STATS:
            components = [_obp_components(p) for p in projections]
            total_num = sum(c[0] for c in components)
            total_den = sum(c[1] for c in components)
            pool_avg = total_num / total_den if total_den != 0.0 else 0.0
            contributions = [c[0] - c[1] * pool_avg for c in components]
            mean_c = statistics.mean(contributions)
            category_sgp[cat] = [(c - mean_c) / denom if denom != 0.0 else 0.0 for c in contributions]
            category_raw[cat] = [extract_batting_stat(p, cat) for p in projections]
        else:
            raw = [extract_batting_stat(p, cat) for p in projections]
            mean_r = statistics.mean(raw)
            category_sgp[cat] = [(v - mean_r) / denom if denom != 0.0 else 0.0 for v in raw]
            category_raw[cat] = raw

    results: list[PlayerValue] = []
    for i, proj in enumerate(projections):
        cat_values = tuple(
            CategoryValue(category=cat, raw_stat=category_raw[cat][i], value=category_sgp[cat][i]) for cat in categories
        )
        total = sum(cv.value for cv in cat_values)
        results.append(
            PlayerValue(player_id=proj.player_id, name=proj.name, category_values=cat_values, total_value=total)
        )
    return results


def sgp_pitching(
    projections: list[PitchingProjection],
    categories: tuple[StatCategory, ...],
    denominators: SGPDenominators,
) -> list[PlayerValue]:
    if not projections:
        return []

    category_sgp: dict[StatCategory, list[float]] = {}
    category_raw: dict[StatCategory, list[float]] = {}

    for cat in categories:
        denom = denominators.denominators[cat]

        if cat in _PITCHING_RATIO_STATS:
            if cat == StatCategory.ERA:
                components = [_era_components(p) for p in projections]
            else:
                components = [_whip_components(p) for p in projections]

            total_num = sum(c[0] for c in components)
            total_den = sum(c[1] for c in components)
            pool_avg = total_num / total_den if total_den != 0.0 else 0.0
            contributions = [c[0] - c[1] * pool_avg for c in components]
            mean_c = statistics.mean(contributions)
            # Negate for lower-is-better
            category_sgp[cat] = [-(c - mean_c) / denom if denom != 0.0 else 0.0 for c in contributions]
            category_raw[cat] = [extract_pitching_stat(p, cat) for p in projections]
        else:
            raw = [extract_pitching_stat(p, cat) for p in projections]
            mean_r = statistics.mean(raw)
            category_sgp[cat] = [(v - mean_r) / denom if denom != 0.0 else 0.0 for v in raw]
            category_raw[cat] = raw

    results: list[PlayerValue] = []
    for i, proj in enumerate(projections):
        cat_values = tuple(
            CategoryValue(category=cat, raw_stat=category_raw[cat][i], value=category_sgp[cat][i]) for cat in categories
        )
        total = sum(cv.value for cv in cat_values)
        results.append(
            PlayerValue(player_id=proj.player_id, name=proj.name, category_values=cat_values, total_value=total)
        )
    return results
