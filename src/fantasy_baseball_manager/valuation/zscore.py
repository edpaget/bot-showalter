import statistics

from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.valuation.models import CategoryValue, PlayerValue, StatCategory
from fantasy_baseball_manager.valuation.stat_extractors import extract_batting_stat, extract_pitching_stat

_BATTING_RATIO_STATS: set[StatCategory] = {StatCategory.OBP}
_PITCHING_RATIO_STATS: set[StatCategory] = {StatCategory.ERA, StatCategory.WHIP}


def _obp_components(proj: BattingProjection) -> tuple[float, float]:
    """Return (numerator, denominator) for OBP: (H+BB+HBP, PA)."""
    return (proj.h + proj.bb + proj.hbp, proj.pa)


def _era_components(proj: PitchingProjection) -> tuple[float, float]:
    """Return (numerator, denominator) for ERA: (ER, IP) — will be scaled by 9."""
    return (proj.er, proj.ip)


def _whip_components(proj: PitchingProjection) -> tuple[float, float]:
    """Return (numerator, denominator) for WHIP: (H+BB, IP)."""
    return (proj.h + proj.bb, proj.ip)


def _zscore(value: float, mean: float, std: float) -> float:
    if std == 0.0:
        return 0.0
    return (value - mean) / std


def zscore_batting(
    projections: list[BattingProjection],
    categories: tuple[StatCategory, ...],
) -> list[PlayerValue]:
    if len(projections) <= 1:
        return [
            PlayerValue(
                player_id=p.player_id,
                name=p.name,
                category_values=tuple(
                    CategoryValue(category=cat, raw_stat=extract_batting_stat(p, cat), value=0.0) for cat in categories
                ),
                total_value=0.0,
            )
            for p in projections
        ]

    category_zscores: dict[StatCategory, list[float]] = {}
    category_raw: dict[StatCategory, list[float]] = {}

    for cat in categories:
        if cat in _BATTING_RATIO_STATS:
            components = [_obp_components(p) for p in projections]
            total_num = sum(c[0] for c in components)
            total_den = sum(c[1] for c in components)
            pool_avg = total_num / total_den if total_den != 0.0 else 0.0
            contributions = [c[0] - c[1] * pool_avg for c in components]
            mean_c = statistics.mean(contributions)
            std_c = statistics.pstdev(contributions)
            category_zscores[cat] = [_zscore(c, mean_c, std_c) for c in contributions]
            category_raw[cat] = [extract_batting_stat(p, cat) for p in projections]
        else:
            raw = [extract_batting_stat(p, cat) for p in projections]
            mean_r = statistics.mean(raw)
            std_r = statistics.pstdev(raw)
            category_zscores[cat] = [_zscore(v, mean_r, std_r) for v in raw]
            category_raw[cat] = raw

    results: list[PlayerValue] = []
    for i, proj in enumerate(projections):
        cat_values = tuple(
            CategoryValue(category=cat, raw_stat=category_raw[cat][i], value=category_zscores[cat][i])
            for cat in categories
        )
        total = sum(cv.value for cv in cat_values)
        results.append(
            PlayerValue(player_id=proj.player_id, name=proj.name, category_values=cat_values, total_value=total)
        )
    return results


def zscore_pitching(
    projections: list[PitchingProjection],
    categories: tuple[StatCategory, ...],
) -> list[PlayerValue]:
    if len(projections) <= 1:
        return [
            PlayerValue(
                player_id=p.player_id,
                name=p.name,
                category_values=tuple(
                    CategoryValue(category=cat, raw_stat=extract_pitching_stat(p, cat), value=0.0) for cat in categories
                ),
                total_value=0.0,
            )
            for p in projections
        ]

    category_zscores: dict[StatCategory, list[float]] = {}
    category_raw: dict[StatCategory, list[float]] = {}

    for cat in categories:
        if cat in _PITCHING_RATIO_STATS:
            if cat == StatCategory.ERA:
                components = [_era_components(p) for p in projections]
                total_num = sum(c[0] for c in components)
                total_den = sum(c[1] for c in components)
                pool_avg_per_unit = total_num / total_den if total_den != 0.0 else 0.0
                contributions = [c[0] - c[1] * pool_avg_per_unit for c in components]
            else:  # WHIP
                components = [_whip_components(p) for p in projections]
                total_num = sum(c[0] for c in components)
                total_den = sum(c[1] for c in components)
                pool_avg = total_num / total_den if total_den != 0.0 else 0.0
                contributions = [c[0] - c[1] * pool_avg for c in components]

            mean_c = statistics.mean(contributions)
            std_c = statistics.pstdev(contributions)
            zscores = [_zscore(c, mean_c, std_c) for c in contributions]
            # Negate for lower-is-better
            category_zscores[cat] = [-z for z in zscores]
            category_raw[cat] = [extract_pitching_stat(p, cat) for p in projections]
        else:
            raw = [extract_pitching_stat(p, cat) for p in projections]
            mean_r = statistics.mean(raw)
            std_r = statistics.pstdev(raw)
            category_zscores[cat] = [_zscore(v, mean_r, std_r) for v in raw]
            category_raw[cat] = raw

    results: list[PlayerValue] = []
    for i, proj in enumerate(projections):
        cat_values = tuple(
            CategoryValue(category=cat, raw_stat=category_raw[cat][i], value=category_zscores[cat][i])
            for cat in categories
        )
        total = sum(cv.value for cv in cat_values)
        results.append(
            PlayerValue(player_id=proj.player_id, name=proj.name, category_values=cat_values, total_value=total)
        )
    return results
