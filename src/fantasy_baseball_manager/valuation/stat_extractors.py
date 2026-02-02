from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.valuation.models import StatCategory

_SUPPORTED_BATTING: dict[StatCategory, str] = {
    StatCategory.HR: "hr",
    StatCategory.SB: "sb",
    StatCategory.R: "r",
    StatCategory.RBI: "rbi",
}

_SUPPORTED_PITCHING_COUNTING: dict[StatCategory, str] = {
    StatCategory.K: "so",
}


def extract_batting_stat(projection: BattingProjection, category: StatCategory) -> float:
    if category in _SUPPORTED_BATTING:
        return float(getattr(projection, _SUPPORTED_BATTING[category]))

    if category == StatCategory.OBP:
        if projection.pa == 0.0:
            return 0.0
        return (projection.h + projection.bb + projection.hbp) / projection.pa

    raise ValueError(f"Cannot extract batting stat for category {category.value}")


def extract_pitching_stat(projection: PitchingProjection, category: StatCategory) -> float:
    if category in _SUPPORTED_PITCHING_COUNTING:
        return float(getattr(projection, _SUPPORTED_PITCHING_COUNTING[category]))

    if category == StatCategory.ERA:
        if projection.ip == 0.0:
            return 0.0
        return projection.er / projection.ip * 9

    if category == StatCategory.WHIP:
        if projection.ip == 0.0:
            return 0.0
        return (projection.h + projection.bb) / projection.ip

    raise ValueError(f"Cannot extract pitching stat for category {category.value}")
