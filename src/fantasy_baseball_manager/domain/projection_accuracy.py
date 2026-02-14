from dataclasses import dataclass

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.projection import Projection

# Stat names that map to BattingStats fields (same names as used in projection stat_json)
_BATTING_STAT_FIELDS: tuple[str, ...] = (
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
    "avg",
    "obp",
    "slg",
    "ops",
    "woba",
    "wrc_plus",
    "war",
)

_PITCHING_STAT_FIELDS: tuple[str, ...] = (
    "w",
    "l",
    "era",
    "g",
    "gs",
    "sv",
    "hld",
    "ip",
    "h",
    "er",
    "hr",
    "bb",
    "so",
    "whip",
    "k_per_9",
    "bb_per_9",
    "fip",
    "xfip",
    "war",
)


@dataclass(frozen=True)
class ProjectionComparison:
    stat_name: str
    projected: float
    actual: float
    error: float


def _compare(
    projection: Projection,
    actual_obj: object,
    stat_fields: tuple[str, ...],
) -> list[ProjectionComparison]:
    comparisons: list[ProjectionComparison] = []
    for stat_name in stat_fields:
        projected_val = projection.stat_json.get(stat_name)
        if projected_val is None:
            continue
        actual_val = getattr(actual_obj, stat_name, None)
        if actual_val is None:
            continue
        p = float(projected_val)
        a = float(actual_val)
        comparisons.append(
            ProjectionComparison(
                stat_name=stat_name,
                projected=p,
                actual=a,
                error=p - a,
            )
        )
    return comparisons


def compare_to_batting_actuals(projection: Projection, actual: BattingStats) -> list[ProjectionComparison]:
    """Compare a batting projection to actual batting stats."""
    return _compare(projection, actual, _BATTING_STAT_FIELDS)


def compare_to_pitching_actuals(projection: Projection, actual: PitchingStats) -> list[ProjectionComparison]:
    """Compare a pitching projection to actual pitching stats."""
    return _compare(projection, actual, _PITCHING_STAT_FIELDS)
