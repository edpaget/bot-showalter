"""Static feature group definitions and parameterized group factories.

Import this module to register all static groups in the feature group registry.
"""

from collections.abc import Sequence

from fantasy_baseball_manager.features import batting, pitching, player, projection
from fantasy_baseball_manager.features.groups import FeatureGroup, register_group
from fantasy_baseball_manager.features.library import (
    STATCAST_BATTED_BALL,
    STATCAST_EXPECTED_STATS,
    STATCAST_PITCH_MIX,
    STATCAST_PLATE_DISCIPLINE,
    STATCAST_SPIN_PROFILE,
)
from fantasy_baseball_manager.features.types import AnyFeature, Feature

# ---------------------------------------------------------------------------
# Static groups — registered at import time
# ---------------------------------------------------------------------------

register_group(
    FeatureGroup(
        name="age",
        description="Player age",
        player_type="both",
        features=(player.age(),),
    )
)

register_group(
    FeatureGroup(
        name="positions",
        description="Player positions",
        player_type="both",
        features=(player.positions(),),
    )
)

register_group(
    FeatureGroup(
        name="statcast_batted_ball",
        description="Statcast batted ball profile",
        player_type="batter",
        features=tuple(STATCAST_BATTED_BALL),
    )
)

register_group(
    FeatureGroup(
        name="statcast_plate_discipline",
        description="Statcast plate discipline profile",
        player_type="both",
        features=tuple(STATCAST_PLATE_DISCIPLINE),
    )
)

register_group(
    FeatureGroup(
        name="statcast_expected_stats",
        description="Statcast expected stats profile",
        player_type="batter",
        features=tuple(STATCAST_EXPECTED_STATS),
    )
)

register_group(
    FeatureGroup(
        name="statcast_pitch_mix",
        description="Statcast pitch mix profile",
        player_type="pitcher",
        features=tuple(STATCAST_PITCH_MIX),
    )
)

register_group(
    FeatureGroup(
        name="statcast_spin_profile",
        description="Statcast spin profile",
        player_type="pitcher",
        features=tuple(STATCAST_SPIN_PROFILE),
    )
)

register_group(
    FeatureGroup(
        name="projected_batting_pt",
        description="Projected batting playing time (PA)",
        player_type="batter",
        features=(projection.col("pa").system("playing_time").lag(0).alias("proj_pa"),),
    )
)

register_group(
    FeatureGroup(
        name="projected_pitching_pt",
        description="Projected pitching playing time (IP)",
        player_type="pitcher",
        features=(projection.col("ip").system("playing_time").lag(0).alias("proj_ip"),),
    )
)


# ---------------------------------------------------------------------------
# Factory functions for parameterized groups (not registered)
# ---------------------------------------------------------------------------


def make_batting_counting_lags(
    categories: Sequence[str],
    lags: Sequence[int],
) -> FeatureGroup:
    """Build a group of lagged batting counting stats.

    For each lag, produces ``pa_{lag}`` plus ``{cat}_{lag}`` for every category.
    """
    features: list[Feature] = []
    for lag_n in lags:
        features.append(batting.col("pa").lag(lag_n).alias(f"pa_{lag_n}"))
        for cat in categories:
            features.append(batting.col(cat).lag(lag_n).alias(f"{cat}_{lag_n}"))
    return FeatureGroup(
        name="batting_counting_lags",
        description="Lagged batting counting stats",
        player_type="batter",
        features=tuple(features),
    )


def make_pitching_counting_lags(
    categories: Sequence[str],
    lags: Sequence[int],
) -> FeatureGroup:
    """Build a group of lagged pitching counting stats.

    For each lag, produces ``ip_{lag}``, ``g_{lag}``, ``gs_{lag}``
    plus ``{cat}_{lag}`` for every category.
    """
    features: list[Feature] = []
    for lag_n in lags:
        features.append(pitching.col("ip").lag(lag_n).alias(f"ip_{lag_n}"))
        features.append(pitching.col("g").lag(lag_n).alias(f"g_{lag_n}"))
        features.append(pitching.col("gs").lag(lag_n).alias(f"gs_{lag_n}"))
        for cat in categories:
            features.append(pitching.col(cat).lag(lag_n).alias(f"{cat}_{lag_n}"))
    return FeatureGroup(
        name="pitching_counting_lags",
        description="Lagged pitching counting stats",
        player_type="pitcher",
        features=tuple(features),
    )


def make_batting_rate_lags(
    columns: Sequence[str],
    lags: Sequence[int],
) -> FeatureGroup:
    """Build a group of lagged batting rate stats.

    For each lag, produces ``{col}_{lag}`` for every column.
    """
    features: list[AnyFeature] = []
    for lag_n in lags:
        for col in columns:
            features.append(batting.col(col).lag(lag_n).alias(f"{col}_{lag_n}"))
    return FeatureGroup(
        name="batting_rate_lags",
        description="Lagged batting rate stats",
        player_type="batter",
        features=tuple(features),
    )
