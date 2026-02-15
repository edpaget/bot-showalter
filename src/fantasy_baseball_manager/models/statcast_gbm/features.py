from collections.abc import Sequence

from fantasy_baseball_manager.features import batting, pitching, player
from fantasy_baseball_manager.features.transforms import (
    BATTED_BALL,
    EXPECTED_STATS,
    PITCH_MIX,
    PLATE_DISCIPLINE,
    SPIN_PROFILE,
)
from fantasy_baseball_manager.features.types import AnyFeature, Feature, FeatureSet, SpineFilter

_BATTER_LAG_STATS = ("pa", "hr", "h", "2b", "3b", "bb", "so", "sb")
_PITCHER_LAG_STATS = ("ip", "so", "bb", "hr", "era", "fip")


def _batter_lag_features() -> list[Feature]:
    features: list[Feature] = []
    for stat in _BATTER_LAG_STATS:
        for lag in (1, 2):
            features.append(batting.col(stat).lag(lag).alias(f"{stat}_{lag}"))
    return features


def _pitcher_lag_features() -> list[Feature]:
    features: list[Feature] = []
    for stat in _PITCHER_LAG_STATS:
        for lag in (1, 2):
            features.append(pitching.col(stat).lag(lag).alias(f"{stat}_{lag}"))
    return features


def build_batter_feature_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_batter_lag_features())
    features.extend([BATTED_BALL, PLATE_DISCIPLINE, EXPECTED_STATS])
    return FeatureSet(
        name="statcast_gbm_batting",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def build_pitcher_feature_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_pitcher_lag_features())
    features.extend([PITCH_MIX, SPIN_PROFILE, PLATE_DISCIPLINE])
    return FeatureSet(
        name="statcast_gbm_pitching",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="pitcher"),
    )
