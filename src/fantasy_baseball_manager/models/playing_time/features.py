from fantasy_baseball_manager.features import batting, il_stint, pitching, player
from fantasy_baseball_manager.features.transforms.playing_time import (
    make_il_summary_transform,
    make_pt_trend_transform,
)
from fantasy_baseball_manager.features.types import DerivedTransformFeature, Feature


def build_batting_pt_features(lags: int = 3) -> list[Feature]:
    """Build features for batting playing-time projection."""
    features: list[Feature] = [player.age()]
    for lag_n in range(1, lags + 1):
        features.append(batting.col("pa").lag(lag_n).alias(f"pa_{lag_n}"))
    # WAR
    for lag_n in range(1, min(lags, 2) + 1):
        features.append(batting.col("war").lag(lag_n).alias(f"war_{lag_n}"))
    # IL
    for lag_n in range(1, lags + 1):
        features.append(il_stint.col("days").lag(lag_n).alias(f"il_days_{lag_n}"))
    for lag_n in range(1, min(lags, 2) + 1):
        features.append(il_stint.col("stint_count").lag(lag_n).alias(f"il_stints_{lag_n}"))
    return features


def build_pitching_pt_features(lags: int = 3) -> list[Feature]:
    """Build features for pitching playing-time projection."""
    features: list[Feature] = [player.age()]
    for lag_n in range(1, lags + 1):
        features.append(pitching.col("ip").lag(lag_n).alias(f"ip_{lag_n}"))
        features.append(pitching.col("g").lag(lag_n).alias(f"g_{lag_n}"))
    features.append(pitching.col("gs").lag(1).alias("gs_1"))
    # WAR
    for lag_n in range(1, min(lags, 2) + 1):
        features.append(pitching.col("war").lag(lag_n).alias(f"war_{lag_n}"))
    # IL
    for lag_n in range(1, lags + 1):
        features.append(il_stint.col("days").lag(lag_n).alias(f"il_days_{lag_n}"))
    for lag_n in range(1, min(lags, 2) + 1):
        features.append(il_stint.col("stint_count").lag(lag_n).alias(f"il_stints_{lag_n}"))
    return features


def build_batting_pt_derived_transforms(lags: int = 3) -> list[DerivedTransformFeature]:
    """Build derived transforms for batting playing-time projection."""
    return [
        DerivedTransformFeature(
            name="il_summary",
            inputs=tuple(f"il_days_{i}" for i in range(1, lags + 1)) + ("il_stints_1", "il_stints_2"),
            group_by=("player_id", "season"),
            transform=make_il_summary_transform(lags),
            outputs=("il_days_3yr", "il_recurrence"),
        ),
        DerivedTransformFeature(
            name="batting_pt_trend",
            inputs=("pa_1", "pa_2"),
            group_by=("player_id", "season"),
            transform=make_pt_trend_transform("pa"),
            outputs=("pt_trend",),
        ),
    ]


def build_pitching_pt_derived_transforms(lags: int = 3) -> list[DerivedTransformFeature]:
    """Build derived transforms for pitching playing-time projection."""
    return [
        DerivedTransformFeature(
            name="il_summary",
            inputs=tuple(f"il_days_{i}" for i in range(1, lags + 1)) + ("il_stints_1", "il_stints_2"),
            group_by=("player_id", "season"),
            transform=make_il_summary_transform(lags),
            outputs=("il_days_3yr", "il_recurrence"),
        ),
        DerivedTransformFeature(
            name="pitching_pt_trend",
            inputs=("ip_1", "ip_2"),
            group_by=("player_id", "season"),
            transform=make_pt_trend_transform("ip"),
            outputs=("pt_trend",),
        ),
    ]
