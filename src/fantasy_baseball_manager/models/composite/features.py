from collections.abc import Sequence

from fantasy_baseball_manager.features import batting, pitching, player, projection
from fantasy_baseball_manager.features.types import (
    AnyFeature,
    DerivedTransformFeature,
    Feature,
    FeatureSet,
    TransformFeature,
)
from fantasy_baseball_manager.models.marcel.features import (
    build_batting_league_averages,
    build_batting_weighted_rates,
    build_pitching_league_averages,
    build_pitching_weighted_rates,
)


def build_composite_batting_features(
    categories: Sequence[str],
    weights: tuple[float, ...],
) -> list[AnyFeature]:
    """Build features for composite batting model: projected PA + raw stats + transforms."""
    lags = len(weights)
    features: list[AnyFeature] = [
        player.age(),
        projection.col("pa").system("playing_time").lag(0).alias("proj_pa"),
    ]

    for lag_n in range(1, lags + 1):
        features.append(batting.col("pa").lag(lag_n).alias(f"pa_{lag_n}"))
        for cat in categories:
            features.append(batting.col(cat).lag(lag_n).alias(f"{cat}_{lag_n}"))

    features.append(build_batting_weighted_rates(categories, weights))
    features.append(build_batting_league_averages(categories))
    return features


def build_composite_pitching_features(
    categories: Sequence[str],
    weights: tuple[float, ...],
) -> list[AnyFeature]:
    """Build features for composite pitching model: projected IP + raw stats + transforms."""
    lags = len(weights)
    features: list[AnyFeature] = [
        player.age(),
        projection.col("ip").system("playing_time").lag(0).alias("proj_ip"),
    ]

    for lag_n in range(1, lags + 1):
        features.append(pitching.col("ip").lag(lag_n).alias(f"ip_{lag_n}"))
        features.append(pitching.col("g").lag(lag_n).alias(f"g_{lag_n}"))
        features.append(pitching.col("gs").lag(lag_n).alias(f"gs_{lag_n}"))
        for cat in categories:
            features.append(pitching.col(cat).lag(lag_n).alias(f"{cat}_{lag_n}"))

    features.append(build_pitching_weighted_rates(categories, weights))
    features.append(build_pitching_league_averages(categories))
    return features


def batter_target_features() -> list[Feature]:
    """Lag-0 batting features with ``target_`` prefix for training."""
    direct = ("avg", "obp", "slg", "woba")
    counting = ("h", "hr", "ab", "so", "sf")
    features: list[Feature] = []
    for stat in direct:
        features.append(batting.col(stat).lag(0).alias(f"target_{stat}"))
    for stat in counting:
        features.append(batting.col(stat).lag(0).alias(f"target_{stat}"))
    return features


def pitcher_target_features() -> list[Feature]:
    """Lag-0 pitching features with ``target_`` prefix for training."""
    direct = ("era", "fip", "k_per_9", "bb_per_9", "whip")
    counting = ("h", "hr", "ip", "so")
    features: list[Feature] = []
    for stat in direct:
        features.append(pitching.col(stat).lag(0).alias(f"target_{stat}"))
    for stat in counting:
        features.append(pitching.col(stat).lag(0).alias(f"target_{stat}"))
    return features


def feature_columns(fs: FeatureSet) -> list[str]:
    """Extract feature column names from a FeatureSet, excluding target columns."""
    columns: list[str] = []
    for f in fs.features:
        if isinstance(f, (TransformFeature, DerivedTransformFeature)):
            columns.extend(f.outputs)
        elif isinstance(f, Feature) and not f.name.startswith("target_"):
            columns.append(f.name)
    return columns


def append_training_targets(fs: FeatureSet, targets: Sequence[Feature]) -> FeatureSet:
    """Return a new FeatureSet with prediction features plus target features."""
    return FeatureSet(
        name=f"{fs.name}_train",
        features=fs.features + tuple(targets),
        seasons=fs.seasons,
        source_filter=fs.source_filter,
        spine_filter=fs.spine_filter,
    )
