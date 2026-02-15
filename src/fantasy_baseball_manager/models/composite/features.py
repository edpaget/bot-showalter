from collections.abc import Sequence

from fantasy_baseball_manager.features import batting, pitching, player, projection
from fantasy_baseball_manager.features.types import AnyFeature
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
