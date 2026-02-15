from fantasy_baseball_manager.features import batting, pitching, player
from fantasy_baseball_manager.features.types import Feature


def build_batting_pt_features(lags: int = 2) -> list[Feature]:
    """Build features for batting playing-time projection: age + lagged PA."""
    features: list[Feature] = [player.age()]
    for lag_n in range(1, lags + 1):
        features.append(batting.col("pa").lag(lag_n).alias(f"pa_{lag_n}"))
    return features


def build_pitching_pt_features(lags: int = 2) -> list[Feature]:
    """Build features for pitching playing-time projection: age + lagged IP/G/GS."""
    features: list[Feature] = [player.age()]
    for lag_n in range(1, lags + 1):
        features.append(pitching.col("ip").lag(lag_n).alias(f"ip_{lag_n}"))
        features.append(pitching.col("g").lag(lag_n).alias(f"g_{lag_n}"))
        features.append(pitching.col("gs").lag(lag_n).alias(f"gs_{lag_n}"))
    return features
