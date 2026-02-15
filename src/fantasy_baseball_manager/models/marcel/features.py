from collections.abc import Sequence

from fantasy_baseball_manager.features import batting, pitching, player
from fantasy_baseball_manager.features.transforms.league_averages import make_league_avg_transform
from fantasy_baseball_manager.features.transforms.weighted_rates import make_weighted_rates_transform
from fantasy_baseball_manager.features.types import DerivedTransformFeature, Feature


def build_batting_features(categories: Sequence[str], lags: int = 3) -> list[Feature]:
    features: list[Feature] = [player.age(), player.positions()]
    for lag_n in range(1, lags + 1):
        features.append(batting.col("pa").lag(lag_n).alias(f"pa_{lag_n}"))
        for cat in categories:
            features.append(batting.col(cat).lag(lag_n).alias(f"{cat}_{lag_n}"))
    return features


def _weights_version(weights: tuple[float, ...], pt_column: str) -> str:
    return f"{pt_column}:{','.join(str(w) for w in weights)}"


def build_batting_weighted_rates(
    categories: Sequence[str],
    weights: tuple[float, ...],
) -> DerivedTransformFeature:
    """Build a DerivedTransformFeature for weighted-average batting rates."""
    n = len(weights)
    cats = list(categories)
    inputs: list[str] = []
    for lag in range(1, n + 1):
        inputs.append(f"pa_{lag}")
        for cat in cats:
            inputs.append(f"{cat}_{lag}")
    outputs = tuple(f"{cat}_wavg" for cat in cats) + ("weighted_pt",)
    return DerivedTransformFeature(
        name="batting_weighted_rates",
        inputs=tuple(inputs),
        group_by=("player_id", "season"),
        transform=make_weighted_rates_transform(categories, weights, "pa"),
        outputs=outputs,
        version=_weights_version(weights, "pa"),
    )


def build_pitching_weighted_rates(
    categories: Sequence[str],
    weights: tuple[float, ...],
) -> DerivedTransformFeature:
    """Build a DerivedTransformFeature for weighted-average pitching rates."""
    n = len(weights)
    cats = list(categories)
    inputs: list[str] = []
    for lag in range(1, n + 1):
        inputs.append(f"ip_{lag}")
        for cat in cats:
            inputs.append(f"{cat}_{lag}")
    outputs = tuple(f"{cat}_wavg" for cat in cats) + ("weighted_pt",)
    return DerivedTransformFeature(
        name="pitching_weighted_rates",
        inputs=tuple(inputs),
        group_by=("player_id", "season"),
        transform=make_weighted_rates_transform(categories, weights, "ip"),
        outputs=outputs,
        version=_weights_version(weights, "ip"),
    )


def build_batting_league_averages(
    categories: Sequence[str],
) -> DerivedTransformFeature:
    """Build a DerivedTransformFeature for league-average batting rates."""
    cats = list(categories)
    inputs = tuple(f"{cat}_1" for cat in cats) + ("pa_1",)
    outputs = tuple(f"league_{cat}_rate" for cat in cats)
    return DerivedTransformFeature(
        name="batting_league_averages",
        inputs=inputs,
        group_by=("season",),
        transform=make_league_avg_transform(categories, "pa"),
        outputs=outputs,
    )


def build_pitching_league_averages(
    categories: Sequence[str],
) -> DerivedTransformFeature:
    """Build a DerivedTransformFeature for league-average pitching rates."""
    cats = list(categories)
    inputs = tuple(f"{cat}_1" for cat in cats) + ("ip_1",)
    outputs = tuple(f"league_{cat}_rate" for cat in cats)
    return DerivedTransformFeature(
        name="pitching_league_averages",
        inputs=inputs,
        group_by=("season",),
        transform=make_league_avg_transform(categories, "ip"),
        outputs=outputs,
    )


def build_pitching_features(categories: Sequence[str], lags: int = 3) -> list[Feature]:
    features: list[Feature] = [player.age(), player.positions()]
    for lag_n in range(1, lags + 1):
        features.append(pitching.col("ip").lag(lag_n).alias(f"ip_{lag_n}"))
        features.append(pitching.col("g").lag(lag_n).alias(f"g_{lag_n}"))
        features.append(pitching.col("gs").lag(lag_n).alias(f"gs_{lag_n}"))
        for cat in categories:
            features.append(pitching.col(cat).lag(lag_n).alias(f"{cat}_{lag_n}"))
    return features
