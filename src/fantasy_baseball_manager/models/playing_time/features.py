from fantasy_baseball_manager.features import batting, il_stint, pitching, player
from fantasy_baseball_manager.features.transforms.playing_time import (
    make_il_severity_transform,
    make_il_summary_transform,
    make_pt_interaction_transform,
    make_pt_trend_transform,
    make_starter_ratio_transform,
    make_war_threshold_transform,
)
from fantasy_baseball_manager.features.types import AnyFeature, DerivedTransformFeature, Feature


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
    il_inputs = tuple(f"il_days_{i}" for i in range(1, lags + 1)) + ("il_stints_1",)
    if lags >= 2:
        il_inputs = il_inputs + ("il_stints_2",)

    pt_inputs: tuple[str, ...] = ("pa_1",)
    if lags >= 2:
        pt_inputs = ("pa_1", "pa_2")

    return [
        DerivedTransformFeature(
            name="il_summary",
            inputs=il_inputs,
            group_by=("player_id", "season"),
            transform=make_il_summary_transform(lags),
            outputs=("il_days_3yr", "il_recurrence"),
        ),
        DerivedTransformFeature(
            name="batting_pt_trend",
            inputs=pt_inputs,
            group_by=("player_id", "season"),
            transform=make_pt_trend_transform("pa"),
            outputs=("pt_trend",),
        ),
        DerivedTransformFeature(
            name="war_threshold",
            inputs=("war_1",),
            group_by=("player_id", "season"),
            transform=make_war_threshold_transform(),
            outputs=("war_above_2", "war_above_4", "war_below_0"),
        ),
        DerivedTransformFeature(
            name="il_severity",
            inputs=("il_days_1",),
            group_by=("player_id", "season"),
            transform=make_il_severity_transform(),
            outputs=("il_minor", "il_moderate", "il_severe"),
        ),
        DerivedTransformFeature(
            name="pt_interaction",
            inputs=("war_1", "pt_trend", "age", "il_recurrence"),
            group_by=("player_id", "season"),
            transform=make_pt_interaction_transform(),
            outputs=("war_trend", "age_il_interact"),
        ),
    ]


def build_pitching_pt_derived_transforms(lags: int = 3) -> list[DerivedTransformFeature]:
    """Build derived transforms for pitching playing-time projection."""
    il_inputs = tuple(f"il_days_{i}" for i in range(1, lags + 1)) + ("il_stints_1",)
    if lags >= 2:
        il_inputs = il_inputs + ("il_stints_2",)

    pt_inputs: tuple[str, ...] = ("ip_1",)
    if lags >= 2:
        pt_inputs = ("ip_1", "ip_2")

    return [
        DerivedTransformFeature(
            name="il_summary",
            inputs=il_inputs,
            group_by=("player_id", "season"),
            transform=make_il_summary_transform(lags),
            outputs=("il_days_3yr", "il_recurrence"),
        ),
        DerivedTransformFeature(
            name="pitching_pt_trend",
            inputs=pt_inputs,
            group_by=("player_id", "season"),
            transform=make_pt_trend_transform("ip"),
            outputs=("pt_trend",),
        ),
        DerivedTransformFeature(
            name="war_threshold",
            inputs=("war_1",),
            group_by=("player_id", "season"),
            transform=make_war_threshold_transform(),
            outputs=("war_above_2", "war_above_4", "war_below_0"),
        ),
        DerivedTransformFeature(
            name="il_severity",
            inputs=("il_days_1",),
            group_by=("player_id", "season"),
            transform=make_il_severity_transform(),
            outputs=("il_minor", "il_moderate", "il_severe"),
        ),
        DerivedTransformFeature(
            name="starter_ratio",
            inputs=("gs_1", "g_1"),
            group_by=("player_id", "season"),
            transform=make_starter_ratio_transform(),
            outputs=("starter_ratio",),
        ),
        DerivedTransformFeature(
            name="pt_interaction",
            inputs=("war_1", "pt_trend", "age", "il_recurrence"),
            group_by=("player_id", "season"),
            transform=make_pt_interaction_transform(),
            outputs=("war_trend", "age_il_interact"),
        ),
    ]


def build_batting_pt_training_features(lags: int = 3) -> list[AnyFeature]:
    """Features + target (pa at lag 0) for training."""
    features: list[AnyFeature] = list(build_batting_pt_features(lags))
    features.extend(build_batting_pt_derived_transforms(lags))
    features.append(batting.col("pa").lag(0).alias("target_pa"))
    return features


def build_pitching_pt_training_features(lags: int = 3) -> list[AnyFeature]:
    """Features + target (ip at lag 0) for training."""
    features: list[AnyFeature] = list(build_pitching_pt_features(lags))
    features.extend(build_pitching_pt_derived_transforms(lags))
    features.append(pitching.col("ip").lag(0).alias("target_ip"))
    return features


def _collect_feature_names(features: list[AnyFeature]) -> list[str]:
    """Collect output column names from a list of features, excluding metadata."""
    columns: list[str] = []
    for f in features:
        if isinstance(f, DerivedTransformFeature):
            columns.extend(f.outputs)
        elif isinstance(f, Feature):
            columns.append(f.name)
    return columns


# IL columns are gathered for residual bucketing but excluded from regression.
_IL_COLUMNS = frozenset(
    {
        "il_days_1",
        "il_days_2",
        "il_days_3",
        "il_stints_1",
        "il_stints_2",
        "il_days_3yr",
        "il_recurrence",
        "il_minor",
        "il_moderate",
        "il_severe",
        "age_il_interact",
    }
)


def batting_pt_feature_columns(lags: int = 3) -> list[str]:
    """Return ordered list of feature column names for batting PT model."""
    features: list[AnyFeature] = list(build_batting_pt_features(lags))
    features.extend(build_batting_pt_derived_transforms(lags))
    return [c for c in _collect_feature_names(features) if c not in _IL_COLUMNS]


def pitching_pt_feature_columns(lags: int = 3) -> list[str]:
    """Return ordered list of feature column names for pitching PT model."""
    features: list[AnyFeature] = list(build_pitching_pt_features(lags))
    features.extend(build_pitching_pt_derived_transforms(lags))
    return [c for c in _collect_feature_names(features) if c not in _IL_COLUMNS]
