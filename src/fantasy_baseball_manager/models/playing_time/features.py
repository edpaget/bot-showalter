from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from fantasy_baseball_manager.features import (
    AnyFeature,
    DerivedTransformFeature,
    Feature,
    batting,
    il_stint,
    make_weighted_consensus_transform,
    pitching,
    player,
    projection,
)
from fantasy_baseball_manager.features.transforms.playing_time import (
    make_il_severity_transform,
    make_il_summary_transform,
    make_pt_interaction_transform,
    make_pt_trend_transform,
    make_starter_ratio_transform,
    make_war_threshold_transform,
)

_DEFAULT_PT_SYSTEMS: tuple[tuple[str, float], ...] = (("steamer", 1.0), ("zips", 1.0))


def _build_projection_features(
    stat: str,
    pt_systems: Sequence[tuple[str, float]] = _DEFAULT_PT_SYSTEMS,
) -> list[Feature]:
    """Build per-system projection features for the given stat.

    Uses ``lag(-1)`` so that the SQL join resolves to
    ``projection.season = spine.season + 1`` — i.e. the projection for the
    *target* season rather than the feature season.  This ensures that NPB
    imports, TJ returns, and other players whose first steamer/zips entry is
    for the target year get populated consensus values.
    """
    if not pt_systems:
        return []
    features: list[Feature] = []
    for system_name, _weight in pt_systems:
        alias = f"{system_name}_{stat}"
        features.append(projection.col(stat).system(system_name).lag(-1).alias(alias))
    return features


def build_batting_pt_features(
    lags: int = 3,
    pt_systems: Sequence[tuple[str, float]] = _DEFAULT_PT_SYSTEMS,
) -> list[Feature]:
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
    features.extend(_build_projection_features("pa", pt_systems))
    return features


def build_pitching_pt_features(
    lags: int = 3,
    pt_systems: Sequence[tuple[str, float]] = _DEFAULT_PT_SYSTEMS,
) -> list[Feature]:
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
    features.extend(_build_projection_features("ip", pt_systems))
    return features


def _build_consensus_derived(
    stat: str,
    pt_systems: Sequence[tuple[str, float]],
) -> DerivedTransformFeature:
    """Build the consensus DerivedTransformFeature for the given stat and systems."""
    source_keys: list[tuple[str, float]] = []
    for system_name, weight in pt_systems:
        source_keys.append((f"{system_name}_{stat}", weight))
    consensus_name = f"consensus_{stat}"
    return DerivedTransformFeature(
        name=consensus_name,
        inputs=tuple(alias for alias, _ in source_keys),
        group_by=("player_id", "season"),
        transform=make_weighted_consensus_transform(source_keys, consensus_name),
        outputs=(consensus_name,),
    )


def build_batting_pt_derived_transforms(
    lags: int = 3,
    pt_systems: Sequence[tuple[str, float]] = _DEFAULT_PT_SYSTEMS,
) -> list[DerivedTransformFeature]:
    """Build derived transforms for batting playing-time projection."""
    il_inputs = tuple(f"il_days_{i}" for i in range(1, lags + 1)) + ("il_stints_1",)
    if lags >= 2:
        il_inputs = il_inputs + ("il_stints_2",)

    pt_inputs: tuple[str, ...] = ("pa_1",)
    if lags >= 2:
        pt_inputs = ("pa_1", "pa_2")

    transforms = [
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
    if pt_systems:
        transforms.append(_build_consensus_derived("pa", pt_systems))
    return transforms


def build_pitching_pt_derived_transforms(
    lags: int = 3,
    pt_systems: Sequence[tuple[str, float]] = _DEFAULT_PT_SYSTEMS,
) -> list[DerivedTransformFeature]:
    """Build derived transforms for pitching playing-time projection."""
    il_inputs = tuple(f"il_days_{i}" for i in range(1, lags + 1)) + ("il_stints_1",)
    if lags >= 2:
        il_inputs = il_inputs + ("il_stints_2",)

    pt_inputs: tuple[str, ...] = ("ip_1",)
    if lags >= 2:
        pt_inputs = ("ip_1", "ip_2")

    transforms = [
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
    if pt_systems:
        transforms.append(_build_consensus_derived("ip", pt_systems))
    return transforms


def build_batting_pt_training_features(
    lags: int = 3,
    pt_systems: Sequence[tuple[str, float]] = _DEFAULT_PT_SYSTEMS,
) -> list[AnyFeature]:
    """Features + target (pa at lag 0) for training."""
    features: list[AnyFeature] = list(build_batting_pt_features(lags, pt_systems))
    features.extend(build_batting_pt_derived_transforms(lags, pt_systems))
    features.append(batting.col("pa").lag(0).alias("target_pa"))
    return features


def build_pitching_pt_training_features(
    lags: int = 3,
    pt_systems: Sequence[tuple[str, float]] = _DEFAULT_PT_SYSTEMS,
) -> list[AnyFeature]:
    """Features + target (ip at lag 0) for training."""
    features: list[AnyFeature] = list(build_pitching_pt_features(lags, pt_systems))
    features.extend(build_pitching_pt_derived_transforms(lags, pt_systems))
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


# IL-related columns that are always excluded from regression.
_IL_EXCLUDED_COLUMNS = frozenset(
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


def _excluded_columns(
    pt_systems: Sequence[tuple[str, float]] = _DEFAULT_PT_SYSTEMS,
) -> frozenset[str]:
    """Compute the set of columns excluded from regression.

    Includes IL columns plus per-system raw projection columns (e.g.
    steamer_pa, zips_ip) which are only used as inputs to the consensus
    derived feature.
    """
    system_cols = frozenset(f"{sys}_{stat}" for sys, _ in pt_systems for stat in ("pa", "ip"))
    return _IL_EXCLUDED_COLUMNS | system_cols


# Backward-compat alias
_EXCLUDED_COLUMNS = _excluded_columns()


def batting_pt_feature_columns(
    lags: int = 3,
    pt_systems: Sequence[tuple[str, float]] = _DEFAULT_PT_SYSTEMS,
) -> list[str]:
    """Return ordered list of feature column names for batting PT model."""
    features: list[AnyFeature] = list(build_batting_pt_features(lags, pt_systems))
    features.extend(build_batting_pt_derived_transforms(lags, pt_systems))
    excluded = _excluded_columns(pt_systems)
    return [c for c in _collect_feature_names(features) if c not in excluded]


def pitching_pt_feature_columns(
    lags: int = 3,
    pt_systems: Sequence[tuple[str, float]] = _DEFAULT_PT_SYSTEMS,
) -> list[str]:
    """Return ordered list of feature column names for pitching PT model."""
    features: list[AnyFeature] = list(build_pitching_pt_features(lags, pt_systems))
    features.extend(build_pitching_pt_derived_transforms(lags, pt_systems))
    excluded = _excluded_columns(pt_systems)
    return [c for c in _collect_feature_names(features) if c not in excluded]
