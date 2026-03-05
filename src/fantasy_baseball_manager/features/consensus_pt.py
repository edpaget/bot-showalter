from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from fantasy_baseball_manager.features import projection
from fantasy_baseball_manager.features.types import (
    AnyFeature,
    DerivedTransformFeature,
    Feature,
    RowTransform,
)

steamer_pa = projection.col("pa").system("steamer").lag(0).alias("steamer_pa")
zips_pa = projection.col("pa").system("zips").lag(0).alias("zips_pa")
steamer_ip = projection.col("ip").system("steamer").lag(0).alias("steamer_ip")
zips_ip = projection.col("ip").system("zips").lag(0).alias("zips_ip")

_DEFAULT_SYSTEMS: tuple[tuple[str, float], ...] = (("steamer", 1.0), ("zips", 1.0))


def _is_missing(val: object) -> bool:
    if val is None:
        return True
    try:
        return math.isnan(float(val))  # type: ignore[arg-type]
    except Exception:
        return True


def make_weighted_consensus_transform(
    source_keys: Sequence[tuple[str, float]],
    output_key: str,
) -> RowTransform:
    """Return a transform that computes a weighted average of N projection columns.

    For each row, collects non-missing values from source columns and computes
    a weighted average, re-normalizing weights over available sources. If all
    sources are missing, returns NaN.
    """

    def _transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
        row = rows[0]
        total_weight = 0.0
        weighted_sum = 0.0
        for col_name, weight in source_keys:
            val = row.get(col_name)
            if not _is_missing(val):
                weighted_sum += float(val) * weight  # type: ignore[arg-type]
                total_weight += weight
        if total_weight == 0.0:
            return {output_key: float("nan")}
        return {output_key: weighted_sum / total_weight}

    return _transform


def make_consensus_transform(steamer_key: str, zips_key: str, output_key: str) -> RowTransform:
    """Return a transform that averages two projection columns with fallback."""
    return make_weighted_consensus_transform(
        [(steamer_key, 1.0), (zips_key, 1.0)],
        output_key,
    )


def build_consensus_features(
    stat: str,
    systems: Sequence[tuple[str, float]] = _DEFAULT_SYSTEMS,
) -> tuple[list[Feature], DerivedTransformFeature]:
    """Build projection features and a consensus derived feature for N systems.

    Returns (list of per-system projection features, one DerivedTransformFeature
    for the weighted consensus).
    """
    proj_features: list[Feature] = []
    source_keys: list[tuple[str, float]] = []
    for system_name, weight in systems:
        alias = f"{system_name}_{stat}"
        feature = projection.col(stat).system(system_name).lag(0).alias(alias)
        proj_features.append(feature)
        source_keys.append((alias, weight))

    consensus_name = f"consensus_{stat}"
    consensus = DerivedTransformFeature(
        name=consensus_name,
        inputs=tuple(alias for alias, _ in source_keys),
        group_by=("player_id", "season"),
        transform=make_weighted_consensus_transform(source_keys, consensus_name),
        outputs=(consensus_name,),
    )
    return proj_features, consensus


CONSENSUS_PA = DerivedTransformFeature(
    name="consensus_pa",
    inputs=("steamer_pa", "zips_pa"),
    group_by=("player_id", "season"),
    transform=make_consensus_transform("steamer_pa", "zips_pa", "consensus_pa"),
    outputs=("consensus_pa",),
)

CONSENSUS_IP = DerivedTransformFeature(
    name="consensus_ip",
    inputs=("steamer_ip", "zips_ip"),
    group_by=("player_id", "season"),
    transform=make_consensus_transform("steamer_ip", "zips_ip", "consensus_ip"),
    outputs=("consensus_ip",),
)


def batting_consensus_features() -> list[AnyFeature]:
    return [steamer_pa, zips_pa, CONSENSUS_PA]


def pitching_consensus_features() -> list[AnyFeature]:
    return [steamer_ip, zips_ip, CONSENSUS_IP]
