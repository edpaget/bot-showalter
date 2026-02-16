from __future__ import annotations

import math
from typing import Any

from fantasy_baseball_manager.features import projection
from fantasy_baseball_manager.features.types import AnyFeature, DerivedTransformFeature, RowTransform

steamer_pa = projection.col("pa").system("steamer").lag(0).alias("steamer_pa")
zips_pa = projection.col("pa").system("zips").lag(0).alias("zips_pa")
steamer_ip = projection.col("ip").system("steamer").lag(0).alias("steamer_ip")
zips_ip = projection.col("ip").system("zips").lag(0).alias("zips_ip")


def _is_missing(val: object) -> bool:
    if val is None:
        return True
    try:
        return math.isnan(float(val))  # type: ignore[arg-type]
    except Exception:
        return True


def make_consensus_transform(steamer_key: str, zips_key: str, output_key: str) -> RowTransform:
    """Return a transform that averages two projection columns with fallback."""

    def _transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
        row = rows[0]
        s_val = row.get(steamer_key)
        z_val = row.get(zips_key)
        s_missing = _is_missing(s_val)
        z_missing = _is_missing(z_val)
        if s_missing and z_missing:
            return {output_key: float("nan")}
        if s_missing:
            return {output_key: float(z_val)}  # type: ignore[arg-type]
        if z_missing:
            return {output_key: float(s_val)}  # type: ignore[arg-type]
        return {output_key: (float(s_val) + float(z_val)) / 2.0}  # type: ignore[arg-type]

    return _transform


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
