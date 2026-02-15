from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import (
    AnyFeature,
    DatasetHandle,
    DatasetSplits,
    DeltaFeature,
    DerivedTransformFeature,
    Feature,
    FeatureBuilder,
    FeatureSet,
    RowTransform,
    Source,
    SourceRef,
    SpineFilter,
    TransformFeature,
)

batting = SourceRef(Source.BATTING)
pitching = SourceRef(Source.PITCHING)
player = SourceRef(Source.PLAYER)
projection = SourceRef(Source.PROJECTION)
statcast = SourceRef(Source.STATCAST)
il_stint = SourceRef(Source.IL_STINT)


def delta(name: str, left: Feature, right: Feature) -> DeltaFeature:
    return DeltaFeature(name=name, left=left, right=right)


__all__ = [
    "AnyFeature",
    "DatasetAssembler",
    "DatasetHandle",
    "DatasetSplits",
    "DeltaFeature",
    "DerivedTransformFeature",
    "Feature",
    "FeatureBuilder",
    "FeatureSet",
    "RowTransform",
    "Source",
    "SourceRef",
    "SpineFilter",
    "TransformFeature",
    "batting",
    "delta",
    "il_stint",
    "pitching",
    "player",
    "projection",
    "statcast",
]
