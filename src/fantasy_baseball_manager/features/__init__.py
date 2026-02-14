from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import (
    AnyFeature,
    DatasetHandle,
    DatasetSplits,
    DeltaFeature,
    Feature,
    FeatureBuilder,
    FeatureSet,
    Source,
    SourceRef,
    SpineFilter,
)

batting = SourceRef(Source.BATTING)
pitching = SourceRef(Source.PITCHING)
player = SourceRef(Source.PLAYER)
projection = SourceRef(Source.PROJECTION)


def delta(name: str, left: Feature, right: Feature) -> DeltaFeature:
    return DeltaFeature(name=name, left=left, right=right)


__all__ = [
    "AnyFeature",
    "DatasetAssembler",
    "DatasetHandle",
    "DatasetSplits",
    "DeltaFeature",
    "Feature",
    "FeatureBuilder",
    "FeatureSet",
    "Source",
    "SourceRef",
    "SpineFilter",
    "batting",
    "delta",
    "pitching",
    "player",
    "projection",
]
