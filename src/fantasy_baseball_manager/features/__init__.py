from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import (
    DatasetHandle,
    DatasetSplits,
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

__all__ = [
    "DatasetAssembler",
    "DatasetHandle",
    "DatasetSplits",
    "Feature",
    "FeatureBuilder",
    "FeatureSet",
    "Source",
    "SourceRef",
    "SpineFilter",
    "batting",
    "pitching",
    "player",
]
