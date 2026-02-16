from fantasy_baseball_manager.features.groups import (
    FeatureGroup,
    compose_feature_set,
    get_group,
    list_groups,
    register_group,
)
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
    "FeatureGroup",
    "FeatureSet",
    "RowTransform",
    "Source",
    "SourceRef",
    "SpineFilter",
    "TransformFeature",
    "batting",
    "compose_feature_set",
    "delta",
    "get_group",
    "il_stint",
    "list_groups",
    "pitching",
    "player",
    "projection",
    "register_group",
    "statcast",
]
