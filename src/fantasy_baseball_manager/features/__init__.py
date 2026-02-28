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

# These imports must come after the SourceRef assignments above because the
# imported modules reference `from fantasy_baseball_manager.features import projection`
# (and other SourceRefs) at module scope.
from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler  # noqa: E402
from fantasy_baseball_manager.features.consensus_pt import (  # noqa: E402
    batting_consensus_features,
    pitching_consensus_features,
)
from fantasy_baseball_manager.features.group_library import (  # noqa: E402
    make_batting_counting_lags,
    make_batting_rate_lags,
    make_pitching_counting_lags,
)


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
    "SqliteDatasetAssembler",
    "TransformFeature",
    "batting",
    "batting_consensus_features",
    "compose_feature_set",
    "delta",
    "get_group",
    "il_stint",
    "list_groups",
    "make_batting_counting_lags",
    "make_batting_rate_lags",
    "make_pitching_counting_lags",
    "pitching",
    "pitching_consensus_features",
    "player",
    "projection",
    "register_group",
    "statcast",
]
