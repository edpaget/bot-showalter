from __future__ import annotations

import enum
import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Protocol


class Source(enum.Enum):
    BATTING = "batting"
    PITCHING = "pitching"
    PLAYER = "player"
    PROJECTION = "projection"
    STATCAST = "statcast"
    IL_STINT = "il_stint"


@dataclass(frozen=True)
class SpineFilter:
    min_pa: int | None = None
    min_ip: float | None = None
    player_type: str | None = None


@dataclass(frozen=True)
class Feature:
    name: str
    source: Source
    column: str
    lag: int = 0
    window: int = 1
    aggregate: str | None = None
    denominator: str | None = None
    computed: str | None = None
    system: str | None = None
    version: str | None = None
    distribution_column: str | None = None


@dataclass(frozen=True)
class DeltaFeature:
    name: str
    left: Feature
    right: Feature


class RowTransform(Protocol):
    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TransformFeature:
    name: str
    source: Source
    columns: tuple[str, ...]
    group_by: tuple[str, ...]
    transform: RowTransform
    outputs: tuple[str, ...]
    version: str | None = None


@dataclass(frozen=True)
class DerivedTransformFeature:
    name: str
    inputs: tuple[str, ...]
    group_by: tuple[str, ...]
    transform: RowTransform
    outputs: tuple[str, ...]
    version: str | None = None


type AnyFeature = Feature | DeltaFeature | TransformFeature | DerivedTransformFeature


class FeatureBuilder:
    def __init__(self, source: Source, column: str) -> None:
        self._source = source
        self._column = column
        self._lag = 0
        self._window = 1
        self._aggregate: str | None = None
        self._denominator: str | None = None
        self._computed: str | None = None
        self._system: str | None = None
        self._version: str | None = None
        self._distribution_column: str | None = None

    def lag(self, n: int) -> FeatureBuilder:
        self._lag = n
        return self

    def rolling_mean(self, n: int) -> FeatureBuilder:
        self._window = n
        self._aggregate = "mean"
        return self

    def rolling_sum(self, n: int) -> FeatureBuilder:
        self._window = n
        self._aggregate = "sum"
        return self

    def per(self, denominator: str) -> FeatureBuilder:
        self._denominator = denominator
        return self

    def system(self, name: str) -> FeatureBuilder:
        self._system = name
        return self

    def version(self, name: str) -> FeatureBuilder:
        self._version = name
        return self

    def percentile(self, p: int) -> FeatureBuilder:
        if self._source != Source.PROJECTION:
            msg = "percentile() is only available on PROJECTION features"
            raise ValueError(msg)
        allowed = {10, 25, 50, 75, 90}
        if p not in allowed:
            msg = f"percentile must be one of 10, 25, 50, 75, 90, got {p}"
            raise ValueError(msg)
        self._distribution_column = f"p{p}"
        return self

    def std(self) -> FeatureBuilder:
        if self._source != Source.PROJECTION:
            msg = "std() is only available on PROJECTION features"
            raise ValueError(msg)
        self._distribution_column = "std"
        return self

    def alias(self, name: str) -> Feature:
        if self._source == Source.PROJECTION and self._system is None:
            msg = "Projection features require .system() to be specified"
            raise ValueError(msg)
        return Feature(
            name=name,
            source=self._source,
            column=self._column,
            lag=self._lag,
            window=self._window,
            aggregate=self._aggregate,
            denominator=self._denominator,
            computed=self._computed,
            system=self._system,
            version=self._version,
            distribution_column=self._distribution_column,
        )


class SourceRef:
    def __init__(self, source: Source) -> None:
        self.source = source

    def col(self, column: str) -> FeatureBuilder:
        return FeatureBuilder(source=self.source, column=column)

    def age(self) -> Feature:
        if self.source != Source.PLAYER:
            msg = "age() is only available on the player source"
            raise ValueError(msg)
        return Feature(name="age", source=self.source, column="", computed="age")

    def positions(self) -> Feature:
        if self.source != Source.PLAYER:
            msg = "positions() is only available on the player source"
            raise ValueError(msg)
        return Feature(name="position", source=self.source, column="", computed="positions")


def _transform_feature_to_dict(f: TransformFeature) -> dict[str, object]:
    if f.version is not None:
        identity = f.version
    else:
        identity = inspect.getsource(f.transform)
    return {
        "type": "transform",
        "name": f.name,
        "source": f.source.value,
        "columns": list(f.columns),
        "group_by": list(f.group_by),
        "outputs": list(f.outputs),
        "transform_identity": identity,
    }


def _derived_transform_feature_to_dict(f: DerivedTransformFeature) -> dict[str, object]:
    if f.version is not None:
        identity = f.version
    else:
        identity = inspect.getsource(f.transform)
    return {
        "type": "derived_transform",
        "name": f.name,
        "inputs": list(f.inputs),
        "group_by": list(f.group_by),
        "outputs": list(f.outputs),
        "transform_identity": identity,
    }


def _feature_to_dict(f: AnyFeature) -> dict[str, object]:
    if isinstance(f, DerivedTransformFeature):
        return _derived_transform_feature_to_dict(f)
    if isinstance(f, TransformFeature):
        return _transform_feature_to_dict(f)
    if isinstance(f, DeltaFeature):
        return {
            "type": "delta",
            "name": f.name,
            "left": _feature_to_dict(f.left),
            "right": _feature_to_dict(f.right),
        }
    assert isinstance(f, Feature)
    return {
        "name": f.name,
        "source": f.source.value,
        "column": f.column,
        "lag": f.lag,
        "window": f.window,
        "aggregate": f.aggregate,
        "denominator": f.denominator,
        "computed": f.computed,
        "system": f.system,
        "version": f.version,
        "distribution_column": f.distribution_column,
    }


def _spine_filter_to_dict(sf: SpineFilter) -> dict[str, object]:
    return {
        "min_pa": sf.min_pa,
        "min_ip": sf.min_ip,
        "player_type": sf.player_type,
    }


def _compute_version(
    features: tuple[AnyFeature, ...],
    seasons: tuple[int, ...],
    source_filter: str | None,
    spine_filter: SpineFilter,
) -> str:
    payload: dict[str, object] = {
        "features": [_feature_to_dict(f) for f in features],
        "seasons": list(seasons),
        "source_filter": source_filter,
        "spine_filter": _spine_filter_to_dict(spine_filter),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return digest[:12]


@dataclass(frozen=True)
class FeatureSet:
    name: str
    features: tuple[AnyFeature, ...]
    seasons: tuple[int, ...]
    source_filter: str | None = None
    spine_filter: SpineFilter = field(default_factory=SpineFilter)
    version: str = field(init=False)

    def __post_init__(self) -> None:
        version = _compute_version(self.features, self.seasons, self.source_filter, self.spine_filter)
        object.__setattr__(self, "version", version)


@dataclass(frozen=True)
class DatasetHandle:
    dataset_id: int
    feature_set_id: int
    table_name: str
    row_count: int
    seasons: tuple[int, ...]


@dataclass(frozen=True)
class DatasetSplits:
    train: DatasetHandle
    validation: DatasetHandle | None
    holdout: DatasetHandle | None
