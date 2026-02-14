from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field


class Source(enum.Enum):
    BATTING = "batting"
    PITCHING = "pitching"
    PLAYER = "player"
    PROJECTION = "projection"


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


class FeatureBuilder:
    def __init__(self, source: Source, column: str) -> None:
        self._source = source
        self._column = column
        self._lag = 0
        self._window = 1
        self._aggregate: str | None = None
        self._denominator: str | None = None
        self._computed: str | None = None

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

    def alias(self, name: str) -> Feature:
        return Feature(
            name=name,
            source=self._source,
            column=self._column,
            lag=self._lag,
            window=self._window,
            aggregate=self._aggregate,
            denominator=self._denominator,
            computed=self._computed,
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


def _feature_to_dict(f: Feature) -> dict[str, object]:
    return {
        "name": f.name,
        "source": f.source.value,
        "column": f.column,
        "lag": f.lag,
        "window": f.window,
        "aggregate": f.aggregate,
        "denominator": f.denominator,
        "computed": f.computed,
    }


def _spine_filter_to_dict(sf: SpineFilter) -> dict[str, object]:
    return {
        "min_pa": sf.min_pa,
        "min_ip": sf.min_ip,
        "player_type": sf.player_type,
    }


def _compute_version(
    features: tuple[Feature, ...],
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
    features: tuple[Feature, ...]
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
