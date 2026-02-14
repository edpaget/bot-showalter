from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from fantasy_baseball_manager.features.types import (
    DatasetHandle,
    DatasetSplits,
    FeatureSet,
)


@runtime_checkable
class DatasetAssembler(Protocol):
    def materialize(self, feature_set: FeatureSet) -> DatasetHandle: ...

    def split(
        self,
        handle: DatasetHandle,
        train: range | list[int],
        validation: list[int] | None = None,
        holdout: list[int] | None = None,
    ) -> DatasetSplits: ...

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle: ...

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]: ...
