from typing import Any

from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import (
    DatasetHandle,
    DatasetSplits,
    FeatureSet,
)


class FakeDatasetAssembler:
    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return DatasetHandle(
            dataset_id=1,
            feature_set_id=1,
            table_name="ds_1",
            row_count=0,
            seasons=feature_set.seasons,
        )

    def split(
        self,
        handle: DatasetHandle,
        train: range | list[int],
        validation: list[int] | None = None,
        holdout: list[int] | None = None,
    ) -> DatasetSplits:
        return DatasetSplits(
            train=handle,
            validation=None,
            holdout=None,
        )

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.materialize(feature_set)

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]:
        return []


class TestDatasetAssemblerProtocol:
    def test_fake_is_instance(self) -> None:
        fake = FakeDatasetAssembler()
        assert isinstance(fake, DatasetAssembler)

    def test_fake_class_is_subclass(self) -> None:
        assert issubclass(FakeDatasetAssembler, DatasetAssembler)
