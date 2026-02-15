from typing import Any

from fantasy_baseball_manager.domain.evaluation import SystemMetrics
from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.models.protocols import (
    Ablatable,
    AblationResult,
    Evaluable,
    FineTunable,
    FeatureIntrospectable,
    Model,
    ModelConfig,
    Predictable,
    PredictResult,
    Preparable,
    PrepareResult,
    Trainable,
    TrainResult,
)
from fantasy_baseball_manager.models.statcast_gbm.model import StatcastGBMModel


class FakeAssembler:
    def __init__(self) -> None:
        self._next_id = 1

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.get_or_materialize(feature_set)

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        handle = DatasetHandle(
            dataset_id=self._next_id,
            feature_set_id=self._next_id,
            table_name=f"ds_{feature_set.name}",
            row_count=0,
            seasons=feature_set.seasons,
        )
        self._next_id += 1
        return handle

    def split(
        self,
        handle: DatasetHandle,
        train: range | list[int],
        validation: list[int] | None = None,
        holdout: list[int] | None = None,
    ) -> DatasetSplits:
        return DatasetSplits(train=handle, validation=None, holdout=None)

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]:
        return []


class TestStatcastGBMProtocol:
    def test_is_model(self) -> None:
        assert isinstance(StatcastGBMModel(), Model)

    def test_is_preparable(self) -> None:
        assert isinstance(StatcastGBMModel(), Preparable)

    def test_is_trainable(self) -> None:
        assert isinstance(StatcastGBMModel(), Trainable)

    def test_is_evaluable(self) -> None:
        assert isinstance(StatcastGBMModel(), Evaluable)

    def test_is_predictable(self) -> None:
        assert isinstance(StatcastGBMModel(), Predictable)

    def test_is_ablatable(self) -> None:
        assert isinstance(StatcastGBMModel(), Ablatable)

    def test_is_feature_introspectable(self) -> None:
        assert isinstance(StatcastGBMModel(), FeatureIntrospectable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(StatcastGBMModel(), FineTunable)

    def test_name(self) -> None:
        assert StatcastGBMModel().name == "statcast-gbm"

    def test_artifact_type(self) -> None:
        assert StatcastGBMModel().artifact_type == "file"

    def test_supported_operations(self) -> None:
        ops = StatcastGBMModel().supported_operations
        assert ops == frozenset({"prepare", "train", "evaluate", "predict", "ablate"})

    def test_declared_features_not_empty(self) -> None:
        features = StatcastGBMModel().declared_features
        assert len(features) > 0


class TestStatcastGBMPrepare:
    def test_prepare_returns_result(self) -> None:
        assembler = FakeAssembler()
        model = StatcastGBMModel(assembler=assembler)
        config = ModelConfig(seasons=[2023])
        result = model.prepare(config)
        assert isinstance(result, PrepareResult)
        assert result.model_name == "statcast-gbm"


class TestStatcastGBMTrain:
    def test_train_returns_result(self) -> None:
        model = StatcastGBMModel()
        config = ModelConfig(seasons=[2023])
        result = model.train(config)
        assert isinstance(result, TrainResult)
        assert result.model_name == "statcast-gbm"
        assert isinstance(result.metrics, dict)


class TestStatcastGBMEvaluate:
    def test_evaluate_returns_result(self) -> None:
        model = StatcastGBMModel()
        config = ModelConfig(seasons=[2023])
        result = model.evaluate(config)
        assert isinstance(result, SystemMetrics)
        assert result.system == "statcast-gbm"


class TestStatcastGBMPredict:
    def test_predict_returns_result(self) -> None:
        model = StatcastGBMModel()
        config = ModelConfig(seasons=[2023])
        result = model.predict(config)
        assert isinstance(result, PredictResult)
        assert result.model_name == "statcast-gbm"
        assert isinstance(result.predictions, list)


class TestStatcastGBMAblate:
    def test_ablate_returns_result(self) -> None:
        model = StatcastGBMModel()
        config = ModelConfig(seasons=[2023])
        result = model.ablate(config)
        assert isinstance(result, AblationResult)
        assert result.model_name == "statcast-gbm"
        assert isinstance(result.feature_impacts, dict)
