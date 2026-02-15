from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.models.protocols import (
    Ablatable,
    AblationResult,
    Evaluable,
    FineTunable,
    ModelConfig,
    Predictable,
    PredictResult,
    Preparable,
    PrepareResult,
    Model,
    Trainable,
    TrainResult,
)


class _StubModel:
    @property
    def name(self) -> str:
        return "stub"

    @property
    def description(self) -> str:
        return "A stub model"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train"})

    @property
    def artifact_type(self) -> str:
        return "none"

    def prepare(self, config: ModelConfig) -> PrepareResult:
        return PrepareResult(model_name="stub", rows_processed=0, artifacts_path="")

    def train(self, config: ModelConfig) -> TrainResult:
        return TrainResult(model_name="stub", metrics={}, artifacts_path="")


class _StubModelWithoutArtifactType:
    @property
    def name(self) -> str:
        return "stub"

    @property
    def description(self) -> str:
        return "A stub model"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train"})


class TestProtocolRuntimeChecks:
    def test_stub_is_model(self) -> None:
        assert isinstance(_StubModel(), Model)

    def test_stub_is_preparable(self) -> None:
        assert isinstance(_StubModel(), Preparable)

    def test_stub_is_trainable(self) -> None:
        assert isinstance(_StubModel(), Trainable)

    def test_stub_is_not_evaluable(self) -> None:
        assert not isinstance(_StubModel(), Evaluable)

    def test_stub_is_not_predictable(self) -> None:
        assert not isinstance(_StubModel(), Predictable)

    def test_stub_is_not_finetuneable(self) -> None:
        assert not isinstance(_StubModel(), FineTunable)

    def test_stub_is_not_ablatable(self) -> None:
        assert not isinstance(_StubModel(), Ablatable)

    def test_model_without_artifact_type_is_not_model(self) -> None:
        assert not isinstance(_StubModelWithoutArtifactType(), Model)


class TestResultDataclasses:
    def test_prepare_result_frozen(self) -> None:
        r = PrepareResult(model_name="m", rows_processed=10, artifacts_path="/tmp")
        with pytest.raises(FrozenInstanceError):
            r.model_name = "x"  # type: ignore[misc]

    def test_train_result_frozen(self) -> None:
        r = TrainResult(model_name="m", metrics={"rmse": 0.5}, artifacts_path="/tmp")
        with pytest.raises(FrozenInstanceError):
            r.model_name = "x"  # type: ignore[misc]
        assert r.metrics == {"rmse": 0.5}

    def test_predict_result_frozen(self) -> None:
        r = PredictResult(model_name="m", predictions=[], output_path="/tmp")
        with pytest.raises(FrozenInstanceError):
            r.model_name = "x"  # type: ignore[misc]

    def test_ablation_result_frozen(self) -> None:
        r = AblationResult(model_name="m", feature_impacts={})
        with pytest.raises(FrozenInstanceError):
            r.model_name = "x"  # type: ignore[misc]


class TestModelConfig:
    def test_model_config_frozen(self) -> None:
        c = ModelConfig(
            data_dir="./data",
            artifacts_dir="./artifacts",
            seasons=[2023],
            model_params={},
        )
        with pytest.raises(FrozenInstanceError):
            c.data_dir = "other"  # type: ignore[misc]

    def test_model_config_defaults(self) -> None:
        c = ModelConfig()
        assert c.data_dir == "./data"
        assert c.artifacts_dir == "./artifacts"
        assert c.seasons == []
        assert c.model_params == {}
        assert c.output_dir is None
        assert c.version is None
        assert c.tags == {}

    def test_model_config_with_version_and_tags(self) -> None:
        c = ModelConfig(version="v1.0", tags={"env": "prod", "owner": "alice"})
        assert c.version == "v1.0"
        assert c.tags == {"env": "prod", "owner": "alice"}
