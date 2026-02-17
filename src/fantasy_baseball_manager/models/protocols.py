from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fantasy_baseball_manager.domain.evaluation import SystemMetrics
from fantasy_baseball_manager.features.types import AnyFeature


@dataclass(frozen=True)
class ModelConfig:
    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    seasons: list[int] = field(default_factory=list)
    model_params: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None
    version: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    top: int | None = None


@dataclass(frozen=True)
class PrepareResult:
    model_name: str
    rows_processed: int
    artifacts_path: str


@dataclass(frozen=True)
class TrainResult:
    model_name: str
    metrics: dict[str, float]
    artifacts_path: str


@dataclass(frozen=True)
class PredictResult:
    model_name: str
    predictions: list[dict[str, Any]]
    output_path: str
    distributions: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class AblationResult:
    model_name: str
    feature_impacts: dict[str, float]


@runtime_checkable
class Model(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def supported_operations(self) -> frozenset[str]: ...
    @property
    def artifact_type(self) -> str: ...


@runtime_checkable
class Preparable(Protocol):
    def prepare(self, config: ModelConfig) -> PrepareResult: ...


@runtime_checkable
class Trainable(Protocol):
    def train(self, config: ModelConfig) -> TrainResult: ...


@runtime_checkable
class Evaluable(Protocol):
    def evaluate(self, config: ModelConfig) -> SystemMetrics: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self,
        system: str,
        version: str,
        season: int,
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
        top: int | None = None,
    ) -> SystemMetrics: ...


@runtime_checkable
class Predictable(Protocol):
    def predict(self, config: ModelConfig) -> PredictResult: ...


@runtime_checkable
class FineTunable(Protocol):
    def finetune(self, config: ModelConfig) -> TrainResult: ...


@runtime_checkable
class Ablatable(Protocol):
    def ablate(self, config: ModelConfig) -> AblationResult: ...


@dataclass(frozen=True)
class TuneResult:
    model_name: str
    batter_params: dict[str, Any]
    pitcher_params: dict[str, Any]
    batter_cv_rmse: dict[str, float]
    pitcher_cv_rmse: dict[str, float]


@runtime_checkable
class Tunable(Protocol):
    def tune(self, config: ModelConfig) -> TuneResult: ...


@runtime_checkable
class FeatureIntrospectable(Protocol):
    @property
    def declared_features(self) -> tuple[AnyFeature, ...]: ...
