from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelConfig:
    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    seasons: list[int] = field(default_factory=list)
    model_params: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None


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
class EvalResult:
    model_name: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class PredictResult:
    model_name: str
    predictions: list[dict[str, Any]]
    output_path: str


@dataclass(frozen=True)
class AblationResult:
    model_name: str
    feature_impacts: dict[str, float]


@runtime_checkable
class ProjectionModel(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def supported_operations(self) -> frozenset[str]: ...


@runtime_checkable
class Preparable(Protocol):
    def prepare(self, config: ModelConfig) -> PrepareResult: ...


@runtime_checkable
class Trainable(Protocol):
    def train(self, config: ModelConfig) -> TrainResult: ...


@runtime_checkable
class Evaluable(Protocol):
    def evaluate(self, config: ModelConfig) -> EvalResult: ...


@runtime_checkable
class Predictable(Protocol):
    def predict(self, config: ModelConfig) -> PredictResult: ...


@runtime_checkable
class FineTunable(Protocol):
    def finetune(self, config: ModelConfig) -> TrainResult: ...


@runtime_checkable
class Ablatable(Protocol):
    def ablate(self, config: ModelConfig) -> AblationResult: ...
