from fantasy_baseball_manager.models.protocols import (
    EvalResult,
    ModelConfig,
    PrepareResult,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import register


@register("marcel")
class MarcelModel:
    @property
    def name(self) -> str:
        return "marcel"

    @property
    def category(self) -> str:
        return "batting"

    @property
    def description(self) -> str:
        return "Marcel the Monkey projection system — a simple, reliable baseline using weighted averages and regression to the mean."

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train", "evaluate"})

    def prepare(self, config: ModelConfig) -> PrepareResult:
        return PrepareResult(model_name=self.name, rows_processed=0, artifacts_path=config.artifacts_dir)

    def train(self, config: ModelConfig) -> TrainResult:
        return TrainResult(model_name=self.name, metrics={}, artifacts_path=config.artifacts_dir)

    def evaluate(self, config: ModelConfig) -> EvalResult:
        return EvalResult(model_name=self.name, metrics={})
