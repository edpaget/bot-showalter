from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.features import batting, player
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import Feature, FeatureSet
from fantasy_baseball_manager.models.protocols import (
    EvalResult,
    ModelConfig,
    PrepareResult,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import register

_MARCEL_BATTING_FEATURES: tuple[Feature, ...] = (
    batting.col("pa").lag(1).alias("pa_1"),
    batting.col("pa").lag(2).alias("pa_2"),
    batting.col("pa").lag(3).alias("pa_3"),
    batting.col("hr").lag(1).alias("hr_1"),
    batting.col("hr").lag(2).alias("hr_2"),
    batting.col("hr").lag(3).alias("hr_3"),
    batting.col("bb").lag(1).alias("bb_1"),
    batting.col("bb").lag(2).alias("bb_2"),
    batting.col("bb").lag(3).alias("bb_3"),
    batting.col("h").lag(1).alias("h_1"),
    batting.col("h").lag(2).alias("h_2"),
    batting.col("h").lag(3).alias("h_3"),
    batting.col("ab").lag(1).alias("ab_1"),
    batting.col("ab").lag(2).alias("ab_2"),
    batting.col("ab").lag(3).alias("ab_3"),
    player.age(),
)


@register("marcel")
class MarcelModel:
    @property
    def name(self) -> str:
        return "marcel"

    @property
    def description(self) -> str:
        return "Marcel the Monkey projection system — a simple, reliable baseline using weighted averages and regression to the mean."

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train", "evaluate"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.NONE.value

    @property
    def declared_features(self) -> tuple[Feature, ...]:
        return _MARCEL_BATTING_FEATURES

    def prepare(self, config: ModelConfig, assembler: DatasetAssembler) -> PrepareResult:
        feature_set = FeatureSet(
            name=f"{self.name}_batting",
            features=_MARCEL_BATTING_FEATURES,
            seasons=tuple(config.seasons),
            source_filter="fangraphs",
        )
        handle = assembler.get_or_materialize(feature_set)
        return PrepareResult(
            model_name=self.name,
            rows_processed=handle.row_count,
            artifacts_path=config.artifacts_dir,
        )

    def train(self, config: ModelConfig) -> TrainResult:
        return TrainResult(model_name=self.name, metrics={}, artifacts_path=config.artifacts_dir)

    def evaluate(self, config: ModelConfig) -> EvalResult:
        return EvalResult(model_name=self.name, metrics={})
