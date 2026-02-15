from fantasy_baseball_manager.domain.evaluation import SystemMetrics
from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import AnyFeature
from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    ModelConfig,
    PredictResult,
    PrepareResult,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.models.statcast_gbm.features import (
    build_batter_feature_set,
    build_pitcher_feature_set,
)


@register("statcast-gbm")
class StatcastGBMModel:
    def __init__(self, assembler: DatasetAssembler | None = None) -> None:
        self._assembler = assembler

    @property
    def name(self) -> str:
        return "statcast-gbm"

    @property
    def description(self) -> str:
        return "Gradient-boosted model using Statcast features for player projections."

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train", "evaluate", "predict", "ablate"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.FILE.value

    @property
    def declared_features(self) -> tuple[AnyFeature, ...]:
        batter_fs = build_batter_feature_set([])
        pitcher_fs = build_pitcher_feature_set([])
        return batter_fs.features + pitcher_fs.features

    def prepare(self, config: ModelConfig) -> PrepareResult:
        assert self._assembler is not None, "assembler is required for prepare"
        batter_fs = build_batter_feature_set(config.seasons)
        pitcher_fs = build_pitcher_feature_set(config.seasons)

        bat_handle = self._assembler.get_or_materialize(batter_fs)
        pitch_handle = self._assembler.get_or_materialize(pitcher_fs)

        return PrepareResult(
            model_name=self.name,
            rows_processed=bat_handle.row_count + pitch_handle.row_count,
            artifacts_path=config.artifacts_dir,
        )

    def train(self, config: ModelConfig) -> TrainResult:
        return TrainResult(
            model_name=self.name,
            metrics={},
            artifacts_path=config.artifacts_dir,
        )

    def evaluate(self, config: ModelConfig) -> SystemMetrics:
        return SystemMetrics(
            system=self.name,
            version=config.version or "latest",
            source_type="first_party",
            metrics={},
        )

    def predict(self, config: ModelConfig) -> PredictResult:
        return PredictResult(
            model_name=self.name,
            predictions=[],
            output_path=config.output_dir or config.artifacts_dir,
        )

    def ablate(self, config: ModelConfig) -> AblationResult:
        return AblationResult(
            model_name=self.name,
            feature_impacts={},
        )
