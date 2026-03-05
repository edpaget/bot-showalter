"""Model protocol types — re-exported from domain.model_protocol.

Intra-``models`` imports can continue using this module.  Cross-package
consumers should import from ``fantasy_baseball_manager.domain`` instead.
"""

from fantasy_baseball_manager.domain import (
    Ablatable,
    AblationResult,
    Evaluable,
    Evaluator,
    Experimentable,
    FeatureIntrospectable,
    FineTunable,
    FittedModels,
    Model,
    ModelConfig,
    PlayerUniverseProvider,
    Predictable,
    PredictResult,
    Preparable,
    PrepareResult,
    Sweepable,
    TargetComparison,
    TargetVector,
    Trainable,
    TrainingBackend,
    TrainResult,
    Tunable,
    TuneResult,
    ValidationResult,
)

__all__ = [
    "Ablatable",
    "AblationResult",
    "Evaluable",
    "Evaluator",
    "Experimentable",
    "FeatureIntrospectable",
    "FineTunable",
    "FittedModels",
    "Model",
    "ModelConfig",
    "PlayerUniverseProvider",
    "Predictable",
    "PredictResult",
    "Preparable",
    "PrepareResult",
    "Sweepable",
    "TargetComparison",
    "TargetVector",
    "Trainable",
    "TrainResult",
    "TrainingBackend",
    "Tunable",
    "TuneResult",
    "ValidationResult",
]
