"""Machine learning module for projection enhancement."""

from fantasy_baseball_manager.ml.features import (
    BatterFeatureExtractor,
    PitcherFeatureExtractor,
)
from fantasy_baseball_manager.ml.persistence import ModelStore
from fantasy_baseball_manager.ml.residual_model import ResidualModelSet, StatResidualModel
from fantasy_baseball_manager.ml.training import ResidualModelTrainer
from fantasy_baseball_manager.ml.validation import (
    EarlyStoppingConfig,
    LeaveOneYearOut,
    StatValidationResult,
    TimeSeriesHoldout,
    ValidationMetrics,
    ValidationReport,
    ValidationStrategy,
)

__all__ = [
    "BatterFeatureExtractor",
    "EarlyStoppingConfig",
    "LeaveOneYearOut",
    "ModelStore",
    "PitcherFeatureExtractor",
    "ResidualModelSet",
    "ResidualModelTrainer",
    "StatResidualModel",
    "StatValidationResult",
    "TimeSeriesHoldout",
    "ValidationMetrics",
    "ValidationReport",
    "ValidationStrategy",
]
