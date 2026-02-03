"""Machine learning module for projection enhancement."""

from fantasy_baseball_manager.ml.features import (
    BatterFeatureExtractor,
    PitcherFeatureExtractor,
)
from fantasy_baseball_manager.ml.persistence import ModelStore
from fantasy_baseball_manager.ml.residual_model import ResidualModelSet, StatResidualModel
from fantasy_baseball_manager.ml.training import ResidualModelTrainer

__all__ = [
    "BatterFeatureExtractor",
    "ModelStore",
    "PitcherFeatureExtractor",
    "ResidualModelSet",
    "ResidualModelTrainer",
    "StatResidualModel",
]
