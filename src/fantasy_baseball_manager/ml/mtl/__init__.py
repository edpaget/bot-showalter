"""Multi-Task Learning neural network module for stat prediction."""

from fantasy_baseball_manager.ml.mtl.config import (
    MTLArchitectureConfig,
    MTLBlenderConfig,
    MTLRateComputerConfig,
    MTLTrainingConfig,
)
from fantasy_baseball_manager.ml.mtl.model import (
    BATTER_STATS,
    PITCHER_STATS,
    MultiTaskBatterModel,
    MultiTaskNet,
    MultiTaskPitcherModel,
)
from fantasy_baseball_manager.ml.mtl.trainer import MTLTrainer

__all__ = [
    "BATTER_STATS",
    "PITCHER_STATS",
    "MTLArchitectureConfig",
    "MTLBlenderConfig",
    "MTLRateComputerConfig",
    "MTLTrainer",
    "MTLTrainingConfig",
    "MultiTaskBatterModel",
    "MultiTaskNet",
    "MultiTaskPitcherModel",
]
