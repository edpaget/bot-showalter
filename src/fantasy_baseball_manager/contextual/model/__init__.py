"""Transformer model architecture for contextual performance prediction."""

from fantasy_baseball_manager.contextual.model.config import ModelConfig
from fantasy_baseball_manager.contextual.model.embedder import EventEmbedder
from fantasy_baseball_manager.contextual.model.heads import (
    MaskedGamestateHead,
    PerformancePredictionHead,
)
from fantasy_baseball_manager.contextual.model.mask import build_player_attention_mask
from fantasy_baseball_manager.contextual.model.model import (
    ContextualPerformanceModel,
)
from fantasy_baseball_manager.contextual.model.positional import (
    SinusoidalPositionalEncoding,
)
from fantasy_baseball_manager.contextual.model.tensorizer import (
    TensorizedBatch,
    TensorizedSingle,
    Tensorizer,
)
from fantasy_baseball_manager.contextual.model.transformer import (
    GamestateTransformer,
)

__all__ = [
    "ContextualPerformanceModel",
    "EventEmbedder",
    "GamestateTransformer",
    "MaskedGamestateHead",
    "ModelConfig",
    "PerformancePredictionHead",
    "SinusoidalPositionalEncoding",
    "TensorizedBatch",
    "TensorizedSingle",
    "Tensorizer",
    "build_player_attention_mask",
]
