"""Pitch event data pipeline for contextual embeddings.

Transforms raw Statcast parquet data into typed, model-ready pitch sequences
exposed through the DataSource[T] protocol.
"""

from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
from fantasy_baseball_manager.contextual.data.cache import SequenceCache
from fantasy_baseball_manager.contextual.data.models import (
    GameSequence,
    PitchEvent,
    PlayerContext,
)
from fantasy_baseball_manager.contextual.data.source import PitchSequenceDataSource
from fantasy_baseball_manager.contextual.data.vocab import Vocabulary

__all__ = [
    "GameSequence",
    "GameSequenceBuilder",
    "PitchEvent",
    "PitchSequenceDataSource",
    "PlayerContext",
    "SequenceCache",
    "Vocabulary",
]
