"""Reusable prediction helper for contextual transformer models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.contextual.training.config import (
    BATTER_TARGET_STATS,
    PITCHER_TARGET_STATS,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
    from fantasy_baseball_manager.contextual.data.models import GameSequence
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
    from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
    from fantasy_baseball_manager.contextual.persistence import ContextualModelStore

logger = logging.getLogger(__name__)


@dataclass
class ContextualPredictor:
    """Shared prediction logic for contextual transformer models.

    Used by both ContextualEmbeddingRateComputer (standalone) and
    ContextualBlender (ensemble) to avoid duplicating model loading,
    tensorization, and inference code.
    """

    sequence_builder: GameSequenceBuilder
    model_store: ContextualModelStore | None = None

    _batter_model: ContextualPerformanceModel | None = field(default=None, init=False, repr=False)
    _pitcher_model: ContextualPerformanceModel | None = field(default=None, init=False, repr=False)
    _model_config: ModelConfig | None = field(default=None, init=False, repr=False)
    _tensorizer: Tensorizer | None = field(default=None, init=False, repr=False)
    _models_loaded: bool = field(default=False, init=False, repr=False)

    def ensure_models_loaded(
        self,
        batter_model_name: str,
        pitcher_model_name: str,
    ) -> None:
        """Lazy-load fine-tuned models on first use."""
        if self._models_loaded:
            return

        store = self._resolve_model_store()

        if store.exists(batter_model_name):
            try:
                batter_config = store.load_model_config(batter_model_name)
                self._model_config = batter_config
                self._batter_model = store.load_finetune_model(
                    batter_model_name, batter_config, len(BATTER_TARGET_STATS),
                )
                self._batter_model.eval()
                logger.debug("Loaded contextual batter model: %s", batter_model_name)
            except (ValueError, FileNotFoundError) as exc:
                logger.warning(
                    "Contextual batter model %s exists but cannot be loaded: %s",
                    batter_model_name,
                    exc,
                )
        else:
            logger.warning(
                "Contextual batter model %s not found",
                batter_model_name,
            )

        if store.exists(pitcher_model_name):
            try:
                pitcher_config = store.load_model_config(pitcher_model_name)
                if self._model_config is None:
                    self._model_config = pitcher_config
                self._pitcher_model = store.load_finetune_model(
                    pitcher_model_name, pitcher_config, len(PITCHER_TARGET_STATS),
                )
                self._pitcher_model.eval()
                logger.debug("Loaded contextual pitcher model: %s", pitcher_model_name)
            except (ValueError, FileNotFoundError) as exc:
                logger.warning(
                    "Contextual pitcher model %s exists but cannot be loaded: %s",
                    pitcher_model_name,
                    exc,
                )
        else:
            logger.warning(
                "Contextual pitcher model %s not found",
                pitcher_model_name,
            )

        self._models_loaded = True

    def has_batter_model(self) -> bool:
        return self._batter_model is not None

    def has_pitcher_model(self) -> bool:
        return self._pitcher_model is not None

    def predict_player(
        self,
        mlbam_id: int,
        data_year: int,
        perspective: str,
        model: ContextualPerformanceModel,
        target_stats: tuple[str, ...],
        min_games: int,
        context_window: int,
    ) -> tuple[dict[str, float], list[GameSequence]] | None:
        """Run model inference for a player.

        Returns (avg_predictions, context_games) or None if insufficient data.
        """
        games = self.sequence_builder.build_player_season(
            data_year, mlbam_id, perspective=perspective,
        )

        if len(games) < min_games:
            return None

        context_games = games[-context_window:]
        avg_predictions = self._run_inference(model, context_games, target_stats)

        return avg_predictions, context_games

    def _run_inference(
        self,
        model: ContextualPerformanceModel,
        games: list[GameSequence],
        target_stats: tuple[str, ...],
    ) -> dict[str, float]:
        """Run model inference on a player's game context."""
        import torch

        from fantasy_baseball_manager.contextual.data.models import PlayerContext

        tensorizer = self._ensure_tensorizer()

        context = PlayerContext(
            player_id=games[0].player_id,
            player_name="",
            season=games[0].season,
            perspective=games[0].perspective,
            games=tuple(games),
        )

        tensorized = tensorizer.tensorize_context(context)
        batch = tensorizer.collate([tensorized])

        with torch.no_grad():
            output = model(batch)

        preds = output["performance_preds"]  # (1, n_targets)
        pred_values = preds.squeeze(0).tolist()

        return dict(zip(target_stats, pred_values, strict=True))

    def _resolve_model_store(self) -> ContextualModelStore:
        if self.model_store is not None:
            return self.model_store
        from fantasy_baseball_manager.contextual.persistence import ContextualModelStore as Store

        self.model_store = Store()
        return self.model_store

    def _ensure_tensorizer(self) -> Tensorizer:
        if self._tensorizer is not None:
            return self._tensorizer

        from fantasy_baseball_manager.contextual.data.vocab import (
            BB_TYPE_VOCAB,
            HANDEDNESS_VOCAB,
            PA_EVENT_VOCAB,
            PITCH_RESULT_VOCAB,
            PITCH_TYPE_VOCAB,
        )
        from fantasy_baseball_manager.contextual.model.config import ModelConfig as MC
        from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer

        config = self._model_config if self._model_config is not None else MC()
        self._tensorizer = Tensorizer(
            config=config,
            pitch_type_vocab=PITCH_TYPE_VOCAB,
            pitch_result_vocab=PITCH_RESULT_VOCAB,
            bb_type_vocab=BB_TYPE_VOCAB,
            handedness_vocab=HANDEDNESS_VOCAB,
            pa_event_vocab=PA_EVENT_VOCAB,
        )
        return self._tensorizer
