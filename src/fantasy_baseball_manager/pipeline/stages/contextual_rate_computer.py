"""Contextual transformer-based rate computer for projection pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.contextual.adapter import PerGameToSeasonAdapter
from fantasy_baseball_manager.contextual.training.config import (
    BATTER_TARGET_STATS,
    PITCHER_TARGET_STATS,
    ContextualRateComputerConfig,
)
from fantasy_baseball_manager.pipeline.stages.rate_computers import (
    MarcelRateComputer,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
    from fantasy_baseball_manager.contextual.data.models import GameSequence
    from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
    from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
    from fantasy_baseball_manager.contextual.persistence import ContextualModelStore
    from fantasy_baseball_manager.data.protocol import DataSource
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
    from fantasy_baseball_manager.player_id.mapper import SfbbMapper

logger = logging.getLogger(__name__)


@dataclass
class ContextualEmbeddingRateComputer:
    """Computes player rates using fine-tuned contextual transformer.

    For players with sufficient pitch sequence data, uses the contextual
    transformer model to predict per-game stats and converts them to rates.
    Falls back to Marcel rates for players without enough data.

    Implements the RateComputer protocol.
    """

    sequence_builder: GameSequenceBuilder
    id_mapper: SfbbMapper
    config: ContextualRateComputerConfig = field(default_factory=ContextualRateComputerConfig)
    model_store: ContextualModelStore | None = None

    _marcel_computer: MarcelRateComputer = field(default_factory=MarcelRateComputer, repr=False)
    _batter_model: ContextualPerformanceModel | None = field(default=None, init=False, repr=False)
    _pitcher_model: ContextualPerformanceModel | None = field(default=None, init=False, repr=False)
    _tensorizer: Tensorizer | None = field(default=None, init=False, repr=False)
    _models_loaded: bool = field(default=False, init=False, repr=False)

    def _ensure_models_loaded(self) -> None:
        """Lazy-load fine-tuned models on first use."""
        if self._models_loaded:
            return

        from fantasy_baseball_manager.contextual.model.config import ModelConfig

        model_config = ModelConfig()
        store = self._resolve_model_store()

        if store.exists(self.config.batter_model_name):
            self._batter_model = store.load_finetune_model(
                self.config.batter_model_name, model_config, len(BATTER_TARGET_STATS),
            )
            self._batter_model.eval()
            logger.debug("Loaded contextual batter model: %s", self.config.batter_model_name)
        else:
            logger.warning(
                "Contextual batter model %s not found, using Marcel fallback",
                self.config.batter_model_name,
            )

        if store.exists(self.config.pitcher_model_name):
            self._pitcher_model = store.load_finetune_model(
                self.config.pitcher_model_name, model_config, len(PITCHER_TARGET_STATS),
            )
            self._pitcher_model.eval()
            logger.debug("Loaded contextual pitcher model: %s", self.config.pitcher_model_name)
        else:
            logger.warning(
                "Contextual pitcher model %s not found, using Marcel fallback",
                self.config.pitcher_model_name,
            )

        self._models_loaded = True

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
        from fantasy_baseball_manager.contextual.model.config import ModelConfig
        from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer

        self._tensorizer = Tensorizer(
            config=ModelConfig(),
            pitch_type_vocab=PITCH_TYPE_VOCAB,
            pitch_result_vocab=PITCH_RESULT_VOCAB,
            bb_type_vocab=BB_TYPE_VOCAB,
            handedness_vocab=HANDEDNESS_VOCAB,
            pa_event_vocab=PA_EVENT_VOCAB,
        )
        return self._tensorizer

    def compute_batting_rates(
        self,
        batting_source: DataSource[BattingSeasonStats],
        team_batting_source: DataSource[BattingSeasonStats],
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        """Compute batting rates using contextual transformer model."""
        self._ensure_models_loaded()

        marcel_rates = self._marcel_computer.compute_batting_rates(
            batting_source, team_batting_source, year, years_back,
        )

        if self._batter_model is None:
            logger.debug("No contextual batter model available, using Marcel rates")
            return marcel_rates

        adapter = PerGameToSeasonAdapter("batter")
        data_year = year - 1

        result: list[PlayerRates] = []
        contextual_count = 0

        for marcel_player in marcel_rates:
            contextual_player = self._try_contextual_prediction(
                marcel_player, self._batter_model, adapter,
                BATTER_TARGET_STATS, "batter", data_year,
            )
            if contextual_player is not None:
                result.append(contextual_player)
                contextual_count += 1
            else:
                result.append(marcel_player)

        logger.info(
            "Contextual rate computer: %d/%d batters used contextual predictions",
            contextual_count, len(result),
        )
        return result

    def compute_pitching_rates(
        self,
        pitching_source: DataSource[PitchingSeasonStats],
        team_pitching_source: DataSource[PitchingSeasonStats],
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        """Compute pitching rates using contextual transformer model."""
        self._ensure_models_loaded()

        marcel_rates = self._marcel_computer.compute_pitching_rates(
            pitching_source, team_pitching_source, year, years_back,
        )

        if self._pitcher_model is None:
            logger.debug("No contextual pitcher model available, using Marcel rates")
            return marcel_rates

        adapter = PerGameToSeasonAdapter("pitcher")
        data_year = year - 1

        result: list[PlayerRates] = []
        contextual_count = 0

        for marcel_player in marcel_rates:
            contextual_player = self._try_contextual_prediction(
                marcel_player, self._pitcher_model, adapter,
                PITCHER_TARGET_STATS, "pitcher", data_year,
            )
            if contextual_player is not None:
                result.append(contextual_player)
                contextual_count += 1
            else:
                result.append(marcel_player)

        logger.info(
            "Contextual rate computer: %d/%d pitchers used contextual predictions",
            contextual_count, len(result),
        )
        return result

    def _try_contextual_prediction(
        self,
        marcel_player: PlayerRates,
        model: ContextualPerformanceModel,
        adapter: PerGameToSeasonAdapter,
        target_stats: tuple[str, ...],
        perspective: str,
        data_year: int,
    ) -> PlayerRates | None:
        """Try to produce contextual predictions for a player.

        Returns None if fallback to Marcel is needed.
        """
        mlbam_id = self.id_mapper.fangraphs_to_mlbam(marcel_player.player_id)
        if mlbam_id is None:
            return None

        games = self.sequence_builder.build_player_season(
            data_year, int(mlbam_id), perspective=perspective,
        )

        if len(games) < self.config.min_games:
            return None

        context_games = games[-self.config.context_window :]

        avg_predictions = self._predict_player(model, context_games, target_stats)
        avg_denominator = sum(adapter.game_denominator(g) for g in context_games) / len(context_games)

        rates = adapter.predictions_to_rates(avg_predictions, avg_denominator, marcel_player.rates)
        if rates is None:
            return None

        return PlayerRates(
            player_id=marcel_player.player_id,
            name=marcel_player.name,
            year=marcel_player.year,
            age=marcel_player.age,
            rates=rates,
            metadata={
                **marcel_player.metadata,
                "contextual_predicted": True,
                "contextual_games_used": len(context_games),
                "marcel_rates": marcel_player.rates,
            },
        )

    def _predict_player(
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

        # performance_preds: (batch=1, n_player_tokens, n_targets)
        preds = output["performance_preds"]
        # Pool over player tokens: (1, n_targets)
        pooled = preds.mean(dim=1)
        # Convert to dict
        pred_values = pooled.squeeze(0).tolist()

        return dict(zip(target_stats, pred_values, strict=True))
