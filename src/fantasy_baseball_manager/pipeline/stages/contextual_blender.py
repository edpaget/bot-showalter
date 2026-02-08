"""Contextual transformer blender for ensemble predictions with Marcel rates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.contextual.adapter import PerGameToSeasonAdapter
from fantasy_baseball_manager.contextual.training.config import (
    BATTER_TARGET_STATS,
    PITCHER_TARGET_STATS,
    ContextualBlenderConfig,
)
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
    from fantasy_baseball_manager.contextual.predictor import ContextualPredictor
    from fantasy_baseball_manager.player_id.mapper import SfbbMapper

logger = logging.getLogger(__name__)


@dataclass
class ContextualBlender:
    """Blends contextual transformer predictions with Marcel rates.

    Ensemble mode: blended_rate = (1 - weight) * marcel + weight * contextual

    Implements the RateAdjuster protocol.
    """

    predictor: ContextualPredictor
    id_mapper: SfbbMapper
    config: ContextualBlenderConfig = field(default_factory=ContextualBlenderConfig)

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        """Blend contextual predictions with Marcel rates.

        For each player:
        1. Look up MLBAM ID from player identity
        2. Get contextual predictions via predictor
        3. Blend: new_rate = (1 - weight) * marcel_rate + weight * contextual_rate
        4. Store both in metadata for analysis

        Players without sufficient data are returned unchanged.
        """
        if not players:
            return []

        self.predictor.ensure_models_loaded(
            self.config.batter_model_name,
            self.config.pitcher_model_name,
        )

        if not self.predictor.has_batter_model() and not self.predictor.has_pitcher_model():
            logger.warning("No contextual models available, skipping blending")
            return players

        result: list[PlayerRates] = []
        for player in players:
            if self._is_batter(player):
                result.append(self._blend_player(
                    player, "batter", self.predictor._batter_model,
                    BATTER_TARGET_STATS,
                ))
            else:
                result.append(self._blend_player(
                    player, "pitcher", self.predictor._pitcher_model,
                    PITCHER_TARGET_STATS,
                ))

        return result

    def _is_batter(self, player: PlayerRates) -> bool:
        """Check if player is a batter (has pa_per_year metadata)."""
        return "pa_per_year" in player.metadata

    def _blend_player(
        self,
        player: PlayerRates,
        perspective: str,
        model: ContextualPerformanceModel | None,
        target_stats: tuple[str, ...],
    ) -> PlayerRates:
        """Blend contextual predictions with Marcel rates for a single player."""
        if model is None:
            return player

        mlbam_id = player.player.mlbam_id if player.player else None
        if mlbam_id is None:
            return player

        data_year = player.year - 1

        prediction = self.predictor.predict_player(
            mlbam_id=int(mlbam_id),
            data_year=data_year,
            perspective=perspective,
            model=model,
            target_stats=target_stats,
            min_games=self.config.min_games,
            context_window=self.config.context_window,
        )
        if prediction is None:
            return player

        avg_predictions, context_games = prediction

        # Convert per-game predictions to rates
        adapter = PerGameToSeasonAdapter(perspective)
        avg_denominator = sum(adapter.game_denominator(g) for g in context_games) / len(context_games)
        contextual_rates = adapter.predictions_to_rates(avg_predictions, avg_denominator, player.rates)
        if contextual_rates is None:
            return player

        # Blend rates
        weight = self.config.contextual_weight
        rates = dict(player.rates)

        for stat, contextual_rate in contextual_rates.items():
            if stat in rates:
                marcel_rate = rates[stat]
                rates[stat] = (1 - weight) * marcel_rate + weight * contextual_rate

        # Uncovered stats are already in rates dict from the copy

        metadata: PlayerMetadata = {
            **player.metadata,
            "contextual_blended": True,
            "contextual_blend_weight": weight,
            "contextual_rates": contextual_rates,
        }

        return PlayerRates(
            player_id=player.player_id,
            name=player.name,
            year=player.year,
            age=player.age,
            rates=rates,
            opportunities=player.opportunities,
            metadata=metadata,
            player=player.player,
        )
