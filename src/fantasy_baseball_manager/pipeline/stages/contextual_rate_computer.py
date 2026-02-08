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
    from fantasy_baseball_manager.contextual.predictor import ContextualPredictor
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

    predictor: ContextualPredictor
    id_mapper: SfbbMapper
    config: ContextualRateComputerConfig = field(default_factory=ContextualRateComputerConfig)

    _marcel_computer: MarcelRateComputer = field(default_factory=MarcelRateComputer, repr=False)

    def compute_batting_rates(
        self,
        batting_source: DataSource[BattingSeasonStats],
        team_batting_source: DataSource[BattingSeasonStats],
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        """Compute batting rates using contextual transformer model."""
        self.predictor.ensure_models_loaded(
            self.config.batter_model_name,
            self.config.pitcher_model_name,
        )

        marcel_rates = self._marcel_computer.compute_batting_rates(
            batting_source, team_batting_source, year, years_back,
        )

        if not self.predictor.has_batter_model():
            logger.debug("No contextual batter model available, using Marcel rates")
            return marcel_rates

        adapter = PerGameToSeasonAdapter("batter")
        data_year = year - 1

        result: list[PlayerRates] = []
        contextual_count = 0

        for marcel_player in marcel_rates:
            contextual_player = self._try_contextual_prediction(
                marcel_player, adapter, BATTER_TARGET_STATS, "batter", data_year,
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
        self.predictor.ensure_models_loaded(
            self.config.batter_model_name,
            self.config.pitcher_model_name,
        )

        marcel_rates = self._marcel_computer.compute_pitching_rates(
            pitching_source, team_pitching_source, year, years_back,
        )

        if not self.predictor.has_pitcher_model():
            logger.debug("No contextual pitcher model available, using Marcel rates")
            return marcel_rates

        adapter = PerGameToSeasonAdapter("pitcher")
        data_year = year - 1

        result: list[PlayerRates] = []
        contextual_count = 0

        for marcel_player in marcel_rates:
            contextual_player = self._try_contextual_prediction(
                marcel_player, adapter, PITCHER_TARGET_STATS, "pitcher", data_year,
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

        model = self.predictor._batter_model if perspective == "batter" else self.predictor._pitcher_model
        if model is None:
            return None

        prediction = self.predictor.predict_player(
            mlbam_id=int(mlbam_id),
            data_year=data_year,
            perspective=perspective,
            model=model,
            target_stats=target_stats,
            min_games=self.config.min_games_for(perspective),
            context_window=self.config.context_window_for(perspective),
        )
        if prediction is None:
            return None

        avg_predictions, context_games = prediction
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
