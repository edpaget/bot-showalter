"""MTL blender for ensemble predictions with Marcel rates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.ml.features import BatterFeatureExtractor, PitcherFeatureExtractor
from fantasy_baseball_manager.ml.mtl.config import MTLBlenderConfig
from fantasy_baseball_manager.ml.mtl.model import BATTER_STATS, PITCHER_STATS
from fantasy_baseball_manager.ml.mtl.persistence import MTLModelStore
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.ml.mtl.model import (
        MultiTaskBatterModel,
        MultiTaskPitcherModel,
    )
    from fantasy_baseball_manager.pipeline.batted_ball_data import (
        PitcherBattedBallDataSource,
        PitcherBattedBallStats,
    )
    from fantasy_baseball_manager.pipeline.feature_store import FeatureStore
    from fantasy_baseball_manager.pipeline.skill_data import BatterSkillStats, SkillDataSource
    from fantasy_baseball_manager.pipeline.statcast_data import (
        FullStatcastDataSource,
        StatcastBatterStats,
        StatcastPitcherStats,
    )

logger = logging.getLogger(__name__)


@dataclass
class MTLBlender:
    """Blends MTL predictions with Marcel rates.

    Ensemble mode: blended_rate = (1 - weight) * marcel + weight * mtl

    This provides a compromise between Marcel's stability and MTL's
    ability to capture skill changes via Statcast data.

    Implements the RateAdjuster protocol.
    """

    statcast_source: FullStatcastDataSource
    batted_ball_source: PitcherBattedBallDataSource
    skill_data_source: SkillDataSource
    config: MTLBlenderConfig = field(default_factory=MTLBlenderConfig)
    model_store: MTLModelStore = field(default_factory=MTLModelStore)
    feature_store: FeatureStore | None = field(default=None)

    # Lazy-loaded models and data
    _batter_model: MultiTaskBatterModel | None = field(default=None, init=False, repr=False)
    _pitcher_model: MultiTaskPitcherModel | None = field(default=None, init=False, repr=False)
    _models_loaded: bool = field(default=False, init=False, repr=False)

    # Cached data per year
    _batter_statcast: dict[str, StatcastBatterStats] | None = field(default=None, init=False, repr=False)
    _pitcher_statcast: dict[str, StatcastPitcherStats] | None = field(default=None, init=False, repr=False)
    _pitcher_batted_ball: dict[str, PitcherBattedBallStats] | None = field(default=None, init=False, repr=False)
    _batter_skill_data: dict[str, BatterSkillStats] | None = field(default=None, init=False, repr=False)
    _cached_year: int | None = field(default=None, init=False, repr=False)

    def _ensure_models_loaded(self) -> None:
        """Lazy-load trained models on first use."""
        if self._models_loaded:
            return

        if self.model_store.exists(self.config.model_name, "batter"):
            self._batter_model = self.model_store.load_batter_model(self.config.model_name)
            logger.debug("Loaded MTL batter model for blending: %s", self.config.model_name)
        else:
            logger.warning(
                "MTL batter model %s not found, skipping batter blending",
                self.config.model_name,
            )

        if self.model_store.exists(self.config.model_name, "pitcher"):
            self._pitcher_model = self.model_store.load_pitcher_model(self.config.model_name)
            logger.debug("Loaded MTL pitcher model for blending: %s", self.config.model_name)
        else:
            logger.warning(
                "MTL pitcher model %s not found, skipping pitcher blending",
                self.config.model_name,
            )

        self._models_loaded = True

    def _ensure_data_loaded(self, year: int) -> None:
        """Lazy-load Statcast and skill data for the projection year."""
        if self._cached_year == year:
            return

        # Load data from prior year (most recent available)
        data_year = year - 1

        if self.feature_store is not None:
            self._batter_statcast = self.feature_store.batter_statcast(data_year)
            self._pitcher_statcast = self.feature_store.pitcher_statcast(data_year)
            self._pitcher_batted_ball = self.feature_store.pitcher_batted_ball(data_year)
            self._batter_skill_data = self.feature_store.batter_skill(data_year)
        else:
            # Batter Statcast
            batter_data = self.statcast_source.batter_expected_stats(data_year)
            self._batter_statcast = {s.player_id: s for s in batter_data}
            logger.debug("Loaded %d batter Statcast records for year %d", len(self._batter_statcast), data_year)

            # Pitcher Statcast
            pitcher_data = self.statcast_source.pitcher_expected_stats(data_year)
            self._pitcher_statcast = {s.player_id: s for s in pitcher_data}
            logger.debug("Loaded %d pitcher Statcast records for year %d", len(self._pitcher_statcast), data_year)

            # Batted ball
            bb_data = self.batted_ball_source.pitcher_batted_ball_stats(data_year)
            self._pitcher_batted_ball = {s.player_id: s for s in bb_data}
            logger.debug("Loaded %d batted ball records for year %d", len(self._pitcher_batted_ball), data_year)

            # Batter skill data
            skill_data = self.skill_data_source.batter_skill_stats(data_year)
            self._batter_skill_data = {s.player_id: s for s in skill_data}
            logger.debug("Loaded %d batter skill records for year %d", len(self._batter_skill_data), data_year)

        self._cached_year = year

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        """Blend MTL predictions with Marcel rates.

        For each player:
        1. Extract features from Statcast data
        2. Get MTL predictions
        3. Blend: new_rate = (1 - weight) * marcel_rate + weight * mtl_rate
        4. Store both in metadata for analysis

        Args:
            players: List of PlayerRates (typically from Marcel rate computer).

        Returns:
            List of PlayerRates with blended rates.
        """
        if not players:
            return []

        self._ensure_models_loaded()
        year = players[0].year
        self._ensure_data_loaded(year)

        result: list[PlayerRates] = []
        for player in players:
            if self._is_batter(player):
                result.append(self._blend_batter(player))
            else:
                result.append(self._blend_pitcher(player))

        return result

    def _is_batter(self, player: PlayerRates) -> bool:
        """Check if player is a batter (has pa_per_year metadata)."""
        return "pa_per_year" in player.metadata

    def _blend_batter(self, player: PlayerRates) -> PlayerRates:
        """Blend MTL predictions with Marcel rates for a batter."""
        if self._batter_model is None or not self._batter_model.is_fitted:
            return player
        if self._batter_statcast is None:
            return player

        # Get MLBAM ID from enriched Player identity
        mlbam_id = player.player.mlbam_id if player.player else None
        if mlbam_id is None:
            return player

        statcast = self._batter_statcast.get(mlbam_id)
        if statcast is None or statcast.pa < self.config.min_pa:
            return player

        # Get skill data (uses FanGraphs ID)
        skill_data = None
        if self._batter_skill_data is not None:
            skill_data = self._batter_skill_data.get(player.player_id)

        # Extract features
        extractor = BatterFeatureExtractor(min_pa=self.config.min_pa)
        features = extractor.extract(player, statcast, skill_data)
        if features is None:
            return player

        # Get MTL predictions
        mtl_predictions = self._batter_model.predict(features)
        if not mtl_predictions:
            return player

        # Blend rates
        weight = self.config.mtl_weight
        rates = dict(player.rates)
        mtl_rates_applied: dict[str, float] = {}

        for stat in BATTER_STATS:
            if stat in mtl_predictions and stat in rates:
                marcel_rate = rates[stat]
                mtl_rate = mtl_predictions[stat]
                blended = (1 - weight) * marcel_rate + weight * mtl_rate
                rates[stat] = blended
                mtl_rates_applied[stat] = mtl_rate

        metadata: PlayerMetadata = {
            **player.metadata,
            "mtl_blended": True,
            "mtl_blend_weight": weight,
            "mtl_rates": mtl_rates_applied,
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

    def _blend_pitcher(self, player: PlayerRates) -> PlayerRates:
        """Blend MTL predictions with Marcel rates for a pitcher."""
        if self._pitcher_model is None or not self._pitcher_model.is_fitted:
            return player
        if self._pitcher_statcast is None:
            return player

        # Get MLBAM ID from enriched Player identity
        mlbam_id = player.player.mlbam_id if player.player else None
        if mlbam_id is None:
            return player

        statcast = self._pitcher_statcast.get(mlbam_id)
        if statcast is None or statcast.pa < self.config.min_pa:
            return player

        # Get batted ball data (uses FanGraphs ID)
        batted_ball = None
        if self._pitcher_batted_ball is not None:
            batted_ball = self._pitcher_batted_ball.get(player.player_id)

        # Extract features
        extractor = PitcherFeatureExtractor(min_pa=self.config.min_pa)
        features = extractor.extract(player, statcast, batted_ball)
        if features is None:
            return player

        # Get MTL predictions
        mtl_predictions = self._pitcher_model.predict(features)
        if not mtl_predictions:
            return player

        # Blend rates
        weight = self.config.mtl_weight
        rates = dict(player.rates)
        mtl_rates_applied: dict[str, float] = {}

        for stat in PITCHER_STATS:
            if stat in mtl_predictions and stat in rates:
                marcel_rate = rates[stat]
                mtl_rate = mtl_predictions[stat]
                blended = (1 - weight) * marcel_rate + weight * mtl_rate
                rates[stat] = blended
                mtl_rates_applied[stat] = mtl_rate

        metadata: PlayerMetadata = {
            **player.metadata,
            "mtl_blended": True,
            "mtl_blend_weight": weight,
            "mtl_rates": mtl_rates_applied,
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
