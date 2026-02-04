"""Gradient boosting residual adjuster for projection corrections."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.ml.features import BatterFeatureExtractor, PitcherFeatureExtractor
from fantasy_baseball_manager.ml.persistence import ModelStore
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.ml.residual_model import ResidualModelSet
    from fantasy_baseball_manager.pipeline.batted_ball_data import (
        PitcherBattedBallDataSource,
        PitcherBattedBallStats,
    )
    from fantasy_baseball_manager.pipeline.skill_data import BatterSkillStats, SkillDataSource
    from fantasy_baseball_manager.pipeline.statcast_data import (
        FullStatcastDataSource,
        StatcastBatterStats,
        StatcastPitcherStats,
    )
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GBResidualConfig:
    """Configuration for gradient boosting residual adjustment."""

    model_name: str = "default"
    batter_min_pa: int = 100
    pitcher_min_pa: int = 100
    max_residual_scale: float = 2.0  # Cap residuals at N * std
    min_rate_denominator_pa: int = 300  # Minimum PA for rate conversion
    min_rate_denominator_ip: int = 100  # Minimum IP for rate conversion


@dataclass
class GBResidualAdjuster:
    """Adjusts rates using gradient boosting predictions of Marcel residuals.

    This adjuster:
    1. Loads pre-trained residual prediction models
    2. Extracts features from current Statcast data
    3. Predicts residuals for each stat
    4. Converts counting stat residuals to rate adjustments
    5. Applies adjustments to player rates
    """

    statcast_source: FullStatcastDataSource
    batted_ball_source: PitcherBattedBallDataSource
    skill_data_source: SkillDataSource
    id_mapper: PlayerIdMapper
    config: GBResidualConfig = field(default_factory=GBResidualConfig)
    model_store: ModelStore = field(default_factory=ModelStore)

    _batter_models: ResidualModelSet | None = field(default=None, init=False, repr=False)
    _pitcher_models: ResidualModelSet | None = field(default=None, init=False, repr=False)
    _batter_statcast: dict[str, StatcastBatterStats] | None = field(default=None, init=False, repr=False)
    _pitcher_statcast: dict[str, StatcastPitcherStats] | None = field(default=None, init=False, repr=False)
    _pitcher_batted_ball: dict[str, PitcherBattedBallStats] | None = field(default=None, init=False, repr=False)
    _batter_skill_data: dict[str, BatterSkillStats] | None = field(default=None, init=False, repr=False)
    _cached_year: int | None = field(default=None, init=False, repr=False)

    def _ensure_models_loaded(self) -> None:
        """Lazy-load trained models on first use."""
        if self._batter_models is None:
            if self.model_store.exists(self.config.model_name, "batter"):
                self._batter_models = self.model_store.load(self.config.model_name, "batter")
                logger.debug(
                    "Loaded batter model %s with stats: %s",
                    self.config.model_name,
                    self._batter_models.get_stats(),
                )
            else:
                logger.warning("Batter model %s not found, skipping batter adjustments", self.config.model_name)

        if self._pitcher_models is None:
            if self.model_store.exists(self.config.model_name, "pitcher"):
                self._pitcher_models = self.model_store.load(self.config.model_name, "pitcher")
                logger.debug(
                    "Loaded pitcher model %s with stats: %s",
                    self.config.model_name,
                    self._pitcher_models.get_stats(),
                )
            else:
                logger.warning("Pitcher model %s not found, skipping pitcher adjustments", self.config.model_name)

    def _ensure_data_loaded(self, year: int) -> None:
        """Lazy-load Statcast, batted ball, and skill data for the projection year."""
        if self._cached_year == year:
            return

        # Load data from prior year (most recent available)
        data_year = year - 1

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

        # Batter skill data (uses FanGraphs ID)
        skill_data = self.skill_data_source.batter_skill_stats(data_year)
        self._batter_skill_data = {s.player_id: s for s in skill_data}
        logger.debug("Loaded %d batter skill records for year %d", len(self._batter_skill_data), data_year)

        self._cached_year = year

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        """Apply gradient boosting residual adjustments to player rates."""
        if not players:
            return []

        self._ensure_models_loaded()
        year = players[0].year
        self._ensure_data_loaded(year)

        result: list[PlayerRates] = []
        for player in players:
            if self._is_batter(player):
                result.append(self._adjust_batter(player))
            else:
                result.append(self._adjust_pitcher(player))

        return result

    def _is_batter(self, player: PlayerRates) -> bool:
        """Check if player is a batter (has pa_per_year metadata)."""
        return "pa_per_year" in player.metadata

    def _adjust_batter(self, player: PlayerRates) -> PlayerRates:
        """Apply residual adjustment to a batter."""
        if self._batter_models is None or self._batter_statcast is None:
            return player

        # Map FanGraphs ID to MLBAM for Statcast lookup
        mlbam_id = self.id_mapper.fangraphs_to_mlbam(player.player_id)
        if mlbam_id is None:
            return player

        statcast = self._batter_statcast.get(mlbam_id)
        if statcast is None or statcast.pa < self.config.batter_min_pa:
            return player

        # Get skill data (uses FanGraphs ID)
        skill_data = None
        if self._batter_skill_data is not None:
            skill_data = self._batter_skill_data.get(player.player_id)

        # Extract features
        extractor = BatterFeatureExtractor(min_pa=self.config.batter_min_pa)
        features = extractor.extract(player, statcast, skill_data)
        if features is None:
            return player

        # Predict residuals
        residuals = self._batter_models.predict_residuals(features)
        if not residuals:
            return player

        # Convert residuals to rate adjustments and apply
        rates = dict(player.rates)
        metadata: PlayerMetadata = {**player.metadata}

        # Use pa_per_year from metadata since opportunities may be 0 at this stage
        # (playing time projector runs after adjusters)
        # Apply minimum threshold to prevent extreme rate adjustments for low-PA players
        pa_per_year = player.metadata.get("pa_per_year", [])
        if isinstance(pa_per_year, list) and pa_per_year:
            avg_pa = sum(pa_per_year) / len(pa_per_year)
        else:
            avg_pa = player.opportunities
        opportunities = max(avg_pa, self.config.min_rate_denominator_pa)

        if opportunities > 0:
            for stat, residual in residuals.items():
                if stat in rates:
                    # Convert counting stat residual to rate adjustment
                    rate_adjustment = residual / opportunities
                    rates[stat] = rates[stat] + rate_adjustment

            metadata["gb_residual_adjustments"] = residuals

        return PlayerRates(
            player_id=player.player_id,
            name=player.name,
            year=player.year,
            age=player.age,
            rates=rates,
            opportunities=player.opportunities,
            metadata=metadata,
        )

    def _adjust_pitcher(self, player: PlayerRates) -> PlayerRates:
        """Apply residual adjustment to a pitcher."""
        if self._pitcher_models is None or self._pitcher_statcast is None:
            return player

        # Map FanGraphs ID to MLBAM for Statcast lookup
        mlbam_id = self.id_mapper.fangraphs_to_mlbam(player.player_id)
        if mlbam_id is None:
            return player

        statcast = self._pitcher_statcast.get(mlbam_id)
        if statcast is None or statcast.pa < self.config.pitcher_min_pa:
            return player

        # Get batted ball data (uses FanGraphs ID)
        batted_ball = None
        if self._pitcher_batted_ball is not None:
            batted_ball = self._pitcher_batted_ball.get(player.player_id)

        # Extract features
        extractor = PitcherFeatureExtractor(min_pa=self.config.pitcher_min_pa)
        features = extractor.extract(player, statcast, batted_ball)
        if features is None:
            return player

        # Predict residuals
        residuals = self._pitcher_models.predict_residuals(features)
        if not residuals:
            return player

        # Convert residuals to rate adjustments and apply
        rates = dict(player.rates)
        metadata: PlayerMetadata = {**player.metadata}

        # Use ip_per_year from metadata since opportunities may be 0 at this stage
        # (playing time projector runs after adjusters)
        # Apply minimum threshold to prevent extreme rate adjustments for low-IP pitchers
        ip_per_year = player.metadata.get("ip_per_year", [])
        if isinstance(ip_per_year, list) and ip_per_year:
            avg_ip = sum(ip_per_year) / len(ip_per_year)
        elif isinstance(ip_per_year, (int, float)) and ip_per_year > 0:
            avg_ip = ip_per_year
        else:
            avg_ip = player.opportunities / 3 if player.opportunities > 0 else 0
        # Use minimum threshold for rate conversion, convert IP to outs
        opportunities = max(avg_ip, self.config.min_rate_denominator_ip) * 3

        if opportunities > 0:
            for stat, residual in residuals.items():
                if stat in rates:
                    # Convert counting stat residual to rate adjustment
                    rate_adjustment = residual / opportunities
                    rates[stat] = rates[stat] + rate_adjustment

            metadata["gb_residual_adjustments"] = residuals

        return PlayerRates(
            player_id=player.player_id,
            name=player.name,
            year=player.year,
            age=player.age,
            rates=rates,
            opportunities=player.opportunities,
            metadata=metadata,
        )
