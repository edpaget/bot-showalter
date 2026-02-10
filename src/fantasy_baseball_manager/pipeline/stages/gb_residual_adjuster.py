"""Gradient boosting residual adjuster for projection corrections."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.ml.features import BatterFeatureExtractor, PitcherFeatureExtractor
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates
from fantasy_baseball_manager.registry.base_store import BaseModelStore
from fantasy_baseball_manager.registry.serializers import JoblibSerializer

if TYPE_CHECKING:
    from fantasy_baseball_manager.ml.residual_model import ResidualModelSet
    from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallStats
    from fantasy_baseball_manager.pipeline.feature_store import FeatureStore
    from fantasy_baseball_manager.pipeline.skill_data import BatterSkillStats
    from fantasy_baseball_manager.pipeline.statcast_data import (
        StatcastBatterStats,
        StatcastPitcherStats,
    )

logger = logging.getLogger(__name__)


def _default_gb_store() -> BaseModelStore:
    from fantasy_baseball_manager.ml.persistence import DEFAULT_MODEL_DIR

    return BaseModelStore(
        model_dir=DEFAULT_MODEL_DIR,
        serializer=JoblibSerializer(),
        model_type_name="gb_residual",
    )


@dataclass(frozen=True)
class GBResidualConfig:
    """Configuration for gradient boosting residual adjustment."""

    model_name: str = "default"
    batter_min_pa: int = 100
    pitcher_min_pa: int = 100
    max_residual_scale: float = 2.0  # Cap residuals at N * std
    min_rate_denominator_pa: int = 300  # Minimum PA for rate conversion
    min_rate_denominator_ip: int = 100  # Minimum IP for rate conversion
    # Which stats to apply adjustments for (None = all available)
    # Use ("hr", "sb") for conservative mode that doesn't hurt OBP
    batter_allowed_stats: tuple[str, ...] | None = None
    pitcher_allowed_stats: tuple[str, ...] | None = None


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

    feature_store: FeatureStore
    config: GBResidualConfig = field(default_factory=GBResidualConfig)
    model_store: BaseModelStore = field(default_factory=_default_gb_store)

    _batter_models: ResidualModelSet | None = field(default=None, init=False, repr=False)
    _pitcher_models: ResidualModelSet | None = field(default=None, init=False, repr=False)
    _batter_statcast: dict[str, StatcastBatterStats] = field(default_factory=dict, init=False, repr=False)
    _pitcher_statcast: dict[str, StatcastPitcherStats] = field(default_factory=dict, init=False, repr=False)
    _pitcher_batted_ball: dict[str, PitcherBattedBallStats] = field(default_factory=dict, init=False, repr=False)
    _batter_skill_data: dict[str, BatterSkillStats] = field(default_factory=dict, init=False, repr=False)
    _cached_year: int | None = field(default=None, init=False, repr=False)

    def _ensure_models_loaded(self) -> None:
        """Lazy-load trained models on first use."""
        from fantasy_baseball_manager.ml.residual_model import ResidualModelSet

        if self._batter_models is None:
            if self.model_store.exists(self.config.model_name, "batter"):
                params = self.model_store.load_params(self.config.model_name, "batter")
                self._batter_models = ResidualModelSet.from_params(params)
                logger.debug(
                    "Loaded batter model %s with stats: %s",
                    self.config.model_name,
                    self._batter_models.get_stats(),
                )
            else:
                logger.warning("Batter model %s not found, skipping batter adjustments", self.config.model_name)

        if self._pitcher_models is None:
            if self.model_store.exists(self.config.model_name, "pitcher"):
                params = self.model_store.load_params(self.config.model_name, "pitcher")
                self._pitcher_models = ResidualModelSet.from_params(params)
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

        data_year = year - 1
        self._batter_statcast = self.feature_store.batter_statcast(data_year)
        self._pitcher_statcast = self.feature_store.pitcher_statcast(data_year)
        self._pitcher_batted_ball = self.feature_store.pitcher_batted_ball(data_year)
        self._batter_skill_data = self.feature_store.batter_skill(data_year)
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
        if self._batter_models is None:
            return player

        # Get MLBAM ID from enriched Player identity
        mlbam_id = player.player.mlbam_id if player.player else None
        if mlbam_id is None:
            return player

        statcast = self._batter_statcast.get(mlbam_id)
        if statcast is None or statcast.pa < self.config.batter_min_pa:
            return player

        # Get skill data (uses FanGraphs ID)
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
            # Filter residuals by allowed_stats if configured
            allowed = self.config.batter_allowed_stats
            applied_residuals: dict[str, float] = {}
            for stat, residual in residuals.items():
                if stat in rates and (allowed is None or stat in allowed):
                    # Convert counting stat residual to rate adjustment
                    rate_adjustment = residual / opportunities
                    rates[stat] = rates[stat] + rate_adjustment
                    applied_residuals[stat] = residual

            if applied_residuals:
                metadata["gb_residual_adjustments"] = applied_residuals

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

    def _adjust_pitcher(self, player: PlayerRates) -> PlayerRates:
        """Apply residual adjustment to a pitcher."""
        if self._pitcher_models is None:
            return player

        # Get MLBAM ID from enriched Player identity
        mlbam_id = player.player.mlbam_id if player.player else None
        if mlbam_id is None:
            return player

        statcast = self._pitcher_statcast.get(mlbam_id)
        if statcast is None or statcast.pa < self.config.pitcher_min_pa:
            return player

        # Get batted ball data (uses FanGraphs ID)
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
            # Filter residuals by allowed_stats if configured
            allowed = self.config.pitcher_allowed_stats
            applied_residuals: dict[str, float] = {}
            for stat, residual in residuals.items():
                if stat in rates and (allowed is None or stat in allowed):
                    # Convert counting stat residual to rate adjustment
                    rate_adjustment = residual / opportunities
                    rates[stat] = rates[stat] + rate_adjustment
                    applied_residuals[stat] = residual

            if applied_residuals:
                metadata["gb_residual_adjustments"] = applied_residuals

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
