"""MTL-based rate computer for standalone neural network projections."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.ml.features import BatterFeatureExtractor, PitcherFeatureExtractor
from fantasy_baseball_manager.ml.mtl.config import MTLRateComputerConfig
from fantasy_baseball_manager.ml.mtl.model import BATTER_STATS, PITCHER_STATS
from fantasy_baseball_manager.ml.mtl.persistence import MTLModelStore
from fantasy_baseball_manager.pipeline.stages.rate_computers import (
    MarcelRateComputer,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import DataSource
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
    from fantasy_baseball_manager.ml.mtl.model import (
        MultiTaskBatterModel,
        MultiTaskPitcherModel,
    )
    from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallDataSource
    from fantasy_baseball_manager.pipeline.skill_data import SkillDataSource
    from fantasy_baseball_manager.pipeline.statcast_data import FullStatcastDataSource
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

logger = logging.getLogger(__name__)


@dataclass
class MTLRateComputer:
    """Computes player rates using trained MTL neural network.

    This is a standalone rate computer that replaces Marcel entirely.
    For players without sufficient Statcast data, falls back to Marcel rates.

    Implements the RateComputer protocol.
    """

    statcast_source: FullStatcastDataSource
    batted_ball_source: PitcherBattedBallDataSource
    skill_data_source: SkillDataSource
    id_mapper: PlayerIdMapper
    config: MTLRateComputerConfig = field(default_factory=MTLRateComputerConfig)
    model_store: MTLModelStore = field(default_factory=MTLModelStore)

    # Fallback to Marcel for players without Statcast data
    _marcel_computer: MarcelRateComputer = field(default_factory=MarcelRateComputer, repr=False)

    # Lazy-loaded models
    _batter_model: MultiTaskBatterModel | None = field(default=None, init=False, repr=False)
    _pitcher_model: MultiTaskPitcherModel | None = field(default=None, init=False, repr=False)
    _models_loaded: bool = field(default=False, init=False, repr=False)

    def _ensure_models_loaded(self) -> None:
        """Lazy-load trained models on first use."""
        if self._models_loaded:
            return

        if self.model_store.exists(self.config.model_name, "batter"):
            self._batter_model = self.model_store.load_batter_model(self.config.model_name)
            logger.debug("Loaded MTL batter model: %s", self.config.model_name)
        else:
            logger.warning(
                "MTL batter model %s not found, using Marcel fallback",
                self.config.model_name,
            )

        if self.model_store.exists(self.config.model_name, "pitcher"):
            self._pitcher_model = self.model_store.load_pitcher_model(self.config.model_name)
            logger.debug("Loaded MTL pitcher model: %s", self.config.model_name)
        else:
            logger.warning(
                "MTL pitcher model %s not found, using Marcel fallback",
                self.config.model_name,
            )

        self._models_loaded = True

    def compute_batting_rates(
        self,
        batting_source: DataSource[BattingSeasonStats],
        team_batting_source: DataSource[BattingSeasonStats],
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        """Compute batting rates using MTL model.

        For players with sufficient Statcast data, uses the MTL model.
        For others, falls back to Marcel rates.
        """
        self._ensure_models_loaded()

        # Get Marcel rates as fallback and for metadata
        marcel_rates = self._marcel_computer.compute_batting_rates(
            batting_source, team_batting_source, year, years_back
        )

        if self._batter_model is None or not self._batter_model.is_fitted:
            logger.debug("No MTL batter model available, using Marcel rates")
            return marcel_rates

        # Get Statcast and skill data from prior year
        statcast_year = year - 1
        statcast_data = self.statcast_source.batter_expected_stats(statcast_year)
        statcast_lookup = {s.player_id: s for s in statcast_data}

        skill_data = self.skill_data_source.batter_skill_stats(statcast_year)
        skill_lookup = {s.player_id: s for s in skill_data}

        extractor = BatterFeatureExtractor(min_pa=self.config.min_pa)

        result: list[PlayerRates] = []
        mtl_count = 0

        for marcel_player in marcel_rates:
            # Try to get MTL prediction
            mlbam_id = self.id_mapper.fangraphs_to_mlbam(marcel_player.player_id)
            statcast = statcast_lookup.get(mlbam_id) if mlbam_id else None

            if statcast is not None and statcast.pa >= self.config.min_pa:
                skill = skill_lookup.get(marcel_player.player_id)
                features = extractor.extract(marcel_player, statcast, skill)

                if features is not None:
                    # Use MTL predictions
                    mtl_predictions = self._batter_model.predict(features)

                    # Build rates dict with MTL predictions
                    rates = {}
                    for stat in BATTER_STATS:
                        if stat in mtl_predictions:
                            rates[stat] = mtl_predictions[stat]
                        elif stat in marcel_player.rates:
                            rates[stat] = marcel_player.rates[stat]

                    # Copy over any rates MTL doesn't predict
                    for stat, rate in marcel_player.rates.items():
                        if stat not in rates:
                            rates[stat] = rate

                    result.append(
                        PlayerRates(
                            player_id=marcel_player.player_id,
                            name=marcel_player.name,
                            year=year,
                            age=marcel_player.age,
                            rates=rates,
                            metadata={
                                **marcel_player.metadata,
                                "mtl_predicted": True,
                                "marcel_rates": marcel_player.rates,
                            },
                        )
                    )
                    mtl_count += 1
                    continue

            # Fallback to Marcel
            result.append(marcel_player)

        logger.info(
            "MTL rate computer: %d/%d batters used MTL predictions",
            mtl_count,
            len(result),
        )
        return result

    def compute_pitching_rates(
        self,
        pitching_source: DataSource[PitchingSeasonStats],
        team_pitching_source: DataSource[PitchingSeasonStats],
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        """Compute pitching rates using MTL model.

        For players with sufficient Statcast data, uses the MTL model.
        For others, falls back to Marcel rates.
        """
        self._ensure_models_loaded()

        # Get Marcel rates as fallback and for metadata
        marcel_rates = self._marcel_computer.compute_pitching_rates(
            pitching_source, team_pitching_source, year, years_back
        )

        if self._pitcher_model is None or not self._pitcher_model.is_fitted:
            logger.debug("No MTL pitcher model available, using Marcel rates")
            return marcel_rates

        # Get Statcast and batted ball data from prior year
        statcast_year = year - 1
        statcast_data = self.statcast_source.pitcher_expected_stats(statcast_year)
        statcast_lookup = {s.player_id: s for s in statcast_data}

        bb_data = self.batted_ball_source.pitcher_batted_ball_stats(statcast_year)
        bb_lookup = {b.player_id: b for b in bb_data}

        extractor = PitcherFeatureExtractor(min_pa=self.config.min_pa)

        result: list[PlayerRates] = []
        mtl_count = 0

        for marcel_player in marcel_rates:
            # Try to get MTL prediction
            mlbam_id = self.id_mapper.fangraphs_to_mlbam(marcel_player.player_id)
            statcast = statcast_lookup.get(mlbam_id) if mlbam_id else None

            if statcast is not None and statcast.pa >= self.config.min_pa:
                batted_ball = bb_lookup.get(marcel_player.player_id)
                features = extractor.extract(marcel_player, statcast, batted_ball)

                if features is not None:
                    # Use MTL predictions
                    mtl_predictions = self._pitcher_model.predict(features)

                    # Build rates dict with MTL predictions
                    rates = {}
                    for stat in PITCHER_STATS:
                        if stat in mtl_predictions:
                            rates[stat] = mtl_predictions[stat]
                        elif stat in marcel_player.rates:
                            rates[stat] = marcel_player.rates[stat]

                    # Copy over any rates MTL doesn't predict
                    for stat, rate in marcel_player.rates.items():
                        if stat not in rates:
                            rates[stat] = rate

                    result.append(
                        PlayerRates(
                            player_id=marcel_player.player_id,
                            name=marcel_player.name,
                            year=year,
                            age=marcel_player.age,
                            rates=rates,
                            metadata={
                                **marcel_player.metadata,
                                "mtl_predicted": True,
                                "marcel_rates": marcel_player.rates,
                            },
                        )
                    )
                    mtl_count += 1
                    continue

            # Fallback to Marcel
            result.append(marcel_player)

        logger.info(
            "MTL rate computer: %d/%d pitchers used MTL predictions",
            mtl_count,
            len(result),
        )
        return result
