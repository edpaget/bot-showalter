"""PyTorch Dataset for Multi-Task Learning training data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from fantasy_baseball_manager.ml.features import BatterFeatureExtractor, PitcherFeatureExtractor
from fantasy_baseball_manager.ml.mtl.model import BATTER_STATS, PITCHER_STATS

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallDataSource
    from fantasy_baseball_manager.pipeline.skill_data import SkillDataSource
    from fantasy_baseball_manager.pipeline.statcast_data import FullStatcastDataSource
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

logger = logging.getLogger(__name__)


class MTLDataset(Dataset):
    """PyTorch Dataset for multi-task learning on raw stat rates.

    Each sample contains:
    - features: Input feature vector
    - rates: Dict of target stat rates (actual stat / actual PA or outs)
    """

    def __init__(
        self,
        features: np.ndarray,
        target_rates: dict[str, np.ndarray],
    ) -> None:
        """Initialize dataset.

        Args:
            features: Feature matrix of shape (N, n_features).
            target_rates: Dict mapping stat name to target rates array of shape (N,).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target_rates = {
            stat: torch.tensor(rates, dtype=torch.float32).unsqueeze(1)
            for stat, rates in target_rates.items()
        }
        self._len = len(features)

    def __len__(self) -> int:
        return self._len

    def __getitem__(  # type: ignore[override]
        self, idx: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get a single sample.

        Returns:
            Tuple of (features tensor, dict of stat rates).
        """
        features = self.features[idx]
        rates = {stat: self.target_rates[stat][idx] for stat in self.target_rates}
        return features, rates


@dataclass
class BatterTrainingDataCollector:
    """Collects training data for batter MTL models.

    Gathers features from Statcast/skill data and actual stat rates
    for training the multi-task neural network.
    """

    data_source: StatsDataSource
    statcast_source: FullStatcastDataSource
    skill_data_source: SkillDataSource
    id_mapper: PlayerIdMapper
    min_pa: int = 100

    def collect(
        self,
        target_years: tuple[int, ...],
    ) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
        """Collect features and actual stat rates for training.

        For each player-year:
        1. Extract features from prior-year Statcast data
        2. Get actual stat rates from target year (actual_stat / actual_pa)

        Args:
            target_years: Years to use as targets (actuals come from these years,
                         features come from year-1).

        Returns:
            Tuple of:
            - features: (N, n_features) array
            - rates: dict mapping stat name to (N,) array of actual rates
            - feature_names: list of feature names
        """
        from fantasy_baseball_manager.pipeline.types import PlayerRates

        extractor = BatterFeatureExtractor(min_pa=self.min_pa)
        feature_names = extractor.feature_names()

        all_features: list[np.ndarray] = []
        all_rates: dict[str, list[float]] = {stat: [] for stat in BATTER_STATS}

        for target_year in target_years:
            logger.info("Collecting batter training data for year %d", target_year)

            # Get actuals for target year
            actuals = self.data_source.batting_stats(target_year)
            actual_lookup = {a.player_id: a for a in actuals}

            # Get Statcast data from prior year
            statcast_year = target_year - 1
            statcast_data = self.statcast_source.batter_expected_stats(statcast_year)
            statcast_lookup = {s.player_id: s for s in statcast_data}

            # Get skill data from prior year
            skill_data = self.skill_data_source.batter_skill_stats(statcast_year)
            skill_lookup = {s.player_id: s for s in skill_data}

            # Process each player with actuals
            for fg_id, actual in actual_lookup.items():
                if actual.pa < self.min_pa:
                    continue

                # Map FanGraphs ID to MLBAM for Statcast lookup
                mlbam_id = self.id_mapper.fangraphs_to_mlbam(fg_id)
                if mlbam_id is None:
                    continue

                statcast = statcast_lookup.get(mlbam_id)
                if statcast is None or statcast.pa < self.min_pa:
                    continue

                player_skill_data = skill_lookup.get(fg_id)

                # Create PlayerRates with actual rates for feature extraction
                # (extractor expects Marcel rates, but we use actual rates here
                # since we're predicting rates directly, not residuals)
                player_rates = PlayerRates(
                    player_id=fg_id,
                    name=actual.name if hasattr(actual, "name") else fg_id,
                    year=target_year,
                    age=actual.age if hasattr(actual, "age") else 30,
                    rates={
                        "hr": actual.hr / actual.pa,
                        "so": actual.so / actual.pa,
                        "bb": actual.bb / actual.pa,
                        "singles": actual.singles / actual.pa,
                        "doubles": actual.doubles / actual.pa,
                        "triples": actual.triples / actual.pa,
                        "sb": actual.sb / actual.pa if hasattr(actual, "sb") else 0,
                    },
                    opportunities=actual.pa,
                )

                features = extractor.extract(player_rates, statcast, player_skill_data)
                if features is None:
                    continue

                all_features.append(features)

                # Collect actual rates (target values)
                for stat in BATTER_STATS:
                    stat_value = getattr(actual, stat, 0)
                    rate = stat_value / actual.pa if actual.pa > 0 else 0.0
                    all_rates[stat].append(rate)

        if not all_features:
            return np.array([]), {stat: np.array([]) for stat in BATTER_STATS}, feature_names

        return (
            np.array(all_features),
            {stat: np.array(rates) for stat, rates in all_rates.items()},
            feature_names,
        )


@dataclass
class PitcherTrainingDataCollector:
    """Collects training data for pitcher MTL models.

    Gathers features from Statcast/batted ball data and actual stat rates
    for training the multi-task neural network.
    """

    data_source: StatsDataSource
    statcast_source: FullStatcastDataSource
    batted_ball_source: PitcherBattedBallDataSource
    id_mapper: PlayerIdMapper
    min_pa: int = 100

    def collect(
        self,
        target_years: tuple[int, ...],
    ) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
        """Collect features and actual stat rates for training.

        For each player-year:
        1. Extract features from prior-year Statcast/batted ball data
        2. Get actual stat rates from target year (actual_stat / actual_outs)

        Args:
            target_years: Years to use as targets (actuals come from these years,
                         features come from year-1).

        Returns:
            Tuple of:
            - features: (N, n_features) array
            - rates: dict mapping stat name to (N,) array of actual rates
            - feature_names: list of feature names
        """
        from fantasy_baseball_manager.pipeline.types import PlayerRates

        extractor = PitcherFeatureExtractor(min_pa=self.min_pa)
        feature_names = extractor.feature_names()

        all_features: list[np.ndarray] = []
        all_rates: dict[str, list[float]] = {stat: [] for stat in PITCHER_STATS}

        for target_year in target_years:
            logger.info("Collecting pitcher training data for year %d", target_year)

            # Get actuals for target year
            actuals = self.data_source.pitching_stats(target_year)
            actual_lookup = {a.player_id: a for a in actuals}

            # Get Statcast data from prior year
            statcast_year = target_year - 1
            statcast_data = self.statcast_source.pitcher_expected_stats(statcast_year)
            statcast_lookup = {s.player_id: s for s in statcast_data}

            # Get batted ball data from prior year
            batted_ball_data = self.batted_ball_source.pitcher_batted_ball_stats(statcast_year)
            bb_lookup = {b.player_id: b for b in batted_ball_data}

            # Process each player with actuals
            for fg_id, actual in actual_lookup.items():
                actual_outs = int(actual.ip * 3)
                if actual_outs < self.min_pa:
                    continue

                # Map FanGraphs ID to MLBAM for Statcast lookup
                mlbam_id = self.id_mapper.fangraphs_to_mlbam(fg_id)
                if mlbam_id is None:
                    continue

                statcast = statcast_lookup.get(mlbam_id)
                if statcast is None or statcast.pa < self.min_pa:
                    continue

                batted_ball = bb_lookup.get(fg_id)

                # Create PlayerRates with actual rates for feature extraction
                player_rates = PlayerRates(
                    player_id=fg_id,
                    name=actual.name if hasattr(actual, "name") else fg_id,
                    year=target_year,
                    age=actual.age if hasattr(actual, "age") else 30,
                    rates={
                        "h": actual.h / actual_outs if actual_outs > 0 else 0,
                        "er": actual.er / actual_outs if actual_outs > 0 else 0,
                        "so": actual.so / actual_outs if actual_outs > 0 else 0,
                        "bb": actual.bb / actual_outs if actual_outs > 0 else 0,
                        "hr": actual.hr / actual_outs if actual_outs > 0 else 0,
                    },
                    opportunities=actual_outs,
                    metadata={"is_starter": actual.gs > actual.g / 2 if actual.g > 0 else False},
                )

                features = extractor.extract(player_rates, statcast, batted_ball)
                if features is None:
                    continue

                all_features.append(features)

                # Collect actual rates (target values)
                for stat in PITCHER_STATS:
                    stat_value = getattr(actual, stat, 0)
                    rate = stat_value / actual_outs if actual_outs > 0 else 0.0
                    all_rates[stat].append(rate)

        if not all_features:
            return np.array([]), {stat: np.array([]) for stat in PITCHER_STATS}, feature_names

        return (
            np.array(all_features),
            {stat: np.array(rates) for stat, rates in all_rates.items()},
            feature_names,
        )
