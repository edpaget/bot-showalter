"""Prediction source abstractions for the ensemble framework.

Provides a unified interface for different model types (MTL, contextual,
gradient boosting) to produce predictions that can be combined by the
EnsembleAdjuster.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fantasy_baseball_manager.ml.features import BatterFeatureExtractor, PitcherFeatureExtractor

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.predictor import ContextualPredictor
    from fantasy_baseball_manager.ml.mtl.model import MultiTaskBatterModel, MultiTaskPitcherModel
    from fantasy_baseball_manager.ml.residual_model import ResidualModelSet
    from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallStats
    from fantasy_baseball_manager.pipeline.feature_store import FeatureStore
    from fantasy_baseball_manager.pipeline.skill_data import BatterSkillStats
    from fantasy_baseball_manager.pipeline.statcast_data import (
        StatcastBatterStats,
        StatcastPitcherStats,
    )
    from fantasy_baseball_manager.pipeline.types import PlayerRates
    from fantasy_baseball_manager.player_id.mapper import SfbbMapper
    from fantasy_baseball_manager.registry.base_store import BaseModelStore
    from fantasy_baseball_manager.registry.mtl_store import MTLBaseModelStore

logger = logging.getLogger(__name__)


class PredictionMode(Enum):
    """How a source's predictions should be combined with the baseline."""

    RATE = "rate"
    RESIDUAL = "residual"


@runtime_checkable
class PredictionSource(Protocol):
    """Unified interface for model prediction sources."""

    @property
    def name(self) -> str: ...

    @property
    def prediction_mode(self) -> PredictionMode: ...

    def predict(self, player: PlayerRates) -> dict[str, float] | None: ...

    def ensure_ready(self, year: int) -> None: ...


# ---------------------------------------------------------------------------
# MTLPredictionSource
# ---------------------------------------------------------------------------


@dataclass
class MTLPredictionSource:
    """Wraps MTL model loading and prediction behind PredictionSource.

    Extracts the prediction logic from MTLBlender — loads models via
    MTLBaseModelStore, extracts features from Statcast/skill data,
    and returns raw MTL rate predictions (not blended).
    """

    feature_store: FeatureStore
    model_store: MTLBaseModelStore | None = None
    model_name: str = "default"
    min_pa: int = 100

    _batter_model: MultiTaskBatterModel | None = field(default=None, init=False, repr=False)
    _pitcher_model: MultiTaskPitcherModel | None = field(default=None, init=False, repr=False)
    _models_loaded: bool = field(default=False, init=False, repr=False)

    _batter_statcast: dict[str, StatcastBatterStats] = field(default_factory=dict, init=False, repr=False)
    _pitcher_statcast: dict[str, StatcastPitcherStats] = field(default_factory=dict, init=False, repr=False)
    _pitcher_batted_ball: dict[str, PitcherBattedBallStats] = field(default_factory=dict, init=False, repr=False)
    _batter_skill_data: dict[str, BatterSkillStats] = field(default_factory=dict, init=False, repr=False)
    _cached_year: int | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return "mtl"

    @property
    def prediction_mode(self) -> PredictionMode:
        return PredictionMode.RATE

    def ensure_ready(self, year: int) -> None:
        self._ensure_models_loaded()
        self._ensure_data_loaded(year)

    def predict(self, player: PlayerRates) -> dict[str, float] | None:
        mlbam_id = player.player.mlbam_id if player.player else None
        if mlbam_id is None:
            return None

        if self._is_batter(player):
            return self._predict_batter(player, mlbam_id)
        return self._predict_pitcher(player, mlbam_id)

    def _is_batter(self, player: PlayerRates) -> bool:
        return "pa_per_year" in player.metadata

    def _predict_batter(self, player: PlayerRates, mlbam_id: str) -> dict[str, float] | None:
        if self._batter_model is None or not self._batter_model.is_fitted:
            return None

        statcast = self._batter_statcast.get(mlbam_id)
        if statcast is None or statcast.pa < self.min_pa:
            return None

        skill_data = self._batter_skill_data.get(player.player_id)
        extractor = BatterFeatureExtractor(min_pa=self.min_pa)
        features = extractor.extract(player, statcast, skill_data)
        if features is None:
            return None

        from fantasy_baseball_manager.ml.mtl.model import BATTER_STATS

        predictions = self._batter_model.predict(features)
        if not predictions:
            return None

        return {stat: predictions[stat] for stat in BATTER_STATS if stat in predictions and stat in player.rates}

    def _predict_pitcher(self, player: PlayerRates, mlbam_id: str) -> dict[str, float] | None:
        if self._pitcher_model is None or not self._pitcher_model.is_fitted:
            return None

        statcast = self._pitcher_statcast.get(mlbam_id)
        if statcast is None or statcast.pa < self.min_pa:
            return None

        batted_ball = self._pitcher_batted_ball.get(player.player_id)
        extractor = PitcherFeatureExtractor(min_pa=self.min_pa)
        features = extractor.extract(player, statcast, batted_ball)
        if features is None:
            return None

        from fantasy_baseball_manager.ml.mtl.model import PITCHER_STATS

        predictions = self._pitcher_model.predict(features)
        if not predictions:
            return None

        return {stat: predictions[stat] for stat in PITCHER_STATS if stat in predictions and stat in player.rates}

    def _ensure_models_loaded(self) -> None:
        if self._models_loaded:
            return

        store = self._resolve_model_store()
        if store.exists(self.model_name, "batter"):
            self._batter_model = store.load_batter(self.model_name)
            logger.debug("Loaded MTL batter model: %s", self.model_name)
        else:
            logger.warning("MTL batter model %s not found", self.model_name)

        if store.exists(self.model_name, "pitcher"):
            self._pitcher_model = store.load_pitcher(self.model_name)
            logger.debug("Loaded MTL pitcher model: %s", self.model_name)
        else:
            logger.warning("MTL pitcher model %s not found", self.model_name)

        self._models_loaded = True

    def _ensure_data_loaded(self, year: int) -> None:
        if self._cached_year == year:
            return
        data_year = year - 1
        self._batter_statcast = self.feature_store.batter_statcast(data_year)
        self._pitcher_statcast = self.feature_store.pitcher_statcast(data_year)
        self._pitcher_batted_ball = self.feature_store.pitcher_batted_ball(data_year)
        self._batter_skill_data = self.feature_store.batter_skill(data_year)
        self._cached_year = year

    def _resolve_model_store(self) -> MTLBaseModelStore:
        if self.model_store is not None:
            return self.model_store
        from fantasy_baseball_manager.ml.mtl.persistence import DEFAULT_MTL_MODEL_DIR
        from fantasy_baseball_manager.registry.mtl_store import MTLBaseModelStore as Store
        from fantasy_baseball_manager.registry.serializers import TorchParamsSerializer

        self.model_store = Store(
            model_dir=DEFAULT_MTL_MODEL_DIR,
            serializer=TorchParamsSerializer(),
            model_type_name="mtl",
        )
        return self.model_store


# ---------------------------------------------------------------------------
# ContextualPredictionSource
# ---------------------------------------------------------------------------


@dataclass
class ContextualPredictionSource:
    """Wraps contextual transformer prediction behind PredictionSource.

    Extracts the prediction logic from ContextualBlender — uses
    ContextualPredictor for inference and PerGameToSeasonAdapter for
    rate conversion. Returns raw contextual rate predictions (not blended).
    """

    predictor: ContextualPredictor
    id_mapper: SfbbMapper
    batter_model_name: str = "finetune_batter_best"
    pitcher_model_name: str = "finetune_pitcher_best"
    batter_context_window: int = 30
    pitcher_context_window: int = 10
    batter_min_games: int = 30
    pitcher_min_games: int = 10

    @property
    def name(self) -> str:
        return "contextual"

    @property
    def prediction_mode(self) -> PredictionMode:
        return PredictionMode.RATE

    def ensure_ready(self, year: int) -> None:
        self.predictor.ensure_models_loaded(
            self.batter_model_name,
            self.pitcher_model_name,
        )

    def predict(self, player: PlayerRates) -> dict[str, float] | None:
        mlbam_id = player.player.mlbam_id if player.player else None
        if mlbam_id is None:
            return None

        is_batter = "pa_per_year" in player.metadata
        perspective = "batter" if is_batter else "pitcher"
        model = self.predictor._batter_model if is_batter else self.predictor._pitcher_model
        if model is None:
            return None

        if is_batter:
            from fantasy_baseball_manager.contextual.training.config import BATTER_TARGET_STATS
            target_stats = BATTER_TARGET_STATS
            min_games = self.batter_min_games
            context_window = self.batter_context_window
        else:
            from fantasy_baseball_manager.contextual.training.config import PITCHER_TARGET_STATS
            target_stats = PITCHER_TARGET_STATS
            min_games = self.pitcher_min_games
            context_window = self.pitcher_context_window

        data_year = player.year - 1
        prediction = self.predictor.predict_player(
            mlbam_id=int(mlbam_id),
            data_year=data_year,
            perspective=perspective,
            model=model,
            target_stats=target_stats,
            min_games=min_games,
            context_window=context_window,
        )
        if prediction is None:
            return None

        avg_predictions, context_games = prediction

        from fantasy_baseball_manager.contextual.adapter import PerGameToSeasonAdapter

        adapter = PerGameToSeasonAdapter(perspective)
        avg_denominator = sum(adapter.game_denominator(g) for g in context_games) / len(context_games)
        return adapter.predictions_to_rates(avg_predictions, avg_denominator, player.rates)


# ---------------------------------------------------------------------------
# GBResidualPredictionSource
# ---------------------------------------------------------------------------


@dataclass
class GBResidualPredictionSource:
    """Wraps gradient boosting residual prediction behind PredictionSource.

    Extracts the prediction logic from GBResidualAdjuster — loads models,
    extracts features, predicts residuals, and converts them to rate
    adjustments. Mode=RESIDUAL so the ensemble applies these additively.
    """

    feature_store: FeatureStore
    model_name: str = "default"
    batter_min_pa: int = 100
    pitcher_min_pa: int = 100
    min_rate_denominator_pa: int = 300
    min_rate_denominator_ip: int = 100
    model_store: BaseModelStore | None = None

    _batter_models: ResidualModelSet | None = field(default=None, init=False, repr=False)
    _pitcher_models: ResidualModelSet | None = field(default=None, init=False, repr=False)
    _batter_statcast: dict[str, StatcastBatterStats] = field(default_factory=dict, init=False, repr=False)
    _pitcher_statcast: dict[str, StatcastPitcherStats] = field(default_factory=dict, init=False, repr=False)
    _pitcher_batted_ball: dict[str, PitcherBattedBallStats] = field(default_factory=dict, init=False, repr=False)
    _batter_skill_data: dict[str, BatterSkillStats] = field(default_factory=dict, init=False, repr=False)
    _cached_year: int | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return "gb_residual"

    @property
    def prediction_mode(self) -> PredictionMode:
        return PredictionMode.RESIDUAL

    def ensure_ready(self, year: int) -> None:
        self._ensure_models_loaded()
        self._ensure_data_loaded(year)

    def predict(self, player: PlayerRates) -> dict[str, float] | None:
        mlbam_id = player.player.mlbam_id if player.player else None
        if mlbam_id is None:
            return None

        if self._is_batter(player):
            return self._predict_batter(player, mlbam_id)
        return self._predict_pitcher(player, mlbam_id)

    def _is_batter(self, player: PlayerRates) -> bool:
        return "pa_per_year" in player.metadata

    def _predict_batter(self, player: PlayerRates, mlbam_id: str) -> dict[str, float] | None:
        if self._batter_models is None:
            return None

        statcast = self._batter_statcast.get(mlbam_id)
        if statcast is None or statcast.pa < self.batter_min_pa:
            return None

        skill_data = self._batter_skill_data.get(player.player_id)
        extractor = BatterFeatureExtractor(min_pa=self.batter_min_pa)
        features = extractor.extract(player, statcast, skill_data)
        if features is None:
            return None

        residuals = self._batter_models.predict_residuals(features)
        if not residuals:
            return None

        pa_per_year = player.metadata.get("pa_per_year", [])
        if isinstance(pa_per_year, list) and pa_per_year:
            avg_pa = sum(pa_per_year) / len(pa_per_year)
        else:
            avg_pa = player.opportunities
        opportunities = max(avg_pa, self.min_rate_denominator_pa)

        if opportunities <= 0:
            return None

        rate_adjustments: dict[str, float] = {}
        for stat, residual in residuals.items():
            if stat in player.rates:
                rate_adjustments[stat] = residual / opportunities

        return rate_adjustments if rate_adjustments else None

    def _predict_pitcher(self, player: PlayerRates, mlbam_id: str) -> dict[str, float] | None:
        if self._pitcher_models is None:
            return None

        statcast = self._pitcher_statcast.get(mlbam_id)
        if statcast is None or statcast.pa < self.pitcher_min_pa:
            return None

        batted_ball = self._pitcher_batted_ball.get(player.player_id)
        extractor = PitcherFeatureExtractor(min_pa=self.pitcher_min_pa)
        features = extractor.extract(player, statcast, batted_ball)
        if features is None:
            return None

        residuals = self._pitcher_models.predict_residuals(features)
        if not residuals:
            return None

        ip_per_year = player.metadata.get("ip_per_year", [])
        if isinstance(ip_per_year, list) and ip_per_year:
            avg_ip = sum(ip_per_year) / len(ip_per_year)
        elif isinstance(ip_per_year, (int, float)) and ip_per_year > 0:
            avg_ip = ip_per_year
        else:
            avg_ip = player.opportunities / 3 if player.opportunities > 0 else 0
        opportunities = max(avg_ip, self.min_rate_denominator_ip) * 3

        if opportunities <= 0:
            return None

        rate_adjustments: dict[str, float] = {}
        for stat, residual in residuals.items():
            if stat in player.rates:
                rate_adjustments[stat] = residual / opportunities

        return rate_adjustments if rate_adjustments else None

    def _ensure_models_loaded(self) -> None:
        from fantasy_baseball_manager.ml.residual_model import ResidualModelSet

        store = self._resolve_model_store()

        if self._batter_models is None:
            if store.exists(self.model_name, "batter"):
                params = store.load_params(self.model_name, "batter")
                self._batter_models = ResidualModelSet.from_params(params)
                logger.debug("Loaded GB batter model: %s", self.model_name)
            else:
                logger.warning("GB batter model %s not found", self.model_name)

        if self._pitcher_models is None:
            if store.exists(self.model_name, "pitcher"):
                params = store.load_params(self.model_name, "pitcher")
                self._pitcher_models = ResidualModelSet.from_params(params)
                logger.debug("Loaded GB pitcher model: %s", self.model_name)
            else:
                logger.warning("GB pitcher model %s not found", self.model_name)

    def _ensure_data_loaded(self, year: int) -> None:
        if self._cached_year == year:
            return
        data_year = year - 1
        self._batter_statcast = self.feature_store.batter_statcast(data_year)
        self._pitcher_statcast = self.feature_store.pitcher_statcast(data_year)
        self._pitcher_batted_ball = self.feature_store.pitcher_batted_ball(data_year)
        self._batter_skill_data = self.feature_store.batter_skill(data_year)
        self._cached_year = year

    def _resolve_model_store(self) -> BaseModelStore:
        if self.model_store is not None:
            return self.model_store
        from fantasy_baseball_manager.ml.persistence import DEFAULT_MODEL_DIR
        from fantasy_baseball_manager.registry.base_store import BaseModelStore as Store
        from fantasy_baseball_manager.registry.serializers import JoblibSerializer

        self.model_store = Store(
            model_dir=DEFAULT_MODEL_DIR,
            serializer=JoblibSerializer(),
            model_type_name="gb_residual",
        )
        return self.model_store
