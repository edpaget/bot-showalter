"""Tests for prediction source abstractions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from fantasy_baseball_manager.ml.residual_model import ResidualModelSet, StatResidualModel
from fantasy_baseball_manager.pipeline.stages.prediction_source import (
    ContextualPredictionSource,
    GBResidualPredictionSource,
    MTLPredictionSource,
    PredictionMode,
)
from fantasy_baseball_manager.pipeline.statcast_data import (
    StatcastBatterStats,
    StatcastPitcherStats,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates
from fantasy_baseball_manager.player.identity import Player
from fantasy_baseball_manager.registry.base_store import BaseModelStore
from fantasy_baseball_manager.registry.mtl_store import MTLBaseModelStore
from fantasy_baseball_manager.registry.serializers import JoblibSerializer, TorchParamsSerializer
from tests.conftest import make_test_feature_store

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fake data sources (reused from test_gb_residual_adjuster patterns)
# ---------------------------------------------------------------------------


class FakeStatcastSource:
    def __init__(
        self,
        batter_stats: dict[int, list[StatcastBatterStats]] | None = None,
        pitcher_stats: dict[int, list[StatcastPitcherStats]] | None = None,
    ) -> None:
        self._batter = batter_stats or {}
        self._pitcher = pitcher_stats or {}

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        return self._batter.get(year, [])

    def pitcher_expected_stats(self, year: int) -> list[StatcastPitcherStats]:
        return self._pitcher.get(year, [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batter(
    player_id: str = "fg123",
    year: int = 2024,
    mlbam_id: str | None = "mlbam123",
) -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name="Test Batter",
        year=year,
        age=28,
        rates={
            "hr": 0.040,
            "so": 0.200,
            "bb": 0.100,
            "singles": 0.150,
            "doubles": 0.050,
            "triples": 0.005,
            "sb": 0.020,
        },
        opportunities=500.0,
        metadata={"pa_per_year": [500.0]},
        player=Player(yahoo_id="", fangraphs_id=player_id, mlbam_id=mlbam_id, name="Test Batter"),
    )


def _make_pitcher(
    player_id: str = "fg456",
    year: int = 2024,
    mlbam_id: str | None = "mlbam456",
) -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name="Test Pitcher",
        year=year,
        age=30,
        rates={
            "h": 0.080,
            "er": 0.030,
            "so": 0.090,
            "bb": 0.030,
            "hr": 0.010,
        },
        opportunities=600.0,
        metadata={"is_starter": True},
        player=Player(yahoo_id="", fangraphs_id=player_id, mlbam_id=mlbam_id, name="Test Pitcher"),
    )


# ---------------------------------------------------------------------------
# PredictionMode
# ---------------------------------------------------------------------------


class TestPredictionMode:
    def test_rate_value(self) -> None:
        assert PredictionMode.RATE.value == "rate"

    def test_residual_value(self) -> None:
        assert PredictionMode.RESIDUAL.value == "residual"


# ---------------------------------------------------------------------------
# MTLPredictionSource
# ---------------------------------------------------------------------------


class TestMTLPredictionSource:
    def test_name(self) -> None:
        source = MTLPredictionSource(feature_store=make_test_feature_store())
        assert source.name == "mtl"

    def test_prediction_mode_is_rate(self) -> None:
        source = MTLPredictionSource(feature_store=make_test_feature_store())
        assert source.prediction_mode == PredictionMode.RATE

    def test_returns_none_when_no_models(self, tmp_path: Path) -> None:
        store = MTLBaseModelStore(
            model_dir=tmp_path / "empty",
            serializer=TorchParamsSerializer(),
            model_type_name="mtl",
        )
        source = MTLPredictionSource(
            feature_store=make_test_feature_store(),
            model_store=store,
        )
        source.ensure_ready(2024)
        batter = _make_batter()
        assert source.predict(batter) is None

    def test_returns_none_when_no_mlbam_id(self) -> None:
        source = MTLPredictionSource(feature_store=make_test_feature_store())
        batter = _make_batter(mlbam_id=None)
        assert source.predict(batter) is None

    def test_returns_none_when_no_statcast(self) -> None:
        source = MTLPredictionSource(feature_store=make_test_feature_store())
        batter = _make_batter()
        source.ensure_ready(2024)
        assert source.predict(batter) is None

    def test_ensure_ready_loads_data(self) -> None:
        source = MTLPredictionSource(feature_store=make_test_feature_store())
        source.ensure_ready(2024)
        assert source._cached_year == 2024

    def test_predict_batter_returns_rates(self) -> None:
        """When model and Statcast data are available, returns rate dict."""
        statcast = StatcastBatterStats(
            player_id="mlbam123",
            name="Test",
            year=2023,
            pa=450,
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )
        feature_store = make_test_feature_store(
            statcast_source=FakeStatcastSource(batter_stats={2023: [statcast]}),
        )

        source = MTLPredictionSource(feature_store=feature_store)
        source.ensure_ready(2024)

        mock_model = MagicMock()
        mock_model.is_fitted = True
        mock_model.predict.return_value = {
            "hr": 0.045,
            "so": 0.190,
            "bb": 0.110,
            "singles": 0.155,
            "doubles": 0.048,
            "triples": 0.004,
            "sb": 0.022,
        }
        source._batter_model = mock_model

        batter = _make_batter()
        result = source.predict(batter)
        assert result is not None
        assert "hr" in result
        assert result["hr"] == pytest.approx(0.045)

    def test_predict_pitcher_returns_rates(self) -> None:
        statcast = StatcastPitcherStats(
            player_id="mlbam456",
            name="Test",
            year=2023,
            pa=500,
            xwoba=0.300,
            xba=0.240,
            xslg=0.380,
            xera=3.50,
            barrel_rate=0.06,
            hard_hit_rate=0.35,
        )
        feature_store = make_test_feature_store(
            statcast_source=FakeStatcastSource(pitcher_stats={2023: [statcast]}),
        )

        source = MTLPredictionSource(feature_store=feature_store)
        source.ensure_ready(2024)

        mock_model = MagicMock()
        mock_model.is_fitted = True
        mock_model.predict.return_value = {
            "h": 0.075,
            "er": 0.025,
            "so": 0.095,
            "bb": 0.028,
            "hr": 0.009,
        }
        source._pitcher_model = mock_model

        pitcher = _make_pitcher()
        result = source.predict(pitcher)
        assert result is not None
        assert "so" in result
        assert result["so"] == pytest.approx(0.095)


# ---------------------------------------------------------------------------
# ContextualPredictionSource
# ---------------------------------------------------------------------------


class TestContextualPredictionSource:
    def test_name(self) -> None:
        source = ContextualPredictionSource(
            predictor=MagicMock(),
            id_mapper=MagicMock(),
        )
        assert source.name == "contextual"

    def test_prediction_mode_is_rate(self) -> None:
        source = ContextualPredictionSource(
            predictor=MagicMock(),
            id_mapper=MagicMock(),
        )
        assert source.prediction_mode == PredictionMode.RATE

    def test_returns_none_when_no_mlbam_id(self) -> None:
        source = ContextualPredictionSource(
            predictor=MagicMock(),
            id_mapper=MagicMock(),
        )
        batter = _make_batter(mlbam_id=None)
        assert source.predict(batter) is None

    def test_returns_none_when_no_model(self) -> None:
        predictor = MagicMock()
        predictor._batter_model = None
        predictor._pitcher_model = None
        source = ContextualPredictionSource(
            predictor=predictor,
            id_mapper=MagicMock(),
        )
        batter = _make_batter()
        assert source.predict(batter) is None

    def test_returns_none_when_prediction_fails(self) -> None:
        predictor = MagicMock()
        predictor._batter_model = MagicMock()
        predictor.predict_player.return_value = None
        source = ContextualPredictionSource(
            predictor=predictor,
            id_mapper=MagicMock(),
        )
        batter = _make_batter(mlbam_id="12345")
        assert source.predict(batter) is None

    def test_ensure_ready_calls_models_loaded(self) -> None:
        predictor = MagicMock()
        source = ContextualPredictionSource(
            predictor=predictor,
            id_mapper=MagicMock(),
        )
        source.ensure_ready(2024)
        predictor.ensure_models_loaded.assert_called_once()

    def test_predict_batter_returns_rates(self) -> None:
        """When predictor returns valid predictions, source returns rate dict."""
        mock_game = MagicMock()
        mock_pitch = MagicMock()
        mock_pitch.pa_event = "single"
        mock_game.pitches = [mock_pitch] * 4  # 4 PA per game

        predictor = MagicMock()
        predictor._batter_model = MagicMock()
        predictor.predict_player.return_value = (
            {"hr": 0.5, "so": 2.0, "bb": 0.8, "h": 1.5, "2b": 0.3, "3b": 0.05},
            [mock_game] * 5,
        )

        source = ContextualPredictionSource(
            predictor=predictor,
            id_mapper=MagicMock(),
        )
        batter = _make_batter(mlbam_id="12345")
        result = source.predict(batter)
        assert result is not None
        assert "hr" in result
        # hr per game / pa per game = 0.5 / 4.0 = 0.125
        assert result["hr"] == pytest.approx(0.5 / 4.0)


# ---------------------------------------------------------------------------
# GBResidualPredictionSource
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_gb_store(tmp_path: Path) -> BaseModelStore:
    return BaseModelStore(
        model_dir=tmp_path / "gb_models",
        serializer=JoblibSerializer(),
        model_type_name="gb_residual",
    )


@pytest.fixture
def trained_batter_gb_models(temp_gb_store: BaseModelStore) -> BaseModelStore:
    np.random.seed(42)
    feature_names = [
        "marcel_hr", "marcel_so", "marcel_bb", "marcel_singles",
        "marcel_doubles", "marcel_triples", "marcel_sb",
        "xba", "xslg", "xwoba", "barrel_rate", "hard_hit_rate",
        "chase_rate", "whiff_rate", "chase_minus_league_avg",
        "whiff_minus_league_avg", "chase_x_whiff", "discipline_score",
        "has_skill_data", "age", "age_squared", "marcel_iso",
        "xba_minus_marcel_avg", "barrel_vs_hr_ratio", "opportunities",
    ]
    X = np.random.randn(50, len(feature_names))
    model_set = ResidualModelSet(
        player_type="batter",
        feature_names=feature_names,
        training_years=(2021, 2022),
    )
    for stat in ["hr", "so", "bb"]:
        y = np.random.randn(50) * 2
        model = StatResidualModel(stat_name=stat)
        model.fit(X, y, feature_names)
        model_set.add_model(model)

    temp_gb_store.save_params(
        model_set.get_params(),
        "default",
        "batter",
        training_years=(2021, 2022),
        stats=model_set.get_stats(),
        feature_names=model_set.feature_names,
    )
    return temp_gb_store


class TestGBResidualPredictionSource:
    def test_name(self) -> None:
        source = GBResidualPredictionSource(feature_store=make_test_feature_store())
        assert source.name == "gb_residual"

    def test_prediction_mode_is_residual(self) -> None:
        source = GBResidualPredictionSource(feature_store=make_test_feature_store())
        assert source.prediction_mode == PredictionMode.RESIDUAL

    def test_returns_none_when_no_models(self, tmp_path: Path) -> None:
        store = BaseModelStore(
            model_dir=tmp_path / "empty",
            serializer=JoblibSerializer(),
            model_type_name="gb_residual",
        )
        source = GBResidualPredictionSource(
            feature_store=make_test_feature_store(),
            model_store=store,
        )
        source.ensure_ready(2024)
        batter = _make_batter()
        assert source.predict(batter) is None

    def test_returns_none_when_no_mlbam_id(self) -> None:
        source = GBResidualPredictionSource(feature_store=make_test_feature_store())
        batter = _make_batter(mlbam_id=None)
        assert source.predict(batter) is None

    def test_returns_none_when_no_statcast(self) -> None:
        source = GBResidualPredictionSource(feature_store=make_test_feature_store())
        source.ensure_ready(2024)
        batter = _make_batter()
        assert source.predict(batter) is None

    def test_predict_batter_returns_rate_adjustments(
        self,
        trained_batter_gb_models: BaseModelStore,
    ) -> None:
        statcast = StatcastBatterStats(
            player_id="mlbam123",
            name="Test",
            year=2023,
            pa=450,
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )
        source = GBResidualPredictionSource(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource(batter_stats={2023: [statcast]}),
            ),
            model_store=trained_batter_gb_models,
        )
        source.ensure_ready(2024)
        batter = _make_batter()
        result = source.predict(batter)
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_ensure_ready_loads_models_and_data(
        self,
        trained_batter_gb_models: BaseModelStore,
    ) -> None:
        source = GBResidualPredictionSource(
            feature_store=make_test_feature_store(),
            model_store=trained_batter_gb_models,
        )
        source.ensure_ready(2024)
        assert source._batter_models is not None
        assert source._cached_year == 2024
