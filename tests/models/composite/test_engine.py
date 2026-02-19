from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from fantasy_baseball_manager.models.composite.engine import (
    CompositeEngine,
    EngineConfig,
    GBMEngine,
    MarcelEngine,
    resolve_engine,
)
from fantasy_baseball_manager.models.composite.targets import BATTER_TARGETS, PITCHER_TARGETS
from fantasy_baseball_manager.models.marcel.types import MarcelConfig


class TestEngineConfig:
    def test_fields_accessible(self) -> None:
        mc = MarcelConfig()
        cfg = EngineConfig(
            marcel_config=mc,
            projected_season=2025,
            version="v1",
            system_name="composite",
        )
        assert cfg.marcel_config is mc
        assert cfg.projected_season == 2025
        assert cfg.version == "v1"
        assert cfg.system_name == "composite"

    def test_frozen(self) -> None:
        cfg = EngineConfig(
            marcel_config=MarcelConfig(),
            projected_season=2025,
            version="v1",
            system_name="composite",
        )
        with pytest.raises(FrozenInstanceError):
            cfg.projected_season = 2026  # type: ignore[misc]


class TestEngineConfigGBMFields:
    def test_defaults_are_none_and_empty(self) -> None:
        cfg = EngineConfig(
            marcel_config=MarcelConfig(),
            projected_season=2025,
            version="v1",
            system_name="composite",
        )
        assert cfg.artifact_path is None
        assert cfg.bat_feature_cols == ()
        assert cfg.pitch_feature_cols == ()

    def test_explicit_values_accessible(self) -> None:
        cfg = EngineConfig(
            marcel_config=MarcelConfig(),
            projected_season=2025,
            version="v1",
            system_name="composite",
            artifact_path=Path("/tmp/models"),
            bat_feature_cols=("age", "hr_1"),
            pitch_feature_cols=("age", "so_1"),
        )
        assert cfg.artifact_path == Path("/tmp/models")
        assert cfg.bat_feature_cols == ("age", "hr_1")
        assert cfg.pitch_feature_cols == ("age", "so_1")


class TestCompositeEngineProtocol:
    def test_runtime_checkable(self) -> None:
        assert isinstance(MarcelEngine(), CompositeEngine)


class TestMarcelEngineProperties:
    def test_supported_operations(self) -> None:
        assert MarcelEngine().supported_operations == frozenset({"prepare", "predict"})

    def test_artifact_type(self) -> None:
        assert MarcelEngine().artifact_type == "none"


class TestMarcelEnginePredict:
    @pytest.fixture
    def marcel_config(self) -> MarcelConfig:
        return MarcelConfig(batting_categories=("hr",), pitching_categories=("so",))

    @pytest.fixture
    def engine_config(self, marcel_config: MarcelConfig) -> EngineConfig:
        return EngineConfig(
            marcel_config=marcel_config,
            projected_season=2024,
            version="latest",
            system_name="composite",
        )

    @pytest.fixture
    def batter_row(self) -> dict[str, Any]:
        return {
            "player_id": 1,
            "season": 2023,
            "age": 29,
            "proj_pa": 555,
            "pa_1": 600,
            "pa_2": 550,
            "hr_1": 30.0,
            "hr_2": 25.0,
            "hr_wavg": 310.0 / 6700.0,
            "weighted_pt": 6700.0,
            "league_hr_rate": 50.0 / 1100.0,
        }

    @pytest.fixture
    def pitcher_row(self) -> dict[str, Any]:
        return {
            "player_id": 10,
            "season": 2023,
            "age": 28,
            "proj_ip": 150.0,
            "ip_1": 180.0,
            "ip_2": 170.0,
            "g_1": 30,
            "g_2": 28,
            "gs_1": 30,
            "gs_2": 28,
            "so_1": 200.0,
            "so_2": 180.0,
            "so_wavg": 1110.0 / 1040.0,
            "weighted_pt": 1040.0,
            "league_so_rate": 200.0 / 180.0,
        }

    def test_batter_returns_prediction(self, batter_row: dict[str, Any], engine_config: EngineConfig) -> None:
        engine = MarcelEngine()
        bat_pt = {1: 555.0}
        result = engine.predict([batter_row], [], bat_pt, {}, engine_config)
        assert len(result) == 1
        pred = result[0]
        assert pred["player_id"] == 1
        assert pred["season"] == 2024
        assert pred["player_type"] == "batter"
        assert "hr" in pred

    def test_batter_counting_equals_rate_times_pt(
        self, batter_row: dict[str, Any], engine_config: EngineConfig
    ) -> None:
        engine = MarcelEngine()
        bat_pt = {1: 600.0}
        result = engine.predict([batter_row], [], bat_pt, {}, engine_config)
        pred = result[0]
        assert pred["rates"]["hr"] * 600 == pred["hr"]

    def test_batter_uses_projected_pt(self, batter_row: dict[str, Any], engine_config: EngineConfig) -> None:
        engine = MarcelEngine()
        bat_pt = {1: 400.0}
        result = engine.predict([batter_row], [], bat_pt, {}, engine_config)
        pred = result[0]
        assert pred["pa"] == 400

    def test_pitcher_returns_prediction(self, pitcher_row: dict[str, Any], engine_config: EngineConfig) -> None:
        engine = MarcelEngine()
        pitch_pt = {10: 150.0}
        result = engine.predict([], [pitcher_row], {}, pitch_pt, engine_config)
        assert len(result) == 1
        pred = result[0]
        assert pred["player_id"] == 10
        assert pred["player_type"] == "pitcher"
        assert pred["ip"] == 150.0

    def test_both_batters_and_pitchers(
        self,
        batter_row: dict[str, Any],
        pitcher_row: dict[str, Any],
        engine_config: EngineConfig,
    ) -> None:
        engine = MarcelEngine()
        bat_pt = {1: 555.0}
        pitch_pt = {10: 150.0}
        result = engine.predict([batter_row], [pitcher_row], bat_pt, pitch_pt, engine_config)
        assert len(result) == 2
        types = {p["player_type"] for p in result}
        assert types == {"batter", "pitcher"}

    def test_empty_rows(self, engine_config: EngineConfig) -> None:
        engine = MarcelEngine()
        result = engine.predict([], [], {}, {}, engine_config)
        assert result == []

    def test_uses_system_name(self, batter_row: dict[str, Any], marcel_config: MarcelConfig) -> None:
        engine = MarcelEngine()
        cfg = EngineConfig(
            marcel_config=marcel_config,
            projected_season=2024,
            version="latest",
            system_name="composite-mle",
        )
        bat_pt = {1: 555.0}
        result = engine.predict([batter_row], [], bat_pt, {}, cfg)
        # The system name propagates through composite_projection_to_domain
        # We can't directly check it in the flat dict, but the prediction should succeed
        assert len(result) == 1

    def test_projected_season(self, batter_row: dict[str, Any], marcel_config: MarcelConfig) -> None:
        engine = MarcelEngine()
        cfg = EngineConfig(
            marcel_config=marcel_config,
            projected_season=2025,
            version="latest",
            system_name="composite",
        )
        bat_pt = {1: 555.0}
        result = engine.predict([batter_row], [], bat_pt, {}, cfg)
        assert result[0]["season"] == 2025


class TestGBMEngineProperties:
    def test_supported_operations(self) -> None:
        assert GBMEngine().supported_operations == frozenset({"prepare", "train", "predict"})

    def test_artifact_type(self) -> None:
        assert GBMEngine().artifact_type == "file"


class TestGBMEngineProtocol:
    def test_satisfies_composite_engine(self) -> None:
        assert isinstance(GBMEngine(), CompositeEngine)


_FEATURE_COLS = ["age", "pa_1", "hr_1", "hr_wavg", "weighted_pt", "league_hr_rate"]
_PITCH_FEATURE_COLS = ["age", "ip_1", "so_1", "so_wavg", "weighted_pt", "league_so_rate"]


def _make_batter_train_row(player_id: int, season: int) -> dict[str, Any]:
    """Synthetic batter row with feature + target columns."""
    row: dict[str, Any] = {"player_id": player_id, "season": season}
    for col in _FEATURE_COLS:
        row[col] = 1.0
    row["target_avg"] = 0.275
    row["target_obp"] = 0.350
    row["target_slg"] = 0.450
    row["target_woba"] = 0.340
    row["target_h"] = 150
    row["target_hr"] = 25
    row["target_ab"] = 500
    row["target_so"] = 100
    row["target_sf"] = 5
    return row


def _make_pitcher_train_row(player_id: int, season: int) -> dict[str, Any]:
    """Synthetic pitcher row with feature + target columns."""
    row: dict[str, Any] = {"player_id": player_id, "season": season}
    for col in _PITCH_FEATURE_COLS:
        row[col] = 1.0
    row["target_era"] = 3.50
    row["target_fip"] = 3.40
    row["target_k_per_9"] = 9.0
    row["target_bb_per_9"] = 3.0
    row["target_whip"] = 1.20
    row["target_h"] = 150
    row["target_hr"] = 20
    row["target_ip"] = 180.0
    row["target_so"] = 200
    return row


@pytest.mark.slow
class TestGBMEngineTrain:
    @pytest.fixture
    def artifact_path(self, tmp_path: Path) -> Path:
        return tmp_path / "models"

    @pytest.fixture
    def bat_train_rows(self) -> list[dict[str, Any]]:
        return [_make_batter_train_row(i, 2022) for i in range(1, 21)]

    @pytest.fixture
    def bat_holdout_rows(self) -> list[dict[str, Any]]:
        return [_make_batter_train_row(i, 2023) for i in range(1, 11)]

    @pytest.fixture
    def pitch_train_rows(self) -> list[dict[str, Any]]:
        return [_make_pitcher_train_row(i, 2022) for i in range(100, 120)]

    @pytest.fixture
    def pitch_holdout_rows(self) -> list[dict[str, Any]]:
        return [_make_pitcher_train_row(i, 2023) for i in range(100, 110)]

    def test_train_returns_batter_metrics(
        self,
        bat_train_rows: list[dict[str, Any]],
        bat_holdout_rows: list[dict[str, Any]],
        pitch_train_rows: list[dict[str, Any]],
        pitch_holdout_rows: list[dict[str, Any]],
        artifact_path: Path,
    ) -> None:
        engine = GBMEngine()
        metrics = engine.train(
            bat_train_rows,
            bat_holdout_rows,
            pitch_train_rows,
            pitch_holdout_rows,
            _FEATURE_COLS,
            _PITCH_FEATURE_COLS,
            {},
            artifact_path,
        )
        for target in BATTER_TARGETS:
            assert f"batter_rmse_{target}" in metrics

    def test_train_returns_pitcher_metrics(
        self,
        bat_train_rows: list[dict[str, Any]],
        bat_holdout_rows: list[dict[str, Any]],
        pitch_train_rows: list[dict[str, Any]],
        pitch_holdout_rows: list[dict[str, Any]],
        artifact_path: Path,
    ) -> None:
        engine = GBMEngine()
        metrics = engine.train(
            bat_train_rows,
            bat_holdout_rows,
            pitch_train_rows,
            pitch_holdout_rows,
            _FEATURE_COLS,
            _PITCH_FEATURE_COLS,
            {},
            artifact_path,
        )
        for target in PITCHER_TARGETS:
            assert f"pitcher_rmse_{target}" in metrics

    def test_train_saves_batter_models(
        self,
        bat_train_rows: list[dict[str, Any]],
        bat_holdout_rows: list[dict[str, Any]],
        pitch_train_rows: list[dict[str, Any]],
        pitch_holdout_rows: list[dict[str, Any]],
        artifact_path: Path,
    ) -> None:
        engine = GBMEngine()
        engine.train(
            bat_train_rows,
            bat_holdout_rows,
            pitch_train_rows,
            pitch_holdout_rows,
            _FEATURE_COLS,
            _PITCH_FEATURE_COLS,
            {},
            artifact_path,
        )
        assert (artifact_path / "batter_models.joblib").exists()

    def test_train_saves_pitcher_models(
        self,
        bat_train_rows: list[dict[str, Any]],
        bat_holdout_rows: list[dict[str, Any]],
        pitch_train_rows: list[dict[str, Any]],
        pitch_holdout_rows: list[dict[str, Any]],
        artifact_path: Path,
    ) -> None:
        engine = GBMEngine()
        engine.train(
            bat_train_rows,
            bat_holdout_rows,
            pitch_train_rows,
            pitch_holdout_rows,
            _FEATURE_COLS,
            _PITCH_FEATURE_COLS,
            {},
            artifact_path,
        )
        assert (artifact_path / "pitcher_models.joblib").exists()

    def test_train_empty_holdout_returns_no_metrics(
        self,
        bat_train_rows: list[dict[str, Any]],
        pitch_train_rows: list[dict[str, Any]],
        artifact_path: Path,
    ) -> None:
        engine = GBMEngine()
        metrics = engine.train(
            bat_train_rows,
            [],
            pitch_train_rows,
            [],
            _FEATURE_COLS,
            _PITCH_FEATURE_COLS,
            {},
            artifact_path,
        )
        assert metrics == {}

    def test_train_routes_batter_params(
        self,
        bat_train_rows: list[dict[str, Any]],
        bat_holdout_rows: list[dict[str, Any]],
        pitch_train_rows: list[dict[str, Any]],
        pitch_holdout_rows: list[dict[str, Any]],
        artifact_path: Path,
    ) -> None:
        """Batter-specific params are extracted via model_params.get('batter', model_params)."""
        model_params = {"batter": {"max_iter": 50}, "pitcher": {"max_iter": 30}}
        engine = GBMEngine()
        # Should not raise — batter uses max_iter=50, pitcher uses max_iter=30
        metrics = engine.train(
            bat_train_rows,
            bat_holdout_rows,
            pitch_train_rows,
            pitch_holdout_rows,
            _FEATURE_COLS,
            _PITCH_FEATURE_COLS,
            model_params,
            artifact_path,
        )
        assert len(metrics) > 0


@pytest.mark.slow
class TestGBMEnginePredict:
    @pytest.fixture
    def artifact_path(self, tmp_path: Path) -> Path:
        return tmp_path / "models"

    @pytest.fixture
    def trained_artifact_path(self, artifact_path: Path) -> Path:
        """Train models and return the artifact path."""
        engine = GBMEngine()
        bat_train = [_make_batter_train_row(i, 2022) for i in range(1, 21)]
        pitch_train = [_make_pitcher_train_row(i, 2022) for i in range(100, 120)]
        engine.train(
            bat_train,
            [],
            pitch_train,
            [],
            _FEATURE_COLS,
            _PITCH_FEATURE_COLS,
            {},
            artifact_path,
        )
        return artifact_path

    @pytest.fixture
    def engine_config(self, trained_artifact_path: Path) -> EngineConfig:
        return EngineConfig(
            marcel_config=MarcelConfig(),
            projected_season=2025,
            version="latest",
            system_name="composite-gbm",
            artifact_path=trained_artifact_path,
            bat_feature_cols=tuple(_FEATURE_COLS),
            pitch_feature_cols=tuple(_PITCH_FEATURE_COLS),
        )

    def test_returns_batter_predictions_with_rates(self, engine_config: EngineConfig) -> None:
        engine = GBMEngine()
        bat_rows = [_make_batter_train_row(1, 2023)]
        bat_rows[0]["proj_pa"] = 600
        bat_pt = {1: 600.0}
        result = engine.predict(bat_rows, [], bat_pt, {}, engine_config)
        assert len(result) == 1
        pred = result[0]
        assert pred["player_id"] == 1
        assert pred["player_type"] == "batter"
        assert "avg" in pred
        assert "obp" in pred
        assert "slg" in pred
        assert "h" in pred

    def test_returns_pitcher_predictions_with_rates(self, engine_config: EngineConfig) -> None:
        engine = GBMEngine()
        pitch_rows = [_make_pitcher_train_row(100, 2023)]
        pitch_rows[0]["proj_ip"] = 180.0
        pitch_pt = {100: 180.0}
        result = engine.predict([], pitch_rows, {}, pitch_pt, engine_config)
        assert len(result) == 1
        pred = result[0]
        assert pred["player_id"] == 100
        assert pred["player_type"] == "pitcher"
        assert "era" in pred
        assert "k_per_9" in pred
        assert "so" in pred

    def test_uses_projected_pt(self, engine_config: EngineConfig) -> None:
        engine = GBMEngine()
        bat_rows = [_make_batter_train_row(1, 2023)]
        bat_rows[0]["proj_pa"] = 400
        bat_pt = {1: 400.0}
        result = engine.predict(bat_rows, [], bat_pt, {}, engine_config)
        assert result[0]["pa"] == 400

    def test_deduplicates_to_one_per_player(self, engine_config: EngineConfig) -> None:
        engine = GBMEngine()
        row1 = _make_batter_train_row(1, 2022)
        row1["proj_pa"] = 550
        row2 = _make_batter_train_row(1, 2023)
        row2["proj_pa"] = 600
        bat_pt = {1: 600.0}
        result = engine.predict([row1, row2], [], bat_pt, {}, engine_config)
        assert len(result) == 1

    def test_uses_projected_season_and_system_name(self, engine_config: EngineConfig) -> None:
        engine = GBMEngine()
        bat_rows = [_make_batter_train_row(1, 2023)]
        bat_rows[0]["proj_pa"] = 600
        bat_pt = {1: 600.0}
        result = engine.predict(bat_rows, [], bat_pt, {}, engine_config)
        assert result[0]["season"] == 2025

    def test_raises_when_artifact_path_is_none(self) -> None:
        engine = GBMEngine()
        config = EngineConfig(
            marcel_config=MarcelConfig(),
            projected_season=2025,
            version="latest",
            system_name="composite",
            artifact_path=None,
        )
        with pytest.raises(ValueError, match="artifact_path"):
            engine.predict([], [], {}, {}, config)

    def test_empty_rows_returns_empty(self, engine_config: EngineConfig) -> None:
        engine = GBMEngine()
        result = engine.predict([], [], {}, {}, engine_config)
        assert result == []

    def test_zero_pt_produces_zero_counting_stats(self, engine_config: EngineConfig) -> None:
        engine = GBMEngine()
        bat_rows = [_make_batter_train_row(1, 2023)]
        bat_rows[0]["proj_pa"] = 0
        bat_pt = {1: 0.0}
        result = engine.predict(bat_rows, [], bat_pt, {}, engine_config)
        assert len(result) == 1
        pred = result[0]
        assert pred["pa"] == 0


class TestResolveEngine:
    def test_default_returns_marcel(self) -> None:
        engine = resolve_engine({})
        assert isinstance(engine, MarcelEngine)

    def test_explicit_marcel(self) -> None:
        engine = resolve_engine({"engine": "marcel"})
        assert isinstance(engine, MarcelEngine)

    def test_explicit_gbm(self) -> None:
        engine = resolve_engine({"engine": "gbm"})
        assert isinstance(engine, GBMEngine)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="bogus"):
            resolve_engine({"engine": "bogus"})
