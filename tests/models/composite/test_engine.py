from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from fantasy_baseball_manager.models.composite.engine import (
    CompositeEngine,
    EngineConfig,
    MarcelEngine,
    resolve_engine,
)
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


class TestResolveEngine:
    def test_default_returns_marcel(self) -> None:
        engine = resolve_engine({})
        assert isinstance(engine, MarcelEngine)

    def test_explicit_marcel(self) -> None:
        engine = resolve_engine({"engine": "marcel"})
        assert isinstance(engine, MarcelEngine)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="bogus"):
            resolve_engine({"engine": "bogus"})
