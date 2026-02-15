from typing import Any

from fantasy_baseball_manager.domain.projection import Projection, StatDistribution
from fantasy_baseball_manager.models.ensemble import EnsembleModel
from fantasy_baseball_manager.models.protocols import (
    Evaluable,
    FineTunable,
    Model,
    ModelConfig,
    Predictable,
    Preparable,
    Trainable,
)
from fantasy_baseball_manager.models.registry import _clear, get, register


class FakeProjectionRepo:
    """In-memory projection repo for testing EnsembleModel."""

    def __init__(self, projections: list[Projection]) -> None:
        self._projections = projections

    def upsert(self, projection: Projection) -> int:
        return 0

    def get_by_player_season(
        self, player_id: int, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Projection]:
        return [
            p
            for p in self._projections
            if p.player_id == player_id and p.season == season and (system is None or p.system == system)
        ]

    def get_by_season(
        self, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Projection]:
        return [p for p in self._projections if p.season == season and (system is None or p.system == system)]

    def get_by_system_version(self, system: str, version: str) -> list[Projection]:
        return [p for p in self._projections if p.system == system and p.version == version]

    def upsert_distributions(self, projection_id: int, distributions: list[StatDistribution]) -> None:
        pass

    def get_distributions(self, projection_id: int) -> list[StatDistribution]:
        return []


class TestEnsembleModelProtocol:
    def test_is_model(self) -> None:
        assert isinstance(EnsembleModel(), Model)

    def test_is_predictable(self) -> None:
        assert isinstance(EnsembleModel(), Predictable)

    def test_is_not_preparable(self) -> None:
        assert not isinstance(EnsembleModel(), Preparable)

    def test_is_not_trainable(self) -> None:
        assert not isinstance(EnsembleModel(), Trainable)

    def test_is_not_evaluable(self) -> None:
        assert not isinstance(EnsembleModel(), Evaluable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(EnsembleModel(), FineTunable)

    def test_name(self) -> None:
        assert EnsembleModel().name == "ensemble"

    def test_supported_operations(self) -> None:
        assert EnsembleModel().supported_operations == frozenset({"predict"})

    def test_artifact_type(self) -> None:
        assert EnsembleModel().artifact_type == "none"


class TestEnsembleRegistration:
    def test_registered(self) -> None:
        _clear()
        register("ensemble")(EnsembleModel)
        assert get("ensemble") is EnsembleModel


def _make_projection(
    player_id: int,
    system: str,
    player_type: str,
    stats: dict[str, Any],
    season: int = 2025,
) -> Projection:
    return Projection(
        player_id=player_id,
        season=season,
        system=system,
        version="v1",
        player_type=player_type,
        stat_json=stats,
    )


class TestEnsemblePredict:
    def test_predict_weighted_average(self) -> None:
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"hr": 30.0, "rbi": 100.0}),
                _make_projection(1, "steamer", "batter", {"hr": 20.0, "rbi": 80.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "weighted_average",
                "season": 2025,
                "stats": ["hr", "rbi"],
            },
        )
        result = model.predict(config)
        assert result.model_name == "ensemble"
        assert len(result.predictions) == 1
        pred = result.predictions[0]
        assert pred["player_id"] == 1
        assert pred["season"] == 2025
        assert pred["player_type"] == "batter"
        expected_hr = (30.0 * 0.6 + 20.0 * 0.4) / (0.6 + 0.4)
        expected_rbi = (100.0 * 0.6 + 80.0 * 0.4) / (0.6 + 0.4)
        assert pred["hr"] == expected_hr
        assert pred["rbi"] == expected_rbi

    def test_predict_blend_rates(self) -> None:
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.300, "obp": 0.400, "pa": 600.0}),
                _make_projection(1, "steamer", "batter", {"avg": 0.250, "obp": 0.350, "pa": 500.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "blend_rates",
                "season": 2025,
                "stats": ["avg", "obp"],
                "pt_stat": "pa",
            },
        )
        result = model.predict(config)
        assert len(result.predictions) == 1
        pred = result.predictions[0]
        expected_avg = (0.300 * 0.6 + 0.250 * 0.4) / (0.6 + 0.4)
        assert pred["avg"] == expected_avg

    def test_predict_missing_system_player(self) -> None:
        """Player in system A but not in system B uses only A's values."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"hr": 30.0}),
                # player 1 not in steamer
                _make_projection(2, "marcel", "batter", {"hr": 25.0}),
                _make_projection(2, "steamer", "batter", {"hr": 20.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "weighted_average",
                "season": 2025,
                "stats": ["hr"],
            },
        )
        result = model.predict(config)
        preds = {p["player_id"]: p for p in result.predictions}
        # Player 1: only marcel, so hr = 30.0
        assert preds[1]["hr"] == 30.0
        # Player 2: weighted average
        assert preds[2]["hr"] == (25.0 * 0.6 + 20.0 * 0.4) / (0.6 + 0.4)

    def test_predict_empty_components(self) -> None:
        repo = FakeProjectionRepo([])
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "weighted_average",
                "season": 2025,
                "stats": ["hr"],
            },
        )
        result = model.predict(config)
        assert result.predictions == []

    def test_predict_output_format(self) -> None:
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"hr": 30.0}),
                _make_projection(2, "marcel", "pitcher", {"era": 3.50}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 1.0},
                "mode": "weighted_average",
                "season": 2025,
                "stats": ["hr", "era"],
            },
            output_dir="/tmp/out",
        )
        result = model.predict(config)
        assert result.model_name == "ensemble"
        assert result.output_path == "/tmp/out"
        assert len(result.predictions) == 2
        batter = next(p for p in result.predictions if p["player_type"] == "batter")
        pitcher = next(p for p in result.predictions if p["player_type"] == "pitcher")
        assert batter["player_id"] == 1
        assert batter["player_type"] == "batter"
        assert pitcher["player_id"] == 2
        assert pitcher["player_type"] == "pitcher"

    def test_predict_multiple_player_types(self) -> None:
        """Same player_id but different player_type should be separate predictions."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"hr": 30.0}),
                _make_projection(1, "marcel", "pitcher", {"era": 3.50}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 1.0},
                "mode": "weighted_average",
                "season": 2025,
                "stats": ["hr", "era"],
            },
        )
        result = model.predict(config)
        assert len(result.predictions) == 2

    def test_predict_uses_all_stats_when_none_specified(self) -> None:
        """When stats not specified in model_params, use all stats found in projections."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"hr": 30.0, "rbi": 100.0}),
                _make_projection(1, "steamer", "batter", {"hr": 20.0, "rbi": 80.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.5, "steamer": 0.5},
                "mode": "weighted_average",
                "season": 2025,
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["hr"] == 25.0
        assert pred["rbi"] == 90.0
