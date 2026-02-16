from typing import Any

import pytest

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

    def test_predict_includes_components_metadata(self) -> None:
        """Prediction dicts include _components dict for lineage display."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"hr": 30.0}),
                _make_projection(1, "steamer", "batter", {"hr": 20.0}),
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
        pred = result.predictions[0]
        assert pred["_components"] == {"marcel": 0.6, "steamer": 0.4}

    def test_predict_includes_mode_metadata(self) -> None:
        """Prediction dicts include _mode string for lineage display."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"hr": 30.0}),
                _make_projection(1, "steamer", "batter", {"hr": 20.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "blend_rates",
                "season": 2025,
                "stats": ["hr"],
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["_mode"] == "blend_rates"

    def test_predict_includes_distributions(self) -> None:
        """With 2+ systems, result.distributions is populated."""
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
        assert result.distributions is not None
        assert len(result.distributions) == 2  # one per stat
        # Check that distributions have correct player info
        hr_dist = next(d for d in result.distributions if d["stat"] == "hr")
        assert hr_dist["player_id"] == 1
        assert hr_dist["player_type"] == "batter"
        assert hr_dist["season"] == 2025
        assert "mean" in hr_dist
        assert "std" in hr_dist
        assert "p10" in hr_dist
        assert "p90" in hr_dist

    def test_predict_single_system_no_distributions(self) -> None:
        """Only 1 system available for a player → no distributions."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"hr": 30.0}),
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
        # Single system → no spread → distributions is None
        assert result.distributions is None

    def test_predict_distributions_none_when_all_single(self) -> None:
        """All players have only 1 system → distributions is None."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"hr": 30.0}),
                _make_projection(2, "steamer", "batter", {"hr": 25.0}),
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
        # Each player only in 1 system → no spread for anyone
        assert result.distributions is None

    def test_predict_distributions_format(self) -> None:
        """Each distribution dict has the required keys for CLI persistence."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"hr": 30.0}),
                _make_projection(1, "steamer", "batter", {"hr": 20.0}),
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
        assert result.distributions is not None
        required_keys = {"player_id", "player_type", "season", "stat", "p10", "p25", "p50", "p75", "p90", "mean", "std"}
        for dist in result.distributions:
            assert required_keys <= set(dist.keys())


class TestEnsembleStatcastGBMIntegration:
    """Integration tests: statcast-gbm blends correctly with Marcel."""

    def test_blend_batter_rates_marcel_and_statcast_gbm(self) -> None:
        """Overlapping batter stats are weighted-averaged; unique stats pass through."""
        repo = FakeProjectionRepo(
            [
                _make_projection(
                    1,
                    "marcel",
                    "batter",
                    {"avg": 0.280, "obp": 0.350, "slg": 0.450, "hr": 25.0, "pa": 600.0},
                ),
                _make_projection(
                    1,
                    "statcast-gbm",
                    "batter",
                    {"avg": 0.270, "obp": 0.340, "slg": 0.430, "woba": 0.330, "iso": 0.180, "babip": 0.300},
                ),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
                "season": 2025,
            },
        )
        result = model.predict(config)
        assert len(result.predictions) == 1
        pred = result.predictions[0]

        # Overlapping stats: weighted average from both systems
        assert pred["avg"] == pytest.approx((0.280 * 0.6 + 0.270 * 0.4) / 1.0)
        assert pred["obp"] == pytest.approx((0.350 * 0.6 + 0.340 * 0.4) / 1.0)
        assert pred["slg"] == pytest.approx((0.450 * 0.6 + 0.430 * 0.4) / 1.0)

        # statcast-gbm only: pass through at own value
        assert pred["woba"] == pytest.approx(0.330)
        assert pred["iso"] == pytest.approx(0.180)
        assert pred["babip"] == pytest.approx(0.300)

        # Marcel only: pass through at own value
        assert pred["hr"] == pytest.approx(25.0)
        assert pred["pa"] == pytest.approx(600.0)

    def test_blend_pitcher_rates_marcel_and_statcast_gbm(self) -> None:
        """Overlapping pitcher stats are weighted-averaged; unique stats pass through."""
        repo = FakeProjectionRepo(
            [
                _make_projection(
                    1,
                    "marcel",
                    "pitcher",
                    {"era": 3.50, "whip": 1.20, "k_per_9": 9.0, "bb_per_9": 3.0, "ip": 180.0, "so": 180.0},
                ),
                _make_projection(
                    1,
                    "statcast-gbm",
                    "pitcher",
                    {
                        "era": 3.80,
                        "whip": 1.25,
                        "k_per_9": 8.5,
                        "bb_per_9": 3.2,
                        "fip": 3.60,
                        "hr_per_9": 1.1,
                        "babip": 0.295,
                    },
                ),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
                "season": 2025,
            },
        )
        result = model.predict(config)
        assert len(result.predictions) == 1
        pred = result.predictions[0]

        # Overlapping stats: weighted average
        assert pred["era"] == pytest.approx(3.50 * 0.6 + 3.80 * 0.4)
        assert pred["whip"] == pytest.approx(1.20 * 0.6 + 1.25 * 0.4)
        assert pred["k_per_9"] == pytest.approx(9.0 * 0.6 + 8.5 * 0.4)
        assert pred["bb_per_9"] == pytest.approx(3.0 * 0.6 + 3.2 * 0.4)

        # statcast-gbm only
        assert pred["fip"] == pytest.approx(3.60)
        assert pred["hr_per_9"] == pytest.approx(1.1)
        assert pred["babip"] == pytest.approx(0.295)

        # Marcel only
        assert pred["ip"] == pytest.approx(180.0)
        assert pred["so"] == pytest.approx(180.0)

    def test_unequal_weights_favor_heavier_system(self) -> None:
        """With marcel=0.6, statcast-gbm=0.4, blended avg is closer to Marcel."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.300}),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.250}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
                "season": 2025,
                "stats": ["avg"],
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        blended = pred["avg"]
        # Blended should be closer to Marcel (0.300) than statcast-gbm (0.250)
        assert abs(blended - 0.300) < abs(blended - 0.250)
        assert blended == pytest.approx(0.280)

    def test_player_in_statcast_gbm_only(self) -> None:
        """Player existing only in statcast-gbm still gets predictions."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.280}),
                _make_projection(2, "statcast-gbm", "batter", {"avg": 0.260, "woba": 0.310}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
                "season": 2025,
            },
        )
        result = model.predict(config)
        preds = {p["player_id"]: p for p in result.predictions}
        assert 2 in preds
        assert preds[2]["avg"] == pytest.approx(0.260)
        assert preds[2]["woba"] == pytest.approx(0.310)

    def test_distributions_for_overlapping_stats_only(self) -> None:
        """Distributions produced for stats in both systems, not for single-system stats."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.280, "obp": 0.350}),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.270, "obp": 0.340, "woba": 0.330}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
                "season": 2025,
            },
        )
        result = model.predict(config)
        assert result.distributions is not None
        dist_stats = {d["stat"] for d in result.distributions}
        # Overlapping stats get distributions
        assert "avg" in dist_stats
        assert "obp" in dist_stats
        # Single-system stat does not
        assert "woba" not in dist_stats

    def test_blend_rates_mode_with_statcast_gbm(self) -> None:
        """blend_rates mode works with statcast-gbm alongside Marcel."""
        repo = FakeProjectionRepo(
            [
                _make_projection(
                    1,
                    "marcel",
                    "batter",
                    {"avg": 0.280, "obp": 0.350, "pa": 600.0},
                ),
                _make_projection(
                    1,
                    "statcast-gbm",
                    "batter",
                    {"avg": 0.270, "obp": 0.340},
                ),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "blend_rates",
                "season": 2025,
                "stats": ["avg", "obp"],
                "pt_stat": "pa",
            },
        )
        result = model.predict(config)
        assert len(result.predictions) == 1
        pred = result.predictions[0]
        # Rates are weighted-averaged
        assert pred["avg"] == pytest.approx(0.280 * 0.6 + 0.270 * 0.4)
        assert pred["obp"] == pytest.approx(0.350 * 0.6 + 0.340 * 0.4)
        # pa comes from Marcel only (statcast-gbm lacks it)
        assert pred["pa"] == pytest.approx(600.0)

    def test_versions_param_filters_by_version(self) -> None:
        """When versions dict is provided, only that version is used per system."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.280}, season=2025),
                Projection(
                    player_id=1,
                    season=2025,
                    system="marcel",
                    version="old",
                    player_type="batter",
                    stat_json={"avg": 0.200},
                ),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.270}, season=2025),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
                "season": 2025,
                "stats": ["avg"],
                "versions": {"marcel": "v1", "statcast-gbm": "v1"},
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # Only marcel v1 (avg=0.280) used, not "old" (avg=0.200)
        assert pred["avg"] == pytest.approx(0.280 * 0.6 + 0.270 * 0.4)
