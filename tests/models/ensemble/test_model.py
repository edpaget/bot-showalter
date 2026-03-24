import logging
from typing import Any

import pytest

from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.ensemble import EnsembleModel
from fantasy_baseball_manager.models.ensemble.stat_groups import BUILTIN_GROUPS
from fantasy_baseball_manager.models.protocols import (
    Evaluable,
    FineTunable,
    Model,
    ModelConfig,
    Predictable,
    Preparable,
    Trainable,
)
from fantasy_baseball_manager.models.registry import get, register
from tests.fakes.repos import FakePitchingStatsRepo, FakeProjectionRepo

_NULL_PROJECTION_REPO = FakeProjectionRepo([])


class TestEnsembleModelProtocol:
    def test_is_model(self) -> None:
        assert isinstance(EnsembleModel(projection_repo=_NULL_PROJECTION_REPO), Model)

    def test_is_predictable(self) -> None:
        assert isinstance(EnsembleModel(projection_repo=_NULL_PROJECTION_REPO), Predictable)

    def test_is_not_preparable(self) -> None:
        assert not isinstance(EnsembleModel(projection_repo=_NULL_PROJECTION_REPO), Preparable)

    def test_is_not_trainable(self) -> None:
        assert not isinstance(EnsembleModel(projection_repo=_NULL_PROJECTION_REPO), Trainable)

    def test_is_not_evaluable(self) -> None:
        assert not isinstance(EnsembleModel(projection_repo=_NULL_PROJECTION_REPO), Evaluable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(EnsembleModel(projection_repo=_NULL_PROJECTION_REPO), FineTunable)

    def test_name(self) -> None:
        assert EnsembleModel(projection_repo=_NULL_PROJECTION_REPO).name == "ensemble"

    def test_supported_operations(self) -> None:
        assert EnsembleModel(projection_repo=_NULL_PROJECTION_REPO).supported_operations == frozenset({"predict"})

    def test_artifact_type(self) -> None:
        assert EnsembleModel(projection_repo=_NULL_PROJECTION_REPO).artifact_type == "none"


class TestEnsembleRegistration:
    def test_registered(self, isolated_model_registry: None) -> None:
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
        player_type=PlayerType(player_type),
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "blend_rates",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 1.0},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 1.0},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.5, "steamer": 0.5},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "blend_rates",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "steamer": 0.4},
                "mode": "weighted_average",
                "stats": ["hr"],
            },
        )
        result = model.predict(config)
        assert result.distributions is not None
        required_keys = {"player_id", "player_type", "season", "stat", "p10", "p25", "p50", "p75", "p90", "mean", "std"}
        for dist in result.distributions:
            assert required_keys <= set(dist.keys())


class TestEnsembleConsensusPT:
    def test_consensus_pt_substitutes_pa_for_batters(self) -> None:
        """Consensus PT replaces weight-averaged PA for batters."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.280, "pa": 600.0}),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.260, "pa": 500.0}),
                # steamer/zips for consensus lookup
                _make_projection(1, "steamer", "batter", {"pa": 550.0}),
                _make_projection(1, "zips", "batter", {"pa": 500.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "blend_rates",
                "stats": ["avg"],
                "pt_stat": "pa",
                "playing_time": "consensus",
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # Consensus PA = avg(550, 500) = 525
        assert pred["pa"] == 525.0

    def test_consensus_pt_substitutes_ip_for_pitchers(self) -> None:
        """Consensus PT replaces weight-averaged IP for pitchers."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "pitcher", {"era": 3.50, "ip": 180.0}),
                _make_projection(1, "statcast-gbm", "pitcher", {"era": 3.80, "ip": 160.0}),
                _make_projection(1, "steamer", "pitcher", {"ip": 190.0}),
                _make_projection(1, "zips", "pitcher", {"ip": 170.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "blend_rates",
                "stats": ["era"],
                "pt_stat": "ip",
                "playing_time": "consensus",
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # Consensus IP = avg(190, 170) = 180
        assert pred["ip"] == 180.0

    def test_consensus_pt_rates_still_blended(self) -> None:
        """Rate stats are still weight-averaged even when consensus PT is used."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.300, "obp": 0.400, "pa": 600.0}),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.250, "obp": 0.350, "pa": 500.0}),
                _make_projection(1, "steamer", "batter", {"pa": 550.0}),
                _make_projection(1, "zips", "batter", {"pa": 520.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "blend_rates",
                "stats": ["avg", "obp"],
                "pt_stat": "pa",
                "playing_time": "consensus",
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        expected_avg = (0.300 * 0.6 + 0.250 * 0.4) / (0.6 + 0.4)
        expected_obp = (0.400 * 0.6 + 0.350 * 0.4) / (0.6 + 0.4)
        assert pred["avg"] == pytest.approx(expected_avg)
        assert pred["obp"] == pytest.approx(expected_obp)
        # PA is consensus, not weight-averaged
        assert pred["pa"] == 535.0  # avg(550, 520)

    def test_consensus_pt_fallback_for_uncovered_player(self) -> None:
        """Player not in steamer/zips falls back to weight-averaged PT."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.280, "pa": 600.0}),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.260, "pa": 500.0}),
                # No steamer/zips for player 1
                _make_projection(2, "steamer", "batter", {"pa": 400.0}),
                _make_projection(2, "zips", "batter", {"pa": 380.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "blend_rates",
                "stats": ["avg"],
                "pt_stat": "pa",
                "playing_time": "consensus",
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # Player 1 not in consensus lookup → falls back to weight-averaged PA
        expected_pa = (600.0 * 0.6 + 500.0 * 0.4) / (0.6 + 0.4)
        assert pred["pa"] == expected_pa

    def test_consensus_pt_mixed_batters_and_pitchers(self) -> None:
        """Each player type gets correct PT stat from consensus."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.280, "pa": 600.0}),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.270, "pa": 550.0}),
                _make_projection(2, "marcel", "pitcher", {"era": 3.50, "ip": 180.0}),
                _make_projection(2, "statcast-gbm", "pitcher", {"era": 3.80, "ip": 160.0}),
                # Consensus sources
                _make_projection(1, "steamer", "batter", {"pa": 580.0}),
                _make_projection(1, "zips", "batter", {"pa": 560.0}),
                _make_projection(2, "steamer", "pitcher", {"ip": 175.0}),
                _make_projection(2, "zips", "pitcher", {"ip": 165.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "blend_rates",
                "stats": ["avg", "era"],
                "pt_stat": "pa",
                "playing_time": "consensus",
            },
        )
        result = model.predict(config)
        preds = {(p["player_id"], p["player_type"]): p for p in result.predictions}
        # Batter gets consensus PA
        assert preds[(1, "batter")]["pa"] == 570.0  # avg(580, 560)
        # Pitcher gets consensus IP
        assert preds[(2, "pitcher")]["ip"] == 170.0  # avg(175, 165)

    def test_consensus_pt_weighted_average_mode(self) -> None:
        """Consensus PT works in weighted_average mode too."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.280, "pa": 600.0}),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.260, "pa": 500.0}),
                _make_projection(1, "steamer", "batter", {"pa": 550.0}),
                _make_projection(1, "zips", "batter", {"pa": 520.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
                "stats": ["avg", "pa"],
                "playing_time": "consensus",
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # PA substituted with consensus value
        assert pred["pa"] == 535.0  # avg(550, 520)
        # AVG still weight-averaged
        expected_avg = (0.280 * 0.6 + 0.260 * 0.4) / (0.6 + 0.4)
        assert pred["avg"] == pytest.approx(expected_avg)

    def test_consensus_inline_three_systems(self) -> None:
        """Inline consensus syntax with 3 systems averages all three."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.280, "pa": 600.0}),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.260, "pa": 500.0}),
                # Three consensus sources
                _make_projection(1, "steamer", "batter", {"pa": 600.0}),
                _make_projection(1, "zips", "batter", {"pa": 500.0}),
                _make_projection(1, "atc", "batter", {"pa": 400.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "blend_rates",
                "stats": ["avg"],
                "pt_stat": "pa",
                "playing_time": "consensus:steamer,zips,atc",
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # Consensus PA = avg(600, 500, 400) = 500
        assert pred["pa"] == pytest.approx(500.0)

    def test_single_system_pt(self) -> None:
        """playing_time='steamer' uses Steamer PA directly."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.280, "pa": 600.0}),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.260, "pa": 500.0}),
                _make_projection(1, "steamer", "batter", {"pa": 550.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "blend_rates",
                "stats": ["avg"],
                "pt_stat": "pa",
                "playing_time": "steamer",
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["pa"] == 550.0

    def test_unknown_system_falls_back_to_native(self) -> None:
        """Unknown system returns empty lookup, so PT falls back to weight-averaged."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.280, "pa": 600.0}),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.260, "pa": 500.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "blend_rates",
                "stats": ["avg"],
                "pt_stat": "pa",
                "playing_time": "no-such-system",
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # Falls back to weight-averaged PA since system has no projections
        expected_pa = (600.0 * 0.6 + 500.0 * 0.4) / (0.6 + 0.4)
        assert pred["pa"] == expected_pa

    def test_native_pt_mode_is_default(self) -> None:
        """When playing_time not specified, behavior unchanged (PT weight-averaged)."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "marcel", "batter", {"avg": 0.280, "pa": 600.0}),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.260, "pa": 500.0}),
                # steamer/zips present but shouldn't be used
                _make_projection(1, "steamer", "batter", {"pa": 550.0}),
                _make_projection(1, "zips", "batter", {"pa": 520.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "blend_rates",
                "stats": ["avg"],
                "pt_stat": "pa",
                # no "playing_time" key
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # PA is weight-averaged, not consensus
        expected_pa = (600.0 * 0.6 + 500.0 * 0.4) / (0.6 + 0.4)
        assert pred["pa"] == expected_pa


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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
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
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "blend_rates",
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
                    player_type=PlayerType.BATTER,
                    stat_json={"avg": 0.200},
                ),
                _make_projection(1, "statcast-gbm", "batter", {"avg": 0.270}, season=2025),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"marcel": 0.6, "statcast-gbm": 0.4},
                "mode": "weighted_average",
                "stats": ["avg"],
                "versions": {"marcel": "v1", "statcast-gbm": "v1"},
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # Only marcel v1 (avg=0.280) used, not "old" (avg=0.200)
        assert pred["avg"] == pytest.approx(0.280 * 0.6 + 0.270 * 0.4)

    def test_versions_param_filters_by_season(self) -> None:
        """Version-pinned fetch must only use projections from the target season."""
        repo = FakeProjectionRepo(
            [
                # Target season (2026) — correct projection
                Projection(
                    player_id=1,
                    season=2026,
                    system="sgbm",
                    version="latest",
                    player_type=PlayerType.PITCHER,
                    stat_json={"era": 3.50},
                ),
                # Old season (2019) — should be ignored, not overwrite 2026
                Projection(
                    player_id=1,
                    season=2019,
                    system="sgbm",
                    version="latest",
                    player_type=PlayerType.PITCHER,
                    stat_json={"era": 9.10},
                ),
                _make_projection(1, "steamer", "pitcher", {"era": 3.80}, season=2026),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2026],
            model_params={
                "components": {"sgbm": 0.5, "steamer": 0.5},
                "mode": "weighted_average",
                "stats": ["era"],
                "versions": {"sgbm": "latest"},
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # Should use 2026 sgbm (3.50), not 2019 (9.10)
        assert pred["era"] == pytest.approx(3.50 * 0.5 + 3.80 * 0.5)

    def test_multi_season_produces_per_season_predictions(self) -> None:
        """config.seasons generates separate predictions per season."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0}, season=2025),
                _make_projection(1, "steamer", "batter", {"hr": 28.0}, season=2026),
                _make_projection(2, "steamer", "batter", {"hr": 20.0}, season=2025),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025, 2026],
            model_params={
                "components": {"steamer": 1.0},
                "mode": "weighted_average",
                "stats": ["hr"],
            },
        )
        result = model.predict(config)
        by_key = {(p["player_id"], p["season"]): p for p in result.predictions}
        assert by_key[(1, 2025)]["hr"] == pytest.approx(30.0)
        assert by_key[(1, 2026)]["hr"] == pytest.approx(28.0)
        assert by_key[(2, 2025)]["hr"] == pytest.approx(20.0)
        assert (2, 2026) not in by_key

    def test_multi_season_with_version_pinning(self) -> None:
        """config.seasons + versions selects correct season from version-pinned system."""
        repo = FakeProjectionRepo(
            [
                Projection(
                    player_id=1,
                    season=2024,
                    system="sgbm",
                    version="latest",
                    player_type=PlayerType.PITCHER,
                    stat_json={"era": 4.00},
                ),
                Projection(
                    player_id=1,
                    season=2025,
                    system="sgbm",
                    version="latest",
                    player_type=PlayerType.PITCHER,
                    stat_json={"era": 3.50},
                ),
                Projection(
                    player_id=1,
                    season=2026,
                    system="sgbm",
                    version="latest",
                    player_type=PlayerType.PITCHER,
                    stat_json={"era": 9.10},  # should be excluded
                ),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2024, 2025],
            model_params={
                "components": {"sgbm": 1.0},
                "mode": "weighted_average",
                "stats": ["era"],
                "versions": {"sgbm": "latest"},
            },
        )
        result = model.predict(config)
        by_key = {p["season"]: p for p in result.predictions}
        assert by_key[2024]["era"] == pytest.approx(4.00)
        assert by_key[2025]["era"] == pytest.approx(3.50)
        # 2026 should NOT be in results
        assert 2026 not in by_key


class TestEnsembleRoutedMode:
    def test_routed_basic(self) -> None:
        """Route HR to steamer, OBP to statcast-gbm."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0, "obp": 0.340}),
                _make_projection(1, "statcast-gbm", "batter", {"hr": 25.0, "obp": 0.360}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "mode": "routed",
                "routes": {"hr": "steamer", "obp": "statcast-gbm"},
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["hr"] == 30.0  # from steamer
        assert pred["obp"] == 0.360  # from statcast-gbm

    def test_routed_with_fallback(self) -> None:
        """Primary system lacks stat, fallback provides it."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0, "obp": 0.340}),
                _make_projection(1, "statcast-gbm", "batter", {"obp": 0.360}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "mode": "routed",
                "routes": {"hr": "statcast-gbm", "obp": "statcast-gbm"},
                "fallback": "steamer",
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["obp"] == 0.360  # from primary
        assert pred["hr"] == 30.0  # from fallback

    def test_routed_missing_stat_no_fallback(self) -> None:
        """Stat omitted when primary lacks it and no fallback."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0}),
                _make_projection(1, "statcast-gbm", "batter", {"obp": 0.360}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "mode": "routed",
                "routes": {"hr": "statcast-gbm"},
            },
        )
        result = model.predict(config)
        # statcast-gbm lacks hr, no fallback → no predictions (empty result_stats)
        assert result.predictions == []

    def test_routed_ignores_global_weights(self) -> None:
        """Component weights don't affect routed output."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0}),
                _make_projection(1, "statcast-gbm", "batter", {"hr": 25.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.9, "statcast-gbm": 0.1},
                "mode": "routed",
                "routes": {"hr": "statcast-gbm"},
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # Weight is 0.9 for steamer, but routed ignores weights
        assert pred["hr"] == 25.0  # from statcast-gbm

    def test_routed_metadata(self) -> None:
        """Prediction includes _mode and _routes metadata."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0}),
                _make_projection(1, "statcast-gbm", "batter", {"obp": 0.360}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        routes = {"hr": "steamer", "obp": "statcast-gbm"}
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "mode": "routed",
                "routes": routes,
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["_mode"] == "routed"
        assert pred["_routes"] == routes


class TestEnsembleRoutedNormalization:
    """Integration tests for post-routing counting stat normalization."""

    def test_pitcher_ip_routed_to_different_system_scales_counting_stats(self) -> None:
        """IP from playing_time, counting stats from steamer → counting stats scaled."""
        repo = FakeProjectionRepo(
            [
                _make_projection(
                    1,
                    "steamer",
                    "pitcher",
                    {"ip": 200.0, "er": 80.0, "so": 200.0, "era": 3.60, "w": 12.0},
                ),
                _make_projection(1, "playing_time", "pitcher", {"ip": 160.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "playing_time": 0.5},
                "mode": "routed",
                "routes": {
                    "ip": "playing_time",
                    "er": "steamer",
                    "so": "steamer",
                    "era": "steamer",
                    "w": "steamer",
                },
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # IP comes from playing_time
        assert pred["ip"] == 160.0
        # Counting stats scaled by 160/200 = 0.8
        assert pred["er"] == pytest.approx(80.0 * 0.8)
        assert pred["so"] == pytest.approx(200.0 * 0.8)
        assert pred["w"] == pytest.approx(12.0 * 0.8)
        # Rate stats unchanged
        assert pred["era"] == 3.60

    def test_all_stats_same_system_no_normalization(self) -> None:
        """When all stats routed to same system, no scaling occurs."""
        repo = FakeProjectionRepo(
            [
                _make_projection(
                    1,
                    "steamer",
                    "pitcher",
                    {"ip": 200.0, "er": 80.0, "era": 3.60},
                ),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 1.0},
                "mode": "routed",
                "routes": {"ip": "steamer", "er": "steamer", "era": "steamer"},
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["ip"] == 200.0
        assert pred["er"] == 80.0
        assert pred["era"] == 3.60

    def test_batter_pa_routed_scales_counting_stats(self) -> None:
        """Batter variant: PA from one system, counting stats scaled."""
        repo = FakeProjectionRepo(
            [
                _make_projection(
                    1,
                    "steamer",
                    "batter",
                    {"pa": 600.0, "hr": 30.0, "rbi": 100.0, "avg": 0.280},
                ),
                _make_projection(1, "playing_time", "batter", {"pa": 500.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "playing_time": 0.5},
                "mode": "routed",
                "routes": {
                    "pa": "playing_time",
                    "hr": "steamer",
                    "rbi": "steamer",
                    "avg": "steamer",
                },
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["pa"] == 500.0
        assert pred["hr"] == pytest.approx(30.0 * 500.0 / 600.0)
        assert pred["rbi"] == pytest.approx(100.0 * 500.0 / 600.0)
        assert pred["avg"] == 0.280  # rate stat unchanged

    def test_rate_stats_always_unchanged_regardless_of_routing(self) -> None:
        """Rate stats are never scaled even when IP is from a different system."""
        repo = FakeProjectionRepo(
            [
                _make_projection(
                    1,
                    "steamer",
                    "pitcher",
                    {"ip": 200.0, "era": 3.60, "whip": 1.20, "fip": 3.80, "er": 80.0},
                ),
                _make_projection(1, "playing_time", "pitcher", {"ip": 120.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "playing_time": 0.5},
                "mode": "routed",
                "routes": {
                    "ip": "playing_time",
                    "era": "steamer",
                    "whip": "steamer",
                    "fip": "steamer",
                    "er": "steamer",
                },
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["era"] == 3.60
        assert pred["whip"] == 1.20
        assert pred["fip"] == 3.80
        # Counting stat is scaled
        assert pred["er"] == pytest.approx(80.0 * 120.0 / 200.0)


class TestEnsembleStatWeights:
    def test_stat_weights_override_global(self) -> None:
        """Per-stat weights used instead of global component weights."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"obp": 0.340, "hr": 30.0}),
                _make_projection(1, "statcast-gbm", "batter", {"obp": 0.360, "hr": 25.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "mode": "weighted_average",
                "stat_weights": {
                    "obp": {"statcast-gbm": 1.0, "steamer": 0.0},
                    "hr": {"steamer": 1.0, "statcast-gbm": 0.0},
                },
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["obp"] == 0.360  # 100% statcast-gbm
        assert pred["hr"] == 30.0  # 100% steamer

    def test_stat_weights_different_per_stat(self) -> None:
        """OBP 70/30 gbm/steamer, HR 0/100 gbm/steamer."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"obp": 0.330, "hr": 30.0}),
                _make_projection(1, "statcast-gbm", "batter", {"obp": 0.350, "hr": 25.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "mode": "weighted_average",
                "stat_weights": {
                    "obp": {"statcast-gbm": 0.7, "steamer": 0.3},
                    "hr": {"statcast-gbm": 0.0, "steamer": 1.0},
                },
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        expected_obp = (0.350 * 0.7 + 0.330 * 0.3) / (0.7 + 0.3)
        assert pred["obp"] == pytest.approx(expected_obp)
        assert pred["hr"] == 30.0

    def test_stat_weights_missing_system(self) -> None:
        """Referenced system absent for player → excluded from average."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"obp": 0.330}),
                # statcast-gbm not present for this player
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "mode": "weighted_average",
                "stat_weights": {
                    "obp": {"statcast-gbm": 0.7, "steamer": 0.3},
                },
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # statcast-gbm not present → only steamer contributes
        assert pred["obp"] == 0.330

    def test_stat_weights_metadata(self) -> None:
        """Prediction includes _stat_weights metadata."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0}),
                _make_projection(1, "statcast-gbm", "batter", {"hr": 25.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        stat_weights = {"hr": {"steamer": 0.6, "statcast-gbm": 0.4}}
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "mode": "weighted_average",
                "stat_weights": stat_weights,
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["_stat_weights"] == stat_weights


class TestEnsembleRouteGroups:
    def test_route_groups_basic(self) -> None:
        """route_groups expands groups to per-stat routes."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0, "obp": 0.340, "avg": 0.280}),
                _make_projection(1, "statcast-gbm", "batter", {"hr": 25.0, "obp": 0.360, "avg": 0.300}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "route_groups": {"batting_rate": "statcast-gbm", "batting_counting": "steamer"},
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # obp and avg are batting_rate → statcast-gbm
        assert pred["obp"] == 0.360
        assert pred["avg"] == 0.300
        # hr is batting_counting → steamer
        assert pred["hr"] == 30.0

    def test_route_groups_with_per_stat_override(self) -> None:
        """Per-stat routes override group assignment."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"obp": 0.340, "avg": 0.280}),
                _make_projection(1, "statcast-gbm", "batter", {"obp": 0.360, "avg": 0.300}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "route_groups": {"batting_rate": "statcast-gbm"},
                "routes": {"obp": "steamer"},
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # obp overridden to steamer
        assert pred["obp"] == 0.340
        # avg still from group → statcast-gbm
        assert pred["avg"] == 0.300

    def test_route_groups_with_custom_groups(self) -> None:
        """Custom stat_groups used in route_groups."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0, "obp": 0.340}),
                _make_projection(1, "statcast-gbm", "batter", {"hr": 25.0, "obp": 0.360}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "stat_groups": {"key_stats": ["hr", "obp"]},
                "route_groups": {"key_stats": "statcast-gbm"},
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["hr"] == 25.0
        assert pred["obp"] == 0.360

    def test_route_groups_with_fallback(self) -> None:
        """route_groups + fallback works."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0, "obp": 0.340}),
                _make_projection(1, "statcast-gbm", "batter", {"obp": 0.360}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "route_groups": {"batting_rate": "statcast-gbm", "batting_counting": "statcast-gbm"},
                "fallback": "steamer",
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # obp from primary (statcast-gbm)
        assert pred["obp"] == 0.360
        # hr from fallback (steamer) since statcast-gbm doesn't have it
        assert pred["hr"] == 30.0

    def test_route_groups_metadata(self) -> None:
        """Prediction includes _routes with expanded per-stat routes."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0, "obp": 0.340}),
                _make_projection(1, "statcast-gbm", "batter", {"hr": 25.0, "obp": 0.360}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "route_groups": {"batting_rate": "statcast-gbm"},
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["_mode"] == "routed"
        # _routes should contain expanded per-stat mapping
        assert pred["_routes"]["obp"] == "statcast-gbm"
        assert pred["_routes"]["avg"] == "statcast-gbm"
        for stat in BUILTIN_GROUPS["batting_rate"]:
            assert pred["_routes"][stat] == "statcast-gbm"

    def test_route_groups_league_required(self) -> None:
        """route_groups with league_required pseudo-group."""
        league = LeagueSettings(
            name="test",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=12,
            budget=260,
            roster_batters=9,
            roster_pitchers=8,
            batting_categories=(
                CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
                CategoryConfig(
                    key="obp",
                    name="OBP",
                    stat_type=StatType.RATE,
                    direction=Direction.HIGHER,
                    numerator="h+bb+hbp",
                    denominator="pa",
                ),
            ),
            pitching_categories=(),
        )
        repo = FakeProjectionRepo(
            [
                _make_projection(
                    1, "steamer", "batter", {"hr": 30.0, "obp": 0.340, "h": 150.0, "bb": 60.0, "hbp": 5.0, "pa": 600.0}
                ),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 1.0},
                "route_groups": {"league_required": "steamer"},
                "league": league,
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["hr"] == 30.0
        assert pred["obp"] == 0.340
        assert pred["h"] == 150.0
        assert pred["pa"] == 600.0


def _make_league(
    batting: list[CategoryConfig],
    pitching: list[CategoryConfig],
) -> LeagueSettings:
    return LeagueSettings(
        name="test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=9,
        roster_pitchers=8,
        batting_categories=tuple(batting),
        pitching_categories=tuple(pitching),
    )


def _counting_cat(key: str) -> CategoryConfig:
    return CategoryConfig(key=key, name=key, stat_type=StatType.COUNTING, direction=Direction.HIGHER)


def _rate_cat(key: str, numerator: str, denominator: str) -> CategoryConfig:
    return CategoryConfig(
        key=key,
        name=key,
        stat_type=StatType.RATE,
        direction=Direction.HIGHER,
        numerator=numerator,
        denominator=denominator,
    )


class TestEnsembleCoverageValidation:
    def test_coverage_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """League present, stat uncovered → warning in caplog."""
        league = _make_league(
            batting=[_counting_cat("hr"), _counting_cat("rbi")],
            pitching=[],
        )
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 1.0},
                "mode": "routed",
                "routes": {"hr": "steamer"},
                "league": league,
            },
        )
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball_manager.models.ensemble.model"):
            model.predict(config)
        assert any("rbi" in msg for msg in caplog.messages)

    def test_coverage_no_warning_when_covered(self, caplog: pytest.LogCaptureFixture) -> None:
        """All covered → no warning."""
        league = _make_league(
            batting=[_counting_cat("hr")],
            pitching=[],
        )
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 1.0},
                "mode": "routed",
                "routes": {"hr": "steamer"},
                "league": league,
            },
        )
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball_manager.models.ensemble.model"):
            model.predict(config)
        assert not any("uncovered" in msg.lower() for msg in caplog.messages)

    def test_coverage_check_raises_on_uncovered(self) -> None:
        """check=True + uncovered → ValueError."""
        league = _make_league(
            batting=[_counting_cat("hr"), _counting_cat("rbi")],
            pitching=[],
        )
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 1.0},
                "mode": "routed",
                "routes": {"hr": "steamer"},
                "league": league,
                "check": True,
            },
        )
        with pytest.raises(ValueError, match="rbi"):
            model.predict(config)

    def test_coverage_check_passes_when_covered(self) -> None:
        """check=True + all covered → no error."""
        league = _make_league(
            batting=[_counting_cat("hr")],
            pitching=[],
        )
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 1.0},
                "mode": "routed",
                "routes": {"hr": "steamer"},
                "league": league,
                "check": True,
            },
        )
        # Should not raise
        model.predict(config)

    def test_no_validation_without_league(self, caplog: pytest.LogCaptureFixture) -> None:
        """No league param → no validation."""
        repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"hr": 30.0}),
            ]
        )
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 1.0},
                "mode": "routed",
                "routes": {"hr": "steamer"},
                # no league
            },
        )
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball_manager.models.ensemble.model"):
            model.predict(config)
        assert not any("uncovered" in msg.lower() for msg in caplog.messages)


class _NeverCalledProjectionRepo:
    """Fake repo that raises if any fetch method is called."""

    def get_by_season(self, season: int, **kwargs: Any) -> list[Any]:
        raise AssertionError("projection repo should not be called in dry_run mode")

    def get_by_system_version(self, system: str, version: str) -> list[Any]:
        raise AssertionError("projection repo should not be called in dry_run mode")


class TestEnsembleDryRun:
    def test_dry_run_returns_empty_predictions(self) -> None:
        """dry_run=True → PredictResult with empty predictions."""
        repo = FakeProjectionRepo([])
        model = EnsembleModel(projection_repo=repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "mode": "routed",
                "routes": {"hr": "steamer", "obp": "statcast-gbm"},
                "dry_run": True,
            },
        )
        result = model.predict(config)
        assert result.model_name == "ensemble"
        # No player-level predictions
        assert len(result.predictions) == 1
        assert "player_id" not in result.predictions[0]

    def test_dry_run_includes_routing_metadata(self) -> None:
        """Result includes _routes in a single metadata dict."""
        repo = FakeProjectionRepo([])
        model = EnsembleModel(projection_repo=repo)
        routes = {"hr": "steamer", "obp": "statcast-gbm"}
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "mode": "routed",
                "routes": routes,
                "dry_run": True,
            },
        )
        result = model.predict(config)
        meta = result.predictions[0]
        assert meta["_routes"] == routes
        assert meta["_mode"] == "routed"
        assert meta["_components"] == {"steamer": 0.5, "statcast-gbm": 0.5}

    def test_dry_run_does_not_fetch_projections(self) -> None:
        """Projection repo is not called when dry_run=True."""
        repo = _NeverCalledProjectionRepo()
        model = EnsembleModel(projection_repo=repo)  # type: ignore[arg-type]
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "mode": "routed",
                "routes": {"hr": "steamer"},
                "dry_run": True,
            },
        )
        # Should not raise — repo methods not called
        result = model.predict(config)
        assert result.predictions

    def test_dry_run_with_route_groups(self) -> None:
        """dry_run works with route_groups expansion."""
        repo = _NeverCalledProjectionRepo()
        model = EnsembleModel(projection_repo=repo)  # type: ignore[arg-type]
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 0.5, "statcast-gbm": 0.5},
                "route_groups": {"batting_rate": "statcast-gbm", "batting_counting": "steamer"},
                "dry_run": True,
            },
        )
        result = model.predict(config)
        meta = result.predictions[0]
        assert meta["_routes"]["obp"] == "statcast-gbm"
        assert meta["_routes"]["hr"] == "steamer"


class TestEnsembleRateCalibration:
    def test_calibration_modifies_pitcher_era(self) -> None:
        proj_repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "pitcher", {"era": 4.0, "whip": 1.3, "ip": 180.0, "so": 200}),
            ]
        )
        pitching_repo = FakePitchingStatsRepo(
            [
                PitchingStats(player_id=1, season=2024, source="fg", ip=150.0),
                PitchingStats(player_id=1, season=2023, source="fg", ip=100.0),
            ]
        )
        model = EnsembleModel(projection_repo=proj_repo, pitching_stats_repo=pitching_repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 1.0},
                "mode": "weighted_average",
                "rate_calibration": {
                    "era_intercept": 0.5,
                    "era_slope": 0.8,
                    "era_prior_coef": 0.02,
                },
            },
        )
        result = model.predict(config)
        pred = result.predictions[0]
        # n_prior = 2 (both 2024 and 2023 have >= 10 IP)
        # corrected_era = 0.5 + 0.8 * 4.0 + 0.02 * 2 = 3.74
        assert pred["era"] == pytest.approx(3.74)
        assert pred["so"] == 200

    def test_calibration_disabled_when_false(self) -> None:
        proj_repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "pitcher", {"era": 4.0, "ip": 180.0}),
            ]
        )
        pitching_repo = FakePitchingStatsRepo([])
        model = EnsembleModel(projection_repo=proj_repo, pitching_stats_repo=pitching_repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 1.0},
                "mode": "weighted_average",
                "rate_calibration": False,
            },
        )
        result = model.predict(config)
        assert result.predictions[0]["era"] == 4.0

    def test_no_pitching_stats_repo_skips_calibration(self) -> None:
        proj_repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "pitcher", {"era": 4.0, "ip": 180.0}),
            ]
        )
        model = EnsembleModel(projection_repo=proj_repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 1.0},
                "mode": "weighted_average",
                "rate_calibration": {
                    "era_intercept": 0.5,
                    "era_slope": 0.8,
                    "era_prior_coef": 0.02,
                },
            },
        )
        result = model.predict(config)
        assert result.predictions[0]["era"] == 4.0

    def test_batters_unaffected_by_calibration(self) -> None:
        proj_repo = FakeProjectionRepo(
            [
                _make_projection(1, "steamer", "batter", {"obp": 0.350, "pa": 600}),
            ]
        )
        pitching_repo = FakePitchingStatsRepo([])
        model = EnsembleModel(projection_repo=proj_repo, pitching_stats_repo=pitching_repo)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "components": {"steamer": 1.0},
                "mode": "weighted_average",
                "rate_calibration": {
                    "era_intercept": 0.5,
                    "era_slope": 0.8,
                    "era_prior_coef": 0.02,
                },
            },
        )
        result = model.predict(config)
        assert result.predictions[0]["obp"] == 0.350
