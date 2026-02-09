"""Tests for ContextualEmbeddingRateComputer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from fantasy_baseball_manager.contextual.data.models import (
    GameSequence,
    PitchEvent,
)
from fantasy_baseball_manager.contextual.predictor import ContextualPredictor
from fantasy_baseball_manager.contextual.training.config import (
    ContextualRateComputerConfig,
)
from fantasy_baseball_manager.pipeline.stages.contextual_rate_computer import (
    ContextualEmbeddingRateComputer,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates

if TYPE_CHECKING:
    import pytest


def _make_pitch(pa_event: str | None = None) -> PitchEvent:
    return PitchEvent(
        batter_id=123,
        pitcher_id=456,
        pitch_type="FF",
        pitch_result="called_strike",
        pitch_result_type="S",
        release_speed=95.0,
        release_spin_rate=2300,
        pfx_x=-5.0,
        pfx_z=10.0,
        plate_x=0.5,
        plate_z=2.5,
        release_extension=6.5,
        launch_speed=None,
        launch_angle=None,
        hit_distance=None,
        bb_type=None,
        estimated_woba=None,
        inning=1,
        is_top=True,
        outs=0,
        balls=0,
        strikes=0,
        runners_on_1b=False,
        runners_on_2b=False,
        runners_on_3b=False,
        bat_score=0,
        fld_score=0,
        stand="R",
        p_throws="R",
        pitch_number=1,
        pa_event=pa_event,
        delta_run_exp=None,
    )


def _make_game(game_pk: int = 1) -> GameSequence:
    """Create a game with 4 PA (3 field_out + 1 single = 3 outs, 4 PA)."""
    pitches = (
        _make_pitch(None),
        _make_pitch("field_out"),
        _make_pitch(None),
        _make_pitch("field_out"),
        _make_pitch(None),
        _make_pitch("single"),
        _make_pitch(None),
        _make_pitch("field_out"),
    )
    return GameSequence(
        game_pk=game_pk,
        game_date="2024-06-01",
        season=2024,
        home_team="NYY",
        away_team="BOS",
        perspective="batter",
        player_id=123,
        pitches=pitches,
    )


def _make_marcel_player(player_id: str = "fg123", name: str = "Test Batter") -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        rates={
            "hr": 0.04, "so": 0.20, "bb": 0.10, "singles": 0.15,
            "doubles": 0.05, "triples": 0.01,
            "hbp": 0.01, "sf": 0.005, "sh": 0.002, "sb": 0.02, "cs": 0.005,
            "r": 0.12, "rbi": 0.14,
        },
        metadata={
            "pa_per_year": [500.0, 450.0, 480.0],
            "avg_league_rates": {"hr": 0.03},
            "target_rates": {"hr": 0.04},
        },
    )


def _make_marcel_pitcher(player_id: str = "fg456", name: str = "Test Pitcher") -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name=name,
        year=2025,
        age=30,
        rates={
            "so": 0.22, "h": 0.25, "bb": 0.08, "hr": 0.03,
            "hbp": 0.01, "er": 0.10, "w": 0.15, "sv": 0.0, "hld": 0.0, "bs": 0.0,
        },
        metadata={
            "ip_per_year": [180.0, 175.0, 190.0],
            "avg_league_rates": {"so": 0.22},
            "target_rates": {"so": 0.22},
        },
    )


def _build_computer(
    model_store: MagicMock | None = None,
    sequence_builder: MagicMock | None = None,
    id_mapper: MagicMock | None = None,
    config: ContextualRateComputerConfig | None = None,
) -> ContextualEmbeddingRateComputer:
    predictor = ContextualPredictor(
        sequence_builder=sequence_builder or MagicMock(),
        model_store=model_store or MagicMock(),
    )
    return ContextualEmbeddingRateComputer(
        predictor=predictor,
        id_mapper=id_mapper or MagicMock(),
        config=config or ContextualRateComputerConfig(),
    )


class TestContextualRateComputerInit:
    def test_init(self) -> None:
        computer = _build_computer()
        assert computer.config.batter_model_name == "finetune_batter_best"
        assert computer.config.pitcher_model_name == "finetune_pitcher_best"
        assert computer.config.batter_min_games == 30
        assert computer.config.pitcher_min_games == 10
        assert computer.config.batter_context_window == 30
        assert computer.config.pitcher_context_window == 10

    def test_custom_config(self) -> None:
        config = ContextualRateComputerConfig(
            batter_model_name="custom_batter",
            batter_min_games=20,
        )
        computer = _build_computer(config=config)
        assert computer.config.batter_model_name == "custom_batter"
        assert computer.config.batter_min_games == 20


class TestContextualFallbackBehavior:
    def test_falls_back_to_marcel_when_no_model(self) -> None:
        """When model checkpoint not found, returns Marcel rates."""
        mock_store = MagicMock()
        mock_store.exists.return_value = False

        computer = _build_computer(model_store=mock_store)
        computer._marcel_computer = MagicMock()
        marcel_player = _make_marcel_player()
        computer._marcel_computer.compute_batting_rates.return_value = [marcel_player]

        result = computer.compute_batting_rates(
            MagicMock(), MagicMock(), 2025, 3,
        )

        assert len(result) == 1
        assert result[0] is marcel_player

    def test_falls_back_for_player_without_mlbam_id(self) -> None:
        """Player with no MLBAM mapping falls back to Marcel."""
        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_model = MagicMock()
        mock_store.load_finetune_model.return_value = mock_model

        mock_mapper = MagicMock()
        mock_mapper.fangraphs_to_mlbam.return_value = None

        computer = _build_computer(model_store=mock_store, id_mapper=mock_mapper)
        marcel_player = _make_marcel_player()
        computer._marcel_computer = MagicMock()
        computer._marcel_computer.compute_batting_rates.return_value = [marcel_player]

        result = computer.compute_batting_rates(
            MagicMock(), MagicMock(), 2025, 3,
        )

        assert len(result) == 1
        assert result[0].metadata.get("contextual_predicted") is not True

    def test_falls_back_for_insufficient_games(self) -> None:
        """Player with fewer games than min_games falls back."""
        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_model = MagicMock()
        mock_store.load_finetune_model.return_value = mock_model

        mock_mapper = MagicMock()
        mock_mapper.fangraphs_to_mlbam.return_value = 123456

        mock_builder = MagicMock()
        mock_builder.build_player_season.return_value = [_make_game(i) for i in range(5)]

        config = ContextualRateComputerConfig(batter_min_games=10, pitcher_min_games=10)
        computer = _build_computer(
            model_store=mock_store,
            id_mapper=mock_mapper,
            sequence_builder=mock_builder,
            config=config,
        )
        marcel_player = _make_marcel_player()
        computer._marcel_computer = MagicMock()
        computer._marcel_computer.compute_batting_rates.return_value = [marcel_player]

        result = computer.compute_batting_rates(
            MagicMock(), MagicMock(), 2025, 3,
        )

        assert len(result) == 1
        assert result[0].metadata.get("contextual_predicted") is not True


class TestContextualPredictions:
    def test_uses_contextual_predictions_for_covered_stats(self) -> None:
        """When model is available and player has enough games, use contextual predictions."""
        import torch

        mock_store = MagicMock()
        mock_store.exists.return_value = True

        mock_model = MagicMock()
        preds_tensor = torch.tensor([[0.5, 2.0, 1.0, 3.0, 0.5, 0.1]])
        mock_model.return_value = {"performance_preds": preds_tensor}
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_store.load_finetune_model.return_value = mock_model

        mock_mapper = MagicMock()
        mock_mapper.fangraphs_to_mlbam.return_value = 123456

        mock_builder = MagicMock()
        games = [_make_game(i) for i in range(12)]
        mock_builder.build_player_season.return_value = games

        config = ContextualRateComputerConfig(batter_min_games=10, batter_context_window=10)
        computer = _build_computer(
            model_store=mock_store,
            id_mapper=mock_mapper,
            sequence_builder=mock_builder,
            config=config,
        )

        marcel_player = _make_marcel_player()
        computer._marcel_computer = MagicMock()
        computer._marcel_computer.compute_batting_rates.return_value = [marcel_player]

        # Mock the tensorizer on the predictor
        mock_tensorizer = MagicMock()
        mock_tensorized = MagicMock()
        mock_tensorizer.tensorize_context.return_value = mock_tensorized
        mock_batch = MagicMock()
        mock_tensorizer.collate.return_value = mock_batch
        computer.predictor._tensorizer = mock_tensorizer

        result = computer.compute_batting_rates(
            MagicMock(), MagicMock(), 2025, 3,
        )

        assert len(result) == 1
        player = result[0]
        assert player.metadata.get("contextual_predicted") is True
        assert player.metadata.get("contextual_games_used") == 10
        assert "marcel_rates" in player.metadata

        assert "hr" in player.rates
        assert "so" in player.rates
        assert "bb" in player.rates
        assert "singles" in player.rates
        assert "doubles" in player.rates
        assert "triples" in player.rates

        assert player.rates["hbp"] == marcel_player.rates["hbp"]
        assert player.rates["sf"] == marcel_player.rates["sf"]
        assert player.rates["sb"] == marcel_player.rates["sb"]

    def test_preserves_marcel_metadata(self) -> None:
        """Contextual player should preserve Marcel metadata fields."""
        import torch

        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_model = MagicMock()
        preds_tensor = torch.tensor([[0.5, 2.0, 1.0, 3.0, 0.5, 0.1]])
        mock_model.return_value = {"performance_preds": preds_tensor}
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_store.load_finetune_model.return_value = mock_model

        mock_mapper = MagicMock()
        mock_mapper.fangraphs_to_mlbam.return_value = 123456

        mock_builder = MagicMock()
        mock_builder.build_player_season.return_value = [_make_game(i) for i in range(12)]

        computer = _build_computer(
            model_store=mock_store,
            id_mapper=mock_mapper,
            sequence_builder=mock_builder,
        )

        marcel_player = _make_marcel_player()
        computer._marcel_computer = MagicMock()
        computer._marcel_computer.compute_batting_rates.return_value = [marcel_player]

        mock_tensorizer = MagicMock()
        mock_tensorizer.tensorize_context.return_value = MagicMock()
        mock_tensorizer.collate.return_value = MagicMock()
        computer.predictor._tensorizer = mock_tensorizer

        result = computer.compute_batting_rates(
            MagicMock(), MagicMock(), 2025, 3,
        )

        player = result[0]
        assert player.metadata.get("pa_per_year") == [500.0, 450.0, 480.0]
        assert player.metadata.get("avg_league_rates") == {"hr": 0.03}
        assert player.metadata.get("target_rates") == {"hr": 0.04}


class TestContextualPitching:
    def test_pitching_falls_back_when_no_model(self) -> None:
        mock_store = MagicMock()
        mock_store.exists.return_value = False

        computer = _build_computer(model_store=mock_store)
        marcel_pitcher = _make_marcel_pitcher()
        computer._marcel_computer = MagicMock()
        computer._marcel_computer.compute_pitching_rates.return_value = [marcel_pitcher]

        result = computer.compute_pitching_rates(
            MagicMock(), MagicMock(), 2025, 3,
        )

        assert len(result) == 1
        assert result[0] is marcel_pitcher


class TestPredictionVarianceLogging:
    """Tests for _log_prediction_variance diagnostic logging."""

    def _make_contextual_player(
        self,
        player_id: str,
        rates: dict[str, float],
    ) -> PlayerRates:
        return PlayerRates(
            player_id=player_id,
            name=f"Player {player_id}",
            year=2025,
            age=28,
            rates=rates,
            metadata={
                "contextual_predicted": True,
                "contextual_games_used": 30,
                "marcel_rates": {"hr": 0.03},
            },
        )

    def test_logs_variance_stats_for_batters(self, caplog: pytest.LogCaptureFixture) -> None:
        """When multiple contextual players exist, logs per-stat distribution."""
        computer = _build_computer()
        players = [
            self._make_contextual_player("fg1", {"hr": 0.02, "so": 0.15}),
            self._make_contextual_player("fg2", {"hr": 0.06, "so": 0.25}),
            self._make_contextual_player("fg3", {"hr": 0.04, "so": 0.20}),
            self._make_contextual_player("fg4", {"hr": 0.08, "so": 0.30}),
        ]

        with caplog.at_level(logging.INFO):
            computer._log_prediction_variance(players, "batter")

        log_text = caplog.text
        assert "hr" in log_text
        assert "so" in log_text
        assert "min=" in log_text
        assert "max=" in log_text
        assert "std=" in log_text
        assert "median=" in log_text

    def test_logs_nothing_with_fewer_than_two_contextual_players(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With fewer than 2 contextual players, logs a skip message."""
        computer = _build_computer()
        marcel_only = _make_marcel_player()
        players = [marcel_only]

        with caplog.at_level(logging.INFO):
            computer._log_prediction_variance(players, "batter")

        assert any("fewer than 2" in r.message.lower() or "skip" in r.message.lower()
                    for r in caplog.records) or len(caplog.records) == 0
        # Should NOT contain per-stat distribution lines
        assert "min=" not in caplog.text

    def test_logs_variance_for_pitchers(self, caplog: pytest.LogCaptureFixture) -> None:
        """Pitching path also triggers variance logging."""
        computer = _build_computer()
        players = [
            self._make_contextual_player("fg10", {"so": 0.20, "h": 0.25}),
            self._make_contextual_player("fg11", {"so": 0.30, "h": 0.20}),
            self._make_contextual_player("fg12", {"so": 0.25, "h": 0.22}),
        ]

        with caplog.at_level(logging.INFO):
            computer._log_prediction_variance(players, "pitcher")

        log_text = caplog.text
        assert "so" in log_text
        assert "h" in log_text
        assert "min=" in log_text
        assert "max=" in log_text

    def test_includes_percentiles(self, caplog: pytest.LogCaptureFixture) -> None:
        """Log output includes p25 and p75 percentiles."""
        computer = _build_computer()
        players = [
            self._make_contextual_player(f"fg{i}", {"hr": 0.01 * (i + 1), "so": 0.10 + 0.02 * i})
            for i in range(10)
        ]

        with caplog.at_level(logging.INFO):
            computer._log_prediction_variance(players, "batter")

        log_text = caplog.text
        assert "p25=" in log_text
        assert "p75=" in log_text
