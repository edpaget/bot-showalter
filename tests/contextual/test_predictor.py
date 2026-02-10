"""Tests for ContextualPredictor helper."""

from __future__ import annotations

from unittest.mock import MagicMock

from fantasy_baseball_manager.contextual.data.models import (
    GameSequence,
    PitchEvent,
)
from fantasy_baseball_manager.contextual.predictor import ContextualPredictor
from fantasy_baseball_manager.contextual.training.config import (
    BATTER_TARGET_STATS,
)


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
    """Create a game with 4 PA (3 field_out + 1 single)."""
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


class TestContextualPredictorInit:
    def test_init_with_defaults(self) -> None:
        predictor = ContextualPredictor(
            sequence_builder=MagicMock(),
        )
        assert predictor._models_loaded is False
        assert predictor._batter_model is None
        assert predictor._pitcher_model is None

    def test_has_batter_model_before_loading(self) -> None:
        predictor = ContextualPredictor(sequence_builder=MagicMock())
        assert predictor.has_batter_model() is False

    def test_has_pitcher_model_before_loading(self) -> None:
        predictor = ContextualPredictor(sequence_builder=MagicMock())
        assert predictor.has_pitcher_model() is False


class TestContextualPredictorModelLoading:
    def test_ensure_models_loaded_loads_both(self) -> None:
        from fantasy_baseball_manager.contextual.model.config import ModelConfig

        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_store.load_model_config.return_value = ModelConfig()
        mock_model = MagicMock()
        mock_store.load_finetune_model.return_value = mock_model

        predictor = ContextualPredictor(
            sequence_builder=MagicMock(),
            model_store=mock_store,
        )
        predictor.ensure_models_loaded(
            batter_model_name="batter_best",
            pitcher_model_name="pitcher_best",
        )

        assert predictor.has_batter_model() is True
        assert predictor.has_pitcher_model() is True
        assert predictor._models_loaded is True

    def test_ensure_models_loaded_skips_if_already_loaded(self) -> None:
        from fantasy_baseball_manager.contextual.model.config import ModelConfig

        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_store.load_model_config.return_value = ModelConfig()
        mock_store.load_finetune_model.return_value = MagicMock()

        predictor = ContextualPredictor(
            sequence_builder=MagicMock(),
            model_store=mock_store,
        )
        predictor.ensure_models_loaded("b", "p")
        predictor.ensure_models_loaded("b", "p")

        # Should only load once
        assert mock_store.load_finetune_model.call_count == 2  # one batter, one pitcher

    def test_model_not_found_returns_false(self) -> None:
        mock_store = MagicMock()
        mock_store.exists.return_value = False

        predictor = ContextualPredictor(
            sequence_builder=MagicMock(),
            model_store=mock_store,
        )
        predictor.ensure_models_loaded("batter_best", "pitcher_best")

        assert predictor.has_batter_model() is False
        assert predictor.has_pitcher_model() is False
        assert predictor._models_loaded is True

    def test_loads_config_from_metadata(self) -> None:
        """Predictor uses model_config from checkpoint metadata, not defaults."""
        from fantasy_baseball_manager.contextual.model.config import ModelConfig

        custom_config = ModelConfig(d_model=64, n_layers=2, n_heads=2, ff_dim=128)
        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_store.load_model_config.return_value = custom_config
        mock_store.load_finetune_model.return_value = MagicMock()

        predictor = ContextualPredictor(
            sequence_builder=MagicMock(),
            model_store=mock_store,
        )
        predictor.ensure_models_loaded("batter_best", "pitcher_best")

        # Verify load_model_config was called for each model
        assert mock_store.load_model_config.call_count == 2
        # Verify the config was passed to load_finetune_model
        calls = mock_store.load_finetune_model.call_args_list
        for call in calls:
            assert call[0][1] == custom_config  # second positional arg is config


class TestContextualPredictorPredict:
    def test_returns_none_when_insufficient_games(self) -> None:
        mock_builder = MagicMock()
        mock_builder.build_player_season.return_value = [_make_game(i) for i in range(5)]

        predictor = ContextualPredictor(sequence_builder=mock_builder)
        mock_model = MagicMock()

        result = predictor.predict_player(
            mlbam_id=123,
            data_year=2024,
            perspective="batter",
            model=mock_model,
            target_stats=BATTER_TARGET_STATS,
            min_games=10,
            context_window=10,
        )

        assert result is None

    def test_returns_predictions_with_sufficient_data(self) -> None:
        import torch

        mock_builder = MagicMock()
        games = [_make_game(i) for i in range(12)]
        mock_builder.build_player_season.return_value = games

        mock_model = MagicMock()
        preds_tensor = torch.tensor([[0.5, 2.0, 1.0, 3.0, 0.5, 0.1]])
        mock_model.return_value = {"performance_preds": preds_tensor}

        mock_tensorizer = MagicMock()
        mock_tensorizer.tensorize_context.return_value = MagicMock()
        mock_tensorizer.collate.return_value = MagicMock()

        predictor = ContextualPredictor(sequence_builder=mock_builder)
        predictor._tensorizer = mock_tensorizer

        result = predictor.predict_player(
            mlbam_id=123,
            data_year=2024,
            perspective="batter",
            model=mock_model,
            target_stats=BATTER_TARGET_STATS,
            min_games=10,
            context_window=10,
        )

        assert result is not None
        predictions, context_games = result
        assert "hr" in predictions
        assert "so" in predictions
        assert "bb" in predictions
        assert len(context_games) == 10  # context_window

    def test_context_window_slices_last_n_games(self) -> None:
        import torch

        mock_builder = MagicMock()
        games = [_make_game(i) for i in range(20)]
        mock_builder.build_player_season.return_value = games

        mock_model = MagicMock()
        preds_tensor = torch.tensor([[0.5, 2.0, 1.0, 3.0, 0.5, 0.1]])
        mock_model.return_value = {"performance_preds": preds_tensor}

        mock_tensorizer = MagicMock()
        mock_tensorizer.tensorize_context.return_value = MagicMock()
        mock_tensorizer.collate.return_value = MagicMock()

        predictor = ContextualPredictor(sequence_builder=mock_builder)
        predictor._tensorizer = mock_tensorizer

        result = predictor.predict_player(
            mlbam_id=123,
            data_year=2024,
            perspective="batter",
            model=mock_model,
            target_stats=BATTER_TARGET_STATS,
            min_games=10,
            context_window=5,
        )

        assert result is not None
        _, context_games = result
        assert len(context_games) == 5
        # Should be the last 5 games
        assert context_games[0].game_pk == 15
        assert context_games[-1].game_pk == 19
