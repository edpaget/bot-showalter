"""Tests for ContextualBlender pipeline stage."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fantasy_baseball_manager.contextual.adapter import PerGameToSeasonAdapter
from fantasy_baseball_manager.contextual.data.models import (
    GameSequence,
    PitchEvent,
)
from fantasy_baseball_manager.contextual.predictor import ContextualPredictor
from fantasy_baseball_manager.contextual.training.config import (
    ContextualBlenderConfig,
)
from fantasy_baseball_manager.pipeline.stages.contextual_blender import (
    ContextualBlender,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates


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


def _make_game(game_pk: int = 1, perspective: str = "batter") -> GameSequence:
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
        perspective=perspective,
        player_id=123,
        pitches=pitches,
    )


def _make_batter(
    player_id: str = "fg123",
    name: str = "Test Batter",
    mlbam_id: int | None = 123456,
) -> PlayerRates:
    player = MagicMock()
    player.mlbam_id = mlbam_id
    return PlayerRates(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        rates={
            "hr": 0.040, "so": 0.200, "bb": 0.100, "singles": 0.150,
            "doubles": 0.050, "triples": 0.010,
            "hbp": 0.010, "sf": 0.005, "sb": 0.020,
        },
        metadata={"pa_per_year": [500.0, 450.0, 480.0]},
        player=player,
    )


def _make_pitcher(
    player_id: str = "fg456",
    name: str = "Test Pitcher",
    mlbam_id: int | None = 789012,
) -> PlayerRates:
    player = MagicMock()
    player.mlbam_id = mlbam_id
    return PlayerRates(
        player_id=player_id,
        name=name,
        year=2025,
        age=30,
        rates={
            "so": 0.220, "h": 0.250, "bb": 0.080, "hr": 0.030,
            "hbp": 0.010, "er": 0.100,
        },
        metadata={"ip_per_year": [180.0, 175.0, 190.0]},
        player=player,
    )


def _build_blender(
    predictor: ContextualPredictor | None = None,
    id_mapper: MagicMock | None = None,
    config: ContextualBlenderConfig | None = None,
) -> ContextualBlender:
    return ContextualBlender(
        predictor=predictor or ContextualPredictor(sequence_builder=MagicMock()),
        id_mapper=id_mapper or MagicMock(),
        config=config or ContextualBlenderConfig(),
    )


def _compute_expected_contextual_rates(
    avg_predictions: dict[str, float],
    games: list[GameSequence],
    perspective: str,
    marcel_rates: dict[str, float],
) -> dict[str, float]:
    """Compute what the adapter would produce from raw predictions."""
    adapter = PerGameToSeasonAdapter(perspective)
    avg_denom = sum(adapter.game_denominator(g) for g in games) / len(games)
    rates = adapter.predictions_to_rates(avg_predictions, avg_denom, marcel_rates)
    assert rates is not None
    return rates


class TestContextualBlenderInit:
    def test_init(self) -> None:
        blender = _build_blender()
        assert blender.config.contextual_weight == 0.3
        assert blender.config.batter_min_games == 30
        assert blender.config.pitcher_min_games == 10
        assert blender.config.batter_context_window == 30
        assert blender.config.pitcher_context_window == 10

    def test_custom_config(self) -> None:
        config = ContextualBlenderConfig(contextual_weight=0.5, pitcher_min_games=20)
        blender = _build_blender(config=config)
        assert blender.config.contextual_weight == 0.5
        assert blender.config.pitcher_min_games == 20


class TestContextualBlenderAdjust:
    def test_returns_empty_for_empty_list(self) -> None:
        blender = _build_blender()
        assert blender.adjust([]) == []

    def test_returns_unchanged_when_no_model(self) -> None:
        """When no model exists, should return players unchanged."""
        mock_store = MagicMock()
        mock_store.exists.return_value = False

        predictor = ContextualPredictor(
            sequence_builder=MagicMock(),
            model_store=mock_store,
        )

        blender = _build_blender(predictor=predictor)
        player = _make_batter()

        result = blender.adjust([player])

        assert len(result) == 1
        assert result[0].rates["hr"] == 0.040

    def test_returns_unchanged_when_no_mlbam_id(self) -> None:
        """Player with no MLBAM ID should be returned unchanged."""
        predictor = ContextualPredictor(sequence_builder=MagicMock())
        predictor._batter_model = MagicMock()
        predictor._pitcher_model = MagicMock()
        predictor._models_loaded = True

        player = _make_batter(mlbam_id=None)

        blender = _build_blender(predictor=predictor)
        result = blender.adjust([player])

        assert len(result) == 1
        assert result[0].rates["hr"] == 0.040
        assert result[0].metadata.get("contextual_blended") is not True

    def test_returns_unchanged_when_insufficient_games(self) -> None:
        """Player without enough games should be returned unchanged."""
        mock_builder = MagicMock()
        mock_builder.build_player_season.return_value = []

        predictor = ContextualPredictor(sequence_builder=mock_builder)
        predictor._batter_model = MagicMock()
        predictor._pitcher_model = MagicMock()
        predictor._models_loaded = True

        blender = _build_blender(predictor=predictor)
        player = _make_batter()

        result = blender.adjust([player])

        assert len(result) == 1
        assert result[0].metadata.get("contextual_blended") is not True

    def test_blends_batter_rates_with_correct_weight(self) -> None:
        """Verify blending math: (1-w)*marcel + w*contextual."""
        # Raw per-game predictions (counts per game)
        avg_predictions = {"hr": 0.5, "so": 2.0, "bb": 1.0, "h": 3.0, "2b": 0.5, "3b": 0.1}
        context_games = [_make_game(i) for i in range(10)]

        predictor = ContextualPredictor(sequence_builder=MagicMock())
        predictor._batter_model = MagicMock()
        predictor._pitcher_model = MagicMock()
        predictor._models_loaded = True
        predictor.predict_player = MagicMock(  # type: ignore[assignment]
            return_value=(avg_predictions, context_games),
        )

        player = _make_batter()
        weight = 0.3
        config = ContextualBlenderConfig(contextual_weight=weight)
        blender = _build_blender(predictor=predictor, config=config)

        result = blender.adjust([player])
        blended = result[0]

        # Compute expected contextual rates
        ctx_rates = _compute_expected_contextual_rates(
            avg_predictions, context_games, "batter", player.rates,
        )

        # Verify blending for each covered stat
        for stat in ctx_rates:
            if stat in player.rates:
                expected = (1 - weight) * player.rates[stat] + weight * ctx_rates[stat]
                assert blended.rates[stat] == pytest.approx(expected, rel=1e-6), f"Mismatch for {stat}"

    def test_custom_weight_threads_through(self) -> None:
        """Custom weight should be used in blending."""
        avg_predictions = {"hr": 0.5, "so": 2.0, "bb": 1.0, "h": 3.0, "2b": 0.5, "3b": 0.1}
        context_games = [_make_game(i) for i in range(10)]

        predictor = ContextualPredictor(sequence_builder=MagicMock())
        predictor._batter_model = MagicMock()
        predictor._pitcher_model = MagicMock()
        predictor._models_loaded = True
        predictor.predict_player = MagicMock(  # type: ignore[assignment]
            return_value=(avg_predictions, context_games),
        )

        weight = 0.5
        config = ContextualBlenderConfig(contextual_weight=weight)
        blender = _build_blender(predictor=predictor, config=config)

        player = _make_batter()
        result = blender.adjust([player])

        ctx_rates = _compute_expected_contextual_rates(
            avg_predictions, context_games, "batter", player.rates,
        )

        blended = result[0]
        expected_hr = (1 - weight) * player.rates["hr"] + weight * ctx_rates["hr"]
        assert blended.rates["hr"] == pytest.approx(expected_hr, rel=1e-6)

    def test_metadata_flags_set_correctly(self) -> None:
        """Blended players should have correct metadata."""
        avg_predictions = {"hr": 0.5, "so": 2.0, "bb": 1.0, "h": 3.0, "2b": 0.5, "3b": 0.1}
        context_games = [_make_game(i) for i in range(10)]

        predictor = ContextualPredictor(sequence_builder=MagicMock())
        predictor._batter_model = MagicMock()
        predictor._pitcher_model = MagicMock()
        predictor._models_loaded = True
        predictor.predict_player = MagicMock(  # type: ignore[assignment]
            return_value=(avg_predictions, context_games),
        )

        config = ContextualBlenderConfig(contextual_weight=0.3)
        blender = _build_blender(predictor=predictor, config=config)

        player = _make_batter()
        result = blender.adjust([player])

        meta = result[0].metadata
        assert meta["contextual_blended"] is True
        assert meta["contextual_blend_weight"] == 0.3
        assert "contextual_rates" in meta

    def test_uncovered_stats_preserved(self) -> None:
        """Stats not covered by the model should be preserved from input."""
        avg_predictions = {"hr": 0.5, "so": 2.0, "bb": 1.0, "h": 3.0, "2b": 0.5, "3b": 0.1}
        context_games = [_make_game(i) for i in range(10)]

        predictor = ContextualPredictor(sequence_builder=MagicMock())
        predictor._batter_model = MagicMock()
        predictor._pitcher_model = MagicMock()
        predictor._models_loaded = True
        predictor.predict_player = MagicMock(  # type: ignore[assignment]
            return_value=(avg_predictions, context_games),
        )

        blender = _build_blender(predictor=predictor)
        player = _make_batter()
        result = blender.adjust([player])

        # hbp, sf, sb are uncovered by adapter (uses Marcel fallback)
        # but they appear in contextual_rates via adapter too. The key thing
        # is that stats not in contextual_rates stay at their original value.
        # Since adapter copies hbp/sf/sb from marcel_rates, they will be in
        # contextual_rates and get blended. But since contextual uses Marcel
        # for these, the blend result should equal the Marcel rate.
        assert result[0].rates["hbp"] == pytest.approx(0.010, rel=1e-6)
        assert result[0].rates["sf"] == pytest.approx(0.005, rel=1e-6)
        assert result[0].rates["sb"] == pytest.approx(0.020, rel=1e-6)

    def test_blends_pitcher_rates(self) -> None:
        """Pitchers should be blended using the pitcher model."""
        avg_predictions = {"so": 3.0, "h": 2.5, "bb": 0.8, "hr": 0.3}
        context_games = [_make_game(i, perspective="pitcher") for i in range(10)]

        predictor = ContextualPredictor(sequence_builder=MagicMock())
        predictor._batter_model = MagicMock()
        predictor._pitcher_model = MagicMock()
        predictor._models_loaded = True
        predictor.predict_player = MagicMock(  # type: ignore[assignment]
            return_value=(avg_predictions, context_games),
        )

        weight = 0.3
        config = ContextualBlenderConfig(contextual_weight=weight)
        blender = _build_blender(predictor=predictor, config=config)

        player = _make_pitcher()
        result = blender.adjust([player])

        ctx_rates = _compute_expected_contextual_rates(
            avg_predictions, context_games, "pitcher", player.rates,
        )

        blended = result[0]
        expected_so = (1 - weight) * player.rates["so"] + weight * ctx_rates["so"]
        assert blended.rates["so"] == pytest.approx(expected_so, rel=1e-6)

        # Uncovered stats preserved (er is uncovered for pitchers)
        assert blended.rates["er"] == pytest.approx(0.100, rel=1e-6)

    def test_player_without_player_object_unchanged(self) -> None:
        """Player without a Player object (no mlbam lookup) is unchanged."""
        predictor = ContextualPredictor(sequence_builder=MagicMock())
        predictor._batter_model = MagicMock()
        predictor._pitcher_model = MagicMock()
        predictor._models_loaded = True

        player = PlayerRates(
            player_id="fg123",
            name="No Player",
            year=2025,
            age=28,
            rates={"hr": 0.040},
            metadata={"pa_per_year": [500.0]},
            player=None,
        )

        blender = _build_blender(predictor=predictor)
        result = blender.adjust([player])

        assert len(result) == 1
        assert result[0].rates["hr"] == 0.040


class TestContextualBlenderMath:
    """Test the blending formula without requiring torch."""

    def test_blend_formula(self) -> None:
        marcel_rate = 0.040
        contextual_rate = 0.060
        weight = 0.3
        expected = (1 - weight) * marcel_rate + weight * contextual_rate
        assert expected == pytest.approx(0.046, rel=1e-3)

    def test_weight_zero_returns_marcel(self) -> None:
        marcel_rate = 0.040
        contextual_rate = 0.060
        weight = 0.0
        result = (1 - weight) * marcel_rate + weight * contextual_rate
        assert result == marcel_rate

    def test_weight_one_returns_contextual(self) -> None:
        marcel_rate = 0.040
        contextual_rate = 0.060
        weight = 1.0
        result = (1 - weight) * marcel_rate + weight * contextual_rate
        assert result == contextual_rate
