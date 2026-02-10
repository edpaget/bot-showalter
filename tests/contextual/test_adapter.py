"""Tests for PerGameToSeasonAdapter."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.contextual.adapter import PerGameToSeasonAdapter
from fantasy_baseball_manager.contextual.data.models import GameSequence, PitchEvent


def _make_pitch(pa_event: str | None = None) -> PitchEvent:
    """Create a minimal PitchEvent for testing."""
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


def _make_game(pa_events: list[str | None]) -> GameSequence:
    """Create a GameSequence with pitches having the given pa_events."""
    pitches = tuple(_make_pitch(pa_event=ev) for ev in pa_events)
    return GameSequence(
        game_pk=1,
        game_date="2024-06-01",
        season=2024,
        home_team="NYY",
        away_team="BOS",
        perspective="batter",
        player_id=123,
        pitches=pitches,
    )


class TestCountPlateAppearances:
    def test_counts_pitches_with_pa_event(self) -> None:
        adapter = PerGameToSeasonAdapter("batter")
        game = _make_game([None, "single", None, None, "strikeout", None, "walk"])
        assert adapter.count_plate_appearances(game) == 3

    def test_zero_when_no_pa_events(self) -> None:
        adapter = PerGameToSeasonAdapter("batter")
        game = _make_game([None, None, None])
        assert adapter.count_plate_appearances(game) == 0

    def test_all_pitches_have_pa_events(self) -> None:
        adapter = PerGameToSeasonAdapter("batter")
        game = _make_game(["single", "home_run"])
        assert adapter.count_plate_appearances(game) == 2


class TestCountOuts:
    def test_counts_out_events(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        game = _make_game([
            "field_out",
            None,
            "strikeout",
            "single",
            "grounded_into_double_play",
            None,
        ])
        # field_out=1, strikeout=1, gidp=2 = 4 outs
        assert adapter.count_outs(game) == 4

    def test_sac_fly_counts_as_out(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        game = _make_game(["sac_fly", "sac_bunt"])
        assert adapter.count_outs(game) == 2

    def test_no_outs(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        game = _make_game(["single", "walk", "home_run", None])
        assert adapter.count_outs(game) == 0

    def test_force_out(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        game = _make_game(["force_out"])
        assert adapter.count_outs(game) == 1

    def test_strikeout_double_play(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        game = _make_game(["strikeout_double_play"])
        assert adapter.count_outs(game) == 2

    def test_double_play(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        game = _make_game(["double_play"])
        assert adapter.count_outs(game) == 2

    def test_triple_play(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        game = _make_game(["triple_play"])
        assert adapter.count_outs(game) == 3

    def test_fielders_choice_out(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        game = _make_game(["fielders_choice_out"])
        assert adapter.count_outs(game) == 1


class TestGameDenominator:
    def test_batter_uses_pa(self) -> None:
        adapter = PerGameToSeasonAdapter("batter")
        game = _make_game(["single", None, "strikeout", None, "walk"])
        assert adapter.game_denominator(game) == 3.0

    def test_pitcher_uses_outs(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        game = _make_game(["field_out", "strikeout", None, "single"])
        assert adapter.game_denominator(game) == 2.0


class TestPredictionsToRates:
    def test_batter_rate_conversion(self) -> None:
        adapter = PerGameToSeasonAdapter("batter")
        # Model predicts per-game counts
        avg_predictions = {"hr": 0.5, "so": 2.0, "bb": 1.0, "h": 3.0, "2b": 0.5, "3b": 0.1}
        avg_denominator = 4.0  # avg PA/game
        marcel_rates = {
            "hr": 0.04, "so": 0.20, "bb": 0.10, "singles": 0.15,
            "doubles": 0.05, "triples": 0.01,
            "hbp": 0.01, "sf": 0.005, "sh": 0.002, "sb": 0.02, "cs": 0.005,
            "r": 0.12, "rbi": 0.14,
        }
        result = adapter.predictions_to_rates(avg_predictions, avg_denominator, marcel_rates)
        assert result is not None

        # hr = 0.5 / 4.0 = 0.125
        assert result["hr"] == pytest.approx(0.125)
        # so = 2.0 / 4.0 = 0.5
        assert result["so"] == pytest.approx(0.5)
        # bb = 1.0 / 4.0 = 0.25
        assert result["bb"] == pytest.approx(0.25)
        # singles = (h - 2b - 3b - hr) / denom = (3.0 - 0.5 - 0.1 - 0.5) / 4.0 = 0.475
        assert result["singles"] == pytest.approx(0.475)
        # doubles = 0.5 / 4.0 = 0.125
        assert result["doubles"] == pytest.approx(0.125)
        # triples = 0.1 / 4.0 = 0.025
        assert result["triples"] == pytest.approx(0.025)

    def test_batter_uncovered_stats_from_marcel(self) -> None:
        adapter = PerGameToSeasonAdapter("batter")
        avg_predictions = {"hr": 0.5, "so": 2.0, "bb": 1.0, "h": 3.0, "2b": 0.5, "3b": 0.1}
        avg_denominator = 4.0
        marcel_rates = {
            "hr": 0.04, "so": 0.20, "bb": 0.10, "singles": 0.15,
            "doubles": 0.05, "triples": 0.01,
            "hbp": 0.01, "sf": 0.005, "sh": 0.002, "sb": 0.02, "cs": 0.005,
            "r": 0.12, "rbi": 0.14,
        }
        result = adapter.predictions_to_rates(avg_predictions, avg_denominator, marcel_rates)
        assert result is not None
        # Uncovered stats should come from Marcel
        assert result["hbp"] == 0.01
        assert result["sf"] == 0.005
        assert result["sh"] == 0.002
        assert result["sb"] == 0.02
        assert result["cs"] == 0.005
        assert result["r"] == 0.12
        assert result["rbi"] == 0.14

    def test_pitcher_rate_conversion(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        avg_predictions = {"so": 6.0, "h": 7.0, "bb": 2.0, "hr": 1.0}
        avg_denominator = 18.0  # avg outs/game
        marcel_rates = {
            "so": 0.20, "h": 0.25, "bb": 0.08, "hr": 0.03,
            "hbp": 0.01, "er": 0.10, "w": 0.15, "sv": 0.0, "hld": 0.0, "bs": 0.0,
        }
        result = adapter.predictions_to_rates(avg_predictions, avg_denominator, marcel_rates)
        assert result is not None

        # so = 6.0 / 18.0 = 0.333...
        assert result["so"] == pytest.approx(1 / 3)
        # h = 7.0 / 18.0
        assert result["h"] == pytest.approx(7 / 18)
        # bb = 2.0 / 18.0
        assert result["bb"] == pytest.approx(2 / 18)
        # hr = 1.0 / 18.0
        assert result["hr"] == pytest.approx(1 / 18)

    def test_pitcher_uncovered_stats_from_marcel(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        avg_predictions = {"so": 6.0, "h": 7.0, "bb": 2.0, "hr": 1.0}
        avg_denominator = 18.0
        marcel_rates = {
            "so": 0.20, "h": 0.25, "bb": 0.08, "hr": 0.03,
            "hbp": 0.01, "er": 0.10, "w": 0.15, "sv": 0.0, "hld": 0.0, "bs": 0.0,
        }
        result = adapter.predictions_to_rates(avg_predictions, avg_denominator, marcel_rates)
        assert result is not None
        assert result["hbp"] == 0.01
        assert result["er"] == 0.10
        assert result["w"] == 0.15
        assert result["sv"] == 0.0
        assert result["hld"] == 0.0
        assert result["bs"] == 0.0

    def test_zero_denominator_returns_none(self) -> None:
        adapter = PerGameToSeasonAdapter("batter")
        avg_predictions = {"hr": 0.5, "so": 2.0, "bb": 1.0, "h": 3.0, "2b": 0.5, "3b": 0.1}
        marcel_rates = {"hr": 0.04}
        result = adapter.predictions_to_rates(avg_predictions, 0.0, marcel_rates)
        assert result is None

    def test_singles_derivation(self) -> None:
        """Singles = h - 2b - 3b - hr, clamped to 0."""
        adapter = PerGameToSeasonAdapter("batter")
        # Edge case: component hits exceed total hits (model noise)
        avg_predictions = {"hr": 1.0, "so": 2.0, "bb": 1.0, "h": 1.5, "2b": 0.5, "3b": 0.2}
        avg_denominator = 4.0
        marcel_rates = {"hr": 0.04, "so": 0.20, "bb": 0.10, "singles": 0.15, "doubles": 0.05, "triples": 0.01}
        result = adapter.predictions_to_rates(avg_predictions, avg_denominator, marcel_rates)
        assert result is not None
        # singles = max(0, 1.5 - 0.5 - 0.2 - 1.0) / 4.0 = max(0, -0.2) / 4.0 = 0.0
        assert result["singles"] == pytest.approx(0.0)


class TestPredictedRatesToPipelineRates:
    def test_batter_rate_mapping(self) -> None:
        adapter = PerGameToSeasonAdapter("batter")
        predicted_rates = {"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.30, "2b": 0.05, "3b": 0.01}
        marcel_rates = {
            "hbp": 0.01, "sf": 0.005, "sh": 0.002, "sb": 0.02, "cs": 0.005,
            "r": 0.12, "rbi": 0.14,
        }
        result = adapter.predicted_rates_to_pipeline_rates(predicted_rates, marcel_rates)

        # Direct mapping — no division
        assert result["hr"] == pytest.approx(0.04)
        assert result["so"] == pytest.approx(0.20)
        assert result["bb"] == pytest.approx(0.10)
        assert result["doubles"] == pytest.approx(0.05)
        assert result["triples"] == pytest.approx(0.01)
        # singles = h - 2b - 3b - hr = 0.30 - 0.05 - 0.01 - 0.04 = 0.20
        assert result["singles"] == pytest.approx(0.20)

    def test_batter_singles_clamped_to_zero(self) -> None:
        adapter = PerGameToSeasonAdapter("batter")
        # Component hits exceed total hits
        predicted_rates = {"hr": 0.10, "so": 0.20, "bb": 0.10, "h": 0.15, "2b": 0.05, "3b": 0.02}
        marcel_rates: dict[str, float] = {}
        result = adapter.predicted_rates_to_pipeline_rates(predicted_rates, marcel_rates)
        # singles = max(0, 0.15 - 0.05 - 0.02 - 0.10) = max(0, -0.02) = 0.0
        assert result["singles"] == pytest.approx(0.0)

    def test_pitcher_rate_mapping(self) -> None:
        adapter = PerGameToSeasonAdapter("pitcher")
        predicted_rates = {"so": 0.25, "h": 0.30, "bb": 0.08, "hr": 0.03}
        marcel_rates = {
            "hbp": 0.01, "er": 0.10, "w": 0.15, "sv": 0.0, "hld": 0.0, "bs": 0.0,
        }
        result = adapter.predicted_rates_to_pipeline_rates(predicted_rates, marcel_rates)

        assert result["so"] == pytest.approx(0.25)
        assert result["h"] == pytest.approx(0.30)
        assert result["bb"] == pytest.approx(0.08)
        assert result["hr"] == pytest.approx(0.03)

    def test_uncovered_stats_from_marcel(self) -> None:
        adapter = PerGameToSeasonAdapter("batter")
        predicted_rates = {"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.30, "2b": 0.05, "3b": 0.01}
        marcel_rates = {
            "hbp": 0.01, "sf": 0.005, "sh": 0.002, "sb": 0.02, "cs": 0.005,
            "r": 0.12, "rbi": 0.14,
        }
        result = adapter.predicted_rates_to_pipeline_rates(predicted_rates, marcel_rates)

        assert result["hbp"] == 0.01
        assert result["sf"] == 0.005
        assert result["sh"] == 0.002
        assert result["sb"] == 0.02
        assert result["cs"] == 0.005
        assert result["r"] == 0.12
        assert result["rbi"] == 0.14
