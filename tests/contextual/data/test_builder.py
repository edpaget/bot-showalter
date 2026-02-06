"""Tests for GameSequenceBuilder."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd

from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder


def _make_statcast_row(**overrides: object) -> dict[str, object]:
    """Create a dict representing one Statcast pitch row with sensible defaults."""
    defaults: dict[str, object] = {
        "game_pk": 717465,
        "game_date": "2024-03-28",
        "home_team": "LAD",
        "away_team": "SD",
        "batter": 660271,
        "pitcher": 477132,
        "player_name": "Ohtani, Shohei",
        "pitch_type": "FF",
        "description": "called_strike",
        "type": "S",
        "release_speed": 95.2,
        "release_spin_rate": 2400,
        "pfx_x": -1.2,
        "pfx_z": 9.5,
        "plate_x": 0.3,
        "plate_z": 2.8,
        "release_extension": 6.3,
        "launch_speed": np.nan,
        "launch_angle": np.nan,
        "hit_distance_sc": np.nan,
        "bb_type": np.nan,
        "estimated_woba_using_speedangle": np.nan,
        "inning": 1,
        "inning_topbot": "Top",
        "outs_when_up": 0,
        "balls": 0,
        "strikes": 0,
        "on_1b": np.nan,
        "on_2b": np.nan,
        "on_3b": np.nan,
        "home_score": 0,
        "away_score": 0,
        "stand": "L",
        "p_throws": "R",
        "at_bat_number": 1,
        "pitch_number": 1,
        "events": np.nan,
        "delta_run_exp": -0.04,
    }
    defaults.update(overrides)
    return defaults


def _make_store(rows: list[dict[str, object]]) -> Mock:
    """Create a mock StatcastStore that returns a DataFrame from rows."""
    store = Mock()
    store.read_season.return_value = pd.DataFrame(rows)
    return store


class TestGameSequenceBuilder:
    """Tests for GameSequenceBuilder."""

    def test_single_game_single_batter(self) -> None:
        """Single game with one batter should produce one GameSequence."""
        rows = [
            _make_statcast_row(at_bat_number=1, pitch_number=1, description="ball", type="B"),
            _make_statcast_row(
                at_bat_number=1,
                pitch_number=2,
                description="hit_into_play",
                type="X",
                events="single",
                launch_speed=98.5,
                launch_angle=15,
                hit_distance_sc=180,
                bb_type="line_drive",
                estimated_woba_using_speedangle=0.850,
            ),
        ]
        store = _make_store(rows)
        builder = GameSequenceBuilder(store)

        result = builder.build_season(2024, perspective="batter")

        assert len(result) == 1
        gs = result[0]
        assert gs.game_pk == 717465
        assert gs.game_date == "2024-03-28"
        assert gs.season == 2024
        assert gs.home_team == "LAD"
        assert gs.away_team == "SD"
        assert gs.perspective == "batter"
        assert gs.player_id == 660271
        assert len(gs.pitches) == 2

        # First pitch
        p1 = gs.pitches[0]
        assert p1.pitch_type == "FF"
        assert p1.pitch_result == "ball"
        assert p1.pitch_result_type == "B"
        assert p1.batter_id == 660271
        assert p1.pitcher_id == 477132
        assert p1.launch_speed is None
        assert p1.pa_event is None

        # Second pitch (with batted ball data and PA event)
        p2 = gs.pitches[1]
        assert p2.pitch_result == "hit_into_play"
        assert p2.launch_speed == 98.5
        assert p2.launch_angle == 15
        assert p2.hit_distance == 180
        assert p2.bb_type == "line_drive"
        assert p2.estimated_woba == 0.850
        assert p2.pa_event == "single"

    def test_runner_encoding(self) -> None:
        """Runners on base should be encoded as booleans from MLBAM IDs."""
        rows = [
            _make_statcast_row(on_1b=592450.0, on_2b=np.nan, on_3b=545361.0),
        ]
        store = _make_store(rows)
        builder = GameSequenceBuilder(store)

        result = builder.build_season(2024, perspective="batter")
        pitch = result[0].pitches[0]
        assert pitch.runners_on_1b is True
        assert pitch.runners_on_2b is False
        assert pitch.runners_on_3b is True

    def test_missing_tracking_data_becomes_none(self) -> None:
        """NaN tracking data should map to None in PitchEvent."""
        rows = [
            _make_statcast_row(
                release_speed=np.nan,
                release_spin_rate=np.nan,
                pfx_x=np.nan,
                pfx_z=np.nan,
                plate_x=np.nan,
                plate_z=np.nan,
                release_extension=np.nan,
                delta_run_exp=np.nan,
            ),
        ]
        store = _make_store(rows)
        builder = GameSequenceBuilder(store)

        result = builder.build_season(2024, perspective="batter")
        pitch = result[0].pitches[0]
        assert pitch.release_speed is None
        assert pitch.release_spin_rate is None
        assert pitch.pfx_x is None
        assert pitch.pfx_z is None
        assert pitch.plate_x is None
        assert pitch.plate_z is None
        assert pitch.release_extension is None
        assert pitch.delta_run_exp is None

    def test_multi_game_grouping(self) -> None:
        """Pitches from different games should produce separate GameSequences."""
        rows = [
            _make_statcast_row(game_pk=717465, game_date="2024-03-28"),
            _make_statcast_row(game_pk=717500, game_date="2024-03-29", home_team="SF", away_team="LAD"),
        ]
        store = _make_store(rows)
        builder = GameSequenceBuilder(store)

        result = builder.build_season(2024, perspective="batter")
        assert len(result) == 2
        game_pks = {gs.game_pk for gs in result}
        assert game_pks == {717465, 717500}

    def test_pitcher_perspective(self) -> None:
        """Builder should group by pitcher when perspective='pitcher'."""
        rows = [
            _make_statcast_row(batter=111, pitcher=222),
            _make_statcast_row(batter=333, pitcher=222),
        ]
        store = _make_store(rows)
        builder = GameSequenceBuilder(store)

        result = builder.build_season(2024, perspective="pitcher")
        assert len(result) == 1
        gs = result[0]
        assert gs.player_id == 222
        assert gs.perspective == "pitcher"
        assert len(gs.pitches) == 2

    def test_build_player_season_filters_to_player(self) -> None:
        """build_player_season should only return games for the given player."""
        rows = [
            _make_statcast_row(batter=660271),
            _make_statcast_row(batter=592450),
        ]
        store = _make_store(rows)
        builder = GameSequenceBuilder(store)

        result = builder.build_player_season(2024, 660271, perspective="batter")
        assert len(result) == 1
        assert result[0].player_id == 660271

    def test_empty_season(self) -> None:
        """Empty DataFrame should return empty list."""
        store = Mock()
        store.read_season.return_value = pd.DataFrame()
        builder = GameSequenceBuilder(store)

        result = builder.build_season(2024, perspective="batter")
        assert result == []

    def test_null_pitch_type_rows_dropped(self) -> None:
        """Rows with null pitch_type should be excluded."""
        rows = [
            _make_statcast_row(pitch_type="FF"),
            _make_statcast_row(pitch_type=np.nan, at_bat_number=2, pitch_number=1),
        ]
        store = _make_store(rows)
        builder = GameSequenceBuilder(store)

        result = builder.build_season(2024, perspective="batter")
        total_pitches = sum(len(gs.pitches) for gs in result)
        assert total_pitches == 1

    def test_bat_score_fld_score_top_inning(self) -> None:
        """In Top inning, batting team is away, fielding team is home."""
        rows = [
            _make_statcast_row(
                inning_topbot="Top",
                home_score=3,
                away_score=1,
            ),
        ]
        store = _make_store(rows)
        builder = GameSequenceBuilder(store)

        result = builder.build_season(2024, perspective="batter")
        pitch = result[0].pitches[0]
        assert pitch.is_top is True
        assert pitch.bat_score == 1  # away team is batting in top
        assert pitch.fld_score == 3  # home team is fielding in top

    def test_bat_score_fld_score_bottom_inning(self) -> None:
        """In Bottom inning, batting team is home, fielding team is away."""
        rows = [
            _make_statcast_row(
                inning_topbot="Bot",
                home_score=3,
                away_score=1,
            ),
        ]
        store = _make_store(rows)
        builder = GameSequenceBuilder(store)

        result = builder.build_season(2024, perspective="batter")
        pitch = result[0].pitches[0]
        assert pitch.is_top is False
        assert pitch.bat_score == 3  # home team is batting in bottom
        assert pitch.fld_score == 1  # away team is fielding in bottom

    def test_pitch_ordering_within_game(self) -> None:
        """Pitches should be ordered by at_bat_number then pitch_number."""
        rows = [
            _make_statcast_row(at_bat_number=2, pitch_number=1, strikes=0),
            _make_statcast_row(at_bat_number=1, pitch_number=2, strikes=1),
            _make_statcast_row(at_bat_number=1, pitch_number=1, strikes=0),
        ]
        store = _make_store(rows)
        builder = GameSequenceBuilder(store)

        result = builder.build_season(2024, perspective="batter")
        pitches = result[0].pitches
        assert len(pitches) == 3
        assert pitches[0].pitch_number == 1  # AB 1, pitch 1
        assert pitches[1].pitch_number == 2  # AB 1, pitch 2
        assert pitches[2].pitch_number == 1  # AB 2, pitch 1

    def test_season_passed_to_store(self) -> None:
        """build_season should call store.read_season with the correct year."""
        store = _make_store([])
        builder = GameSequenceBuilder(store)

        builder.build_season(2023, perspective="batter")

        store.read_season.assert_called_once_with(2023)

    def test_build_player_season_pitcher_perspective(self) -> None:
        """build_player_season with pitcher perspective filters by pitcher column."""
        rows = [
            _make_statcast_row(batter=111, pitcher=222),
            _make_statcast_row(batter=333, pitcher=444),
        ]
        store = _make_store(rows)
        builder = GameSequenceBuilder(store)

        result = builder.build_player_season(2024, 222, perspective="pitcher")
        assert len(result) == 1
        assert result[0].player_id == 222
