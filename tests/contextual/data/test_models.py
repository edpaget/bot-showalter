"""Tests for contextual pitch event data models."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.contextual.data.models import (
    GameSequence,
    PitchEvent,
    PlayerContext,
)


def _make_pitch(**overrides: object) -> PitchEvent:
    """Create a PitchEvent with sensible defaults, overridable per-field."""
    defaults: dict[str, object] = {
        "batter_id": 660271,
        "pitcher_id": 477132,
        "pitch_type": "FF",
        "pitch_result": "called_strike",
        "pitch_result_type": "S",
        "release_speed": 95.2,
        "release_spin_rate": 2400,
        "pfx_x": -1.2,
        "pfx_z": 9.5,
        "plate_x": 0.3,
        "plate_z": 2.8,
        "release_extension": 6.3,
        "launch_speed": None,
        "launch_angle": None,
        "hit_distance": None,
        "bb_type": None,
        "estimated_woba": None,
        "inning": 1,
        "is_top": True,
        "outs": 0,
        "balls": 0,
        "strikes": 0,
        "runners_on_1b": False,
        "runners_on_2b": False,
        "runners_on_3b": False,
        "bat_score": 0,
        "fld_score": 0,
        "stand": "L",
        "p_throws": "R",
        "pitch_number": 1,
        "pa_event": None,
        "delta_run_exp": -0.04,
    }
    defaults.update(overrides)
    return PitchEvent(**defaults)  # type: ignore[arg-type]


class TestPitchEvent:
    """Tests for the PitchEvent frozen dataclass."""

    def test_creation_with_all_fields(self) -> None:
        pitch = _make_pitch()
        assert pitch.batter_id == 660271
        assert pitch.pitcher_id == 477132
        assert pitch.pitch_type == "FF"
        assert pitch.release_speed == 95.2
        assert pitch.inning == 1
        assert pitch.is_top is True

    def test_frozen_immutability(self) -> None:
        pitch = _make_pitch()
        with pytest.raises(AttributeError):
            pitch.pitch_type = "SL"  # type: ignore[misc]

    def test_none_fields(self) -> None:
        pitch = _make_pitch(
            launch_speed=None,
            launch_angle=None,
            hit_distance=None,
            bb_type=None,
            estimated_woba=None,
            pa_event=None,
            delta_run_exp=None,
        )
        assert pitch.launch_speed is None
        assert pitch.launch_angle is None
        assert pitch.hit_distance is None
        assert pitch.bb_type is None
        assert pitch.estimated_woba is None
        assert pitch.pa_event is None
        assert pitch.delta_run_exp is None

    def test_equality(self) -> None:
        p1 = _make_pitch()
        p2 = _make_pitch()
        assert p1 == p2

    def test_inequality_on_different_values(self) -> None:
        p1 = _make_pitch(pitch_type="FF")
        p2 = _make_pitch(pitch_type="SL")
        assert p1 != p2

    def test_hashable(self) -> None:
        pitch = _make_pitch()
        assert isinstance(hash(pitch), int)
        assert {pitch} == {pitch}

    def test_batted_ball_fields_populated(self) -> None:
        pitch = _make_pitch(
            launch_speed=105.3,
            launch_angle=28,
            hit_distance=410,
            bb_type="fly_ball",
            estimated_woba=1.120,
        )
        assert pitch.launch_speed == 105.3
        assert pitch.launch_angle == 28
        assert pitch.hit_distance == 410
        assert pitch.bb_type == "fly_ball"
        assert pitch.estimated_woba == 1.120

    def test_pa_event_on_last_pitch(self) -> None:
        pitch = _make_pitch(pa_event="home_run")
        assert pitch.pa_event == "home_run"


class TestGameSequence:
    """Tests for the GameSequence frozen dataclass."""

    def test_creation(self) -> None:
        pitch = _make_pitch()
        gs = GameSequence(
            game_pk=717465,
            game_date="2024-03-28",
            season=2024,
            home_team="LAD",
            away_team="SD",
            perspective="batter",
            player_id=660271,
            pitches=(pitch,),
        )
        assert gs.game_pk == 717465
        assert gs.game_date == "2024-03-28"
        assert gs.season == 2024
        assert gs.home_team == "LAD"
        assert gs.away_team == "SD"
        assert gs.perspective == "batter"
        assert gs.player_id == 660271
        assert len(gs.pitches) == 1
        assert gs.pitches[0] is pitch

    def test_frozen_immutability(self) -> None:
        gs = GameSequence(
            game_pk=717465,
            game_date="2024-03-28",
            season=2024,
            home_team="LAD",
            away_team="SD",
            perspective="batter",
            player_id=660271,
            pitches=(),
        )
        with pytest.raises(AttributeError):
            gs.game_pk = 0  # type: ignore[misc]

    def test_empty_pitches(self) -> None:
        gs = GameSequence(
            game_pk=717465,
            game_date="2024-03-28",
            season=2024,
            home_team="LAD",
            away_team="SD",
            perspective="batter",
            player_id=660271,
            pitches=(),
        )
        assert gs.pitches == ()

    def test_equality(self) -> None:
        pitch = _make_pitch()
        gs1 = GameSequence(
            game_pk=717465,
            game_date="2024-03-28",
            season=2024,
            home_team="LAD",
            away_team="SD",
            perspective="batter",
            player_id=660271,
            pitches=(pitch,),
        )
        gs2 = GameSequence(
            game_pk=717465,
            game_date="2024-03-28",
            season=2024,
            home_team="LAD",
            away_team="SD",
            perspective="batter",
            player_id=660271,
            pitches=(pitch,),
        )
        assert gs1 == gs2

    def test_hashable(self) -> None:
        gs = GameSequence(
            game_pk=717465,
            game_date="2024-03-28",
            season=2024,
            home_team="LAD",
            away_team="SD",
            perspective="batter",
            player_id=660271,
            pitches=(),
        )
        assert isinstance(hash(gs), int)


class TestPlayerContext:
    """Tests for the PlayerContext frozen dataclass."""

    def test_creation(self) -> None:
        gs = GameSequence(
            game_pk=717465,
            game_date="2024-03-28",
            season=2024,
            home_team="LAD",
            away_team="SD",
            perspective="batter",
            player_id=660271,
            pitches=(_make_pitch(),),
        )
        ctx = PlayerContext(
            player_id=660271,
            player_name="Shohei Ohtani",
            season=2024,
            perspective="batter",
            games=(gs,),
        )
        assert ctx.player_id == 660271
        assert ctx.player_name == "Shohei Ohtani"
        assert ctx.season == 2024
        assert ctx.perspective == "batter"
        assert len(ctx.games) == 1

    def test_frozen_immutability(self) -> None:
        ctx = PlayerContext(
            player_id=660271,
            player_name="Shohei Ohtani",
            season=2024,
            perspective="batter",
            games=(),
        )
        with pytest.raises(AttributeError):
            ctx.player_id = 0  # type: ignore[misc]

    def test_empty_games(self) -> None:
        ctx = PlayerContext(
            player_id=660271,
            player_name="Shohei Ohtani",
            season=2024,
            perspective="batter",
            games=(),
        )
        assert ctx.games == ()

    def test_equality(self) -> None:
        ctx1 = PlayerContext(
            player_id=660271,
            player_name="Shohei Ohtani",
            season=2024,
            perspective="batter",
            games=(),
        )
        ctx2 = PlayerContext(
            player_id=660271,
            player_name="Shohei Ohtani",
            season=2024,
            perspective="batter",
            games=(),
        )
        assert ctx1 == ctx2

    def test_hashable(self) -> None:
        ctx = PlayerContext(
            player_id=660271,
            player_name="Shohei Ohtani",
            season=2024,
            perspective="batter",
            games=(),
        )
        assert isinstance(hash(ctx), int)
