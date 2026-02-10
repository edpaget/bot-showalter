"""Shared test fixtures for contextual model tests."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.contextual.data.models import (
    GameSequence,
    PitchEvent,
    PlayerContext,
)
from fantasy_baseball_manager.contextual.model.config import ModelConfig


def make_pitch(**overrides: object) -> PitchEvent:
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
        "at_bat_number": None,
    }
    defaults.update(overrides)
    return PitchEvent(**defaults)  # type: ignore[arg-type]


def make_game_sequence(
    n_pitches: int = 3,
    game_pk: int = 717465,
    game_date: str = "2024-03-28",
    season: int = 2024,
    player_id: int = 660271,
    **pitch_overrides: object,
) -> GameSequence:
    """Create a GameSequence with n_pitches pitches."""
    pitches = tuple(make_pitch(pitch_number=i + 1, **pitch_overrides) for i in range(n_pitches))
    return GameSequence(
        game_pk=game_pk,
        game_date=game_date,
        season=season,
        home_team="LAD",
        away_team="SD",
        perspective="batter",
        player_id=player_id,
        pitches=pitches,
    )


def make_player_context(
    n_games: int = 1,
    pitches_per_game: int = 3,
    player_id: int = 660271,
) -> PlayerContext:
    """Create a PlayerContext with n_games games."""
    games = tuple(
        make_game_sequence(
            n_pitches=pitches_per_game,
            game_pk=717465 + i,
            game_date=f"2024-03-{28 + i:02d}",
            player_id=player_id,
        )
        for i in range(n_games)
    )
    return PlayerContext(
        player_id=player_id,
        player_name="Shohei Ohtani",
        season=2024,
        perspective="batter",
        games=games,
    )


@pytest.fixture
def small_config() -> ModelConfig:
    """A small ModelConfig for fast tests."""
    return ModelConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        ff_dim=64,
        dropout=0.0,
        max_seq_len=128,
        pitch_type_embed_dim=8,
        pitch_result_embed_dim=6,
        bb_type_embed_dim=4,
        stand_embed_dim=4,
        p_throws_embed_dim=4,
        pa_event_embed_dim=8,
    )
