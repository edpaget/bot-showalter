"""Data models for contextual pitch event sequences.

Provides three frozen dataclasses representing pitch-level data at increasing
levels of aggregation: individual pitches, game sequences, and player contexts.

All missing values use None (not imputed). The tensor layer handles masking
in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PitchEvent:
    """A single pitch with tracking and gamestate data.

    Fields use absolute gamestate (not deltas). Deltas are trivially computed
    at tensor-creation time and keeping absolutes decouples data pipeline
    from model architecture.
    """

    # Identity
    batter_id: int
    pitcher_id: int

    # Pitch categorical
    pitch_type: str
    pitch_result: str
    pitch_result_type: str

    # Pitch continuous
    release_speed: float | None
    release_spin_rate: int | None
    pfx_x: float | None
    pfx_z: float | None
    plate_x: float | None
    plate_z: float | None
    release_extension: float | None

    # Batted ball
    launch_speed: float | None
    launch_angle: int | None
    hit_distance: int | None
    bb_type: str | None
    estimated_woba: float | None

    # Gamestate
    inning: int
    is_top: bool
    outs: int
    balls: int
    strikes: int
    runners_on_1b: bool
    runners_on_2b: bool
    runners_on_3b: bool
    bat_score: int
    fld_score: int

    # Context
    stand: str
    p_throws: str
    pitch_number: int

    # PA outcome (only on last pitch of PA)
    pa_event: str | None

    # Run expectancy
    delta_run_exp: float | None

    # Plate appearance identifier (optional, from Statcast at_bat_number column)
    at_bat_number: int | None = None


@dataclass(frozen=True, slots=True)
class GameSequence:
    """Ordered pitches for one game from one player's perspective.

    Pitches are ordered by at_bat_number then pitch_number.
    """

    game_pk: int
    game_date: str
    season: int
    home_team: str
    away_team: str
    perspective: str
    player_id: int
    pitches: tuple[PitchEvent, ...]


@dataclass(frozen=True, slots=True)
class PlayerContext:
    """N games of pitch sequences for one player.

    Games are in chronological order.
    """

    player_id: int
    player_name: str
    season: int
    perspective: str
    games: tuple[GameSequence, ...]
