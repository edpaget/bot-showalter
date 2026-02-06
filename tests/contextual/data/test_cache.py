"""Tests for SequenceCache."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from fantasy_baseball_manager.contextual.data.cache import SequenceCache
from fantasy_baseball_manager.contextual.data.models import (
    GameSequence,
    PitchEvent,
    PlayerContext,
)


def _make_pitch(**overrides: object) -> PitchEvent:
    """Create a PitchEvent with sensible defaults."""
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


def _make_context(
    player_id: int = 660271,
    player_name: str = "Shohei Ohtani",
    season: int = 2024,
    perspective: str = "batter",
    num_games: int = 1,
    pitches_per_game: int = 2,
) -> PlayerContext:
    """Create a PlayerContext with configurable size."""
    games: list[GameSequence] = []
    for g in range(num_games):
        pitches = tuple(_make_pitch(pitch_number=p + 1, inning=g + 1) for p in range(pitches_per_game))
        games.append(
            GameSequence(
                game_pk=717465 + g,
                game_date=f"2024-03-{28 + g:02d}",
                season=season,
                home_team="LAD",
                away_team="SD",
                perspective=perspective,
                player_id=player_id,
                pitches=pitches,
            )
        )
    return PlayerContext(
        player_id=player_id,
        player_name=player_name,
        season=season,
        perspective=perspective,
        games=tuple(games),
    )


class TestSequenceCache:
    """Tests for SequenceCache."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """put then get should return equivalent PlayerContext."""
        cache = SequenceCache(cache_dir=tmp_path)
        ctx = _make_context()

        cache.put(ctx)
        result = cache.get(2024, 660271, "batter")

        assert result is not None
        assert result.player_id == ctx.player_id
        assert result.player_name == ctx.player_name
        assert result.season == ctx.season
        assert result.perspective == ctx.perspective
        assert len(result.games) == len(ctx.games)

        # Verify pitch data survives round-trip
        orig_pitch = ctx.games[0].pitches[0]
        cached_pitch = result.games[0].pitches[0]
        assert cached_pitch.pitch_type == orig_pitch.pitch_type
        assert cached_pitch.batter_id == orig_pitch.batter_id
        assert cached_pitch.release_speed == orig_pitch.release_speed
        assert cached_pitch.is_top == orig_pitch.is_top
        assert cached_pitch.runners_on_1b == orig_pitch.runners_on_1b

    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        """get on empty cache should return None."""
        cache = SequenceCache(cache_dir=tmp_path)
        result = cache.get(2024, 660271, "batter")
        assert result is None

    def test_has_before_put(self, tmp_path: Path) -> None:
        """has() should return False before any put."""
        cache = SequenceCache(cache_dir=tmp_path)
        assert cache.has(2024, 660271, "batter") is False

    def test_has_after_put(self, tmp_path: Path) -> None:
        """has() should return True after put."""
        cache = SequenceCache(cache_dir=tmp_path)
        cache.put(_make_context())
        assert cache.has(2024, 660271, "batter") is True

    def test_invalidate_by_player(self, tmp_path: Path) -> None:
        """invalidate with player_id removes only that player."""
        cache = SequenceCache(cache_dir=tmp_path)
        cache.put(_make_context(player_id=660271))
        cache.put(_make_context(player_id=592450, player_name="Aaron Judge"))

        cache.invalidate(2024, player_id=660271, perspective="batter")

        assert cache.has(2024, 660271, "batter") is False
        assert cache.has(2024, 592450, "batter") is True

    def test_invalidate_by_season(self, tmp_path: Path) -> None:
        """invalidate with only season removes all for that season."""
        cache = SequenceCache(cache_dir=tmp_path)
        cache.put(_make_context(season=2024))
        cache.put(_make_context(season=2023))

        cache.invalidate(2024)

        assert cache.has(2024, 660271, "batter") is False
        assert cache.has(2023, 660271, "batter") is True

    def test_none_fields_survive_round_trip(self, tmp_path: Path) -> None:
        """None optional fields should remain None after cache round-trip."""
        ctx = _make_context()
        cache = SequenceCache(cache_dir=tmp_path)
        cache.put(ctx)

        result = cache.get(2024, 660271, "batter")
        assert result is not None
        pitch = result.games[0].pitches[0]
        assert pitch.launch_speed is None
        assert pitch.launch_angle is None
        assert pitch.hit_distance is None
        assert pitch.bb_type is None
        assert pitch.estimated_woba is None
        assert pitch.pa_event is None

    def test_multi_game_ordering_preserved(self, tmp_path: Path) -> None:
        """Games should maintain chronological order after round-trip."""
        ctx = _make_context(num_games=3)
        cache = SequenceCache(cache_dir=tmp_path)
        cache.put(ctx)

        result = cache.get(2024, 660271, "batter")
        assert result is not None
        game_pks = [g.game_pk for g in result.games]
        assert game_pks == [717465, 717466, 717467]

    def test_batted_ball_fields_survive_round_trip(self, tmp_path: Path) -> None:
        """Batted ball fields with values should survive round-trip."""
        pitch = _make_pitch(
            launch_speed=105.3,
            launch_angle=28,
            hit_distance=410,
            bb_type="fly_ball",
            estimated_woba=1.120,
            pa_event="home_run",
        )
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
        ctx = PlayerContext(
            player_id=660271,
            player_name="Shohei Ohtani",
            season=2024,
            perspective="batter",
            games=(gs,),
        )
        cache = SequenceCache(cache_dir=tmp_path)
        cache.put(ctx)

        result = cache.get(2024, 660271, "batter")
        assert result is not None
        p = result.games[0].pitches[0]
        assert p.launch_speed == 105.3
        assert p.launch_angle == 28
        assert p.hit_distance == 410
        assert p.bb_type == "fly_ball"
        assert p.estimated_woba == pytest.approx(1.120)
        assert p.pa_event == "home_run"
