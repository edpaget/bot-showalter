"""Tests for PitchSequenceDataSource."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from fantasy_baseball_manager.context import init_context
from fantasy_baseball_manager.contextual.data.models import (
    GameSequence,
    PitchEvent,
    PlayerContext,
)
from fantasy_baseball_manager.contextual.data.source import PitchSequenceDataSource
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.player.identity import Player


def _make_pitch(**overrides: object) -> PitchEvent:
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


def _make_game_sequence(game_pk: int = 717465) -> GameSequence:
    return GameSequence(
        game_pk=game_pk,
        game_date="2024-03-28",
        season=2024,
        home_team="LAD",
        away_team="SD",
        perspective="batter",
        player_id=660271,
        pitches=(_make_pitch(),),
    )


@pytest.fixture(autouse=True)
def _setup_context() -> None:
    init_context(year=2024)


@pytest.fixture
def mock_builder() -> Mock:
    builder = Mock()
    builder.build_player_season.return_value = [_make_game_sequence()]
    return builder


@pytest.fixture
def player() -> Player:
    return Player(name="Shohei Ohtani", yahoo_id="10835", mlbam_id="660271")


class TestPitchSequenceDataSource:
    """Tests for PitchSequenceDataSource."""

    def test_single_player_returns_ok(self, mock_builder: Mock, player: Player) -> None:
        source = PitchSequenceDataSource(builder=mock_builder)
        result = source(player)

        assert result.is_ok()
        ctx = result.unwrap()
        assert isinstance(ctx, PlayerContext)
        assert ctx.player_id == 660271
        assert len(ctx.games) == 1

    def test_missing_mlbam_id_returns_err(self, mock_builder: Mock) -> None:
        player = Player(name="Unknown", yahoo_id="9999")
        source = PitchSequenceDataSource(builder=mock_builder)

        result = source(player)

        assert result.is_err()
        error = result.unwrap_err()
        assert isinstance(error, DataSourceError)
        assert "mlbam_id" in error.message

    def test_all_players_returns_err(self, mock_builder: Mock) -> None:
        source = PitchSequenceDataSource(builder=mock_builder)

        result = source(ALL_PLAYERS)

        assert result.is_err()
        error = result.unwrap_err()
        assert isinstance(error, DataSourceError)

    def test_list_query_delegates_to_single(self, mock_builder: Mock, player: Player) -> None:
        source = PitchSequenceDataSource(builder=mock_builder)

        result = source([player])

        assert result.is_ok()
        contexts = result.unwrap()
        assert isinstance(contexts, list)
        assert len(contexts) == 1
        assert contexts[0].player_id == 660271

    def test_game_window_truncation(self, mock_builder: Mock, player: Player) -> None:
        mock_builder.build_player_season.return_value = [
            _make_game_sequence(game_pk=1),
            _make_game_sequence(game_pk=2),
            _make_game_sequence(game_pk=3),
        ]
        source = PitchSequenceDataSource(builder=mock_builder, game_window=2)

        result = source(player)

        assert result.is_ok()
        ctx = result.unwrap()
        assert len(ctx.games) == 2
        # Should keep the last N games
        assert ctx.games[0].game_pk == 2
        assert ctx.games[1].game_pk == 3

    def test_builder_exception_wrapped_in_err(self, mock_builder: Mock, player: Player) -> None:
        mock_builder.build_player_season.side_effect = RuntimeError("disk error")
        source = PitchSequenceDataSource(builder=mock_builder)

        result = source(player)

        assert result.is_err()
        error = result.unwrap_err()
        assert isinstance(error, DataSourceError)
        assert isinstance(error.cause, RuntimeError)

    def test_perspective_forwarded_to_builder(self, mock_builder: Mock, player: Player) -> None:
        source = PitchSequenceDataSource(builder=mock_builder, perspective="pitcher")
        source(player)

        mock_builder.build_player_season.assert_called_once_with(2024, 660271, perspective="pitcher")

    def test_context_year_used(self, mock_builder: Mock, player: Player) -> None:
        init_context(year=2023)
        source = PitchSequenceDataSource(builder=mock_builder)
        source(player)

        mock_builder.build_player_season.assert_called_once_with(2023, 660271, perspective="batter")

    def test_cache_is_used_on_hit(self, mock_builder: Mock, player: Player) -> None:
        """When cache has data, builder should not be called."""
        ctx = PlayerContext(
            player_id=660271,
            player_name="Shohei Ohtani",
            season=2024,
            perspective="batter",
            games=(_make_game_sequence(),),
        )
        cache = Mock()
        cache.get.return_value = ctx
        source = PitchSequenceDataSource(builder=mock_builder, cache=cache)

        result = source(player)

        assert result.is_ok()
        mock_builder.build_player_season.assert_not_called()

    def test_cache_miss_falls_through_to_builder(self, mock_builder: Mock, player: Player) -> None:
        """When cache misses, builder should be called and result cached."""
        cache = Mock()
        cache.get.return_value = None
        source = PitchSequenceDataSource(builder=mock_builder, cache=cache)

        result = source(player)

        assert result.is_ok()
        mock_builder.build_player_season.assert_called_once()
        cache.put.assert_called_once()

    def test_empty_builder_result(self, mock_builder: Mock, player: Player) -> None:
        """Builder returning empty list should still produce Ok with empty games."""
        mock_builder.build_player_season.return_value = []
        source = PitchSequenceDataSource(builder=mock_builder)

        result = source(player)

        assert result.is_ok()
        ctx = result.unwrap()
        assert len(ctx.games) == 0
