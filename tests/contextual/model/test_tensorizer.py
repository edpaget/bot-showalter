"""Tests for the Tensorizer that converts PitchEvents to tensors."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.contextual.data.models import PlayerContext
from fantasy_baseball_manager.contextual.data.vocab import (
    BB_TYPE_VOCAB,
    HANDEDNESS_VOCAB,
    PA_EVENT_VOCAB,
    PITCH_RESULT_VOCAB,
    PITCH_TYPE_VOCAB,
)
from fantasy_baseball_manager.contextual.model.config import ModelConfig
from fantasy_baseball_manager.contextual.model.tensorizer import (
    Tensorizer,
)

from .conftest import make_game_sequence, make_pitch, make_player_context


def _make_tensorizer(config: ModelConfig | None = None) -> Tensorizer:
    """Create a Tensorizer with real vocabs and optional config."""
    if config is None:
        config = ModelConfig(max_seq_len=128)
    return Tensorizer(
        config=config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )


class TestCategoricalEncoding:
    """Tests for categorical field encoding."""

    def test_known_pitch_type_encodes_to_expected_index(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=1)
        result = tensorizer.tensorize_context(ctx)
        # Pitch at position 2 (after CLS at 0 and player token at 1)
        assert result.pitch_type_ids[2].item() == PITCH_TYPE_VOCAB.encode("FF")

    def test_unknown_pitch_type_encodes_to_unk(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=1)
        # Override with unknown type
        pitch = make_pitch(pitch_type="UNKNOWN_TYPE")
        game = make_game_sequence(n_pitches=0)
        from dataclasses import replace

        game = replace(game, pitches=(pitch,))
        ctx = replace(ctx, games=(game,))
        result = tensorizer.tensorize_context(ctx)
        assert result.pitch_type_ids[2].item() == 1  # UNK index

    def test_known_pitch_result_encodes_correctly(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=1)
        result = tensorizer.tensorize_context(ctx)
        assert result.pitch_result_ids[2].item() == PITCH_RESULT_VOCAB.encode("called_strike")

    def test_none_bb_type_encodes_to_pad(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=1)
        result = tensorizer.tensorize_context(ctx)
        assert result.bb_type_ids[2].item() == 0  # PAD for None

    def test_present_bb_type_encodes_correctly(self) -> None:
        tensorizer = _make_tensorizer()
        pitch = make_pitch(bb_type="fly_ball")
        game = make_game_sequence(n_pitches=0)
        from dataclasses import replace

        game = replace(game, pitches=(pitch,))
        ctx = PlayerContext(
            player_id=660271,
            player_name="Shohei Ohtani",
            season=2024,
            perspective="batter",
            games=(game,),
        )
        result = tensorizer.tensorize_context(ctx)
        assert result.bb_type_ids[2].item() == BB_TYPE_VOCAB.encode("fly_ball")

    def test_none_pa_event_encodes_to_pad(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=1)
        result = tensorizer.tensorize_context(ctx)
        assert result.pa_event_ids[2].item() == 0  # PAD for None

    def test_present_pa_event_encodes_correctly(self) -> None:
        tensorizer = _make_tensorizer()
        pitch = make_pitch(pa_event="home_run")
        game = make_game_sequence(n_pitches=0)
        from dataclasses import replace

        game = replace(game, pitches=(pitch,))
        ctx = PlayerContext(
            player_id=660271,
            player_name="Shohei Ohtani",
            season=2024,
            perspective="batter",
            games=(game,),
        )
        result = tensorizer.tensorize_context(ctx)
        assert result.pa_event_ids[2].item() == PA_EVENT_VOCAB.encode("home_run")


class TestNumericEncoding:
    """Tests for numeric field encoding."""

    def test_present_values_become_float_with_mask_true(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=1)
        result = tensorizer.tensorize_context(ctx)
        # release_speed (idx 0) is 95.2 — pitch at position 2 (after CLS + player)
        assert result.numeric_features[2, 0].item() == pytest.approx(95.2)
        assert result.numeric_mask[2, 0].item() is True

    def test_none_values_become_zero_with_mask_false(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=1)
        result = tensorizer.tensorize_context(ctx)
        # launch_speed (idx 7) is None in default pitch — pitch at position 2
        assert result.numeric_features[2, 7].item() == 0.0
        assert result.numeric_mask[2, 7].item() is False

    def test_bool_fields_convert_to_float(self) -> None:
        tensorizer = _make_tensorizer()
        pitch = make_pitch(is_top=True, runners_on_1b=True, runners_on_2b=False)
        game = make_game_sequence(n_pitches=0)
        from dataclasses import replace

        game = replace(game, pitches=(pitch,))
        ctx = PlayerContext(
            player_id=660271,
            player_name="Shohei Ohtani",
            season=2024,
            perspective="batter",
            games=(game,),
        )
        result = tensorizer.tensorize_context(ctx)
        # is_top (idx 18) — pitch at position 2
        assert result.numeric_features[2, 18].item() == 1.0
        # runners_on_1b (idx 19)
        assert result.numeric_features[2, 19].item() == 1.0
        # runners_on_2b (idx 20)
        assert result.numeric_features[2, 20].item() == 0.0

    def test_all_23_numeric_features(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=1)
        result = tensorizer.tensorize_context(ctx)
        assert result.numeric_features.shape[1] == 23
        assert result.numeric_mask.shape[1] == 23


class TestTensorizeSingle:
    """Tests for tensorize_context producing TensorizedSingle."""

    def test_one_game_three_pitches_shape(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=3)
        result = tensorizer.tensorize_context(ctx)
        # 1 CLS + 1 player token + 3 pitches = 5
        assert result.seq_length == 5
        assert result.pitch_type_ids.shape == (5,)
        assert result.numeric_features.shape == (5, 23)
        assert result.numeric_mask.shape == (5, 23)
        assert result.padding_mask.shape == (5,)
        assert result.player_token_mask.shape == (5,)
        assert result.game_ids.shape == (5,)

    def test_cls_at_position_zero(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=3)
        result = tensorizer.tensorize_context(ctx)
        # CLS at position 0: not a player token, game_id=-1
        assert result.player_token_mask[0].item() is False
        assert result.padding_mask[0].item() is True
        assert result.game_ids[0].item() == -1
        assert result.pitch_type_ids[0].item() == 0
        assert result.pitch_result_ids[0].item() == 0
        assert (result.numeric_features[0] == 0.0).all()
        assert (result.numeric_mask[0] == False).all()  # noqa: E712

    def test_player_token_at_position_one(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=3)
        result = tensorizer.tensorize_context(ctx)
        # Player token at position 1 (after CLS)
        assert result.player_token_mask[0].item() is False  # CLS
        assert result.player_token_mask[1].item() is True   # player
        assert result.player_token_mask[2].item() is False
        assert result.player_token_mask[3].item() is False
        assert result.player_token_mask[4].item() is False

    def test_player_token_categoricals_are_pad(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=1)
        result = tensorizer.tensorize_context(ctx)
        # Player token at position 1 (after CLS)
        assert result.pitch_type_ids[1].item() == 0
        assert result.pitch_result_ids[1].item() == 0
        assert result.bb_type_ids[1].item() == 0
        assert result.stand_ids[1].item() == 0
        assert result.p_throws_ids[1].item() == 0
        assert result.pa_event_ids[1].item() == 0

    def test_player_token_numerics_are_zero_masked(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=1)
        result = tensorizer.tensorize_context(ctx)
        # Player token at position 1
        assert (result.numeric_features[1] == 0.0).all()
        assert (result.numeric_mask[1] == False).all()  # noqa: E712

    def test_padding_mask_all_true(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=1, pitches_per_game=3)
        result = tensorizer.tensorize_context(ctx)
        assert result.padding_mask.all()

    def test_two_games_player_tokens_at_correct_positions(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=2, pitches_per_game=3)
        result = tensorizer.tensorize_context(ctx)
        # [CLS] [P0] p0 p1 p2 [P1] p3 p4 p5 → 9 positions
        assert result.seq_length == 9
        assert result.player_token_mask[0].item() is False  # CLS
        assert result.player_token_mask[1].item() is True   # P0
        assert result.player_token_mask[5].item() is True   # P1
        # Non-player positions
        assert result.player_token_mask[2].item() is False
        assert result.player_token_mask[6].item() is False

    def test_two_games_game_ids_increment(self) -> None:
        tensorizer = _make_tensorizer()
        ctx = make_player_context(n_games=2, pitches_per_game=3)
        result = tensorizer.tensorize_context(ctx)
        # CLS at position 0 has game_id=-1
        assert result.game_ids[0].item() == -1
        # Game 0 positions (1..4)
        assert (result.game_ids[1:5] == 0).all()
        # Game 1 positions (5..8)
        assert (result.game_ids[5:] == 1).all()


class TestTruncation:
    """Tests for sequence truncation."""

    def test_truncation_drops_oldest_games(self) -> None:
        config = ModelConfig(max_seq_len=10)
        tensorizer = _make_tensorizer(config)
        # 1 CLS + 3 games x (1 player + 3 pitches) = 13, exceeds 10
        ctx = make_player_context(n_games=3, pitches_per_game=3)
        result = tensorizer.tensorize_context(ctx)
        # Should drop game 0, keep games 1 and 2 → 1 CLS + 8 = 9 positions
        assert result.seq_length == 9
        # CLS at position 0
        assert result.player_token_mask[0].item() is False
        # Both remaining games should have player tokens
        assert result.player_token_mask[1].item() is True
        assert result.player_token_mask[5].item() is True

    def test_no_truncation_when_within_limit(self) -> None:
        config = ModelConfig(max_seq_len=128)
        tensorizer = _make_tensorizer(config)
        ctx = make_player_context(n_games=2, pitches_per_game=3)
        result = tensorizer.tensorize_context(ctx)
        # 1 CLS + 2 games x (1 player + 3 pitches) = 9
        assert result.seq_length == 9


class TestCollate:
    """Tests for collating TensorizedSingle into TensorizedBatch."""

    def test_collate_same_lengths(self) -> None:
        tensorizer = _make_tensorizer()
        ctx1 = make_player_context(n_games=1, pitches_per_game=3)
        ctx2 = make_player_context(n_games=1, pitches_per_game=3)
        items = [
            tensorizer.tensorize_context(ctx1),
            tensorizer.tensorize_context(ctx2),
        ]
        batch = tensorizer.collate(items)
        # 1 CLS + 1 player + 3 pitches = 5
        assert batch.pitch_type_ids.shape == (2, 5)
        assert batch.numeric_features.shape == (2, 5, 23)
        assert batch.seq_lengths.shape == (2,)

    def test_collate_variable_lengths_pads_correctly(self) -> None:
        tensorizer = _make_tensorizer()
        ctx1 = make_player_context(n_games=1, pitches_per_game=2)  # 1 CLS + 1 player + 2 = 4
        ctx2 = make_player_context(n_games=1, pitches_per_game=5)  # 1 CLS + 1 player + 5 = 7
        items = [
            tensorizer.tensorize_context(ctx1),
            tensorizer.tensorize_context(ctx2),
        ]
        batch = tensorizer.collate(items)
        # Padded to max length of 7
        assert batch.pitch_type_ids.shape == (2, 7)
        assert batch.seq_lengths[0].item() == 4
        assert batch.seq_lengths[1].item() == 7

    def test_collate_padding_mask_false_for_pad_positions(self) -> None:
        tensorizer = _make_tensorizer()
        ctx1 = make_player_context(n_games=1, pitches_per_game=2)  # 4 total
        ctx2 = make_player_context(n_games=1, pitches_per_game=5)  # 7 total
        items = [
            tensorizer.tensorize_context(ctx1),
            tensorizer.tensorize_context(ctx2),
        ]
        batch = tensorizer.collate(items)
        # First item: positions 0-3 real, 4-6 padding
        assert batch.padding_mask[0, :4].all()
        assert not batch.padding_mask[0, 4:].any()
        # Second item: all positions real
        assert batch.padding_mask[1].all()

    def test_collate_pad_categoricals_are_zero(self) -> None:
        tensorizer = _make_tensorizer()
        ctx1 = make_player_context(n_games=1, pitches_per_game=1)  # 1 CLS + 1 player + 1 = 3
        ctx2 = make_player_context(n_games=1, pitches_per_game=3)  # 1 CLS + 1 player + 3 = 5
        items = [
            tensorizer.tensorize_context(ctx1),
            tensorizer.tensorize_context(ctx2),
        ]
        batch = tensorizer.collate(items)
        # Padding positions for first item (positions 3, 4)
        assert (batch.pitch_type_ids[0, 3:] == 0).all()
        assert (batch.pitch_result_ids[0, 3:] == 0).all()

    def test_collate_pad_player_token_mask_false(self) -> None:
        tensorizer = _make_tensorizer()
        ctx1 = make_player_context(n_games=1, pitches_per_game=1)  # 3 total
        ctx2 = make_player_context(n_games=1, pitches_per_game=3)  # 5 total
        items = [
            tensorizer.tensorize_context(ctx1),
            tensorizer.tensorize_context(ctx2),
        ]
        batch = tensorizer.collate(items)
        assert not batch.player_token_mask[0, 3:].any()
