"""Tests for build_player_attention_mask."""

from __future__ import annotations

import torch

from fantasy_baseball_manager.contextual.model.mask import build_player_attention_mask


class TestBuildPlayerAttentionMask:
    """Tests for the player attention mask builder."""

    def test_one_game_player_attends_to_pitches(self) -> None:
        """Player token attends to 3 pitches in its game + self."""
        # [P0] p0 p1 p2
        padding_mask = torch.tensor([[True, True, True, True]])
        player_token_mask = torch.tensor([[True, False, False, False]])
        game_ids = torch.tensor([[0, 0, 0, 0]])

        mask = build_player_attention_mask(padding_mask, player_token_mask, game_ids)
        # mask[b, i, j] = True means position i CANNOT attend to position j

        # Player (pos 0) can attend to 3 pitches in same game + self
        assert mask[0, 0, 0].item() is False  # self-attention included
        assert mask[0, 0, 1].item() is False  # pitch 0
        assert mask[0, 0, 2].item() is False  # pitch 1
        assert mask[0, 0, 3].item() is False  # pitch 2

    def test_one_game_pitches_attend_to_all(self) -> None:
        """Pitch tokens attend to all non-padding positions."""
        padding_mask = torch.tensor([[True, True, True, True]])
        player_token_mask = torch.tensor([[True, False, False, False]])
        game_ids = torch.tensor([[0, 0, 0, 0]])

        mask = build_player_attention_mask(padding_mask, player_token_mask, game_ids)

        # Pitch at pos 1 can attend to all 4 positions
        assert mask[0, 1, 0].item() is False
        assert mask[0, 1, 1].item() is False
        assert mask[0, 1, 2].item() is False
        assert mask[0, 1, 3].item() is False

    def test_two_games_player_isolation(self) -> None:
        """Player tokens only attend to their own game's events + self."""
        # [P0] p0 p1 [P1] p2 p3
        padding_mask = torch.tensor([[True, True, True, True, True, True]])
        player_token_mask = torch.tensor([[True, False, False, True, False, False]])
        game_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

        mask = build_player_attention_mask(padding_mask, player_token_mask, game_ids)

        # P0 (pos 0): attends to game 0 pitches (pos 1, 2) + self, NOT to game 1
        assert mask[0, 0, 0].item() is False  # self-attention included
        assert mask[0, 0, 1].item() is False  # game 0 pitch
        assert mask[0, 0, 2].item() is False  # game 0 pitch
        assert mask[0, 0, 3].item() is True  # P1 (other player token)
        assert mask[0, 0, 4].item() is True  # game 1 pitch
        assert mask[0, 0, 5].item() is True  # game 1 pitch

        # P1 (pos 3): attends to game 1 pitches (pos 4, 5) + self, NOT to game 0
        assert mask[0, 3, 0].item() is True  # P0
        assert mask[0, 3, 1].item() is True  # game 0 pitch
        assert mask[0, 3, 2].item() is True  # game 0 pitch
        assert mask[0, 3, 3].item() is False  # self-attention included
        assert mask[0, 3, 4].item() is False  # game 1 pitch
        assert mask[0, 3, 5].item() is False  # game 1 pitch

    def test_two_games_pitches_game_local(self) -> None:
        """Pitch tokens attend only within their own game, not across games."""
        # [P0] p0 [P1] p1
        padding_mask = torch.tensor([[True, True, True, True]])
        player_token_mask = torch.tensor([[True, False, True, False]])
        game_ids = torch.tensor([[0, 0, 1, 1]])

        mask = build_player_attention_mask(padding_mask, player_token_mask, game_ids)

        # Pitch at pos 1 (game 0) attends to same-game only
        assert mask[0, 1, 0].item() is False  # P0 (same game)
        assert mask[0, 1, 1].item() is False  # self (same game)
        assert mask[0, 1, 2].item() is True   # P1 (different game — blocked)
        assert mask[0, 1, 3].item() is True   # pitch in game 1 (blocked)

    def test_pitch_tokens_blocked_cross_game(self) -> None:
        """Pitch tokens cannot attend to positions in other games."""
        # [P0] p0 p1 [P1] p2 p3
        padding_mask = torch.tensor([[True, True, True, True, True, True]])
        player_token_mask = torch.tensor([[True, False, False, True, False, False]])
        game_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

        mask = build_player_attention_mask(padding_mask, player_token_mask, game_ids)

        # Pitch at pos 1 (game 0) blocked from game 1
        assert mask[0, 1, 3].item() is True  # P1 (game 1)
        assert mask[0, 1, 4].item() is True  # pitch game 1
        assert mask[0, 1, 5].item() is True  # pitch game 1

        # Pitch at pos 4 (game 1) blocked from game 0
        assert mask[0, 4, 0].item() is True  # P0 (game 0)
        assert mask[0, 4, 1].item() is True  # pitch game 0
        assert mask[0, 4, 2].item() is True  # pitch game 0

    def test_pitch_attends_to_own_player_token(self) -> None:
        """Pitch tokens can attend to the player token in their own game."""
        # [P0] p0 p1 [P1] p2 p3
        padding_mask = torch.tensor([[True, True, True, True, True, True]])
        player_token_mask = torch.tensor([[True, False, False, True, False, False]])
        game_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

        mask = build_player_attention_mask(padding_mask, player_token_mask, game_ids)

        # Pitch at pos 1 (game 0) can attend to P0 (pos 0, same game)
        assert mask[0, 1, 0].item() is False
        # Pitch at pos 4 (game 1) can attend to P1 (pos 3, same game)
        assert mask[0, 4, 3].item() is False

    def test_padding_always_masked(self) -> None:
        """No token attends to padding positions."""
        # [P0] p0 <PAD> <PAD>
        padding_mask = torch.tensor([[True, True, False, False]])
        player_token_mask = torch.tensor([[True, False, False, False]])
        game_ids = torch.tensor([[0, 0, 0, 0]])

        mask = build_player_attention_mask(padding_mask, player_token_mask, game_ids)

        # All positions cannot attend to padding (pos 2, 3)
        assert mask[0, 0, 2].item() is True
        assert mask[0, 0, 3].item() is True
        assert mask[0, 1, 2].item() is True
        assert mask[0, 1, 3].item() is True

    def test_batched_different_lengths(self) -> None:
        """Batch with different lengths handled correctly."""
        # Batch 0: [P0] p0 p1 <PAD>
        # Batch 1: [P0] p0 p1 p2
        padding_mask = torch.tensor(
            [
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        player_token_mask = torch.tensor(
            [
                [True, False, False, False],
                [True, False, False, False],
            ]
        )
        game_ids = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        mask = build_player_attention_mask(padding_mask, player_token_mask, game_ids)
        assert mask.shape == (2, 4, 4)

        # Batch 0: pos 3 is padding, should be masked
        assert mask[0, 0, 3].item() is True
        assert mask[0, 1, 3].item() is True
        # Batch 1: pos 3 is real
        assert mask[1, 0, 3].item() is False
        assert mask[1, 1, 3].item() is False

    def test_cls_attends_to_all_non_padding(self) -> None:
        """CLS token (game_id=-1, not a player token) attends to all non-padding."""
        # [CLS] [P0] p0 p1 [P1] p2 p3 <PAD>
        padding_mask = torch.tensor([[True, True, True, True, True, True, True, False]])
        player_token_mask = torch.tensor([[False, True, False, False, True, False, False, False]])
        game_ids = torch.tensor([[-1, 0, 0, 0, 1, 1, 1, 0]])

        mask = build_player_attention_mask(padding_mask, player_token_mask, game_ids)

        # CLS (pos 0) can attend to all non-padding positions
        assert mask[0, 0, 0].item() is False  # self (CLS is not a player token)
        assert mask[0, 0, 1].item() is False  # P0
        assert mask[0, 0, 2].item() is False  # pitch game 0
        assert mask[0, 0, 3].item() is False  # pitch game 0
        assert mask[0, 0, 4].item() is False  # P1
        assert mask[0, 0, 5].item() is False  # pitch game 1
        assert mask[0, 0, 6].item() is False  # pitch game 1
        assert mask[0, 0, 7].item() is True   # padding — blocked

        # Player tokens still isolated to their own game
        assert mask[0, 1, 2].item() is False  # P0 → game 0 pitch
        assert mask[0, 1, 5].item() is True   # P0 → game 1 pitch (blocked)

    def test_output_shape(self) -> None:
        batch, seq_len = 3, 10
        padding_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        player_token_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
        player_token_mask[:, 0] = True
        game_ids = torch.zeros(batch, seq_len, dtype=torch.long)

        mask = build_player_attention_mask(padding_mask, player_token_mask, game_ids)
        assert mask.shape == (batch, seq_len, seq_len)
        assert mask.dtype == torch.bool
