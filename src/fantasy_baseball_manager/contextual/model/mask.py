"""Attention mask construction for the player-aware transformer."""

from __future__ import annotations

import torch
from torch import Tensor


def build_player_attention_mask(
    padding_mask: Tensor,
    player_token_mask: Tensor,
    game_ids: Tensor,
) -> Tensor:
    """Build a 3D attention mask with player token isolation.

    Attention rules:
      1. No token attends to padding positions.
      2. Player tokens attend only to pitch events in their own game
         (same game_id, not padding, not other player tokens) + themselves.
      3. Pitch tokens attend to same-game non-padding positions only.
      4. CLS token (game_id=-1) attends to all non-padding positions.

    Args:
        padding_mask: (batch, seq_len) bool, True=real token
        player_token_mask: (batch, seq_len) bool, True=player token
        game_ids: (batch, seq_len) long, game index per position

    Returns:
        (batch, seq_len, seq_len) bool mask where True=MASKED (PyTorch convention)
    """
    batch, seq_len = padding_mask.shape

    # Start: everything masked
    mask = torch.ones(batch, seq_len, seq_len, dtype=torch.bool, device=padding_mask.device)

    # Expand padding_mask to (batch, 1, seq_len) for column-wise broadcasting
    can_attend_to = padding_mask.unsqueeze(1).expand(batch, seq_len, seq_len)

    # Same game: game_ids[b, i] == game_ids[b, j]
    same_game = game_ids.unsqueeze(2) == game_ids.unsqueeze(1)  # (batch, seq_len, seq_len)

    # Rule 3a: Pitch tokens attend to same-game non-padding positions only
    pitch_token_mask = ~player_token_mask & padding_mask  # (batch, seq_len)
    pitch_rows = pitch_token_mask.unsqueeze(2).expand(batch, seq_len, seq_len)
    mask[pitch_rows & same_game & can_attend_to] = False

    # Rule 3b: CLS token (game_id=-1) attends to all non-padding positions
    cls_rows = (game_ids == -1).unsqueeze(2).expand(batch, seq_len, seq_len)
    mask[cls_rows & can_attend_to] = False

    # Rule 2: Player tokens attend to same-game non-player non-padding positions + self

    # Non-player, non-padding columns
    non_player_real = ~player_token_mask & padding_mask  # (batch, seq_len)
    non_player_real_cols = non_player_real.unsqueeze(1).expand(batch, seq_len, seq_len)

    # Player rows
    player_rows = player_token_mask.unsqueeze(2).expand(batch, seq_len, seq_len)

    # Player attends to same-game non-player real tokens
    player_can_attend = player_rows & same_game & non_player_real_cols
    mask[player_can_attend] = False

    return mask
