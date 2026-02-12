"""Precompute per-game embeddings from a frozen backbone.

Runs the frozen backbone once per unique game, caching the resulting
game-level embedding vector.  Deduplicates by (player_id, game_pk)
so overlapping sliding windows share embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from fantasy_baseball_manager.contextual.data.models import PlayerContext
from fantasy_baseball_manager.contextual.model.hierarchical import GamePooler
from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedBatch

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.model import (
        ContextualPerformanceModel,
    )
    from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer


@dataclass
class GameEmbeddingIndex:
    """Holds precomputed per-game embeddings with a lookup index.

    Attributes:
        embeddings: (total_games, d_model) tensor of game embeddings.
        index: Mapping from (player_id, game_pk) to row in embeddings.
    """

    embeddings: Tensor
    index: dict[tuple[int, int], int] = field(default_factory=dict)


class GameEmbeddingPrecomputer:
    """Extracts per-game embeddings from a frozen backbone.

    Each game is processed independently through the backbone as a
    single-game PlayerContext, then GamePooler extracts the game embedding.
    """

    def __init__(
        self,
        model: ContextualPerformanceModel,
        tensorizer: Tensorizer,
        micro_batch_size: int = 8,
        device: torch.device | None = None,
    ) -> None:
        self._model = model
        self._tensorizer = tensorizer
        self._micro_batch_size = micro_batch_size
        self._device = device or torch.device("cpu")
        self._pooler = GamePooler()
        self._model.to(self._device)
        self._model.eval()

    def precompute(
        self,
        player_contexts: list[PlayerContext],
    ) -> GameEmbeddingIndex:
        """Precompute game embeddings for all unique games across contexts.

        Args:
            player_contexts: List of PlayerContext objects to process.

        Returns:
            GameEmbeddingIndex with deduplicated game embeddings.
        """
        # Collect unique (player_id, game_pk) → single-game PlayerContext
        seen: dict[tuple[int, int], PlayerContext] = {}
        for ctx in player_contexts:
            for game in ctx.games:
                key = (ctx.player_id, game.game_pk)
                if key not in seen:
                    single_ctx = PlayerContext(
                        player_id=ctx.player_id,
                        player_name=ctx.player_name,
                        season=ctx.season,
                        perspective=ctx.perspective,
                        games=(game,),
                    )
                    seen[key] = single_ctx

        keys = list(seen.keys())
        single_contexts = [seen[k] for k in keys]

        # Tensorize all single-game contexts
        tensorized = [self._tensorizer.tensorize_context(ctx) for ctx in single_contexts]

        # Process in micro-batches
        embeddings_list: list[Tensor] = []
        mb = self._micro_batch_size

        with torch.no_grad():
            for start in range(0, len(tensorized), mb):
                end = min(start + mb, len(tensorized))
                batch = self._tensorizer.collate(tensorized[start:end])

                # Move batch to device
                batch = self._batch_to_device(batch)

                # Run backbone
                out = self._model(batch)
                hidden = out["transformer_output"]  # (mb, seq_len, d_model)

                # Pool per game
                game_embs, _game_mask = self._pooler(
                    hidden, batch.game_ids, batch.padding_mask, batch.player_token_mask,
                )
                # Each single-game context produces exactly 1 game
                # game_embs: (mb, 1, d_model) → squeeze to (mb, d_model)
                embeddings_list.append(game_embs[:, 0, :].cpu())

        all_embeddings = torch.cat(embeddings_list, dim=0)  # (total_games, d_model)

        index = {key: i for i, key in enumerate(keys)}

        return GameEmbeddingIndex(embeddings=all_embeddings, index=index)

    def _batch_to_device(self, batch: TensorizedBatch) -> TensorizedBatch:
        """Move a TensorizedBatch to the target device."""
        if self._device.type == "cpu":
            return batch
        return TensorizedBatch(
            pitch_type_ids=batch.pitch_type_ids.to(self._device),
            pitch_result_ids=batch.pitch_result_ids.to(self._device),
            bb_type_ids=batch.bb_type_ids.to(self._device),
            stand_ids=batch.stand_ids.to(self._device),
            p_throws_ids=batch.p_throws_ids.to(self._device),
            pa_event_ids=batch.pa_event_ids.to(self._device),
            numeric_features=batch.numeric_features.to(self._device),
            numeric_mask=batch.numeric_mask.to(self._device),
            padding_mask=batch.padding_mask.to(self._device),
            player_token_mask=batch.player_token_mask.to(self._device),
            game_ids=batch.game_ids.to(self._device),
            seq_lengths=batch.seq_lengths.to(self._device),
        )
