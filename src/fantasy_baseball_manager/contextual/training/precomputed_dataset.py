"""Dataset classes for precomputed game embedding training.

When the frozen backbone has been run once and per-game embeddings cached,
training only needs the lightweight Level 3 + Identity + Head modules.
This module provides the dataset, sample/batch types, collation, and
columnar builder for that precomputed path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.data.models import PlayerContext
    from fantasy_baseball_manager.contextual.identity.archetypes import ArchetypeModel
    from fantasy_baseball_manager.contextual.identity.stat_profile import (
        PlayerStatProfile,
    )
    from fantasy_baseball_manager.contextual.training.config import (
        HierarchicalFineTuneConfig,
    )
    from fantasy_baseball_manager.contextual.training.game_embedding_precomputer import (
        GameEmbeddingIndex,
    )


@dataclass
class PrecomputedSample:
    """A single precomputed fine-tuning example.

    Instead of raw pitch tokens, this holds pre-extracted game embeddings.
    """

    game_embeddings: Tensor  # (n_games, d_model)
    game_mask: Tensor  # (n_games,) bool — True=valid
    targets: Tensor  # (n_targets,)
    context_mean: Tensor  # (n_targets,)
    identity_features: Tensor  # (stat_input_dim,)
    archetype_id: int


@dataclass
class PrecomputedBatch:
    """Collated batch of PrecomputedSample instances."""

    game_embeddings: Tensor  # (batch, max_n_games, d_model)
    game_mask: Tensor  # (batch, max_n_games) bool
    targets: Tensor  # (batch, n_targets)
    context_mean: Tensor  # (batch, n_targets)
    identity_features: Tensor  # (batch, stat_input_dim)
    archetype_ids: Tensor  # (batch,) long


def collate_precomputed_samples(
    samples: list[PrecomputedSample],
) -> PrecomputedBatch:
    """Pad game_embeddings to max_n_games and stack into a batch."""
    batch_size = len(samples)
    max_n_games = max(s.game_embeddings.shape[0] for s in samples)
    d_model = samples[0].game_embeddings.shape[1]

    game_embeddings = torch.zeros(batch_size, max_n_games, d_model)
    game_mask = torch.zeros(batch_size, max_n_games, dtype=torch.bool)

    for i, s in enumerate(samples):
        n = s.game_embeddings.shape[0]
        game_embeddings[i, :n] = s.game_embeddings
        game_mask[i, :n] = s.game_mask

    targets = torch.stack([s.targets for s in samples])
    context_mean = torch.stack([s.context_mean for s in samples])
    identity_features = torch.stack([s.identity_features for s in samples])
    archetype_ids = torch.tensor([s.archetype_id for s in samples], dtype=torch.long)

    return PrecomputedBatch(
        game_embeddings=game_embeddings,
        game_mask=game_mask,
        targets=targets,
        context_mean=context_mean,
        identity_features=identity_features,
        archetype_ids=archetype_ids,
    )


# ---------------------------------------------------------------------------
# Columnar builder
# ---------------------------------------------------------------------------


def build_precomputed_columnar(
    player_contexts: list[PlayerContext],
    config: HierarchicalFineTuneConfig,
    target_stats: tuple[str, ...],
    profile_lookup: dict[int, PlayerStatProfile],
    archetype_model: ArchetypeModel,
    game_embedding_index: GameEmbeddingIndex,
    stat_input_dim: int,
) -> dict[str, Tensor | str]:
    """Build precomputed columnar data for training.

    Iterates over eligible players and sliding windows, looks up game
    embeddings from the precomputed index, and produces a flat columnar
    dict with format marker ``"precomputed_v1"``.

    Returns:
        Columnar dict with:
        - game_embeddings_flat: (total_game_slots, d_model)
        - game_offsets: (n_windows,) int64
        - n_games_per_window: (n_windows,) int64
        - targets, context_mean, identity_features, archetype_ids
    """
    from fantasy_baseball_manager.contextual.training.dataset import (
        compute_rate_targets,
        extract_game_stats,
    )

    n = config.context_window
    target_mode = config.target_mode
    target_window = config.target_window
    min_required = n + target_window if target_mode == "rates" else n + 1

    eligible = [p for p in player_contexts if len(p.games) >= min_required]

    # Accumulator lists
    game_emb_parts: list[Tensor] = []
    n_games_list: list[int] = []
    targets_list: list[Tensor] = []
    context_mean_list: list[Tensor] = []
    identity_features_list: list[Tensor] = []
    archetype_ids_list: list[int] = []

    for player_ctx in eligible:
        # Look up identity (once per player)
        profile = profile_lookup.get(player_ctx.player_id)
        if profile is not None:
            feat_vec = profile.to_feature_vector()
            identity_features = torch.tensor(feat_vec, dtype=torch.float32)
            archetype_id = int(archetype_model.predict_single(feat_vec))
        else:
            identity_features = torch.zeros(stat_input_dim, dtype=torch.float32)
            archetype_id = 0

        games = player_ctx.games

        if target_mode == "rates":
            k = target_window
            for i in range(len(games) - n - k + 1):
                context_games = games[i : i + n]
                target_games = games[i + n : i + n + k]

                result = compute_rate_targets(
                    context_games, target_games, target_stats, player_ctx.perspective,
                )
                if result is None:
                    continue

                target_rate, context_rate = result

                # Look up game embeddings
                embs = []
                for g in context_games:
                    key = (player_ctx.player_id, g.game_pk)
                    row = game_embedding_index.index[key]
                    embs.append(game_embedding_index.embeddings[row])

                game_emb_parts.append(torch.stack(embs))
                n_games_list.append(len(context_games))
                targets_list.append(target_rate)
                context_mean_list.append(context_rate)
                identity_features_list.append(identity_features)
                archetype_ids_list.append(archetype_id)
        else:
            for i in range(len(games) - n):
                context_games = games[i : i + n]
                target_game = games[i + n]

                targets = extract_game_stats(target_game, target_stats)
                context_mean = torch.stack(
                    [extract_game_stats(g, target_stats) for g in context_games]
                ).mean(dim=0)

                # Look up game embeddings
                embs = []
                for g in context_games:
                    key = (player_ctx.player_id, g.game_pk)
                    row = game_embedding_index.index[key]
                    embs.append(game_embedding_index.embeddings[row])

                game_emb_parts.append(torch.stack(embs))
                n_games_list.append(len(context_games))
                targets_list.append(targets)
                context_mean_list.append(context_mean)
                identity_features_list.append(identity_features)
                archetype_ids_list.append(archetype_id)

    if not n_games_list:
        raise ValueError("No windows produced — check eligibility criteria")

    n_windows = len(n_games_list)
    n_games_per_window = torch.tensor(n_games_list, dtype=torch.long)

    # Build flat game embeddings and offsets
    game_embeddings_flat = torch.cat(game_emb_parts, dim=0)  # (total_game_slots, d_model)
    game_offsets = torch.zeros(n_windows, dtype=torch.long)
    if n_windows > 1:
        game_offsets[1:] = n_games_per_window[:-1].cumsum(0)

    return {
        "__format__": "precomputed_v1",
        "game_embeddings_flat": game_embeddings_flat,
        "game_offsets": game_offsets,
        "n_games_per_window": n_games_per_window,
        "targets": torch.stack(targets_list),
        "context_mean": torch.stack(context_mean_list),
        "identity_features": torch.stack(identity_features_list),
        "archetype_ids": torch.tensor(archetype_ids_list, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PrecomputedDataset(Dataset[PrecomputedSample]):
    """Dataset wrapping precomputed columnar game embeddings.

    Each item returns a PrecomputedSample with pre-extracted game embeddings
    instead of raw pitch tokens.
    """

    def __init__(self, data: dict[str, Tensor | str]) -> None:
        self._data = data

    def __len__(self) -> int:
        return int(self._data["n_games_per_window"].shape[0])  # type: ignore[union-attr]

    def __getitem__(self, index: int) -> PrecomputedSample:
        d = self._data
        off = int(d["game_offsets"][index].item())  # type: ignore[union-attr]
        n_games = int(d["n_games_per_window"][index].item())  # type: ignore[union-attr]

        emb_flat = d["game_embeddings_flat"]
        assert isinstance(emb_flat, Tensor)
        game_embeddings = emb_flat[off : off + n_games]
        game_mask = torch.ones(n_games, dtype=torch.bool)

        targets_t = d["targets"]
        assert isinstance(targets_t, Tensor)
        cm_t = d["context_mean"]
        assert isinstance(cm_t, Tensor)
        id_t = d["identity_features"]
        assert isinstance(id_t, Tensor)

        return PrecomputedSample(
            game_embeddings=game_embeddings,
            game_mask=game_mask,
            targets=targets_t[index],
            context_mean=cm_t[index],
            identity_features=id_t[index],
            archetype_id=int(d["archetype_ids"][index].item()),  # type: ignore[union-attr]
        )

    def compute_target_std(self) -> Tensor:
        """Per-stat standard deviation of targets, clamped to >= 1e-6."""
        return self._data["targets"].std(dim=0).clamp(min=1e-6)  # type: ignore[union-attr]
