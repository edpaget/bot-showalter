"""Tests for GameEmbeddingPrecomputer."""

from __future__ import annotations

import torch

from fantasy_baseball_manager.contextual.data.vocab import (
    BB_TYPE_VOCAB,
    HANDEDNESS_VOCAB,
    PA_EVENT_VOCAB,
    PITCH_RESULT_VOCAB,
    PITCH_TYPE_VOCAB,
)
from fantasy_baseball_manager.contextual.model.config import ModelConfig
from fantasy_baseball_manager.contextual.model.heads import PerformancePredictionHead
from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
from fantasy_baseball_manager.contextual.training.game_embedding_precomputer import (
    GameEmbeddingPrecomputer,
)
from tests.contextual.model.conftest import make_player_context


def _small_config() -> ModelConfig:
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


def _make_tensorizer(config: ModelConfig) -> Tensorizer:
    return Tensorizer(
        config=config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )


def _make_backbone(config: ModelConfig) -> ContextualPerformanceModel:
    head = PerformancePredictionHead(config, config.n_batter_targets)
    return ContextualPerformanceModel(config, head)


class TestGameEmbeddingPrecomputer:
    def test_single_game_produces_correct_shape(self) -> None:
        config = _small_config()
        backbone = _make_backbone(config)
        tensorizer = _make_tensorizer(config)

        precomputer = GameEmbeddingPrecomputer(
            model=backbone,
            tensorizer=tensorizer,
            micro_batch_size=4,
            device=torch.device("cpu"),
        )

        ctx = make_player_context(n_games=1, pitches_per_game=5, player_id=1)
        index = precomputer.precompute([ctx])

        assert index.embeddings.shape == (1, config.d_model)
        assert (1, ctx.games[0].game_pk) in index.index

    def test_multiple_players_indexed_correctly(self) -> None:
        config = _small_config()
        backbone = _make_backbone(config)
        tensorizer = _make_tensorizer(config)

        precomputer = GameEmbeddingPrecomputer(
            model=backbone,
            tensorizer=tensorizer,
            micro_batch_size=4,
            device=torch.device("cpu"),
        )

        ctx1 = make_player_context(n_games=2, pitches_per_game=5, player_id=1)
        ctx2 = make_player_context(n_games=3, pitches_per_game=5, player_id=2)
        index = precomputer.precompute([ctx1, ctx2])

        # 2 + 3 = 5 unique games
        assert index.embeddings.shape[0] == 5
        assert index.embeddings.shape[1] == config.d_model

        # Check indexing
        for game in ctx1.games:
            assert (1, game.game_pk) in index.index
        for game in ctx2.games:
            assert (2, game.game_pk) in index.index

        # Verify rows map correctly
        for _key, row_idx in index.index.items():
            assert 0 <= row_idx < 5

    def test_deduplication(self) -> None:
        """Same game appearing in overlapping contexts should be computed once."""
        config = _small_config()
        backbone = _make_backbone(config)
        tensorizer = _make_tensorizer(config)

        precomputer = GameEmbeddingPrecomputer(
            model=backbone,
            tensorizer=tensorizer,
            micro_batch_size=4,
            device=torch.device("cpu"),
        )

        # Create two contexts for the same player with overlapping games
        ctx = make_player_context(n_games=5, pitches_per_game=5, player_id=1)
        # Two contexts with different subsets of the same games (overlap on middle games)
        from fantasy_baseball_manager.contextual.data.models import PlayerContext

        ctx1 = PlayerContext(
            player_id=1,
            player_name="Test",
            season=2024,
            perspective="batter",
            games=ctx.games[:3],  # games 0,1,2
        )
        ctx2 = PlayerContext(
            player_id=1,
            player_name="Test",
            season=2024,
            perspective="batter",
            games=ctx.games[1:4],  # games 1,2,3
        )

        index = precomputer.precompute([ctx1, ctx2])

        # Should deduplicate: unique games are 0,1,2,3 = 4 games
        assert index.embeddings.shape[0] == 4

    def test_micro_batching(self) -> None:
        """Results should be identical regardless of micro_batch_size."""
        config = _small_config()
        backbone = _make_backbone(config)
        tensorizer = _make_tensorizer(config)

        ctx = make_player_context(n_games=5, pitches_per_game=5, player_id=1)

        precomputer_1 = GameEmbeddingPrecomputer(
            model=backbone,
            tensorizer=tensorizer,
            micro_batch_size=1,
            device=torch.device("cpu"),
        )
        precomputer_4 = GameEmbeddingPrecomputer(
            model=backbone,
            tensorizer=tensorizer,
            micro_batch_size=4,
            device=torch.device("cpu"),
        )

        index_1 = precomputer_1.precompute([ctx])
        index_4 = precomputer_4.precompute([ctx])

        assert index_1.embeddings.shape == index_4.embeddings.shape

        # Compare embeddings in same order
        for key in index_1.index:
            row_1 = index_1.index[key]
            row_4 = index_4.index[key]
            assert torch.allclose(
                index_1.embeddings[row_1],
                index_4.embeddings[row_4],
                atol=1e-5,
            )
