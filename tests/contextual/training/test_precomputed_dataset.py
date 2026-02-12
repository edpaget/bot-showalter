"""Tests for PrecomputedDataset, PrecomputedSample/Batch, and collation."""

from __future__ import annotations

import numpy as np
import torch

from fantasy_baseball_manager.contextual.data.vocab import (
    BB_TYPE_VOCAB,
    HANDEDNESS_VOCAB,
    PA_EVENT_VOCAB,
    PITCH_RESULT_VOCAB,
    PITCH_TYPE_VOCAB,
)
from fantasy_baseball_manager.contextual.identity.archetypes import ArchetypeModel
from fantasy_baseball_manager.contextual.identity.stat_profile import PlayerStatProfile
from fantasy_baseball_manager.contextual.model.config import ModelConfig
from fantasy_baseball_manager.contextual.model.heads import PerformancePredictionHead
from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
from fantasy_baseball_manager.contextual.training.config import (
    BATTER_TARGET_STATS,
    HierarchicalFineTuneConfig,
)
from fantasy_baseball_manager.contextual.training.game_embedding_precomputer import (
    GameEmbeddingPrecomputer,
)
from fantasy_baseball_manager.contextual.training.precomputed_dataset import (
    PrecomputedDataset,
    PrecomputedSample,
    build_precomputed_columnar,
    collate_precomputed_samples,
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


def _make_profile(player_id: int) -> PlayerStatProfile:
    return PlayerStatProfile(
        player_id=str(player_id),
        name=f"Player {player_id}",
        year=2024,
        player_type="batter",
        age=28,
        handedness=None,
        rates_career={"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.25, "2b": 0.05, "3b": 0.01},
        rates_3yr=None,
        rates_1yr=None,
        rates_30d=None,
        opportunities_career=1000.0,
        opportunities_3yr=None,
        opportunities_1yr=None,
    )


# ---------------------------------------------------------------------------
# Collation tests
# ---------------------------------------------------------------------------


class TestCollatePrecomputedSamples:
    def test_collate_pads_correctly(self) -> None:
        """Two samples with different n_games should be padded to max."""
        d_model = 16
        s1 = PrecomputedSample(
            game_embeddings=torch.randn(3, d_model),
            game_mask=torch.ones(3, dtype=torch.bool),
            targets=torch.randn(6),
            context_mean=torch.randn(6),
            identity_features=torch.randn(19),
            archetype_id=0,
        )
        s2 = PrecomputedSample(
            game_embeddings=torch.randn(5, d_model),
            game_mask=torch.ones(5, dtype=torch.bool),
            targets=torch.randn(6),
            context_mean=torch.randn(6),
            identity_features=torch.randn(19),
            archetype_id=1,
        )
        batch = collate_precomputed_samples([s1, s2])

        assert batch.game_embeddings.shape == (2, 5, d_model)
        assert batch.game_mask.shape == (2, 5)

        # s1 is padded: first 3 valid, last 2 zero
        assert batch.game_mask[0, :3].all()
        assert not batch.game_mask[0, 3:].any()

        # s2 is fully valid
        assert batch.game_mask[1].all()

        # Padding positions should be zero
        assert torch.allclose(batch.game_embeddings[0, 3:], torch.zeros(2, d_model))

    def test_collate_preserves_values(self) -> None:
        """Verify unpadded values match originals."""
        d_model = 8
        embs = torch.randn(4, d_model)
        s = PrecomputedSample(
            game_embeddings=embs,
            game_mask=torch.ones(4, dtype=torch.bool),
            targets=torch.tensor([1.0, 2.0]),
            context_mean=torch.tensor([0.5, 1.0]),
            identity_features=torch.randn(13),
            archetype_id=2,
        )
        batch = collate_precomputed_samples([s])

        assert torch.allclose(batch.game_embeddings[0, :4], embs)
        assert torch.allclose(batch.targets[0], s.targets)
        assert torch.allclose(batch.context_mean[0], s.context_mean)
        assert batch.archetype_ids[0].item() == 2


# ---------------------------------------------------------------------------
# Dataset + columnar builder tests
# ---------------------------------------------------------------------------


def _build_test_dataset() -> tuple[PrecomputedDataset, HierarchicalFineTuneConfig, list]:
    """Helper to build a small precomputed dataset for testing."""
    config = _small_config()
    tensorizer = _make_tensorizer(config)
    backbone = ContextualPerformanceModel(
        config, PerformancePredictionHead(config, config.n_batter_targets),
    )

    ft_config = HierarchicalFineTuneConfig(
        epochs=1, batch_size=4, context_window=2, min_games=3,
        target_mode="counts",
    )

    player_ids = [1, 2, 3]
    contexts = [
        make_player_context(n_games=5, pitches_per_game=5, player_id=pid)
        for pid in player_ids
    ]

    all_profiles = [_make_profile(pid) for pid in player_ids]
    X = np.array([p.to_feature_vector() for p in all_profiles])
    archetype_model = ArchetypeModel(n_archetypes=min(3, len(all_profiles)))
    archetype_model.fit(X)
    profile_lookup = {int(p.player_id): p for p in all_profiles}

    # Precompute embeddings
    precomputer = GameEmbeddingPrecomputer(
        model=backbone,
        tensorizer=tensorizer,
        micro_batch_size=4,
        device=torch.device("cpu"),
    )
    game_index = precomputer.precompute(contexts)

    # Build columnar
    columnar = build_precomputed_columnar(
        contexts, ft_config, BATTER_TARGET_STATS,
        profile_lookup, archetype_model, game_index,
        stat_input_dim=19,
    )

    dataset = PrecomputedDataset(columnar)
    return dataset, ft_config, contexts


class TestBuildPrecomputedColumnar:
    def test_window_count_matches_existing(self) -> None:
        """Precomputed windows should match count from standard hierarchical path."""
        from fantasy_baseball_manager.contextual.training.hierarchical_dataset import (
            build_hierarchical_windows,
        )

        config = _small_config()
        tensorizer = _make_tensorizer(config)
        backbone = ContextualPerformanceModel(
            config, PerformancePredictionHead(config, config.n_batter_targets),
        )

        ft_config = HierarchicalFineTuneConfig(
            epochs=1, batch_size=4, context_window=2, min_games=3,
            target_mode="counts",
        )

        player_ids = [1, 2]
        contexts = [
            make_player_context(n_games=5, pitches_per_game=5, player_id=pid)
            for pid in player_ids
        ]

        all_profiles = [_make_profile(pid) for pid in player_ids]
        X = np.array([p.to_feature_vector() for p in all_profiles])
        archetype_model = ArchetypeModel(n_archetypes=2)
        archetype_model.fit(X)
        profile_lookup = {int(p.player_id): p for p in all_profiles}

        # Standard path
        std_windows = build_hierarchical_windows(
            contexts, tensorizer, ft_config, BATTER_TARGET_STATS,
            profile_lookup, archetype_model, stat_input_dim=19,
        )

        # Precomputed path
        precomputer = GameEmbeddingPrecomputer(
            model=backbone, tensorizer=tensorizer,
            micro_batch_size=4, device=torch.device("cpu"),
        )
        game_index = precomputer.precompute(contexts)
        columnar = build_precomputed_columnar(
            contexts, ft_config, BATTER_TARGET_STATS,
            profile_lookup, archetype_model, game_index,
            stat_input_dim=19,
        )

        n_games = columnar["n_games_per_window"]
        assert isinstance(n_games, torch.Tensor)
        assert n_games.shape[0] == len(std_windows)

    def test_targets_match_existing(self) -> None:
        """Targets and context_mean should match standard hierarchical dataset."""
        from fantasy_baseball_manager.contextual.training.hierarchical_dataset import (
            build_hierarchical_windows,
        )

        config = _small_config()
        tensorizer = _make_tensorizer(config)
        backbone = ContextualPerformanceModel(
            config, PerformancePredictionHead(config, config.n_batter_targets),
        )

        ft_config = HierarchicalFineTuneConfig(
            epochs=1, batch_size=4, context_window=2, min_games=3,
            target_mode="counts",
        )

        player_ids = [1, 2]
        contexts = [
            make_player_context(n_games=5, pitches_per_game=5, player_id=pid)
            for pid in player_ids
        ]

        all_profiles = [_make_profile(pid) for pid in player_ids]
        X = np.array([p.to_feature_vector() for p in all_profiles])
        archetype_model = ArchetypeModel(n_archetypes=2)
        archetype_model.fit(X)
        profile_lookup = {int(p.player_id): p for p in all_profiles}

        # Standard path
        std_windows = build_hierarchical_windows(
            contexts, tensorizer, ft_config, BATTER_TARGET_STATS,
            profile_lookup, archetype_model, stat_input_dim=19,
        )
        std_targets = torch.stack([w[1] for w in std_windows])
        std_context_mean = torch.stack([w[2] for w in std_windows])

        # Precomputed path
        precomputer = GameEmbeddingPrecomputer(
            model=backbone, tensorizer=tensorizer,
            micro_batch_size=4, device=torch.device("cpu"),
        )
        game_index = precomputer.precompute(contexts)
        columnar = build_precomputed_columnar(
            contexts, ft_config, BATTER_TARGET_STATS,
            profile_lookup, archetype_model, game_index,
            stat_input_dim=19,
        )

        targets_t = columnar["targets"]
        assert isinstance(targets_t, torch.Tensor)
        cm_t = columnar["context_mean"]
        assert isinstance(cm_t, torch.Tensor)
        assert torch.allclose(targets_t, std_targets)
        assert torch.allclose(cm_t, std_context_mean)


class TestPrecomputedDataset:
    def test_getitem_shapes(self) -> None:
        dataset, ft_config, _ = _build_test_dataset()
        sample = dataset[0]

        assert sample.game_embeddings.shape == (ft_config.context_window, 32)  # d_model=32
        assert sample.game_mask.shape == (ft_config.context_window,)
        assert sample.game_mask.all()
        assert sample.targets.shape == (len(BATTER_TARGET_STATS),)
        assert sample.context_mean.shape == (len(BATTER_TARGET_STATS),)
        assert sample.identity_features.shape == (19,)

    def test_compute_target_std(self) -> None:
        dataset, _, _ = _build_test_dataset()
        std = dataset.compute_target_std()

        assert std.shape == (len(BATTER_TARGET_STATS),)
        assert (std >= 1e-6).all()
