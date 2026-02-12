"""Tests for HierarchicalFineTuneTrainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from fantasy_baseball_manager.contextual.model.heads import PerformancePredictionHead
from fantasy_baseball_manager.contextual.model.hierarchical import HierarchicalModel
from fantasy_baseball_manager.contextual.model.hierarchical_config import (
    HierarchicalModelConfig,
)
from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
from fantasy_baseball_manager.contextual.persistence import ContextualModelStore
from fantasy_baseball_manager.contextual.training.config import (
    BATTER_TARGET_STATS,
    HierarchicalFineTuneConfig,
)
from fantasy_baseball_manager.contextual.training.hierarchical_dataset import (
    HierarchicalFineTuneDataset,
    build_hierarchical_windows,
)
from fantasy_baseball_manager.contextual.training.hierarchical_finetune import (
    HierarchicalFineTuneTrainer,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fantasy_baseball_manager.contextual.model.config import ModelConfig

from tests.contextual.model.conftest import make_player_context


def _build_tensorizer(config: ModelConfig) -> Tensorizer:
    return Tensorizer(
        config=config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )


def _make_profile(player_id: int, rates_override: dict[str, float] | None = None) -> PlayerStatProfile:
    default_rates = {"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.25, "2b": 0.05, "3b": 0.01}
    rates = rates_override or default_rates
    return PlayerStatProfile(
        player_id=str(player_id),
        name=f"Player {player_id}",
        year=2024,
        player_type="batter",
        age=28,
        handedness=None,
        rates_career=rates,
        rates_3yr=None,
        rates_1yr=None,
        rates_30d=None,
        opportunities_career=1000.0,
        opportunities_3yr=None,
        opportunities_1yr=None,
    )


def _small_hier_config(d_model: int = 32) -> HierarchicalModelConfig:
    return HierarchicalModelConfig(
        identity_stat_dim=16,
        identity_archetype_dim=8,
        identity_repr_dim=24,
        n_archetypes=5,
        batter_stat_input_dim=19,
        pitcher_stat_input_dim=13,
        level3_d_model=d_model,
        level3_n_heads=2,
        level3_n_layers=1,
        level3_ff_dim=64,
        level3_dropout=0.0,
        level3_max_games=10,
    )


def _make_datasets(
    config: ModelConfig,
    hc: HierarchicalModelConfig,
    ft_config: HierarchicalFineTuneConfig,
    n_train_players: int = 4,
    n_val_players: int = 2,
    n_games: int = 6,
    pitches_per_game: int = 5,
) -> tuple[HierarchicalFineTuneDataset, HierarchicalFineTuneDataset]:
    tensorizer = _build_tensorizer(config)

    train_ids = [660271 + i for i in range(n_train_players)]
    val_ids = [770271 + i for i in range(n_val_players)]

    train_contexts = [
        make_player_context(n_games=n_games, pitches_per_game=pitches_per_game, player_id=pid)
        for pid in train_ids
    ]
    val_contexts = [
        make_player_context(n_games=n_games, pitches_per_game=pitches_per_game, player_id=pid)
        for pid in val_ids
    ]

    # Create profiles and archetype model
    all_profiles = [_make_profile(pid) for pid in train_ids + val_ids]
    X = np.array([p.to_feature_vector() for p in all_profiles])
    archetype_model = ArchetypeModel(n_archetypes=min(3, len(all_profiles)))
    archetype_model.fit(X)

    profile_lookup = {int(p.player_id): p for p in all_profiles}

    train_windows = build_hierarchical_windows(
        train_contexts, tensorizer, ft_config, BATTER_TARGET_STATS,
        profile_lookup, archetype_model, stat_input_dim=hc.batter_stat_input_dim,
    )
    val_windows = build_hierarchical_windows(
        val_contexts, tensorizer, ft_config, BATTER_TARGET_STATS,
        profile_lookup, archetype_model, stat_input_dim=hc.batter_stat_input_dim,
    )

    return (
        HierarchicalFineTuneDataset.from_windows(train_windows),
        HierarchicalFineTuneDataset.from_windows(val_windows),
    )


def _make_model(config: ModelConfig, hc: HierarchicalModelConfig) -> HierarchicalModel:
    backbone = ContextualPerformanceModel(
        config, PerformancePredictionHead(config, config.n_batter_targets),
    )
    return HierarchicalModel(
        backbone=backbone,
        hier_config=hc,
        n_targets=config.n_batter_targets,
        stat_input_dim=hc.batter_stat_input_dim,
    )


class TestHierarchicalFineTuneTrainer:
    def test_training_completes(self, small_config: ModelConfig, tmp_path: Path) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        ft_config = HierarchicalFineTuneConfig(
            epochs=2, batch_size=4, context_window=2, min_games=3,
            target_mode="counts",
        )
        model = _make_model(small_config, hc)
        store = ContextualModelStore(model_dir=tmp_path)
        train_ds, val_ds = _make_datasets(small_config, hc, ft_config)

        trainer = HierarchicalFineTuneTrainer(
            model=model, config=ft_config, model_store=store,
            target_stats=BATTER_TARGET_STATS,
        )
        result = trainer.train(train_ds, val_ds)
        assert "val_loss" in result
        assert result["val_loss"] >= 0

    def test_frozen_backbone_unchanged(self, small_config: ModelConfig, tmp_path: Path) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        ft_config = HierarchicalFineTuneConfig(
            epochs=1, batch_size=4, context_window=2, min_games=3,
            target_mode="counts",
        )
        model = _make_model(small_config, hc)
        store = ContextualModelStore(model_dir=tmp_path)
        train_ds, val_ds = _make_datasets(small_config, hc, ft_config)

        backbone_before = {
            name: p.clone() for name, p in model.backbone.named_parameters()
        }

        trainer = HierarchicalFineTuneTrainer(
            model=model, config=ft_config, model_store=store,
            target_stats=BATTER_TARGET_STATS,
        )
        trainer.train(train_ds, val_ds)

        for name, p in model.backbone.named_parameters():
            assert torch.equal(p, backbone_before[name]), f"Backbone param {name} changed"

    def test_learnable_params_change(self, small_config: ModelConfig, tmp_path: Path) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        ft_config = HierarchicalFineTuneConfig(
            epochs=2, batch_size=4, context_window=2, min_games=3,
            target_mode="counts",
            identity_learning_rate=0.01, level3_learning_rate=0.01, head_learning_rate=0.01,
            min_warmup_steps=1,
        )
        model = _make_model(small_config, hc)
        store = ContextualModelStore(model_dir=tmp_path)
        train_ds, val_ds = _make_datasets(small_config, hc, ft_config)

        # Snapshot learnable params
        head_before = {
            name: p.clone() for name, p in model.head.named_parameters()
        }

        trainer = HierarchicalFineTuneTrainer(
            model=model, config=ft_config, model_store=store,
            target_stats=BATTER_TARGET_STATS,
        )
        trainer.train(train_ds, val_ds)

        any_changed = False
        for name, p in model.head.named_parameters():
            if not torch.equal(p, head_before[name]):
                any_changed = True
                break
        assert any_changed, "No head parameters changed after training"

    def test_different_identity_different_predictions(self, small_config: ModelConfig) -> None:
        """Key diagnostic: different identity inputs produce different predictions."""
        hc = _small_hier_config(d_model=small_config.d_model)
        backbone = ContextualPerformanceModel(
            small_config, PerformancePredictionHead(small_config, small_config.n_batter_targets),
        )
        model = HierarchicalModel(
            backbone=backbone, hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )

        tensorizer = _build_tensorizer(small_config)
        ctx = make_player_context(n_games=3, pitches_per_game=5)
        single = tensorizer.tensorize_context(ctx)
        batch = tensorizer.collate([single])

        # Two very different identity inputs
        stat_a = torch.zeros(1, hc.batter_stat_input_dim)
        stat_b = torch.ones(1, hc.batter_stat_input_dim) * 5.0
        arch_a = torch.tensor([0])
        arch_b = torch.tensor([2])

        model.eval()
        with torch.no_grad():
            out_a = model(batch, stat_a, arch_a)["performance_preds"]
            out_b = model(batch, stat_b, arch_b)["performance_preds"]

        assert not torch.allclose(out_a, out_b, atol=1e-4)

    def test_checkpoint_save(self, small_config: ModelConfig, tmp_path: Path) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        ft_config = HierarchicalFineTuneConfig(
            epochs=1, batch_size=4, context_window=2, min_games=3,
            target_mode="counts",
        )
        model = _make_model(small_config, hc)
        store = ContextualModelStore(model_dir=tmp_path)
        train_ds, val_ds = _make_datasets(small_config, hc, ft_config)

        trainer = HierarchicalFineTuneTrainer(
            model=model, config=ft_config, model_store=store,
            target_stats=BATTER_TARGET_STATS,
        )
        trainer.train(train_ds, val_ds)

        # Check that checkpoints were saved
        assert store.exists("hierarchical_pitcher_best") or store.exists("hierarchical_pitcher_latest")


# ---------------------------------------------------------------------------
# Precomputed path tests
# ---------------------------------------------------------------------------


def _make_precomputed_datasets(
    config: ModelConfig,
    hc: HierarchicalModelConfig,
    ft_config: HierarchicalFineTuneConfig,
    n_train_players: int = 4,
    n_val_players: int = 2,
    n_games: int = 6,
    pitches_per_game: int = 5,
) -> tuple:
    from fantasy_baseball_manager.contextual.training.game_embedding_precomputer import (
        GameEmbeddingPrecomputer,
    )
    from fantasy_baseball_manager.contextual.training.precomputed_dataset import (
        PrecomputedDataset,
        build_precomputed_columnar,
    )

    tensorizer = _build_tensorizer(config)

    train_ids = [660271 + i for i in range(n_train_players)]
    val_ids = [770271 + i for i in range(n_val_players)]

    train_contexts = [
        make_player_context(n_games=n_games, pitches_per_game=pitches_per_game, player_id=pid)
        for pid in train_ids
    ]
    val_contexts = [
        make_player_context(n_games=n_games, pitches_per_game=pitches_per_game, player_id=pid)
        for pid in val_ids
    ]

    all_profiles = [_make_profile(pid) for pid in train_ids + val_ids]
    X = np.array([p.to_feature_vector() for p in all_profiles])
    archetype_model = ArchetypeModel(n_archetypes=min(3, len(all_profiles)))
    archetype_model.fit(X)
    profile_lookup = {int(p.player_id): p for p in all_profiles}

    backbone = ContextualPerformanceModel(
        config, PerformancePredictionHead(config, config.n_batter_targets),
    )

    precomputer = GameEmbeddingPrecomputer(
        model=backbone, tensorizer=tensorizer,
        micro_batch_size=4, device=torch.device("cpu"),
    )

    all_contexts = train_contexts + val_contexts
    game_index = precomputer.precompute(all_contexts)

    train_columnar = build_precomputed_columnar(
        train_contexts, ft_config, BATTER_TARGET_STATS,
        profile_lookup, archetype_model, game_index,
        stat_input_dim=hc.batter_stat_input_dim,
    )
    val_columnar = build_precomputed_columnar(
        val_contexts, ft_config, BATTER_TARGET_STATS,
        profile_lookup, archetype_model, game_index,
        stat_input_dim=hc.batter_stat_input_dim,
    )

    return PrecomputedDataset(train_columnar), PrecomputedDataset(val_columnar)


class TestHierarchicalFineTuneTrainerPrecomputed:
    def test_training_completes_precomputed(self, small_config: ModelConfig, tmp_path: Path) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        ft_config = HierarchicalFineTuneConfig(
            epochs=2, batch_size=4, context_window=2, min_games=3,
            target_mode="counts",
        )
        model = _make_model(small_config, hc)
        store = ContextualModelStore(model_dir=tmp_path)
        train_ds, val_ds = _make_precomputed_datasets(small_config, hc, ft_config)

        trainer = HierarchicalFineTuneTrainer(
            model=model, config=ft_config, model_store=store,
            target_stats=BATTER_TARGET_STATS,
        )
        result = trainer.train(train_ds, val_ds)
        assert "val_loss" in result
        assert result["val_loss"] >= 0

    def test_learnable_params_change_precomputed(self, small_config: ModelConfig, tmp_path: Path) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        ft_config = HierarchicalFineTuneConfig(
            epochs=2, batch_size=4, context_window=2, min_games=3,
            target_mode="counts",
            identity_learning_rate=0.01, level3_learning_rate=0.01, head_learning_rate=0.01,
            min_warmup_steps=1,
        )
        model = _make_model(small_config, hc)
        store = ContextualModelStore(model_dir=tmp_path)
        train_ds, val_ds = _make_precomputed_datasets(small_config, hc, ft_config)

        head_before = {
            name: p.clone() for name, p in model.head.named_parameters()
        }

        trainer = HierarchicalFineTuneTrainer(
            model=model, config=ft_config, model_store=store,
            target_stats=BATTER_TARGET_STATS,
        )
        trainer.train(train_ds, val_ds)

        any_changed = False
        for name, p in model.head.named_parameters():
            if not torch.equal(p, head_before[name]):
                any_changed = True
                break
        assert any_changed, "No head parameters changed after precomputed training"

    def test_checkpoint_save_load_precomputed(self, small_config: ModelConfig, tmp_path: Path) -> None:
        hc = _small_hier_config(d_model=small_config.d_model)
        ft_config = HierarchicalFineTuneConfig(
            epochs=1, batch_size=4, context_window=2, min_games=3,
            target_mode="counts",
        )
        model = _make_model(small_config, hc)
        store = ContextualModelStore(model_dir=tmp_path)
        train_ds, val_ds = _make_precomputed_datasets(small_config, hc, ft_config)

        trainer = HierarchicalFineTuneTrainer(
            model=model, config=ft_config, model_store=store,
            target_stats=BATTER_TARGET_STATS,
        )
        trainer.train(train_ds, val_ds)

        # Load and verify round-trip
        assert store.exists("hierarchical_pitcher_best") or store.exists("hierarchical_pitcher_latest")
        loaded = store.load_hierarchical_model(
            "hierarchical_pitcher_latest",
            backbone_config=small_config,
            hier_config=hc,
            n_targets=small_config.n_batter_targets,
            stat_input_dim=hc.batter_stat_input_dim,
        )
        assert loaded is not None

    def test_precomputed_pipeline_end_to_end(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """End-to-end: precompute → build dataset → train → verify identity differentiation."""
        hc = _small_hier_config(d_model=small_config.d_model)
        ft_config = HierarchicalFineTuneConfig(
            epochs=2, batch_size=4, context_window=2, min_games=3,
            target_mode="counts",
            identity_learning_rate=0.01, level3_learning_rate=0.01, head_learning_rate=0.01,
            min_warmup_steps=1,
        )
        model = _make_model(small_config, hc)
        store = ContextualModelStore(model_dir=tmp_path)
        train_ds, val_ds = _make_precomputed_datasets(small_config, hc, ft_config)

        trainer = HierarchicalFineTuneTrainer(
            model=model, config=ft_config, model_store=store,
            target_stats=BATTER_TARGET_STATS,
        )
        trainer.train(train_ds, val_ds)

        # Verify predictions differ by identity via forward_precomputed
        model.eval()
        game_embs = torch.randn(1, 2, hc.level3_d_model)
        game_mask = torch.ones(1, 2, dtype=torch.bool)
        stat_a = torch.zeros(1, hc.batter_stat_input_dim)
        stat_b = torch.ones(1, hc.batter_stat_input_dim) * 5.0
        arch_a = torch.tensor([0])
        arch_b = torch.tensor([2])

        with torch.no_grad():
            out_a = model.forward_precomputed(game_embs, game_mask, stat_a, arch_a)["performance_preds"]
            out_b = model.forward_precomputed(game_embs, game_mask, stat_b, arch_b)["performance_preds"]

        assert not torch.allclose(out_a, out_b, atol=1e-4)
