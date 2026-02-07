"""Tests for ContextualModelStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

from fantasy_baseball_manager.contextual.model.config import ModelConfig
from fantasy_baseball_manager.contextual.model.heads import (
    MaskedGamestateHead,
    PerformancePredictionHead,
)
from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
from fantasy_baseball_manager.contextual.persistence import (
    ContextualModelMetadata,
    ContextualModelStore,
)


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


def _make_model(config: ModelConfig) -> ContextualPerformanceModel:
    head = MaskedGamestateHead(config)
    return ContextualPerformanceModel(config, head)


class TestContextualModelStore:
    def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
        config = _small_config()
        model = _make_model(config)
        store = ContextualModelStore(model_dir=tmp_path)

        metadata = ContextualModelMetadata(
            name="test_checkpoint",
            epoch=5,
            train_loss=0.5,
            val_loss=0.6,
        )
        path = store.save_checkpoint("test_checkpoint", model, metadata)
        assert path.exists()

        loaded = store.load_model("test_checkpoint", config)
        # Check weights match
        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], loaded.state_dict()[key])

    def test_save_and_load_training_state(self, tmp_path: Path) -> None:
        config = _small_config()
        model = _make_model(config)
        store = ContextualModelStore(model_dir=tmp_path)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler_state = {"step": 100}

        metadata = ContextualModelMetadata(name="resume_test", epoch=3, train_loss=0.4, val_loss=0.5)
        store.save_checkpoint(
            "resume_test", model, metadata,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler_state,
        )

        state_dict, opt_state, sched_state = store.load_training_state("resume_test")
        assert state_dict is not None
        assert opt_state is not None
        assert sched_state == {"step": 100}

    def test_get_metadata(self, tmp_path: Path) -> None:
        config = _small_config()
        model = _make_model(config)
        store = ContextualModelStore(model_dir=tmp_path)

        metadata = ContextualModelMetadata(
            name="meta_test", epoch=10, train_loss=0.3, val_loss=0.4,
            pitch_type_accuracy=0.75, pitch_result_accuracy=0.65,
        )
        store.save_checkpoint("meta_test", model, metadata)

        loaded_meta = store.get_metadata("meta_test")
        assert loaded_meta is not None
        assert loaded_meta.name == "meta_test"
        assert loaded_meta.epoch == 10
        assert loaded_meta.train_loss == 0.3
        assert loaded_meta.val_loss == 0.4
        assert loaded_meta.pitch_type_accuracy == 0.75
        assert loaded_meta.pitch_result_accuracy == 0.65

    def test_list_checkpoints(self, tmp_path: Path) -> None:
        config = _small_config()
        model = _make_model(config)
        store = ContextualModelStore(model_dir=tmp_path)

        for i in range(3):
            meta = ContextualModelMetadata(name=f"cp_{i}", epoch=i, train_loss=0.5, val_loss=0.6)
            store.save_checkpoint(f"cp_{i}", model, meta)

        checkpoints = store.list_checkpoints()
        assert len(checkpoints) == 3
        names = {c.name for c in checkpoints}
        assert names == {"cp_0", "cp_1", "cp_2"}

    def test_exists(self, tmp_path: Path) -> None:
        config = _small_config()
        model = _make_model(config)
        store = ContextualModelStore(model_dir=tmp_path)

        assert not store.exists("nonexistent")
        meta = ContextualModelMetadata(name="exists_test", epoch=1, train_loss=0.5, val_loss=0.6)
        store.save_checkpoint("exists_test", model, meta)
        assert store.exists("exists_test")

    def test_delete(self, tmp_path: Path) -> None:
        config = _small_config()
        model = _make_model(config)
        store = ContextualModelStore(model_dir=tmp_path)

        meta = ContextualModelMetadata(name="del_test", epoch=1, train_loss=0.5, val_loss=0.6)
        store.save_checkpoint("del_test", model, meta)
        assert store.exists("del_test")

        deleted = store.delete("del_test")
        assert deleted
        assert not store.exists("del_test")

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        store = ContextualModelStore(model_dir=tmp_path)
        assert not store.delete("nonexistent")

    def test_get_metadata_nonexistent_returns_none(self, tmp_path: Path) -> None:
        store = ContextualModelStore(model_dir=tmp_path)
        assert store.get_metadata("nonexistent") is None

    def test_load_model_not_found_raises(self, tmp_path: Path) -> None:
        config = _small_config()
        store = ContextualModelStore(model_dir=tmp_path)
        try:
            store.load_model("nonexistent", config)
            raise AssertionError("Should raise FileNotFoundError")
        except FileNotFoundError:
            pass

    def test_save_without_optimizer_state(self, tmp_path: Path) -> None:
        config = _small_config()
        model = _make_model(config)
        store = ContextualModelStore(model_dir=tmp_path)

        meta = ContextualModelMetadata(name="no_opt", epoch=1, train_loss=0.5, val_loss=0.6)
        store.save_checkpoint("no_opt", model, meta)

        # Optimizer file should not exist
        opt_path = tmp_path / "no_opt_optimizer.pt"
        assert not opt_path.exists()

    def test_save_and_load_finetune_metadata(self, tmp_path: Path) -> None:
        config = _small_config()
        n_targets = 4
        head = PerformancePredictionHead(config, n_targets)
        model = ContextualPerformanceModel(config, head)
        store = ContextualModelStore(model_dir=tmp_path)

        metadata = ContextualModelMetadata(
            name="finetune_best",
            epoch=10,
            train_loss=0.1,
            val_loss=0.15,
            perspective="pitcher",
            target_stats=("so", "h", "bb", "hr"),
            per_stat_mse={"so": 0.5, "h": 0.3, "bb": 0.2, "hr": 0.1},
            base_model="pretrain_best",
        )
        store.save_checkpoint("finetune_best", model, metadata)

        loaded_meta = store.get_metadata("finetune_best")
        assert loaded_meta is not None
        assert loaded_meta.perspective == "pitcher"
        assert loaded_meta.target_stats == ("so", "h", "bb", "hr")
        assert loaded_meta.per_stat_mse == {"so": 0.5, "h": 0.3, "bb": 0.2, "hr": 0.1}
        assert loaded_meta.base_model == "pretrain_best"
        # Pre-training fields should be None
        assert loaded_meta.pitch_type_accuracy is None
        assert loaded_meta.pitch_result_accuracy is None

    def test_load_finetune_model(self, tmp_path: Path) -> None:
        config = _small_config()
        n_targets = 6
        head = PerformancePredictionHead(config, n_targets)
        model = ContextualPerformanceModel(config, head)
        store = ContextualModelStore(model_dir=tmp_path)

        metadata = ContextualModelMetadata(
            name="ft_load_test", epoch=5, train_loss=0.2, val_loss=0.3,
            perspective="batter", target_stats=("hr", "so", "bb", "h", "2b", "3b"),
        )
        store.save_checkpoint("ft_load_test", model, metadata)

        loaded = store.load_finetune_model("ft_load_test", config, n_targets)
        assert isinstance(loaded.head, PerformancePredictionHead)
        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], loaded.state_dict()[key])

    def test_load_finetune_model_not_found_raises(self, tmp_path: Path) -> None:
        config = _small_config()
        store = ContextualModelStore(model_dir=tmp_path)
        try:
            store.load_finetune_model("nonexistent", config, 4)
            raise AssertionError("Should raise FileNotFoundError")
        except FileNotFoundError:
            pass

    def test_list_checkpoints_with_finetune_fields(self, tmp_path: Path) -> None:
        config = _small_config()
        store = ContextualModelStore(model_dir=tmp_path)

        # Save a pre-training checkpoint
        model = _make_model(config)
        meta_pre = ContextualModelMetadata(name="pretrain", epoch=1, train_loss=0.5, val_loss=0.6)
        store.save_checkpoint("pretrain", model, meta_pre)

        # Save a fine-tune checkpoint
        ft_head = PerformancePredictionHead(config, 4)
        ft_model = ContextualPerformanceModel(config, ft_head)
        meta_ft = ContextualModelMetadata(
            name="finetune", epoch=10, train_loss=0.1, val_loss=0.15,
            perspective="pitcher", target_stats=("so", "h", "bb", "hr"),
        )
        store.save_checkpoint("finetune", ft_model, meta_ft)

        checkpoints = store.list_checkpoints()
        assert len(checkpoints) == 2
        names = {c.name for c in checkpoints}
        assert names == {"pretrain", "finetune"}

        ft_cp = next(c for c in checkpoints if c.name == "finetune")
        assert ft_cp.perspective == "pitcher"
        assert ft_cp.target_stats == ("so", "h", "bb", "hr")
