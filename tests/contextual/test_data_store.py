"""Tests for PreparedDataStore persistence layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

from fantasy_baseball_manager.contextual.data_store import PreparedDataStore
from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedSingle


def _make_tensorized_single(seq_len: int = 5) -> TensorizedSingle:
    """Create a minimal TensorizedSingle for testing."""
    return TensorizedSingle(
        pitch_type_ids=torch.randint(0, 10, (seq_len,)),
        pitch_result_ids=torch.randint(0, 5, (seq_len,)),
        bb_type_ids=torch.zeros(seq_len, dtype=torch.long),
        stand_ids=torch.ones(seq_len, dtype=torch.long),
        p_throws_ids=torch.ones(seq_len, dtype=torch.long),
        pa_event_ids=torch.zeros(seq_len, dtype=torch.long),
        numeric_features=torch.randn(seq_len, 23),
        numeric_mask=torch.ones(seq_len, 23, dtype=torch.bool),
        padding_mask=torch.ones(seq_len, dtype=torch.bool),
        player_token_mask=torch.zeros(seq_len, dtype=torch.bool),
        game_ids=torch.zeros(seq_len, dtype=torch.long),
        seq_length=seq_len,
    )


class TestPreparedDataStore:
    def test_save_and_load_pretrain_data(self, tmp_path: Path) -> None:
        store = PreparedDataStore(data_dir=tmp_path)
        sequences = [_make_tensorized_single(5), _make_tensorized_single(10)]
        meta = {"seasons": [2022], "max_seq_len": 512}

        store.save_pretrain_data("pretrain_train", sequences, meta)
        loaded = store.load_pretrain_data("pretrain_train")

        assert len(loaded) == 2
        assert isinstance(loaded[0], TensorizedSingle)
        assert loaded[0].seq_length == 5
        assert loaded[1].seq_length == 10
        assert torch.equal(loaded[0].pitch_type_ids, sequences[0].pitch_type_ids)
        assert torch.equal(loaded[1].numeric_features, sequences[1].numeric_features)

    def test_save_and_load_finetune_data(self, tmp_path: Path) -> None:
        store = PreparedDataStore(data_dir=tmp_path)
        ts1 = _make_tensorized_single(5)
        ts2 = _make_tensorized_single(8)
        targets1 = torch.tensor([1.0, 0.0, 2.0])
        targets2 = torch.tensor([0.0, 1.0, 0.0])
        cm1 = torch.tensor([0.5, 0.1, 1.0])
        cm2 = torch.tensor([0.2, 0.3, 0.0])
        windows = [(ts1, targets1, cm1), (ts2, targets2, cm2)]
        meta = {"perspective": "pitcher", "context_window": 10}

        store.save_finetune_data("finetune_pitcher_train", windows, meta)
        loaded = store.load_finetune_data("finetune_pitcher_train")

        assert len(loaded) == 2
        loaded_ts, loaded_targets, loaded_cm = loaded[0]
        assert isinstance(loaded_ts, TensorizedSingle)
        assert loaded_ts.seq_length == 5
        assert torch.equal(loaded_targets, targets1)
        assert torch.equal(loaded_cm, cm1)
        assert torch.equal(loaded[1][1], targets2)
        assert torch.equal(loaded[1][2], cm2)

    def test_exists_false_when_missing(self, tmp_path: Path) -> None:
        store = PreparedDataStore(data_dir=tmp_path)
        assert store.exists("nonexistent") is False

    def test_exists_true_after_save(self, tmp_path: Path) -> None:
        store = PreparedDataStore(data_dir=tmp_path)
        store.save_pretrain_data("test_data", [_make_tensorized_single()], {"k": "v"})
        assert store.exists("test_data") is True

    def test_load_meta_round_trip(self, tmp_path: Path) -> None:
        store = PreparedDataStore(data_dir=tmp_path)
        meta = {
            "seasons": [2021, 2022],
            "val_seasons": [2023],
            "perspectives": ["batter", "pitcher"],
            "max_seq_len": 512,
            "min_pitch_count": 10,
        }
        store.save_pretrain_data("pretrain_train", [_make_tensorized_single()], meta)
        loaded_meta = store.load_meta("pretrain_train")

        assert loaded_meta == meta

    def test_load_meta_returns_none_when_missing(self, tmp_path: Path) -> None:
        store = PreparedDataStore(data_dir=tmp_path)
        assert store.load_meta("nonexistent") is None

    def test_load_pretrain_data_raises_when_missing(self, tmp_path: Path) -> None:
        store = PreparedDataStore(data_dir=tmp_path)
        try:
            store.load_pretrain_data("nonexistent")
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass

    def test_load_finetune_data_raises_when_missing(self, tmp_path: Path) -> None:
        store = PreparedDataStore(data_dir=tmp_path)
        try:
            store.load_finetune_data("nonexistent")
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass
