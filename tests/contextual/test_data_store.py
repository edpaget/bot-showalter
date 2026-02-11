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

    def test_save_and_load_hierarchical_columnar(self, tmp_path: Path) -> None:
        """Hierarchical save produces columnar dict; load round-trips correctly."""
        store = PreparedDataStore(data_dir=tmp_path)
        ts1 = _make_tensorized_single(5)
        ts2 = _make_tensorized_single(8)
        targets1 = torch.tensor([1.0, 0.5])
        targets2 = torch.tensor([0.0, 1.0])
        cm1 = torch.tensor([0.3, 0.2])
        cm2 = torch.tensor([0.1, 0.4])
        id1 = torch.randn(13)
        id2 = torch.randn(13)
        windows = [
            (ts1, targets1, cm1, id1, 2),
            (ts2, targets2, cm2, id2, 0),
        ]
        meta = {"perspective": "pitcher"}

        store.save_hierarchical_finetune_data("hier_test", windows, meta)
        loaded = store.load_hierarchical_finetune_data("hier_test")

        assert isinstance(loaded, dict)
        assert loaded["__format__"] == "columnar_v1"

        def t(key: str) -> torch.Tensor:
            v = loaded[key]
            assert isinstance(v, torch.Tensor)
            return v

        assert t("pitch_type_ids").shape == (2, 8)  # max seq_len
        assert t("targets").shape == (2, 2)
        assert t("identity_features").shape == (2, 13)
        assert t("archetype_ids").shape == (2,)
        assert t("seq_lengths").shape == (2,)
        assert t("seq_lengths")[0].item() == 5
        assert t("seq_lengths")[1].item() == 8
        # Values preserved
        assert torch.equal(t("targets")[0], targets1)
        assert torch.equal(t("targets")[1], targets2)
        assert torch.allclose(t("identity_features")[0], id1)
        assert t("archetype_ids")[0].item() == 2
        assert t("archetype_ids")[1].item() == 0

    def test_load_hierarchical_legacy_format(self, tmp_path: Path) -> None:
        """Loading a legacy list-of-tuples .pt file returns columnar dict with warning."""
        store = PreparedDataStore(data_dir=tmp_path)
        ts1 = _make_tensorized_single(5)
        targets = torch.tensor([1.0, 0.5])
        cm = torch.tensor([0.3, 0.2])
        identity = torch.randn(13)
        legacy_windows = [(ts1, targets, cm, identity, 1)]

        # Save in legacy format directly (bypass the new save)
        pt_path = tmp_path / "legacy_hier.pt"
        torch.save(legacy_windows, pt_path)

        # Load should detect legacy and convert on-the-fly
        loaded = store.load_hierarchical_finetune_data("legacy_hier")

        assert isinstance(loaded, dict)
        assert loaded["__format__"] == "columnar_v1"
        targets_t = loaded["targets"]
        assert isinstance(targets_t, torch.Tensor)
        assert targets_t.shape == (1, 2)
        seq_lengths_t = loaded["seq_lengths"]
        assert isinstance(seq_lengths_t, torch.Tensor)
        assert seq_lengths_t[0].item() == 5
