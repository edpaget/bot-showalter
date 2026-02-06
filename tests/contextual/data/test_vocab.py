"""Tests for contextual pitch event vocabularies."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.contextual.data.vocab import (
    BB_TYPE_VOCAB,
    HANDEDNESS_VOCAB,
    PA_EVENT_VOCAB,
    PITCH_RESULT_VOCAB,
    PITCH_TYPE_VOCAB,
    Vocabulary,
)


class TestVocabulary:
    """Tests for the Vocabulary class."""

    def test_encode_known_token(self) -> None:
        vocab = Vocabulary(name="test", token_to_index={"<PAD>": 0, "<UNK>": 1, "FF": 2})
        assert vocab.encode("FF") == 2

    def test_encode_unknown_token_returns_unk(self) -> None:
        vocab = Vocabulary(name="test", token_to_index={"<PAD>": 0, "<UNK>": 1, "FF": 2})
        assert vocab.encode("UNKNOWN_TYPE") == 1

    def test_size(self) -> None:
        vocab = Vocabulary(name="test", token_to_index={"<PAD>": 0, "<UNK>": 1, "FF": 2, "SL": 3})
        assert vocab.size == 4

    def test_index_to_token_round_trip(self) -> None:
        vocab = Vocabulary(name="test", token_to_index={"<PAD>": 0, "<UNK>": 1, "FF": 2, "SL": 3})
        reverse = vocab.index_to_token
        assert reverse[0] == "<PAD>"
        assert reverse[1] == "<UNK>"
        assert reverse[2] == "FF"
        assert reverse[3] == "SL"

    def test_no_duplicate_indices(self) -> None:
        vocab = Vocabulary(name="test", token_to_index={"<PAD>": 0, "<UNK>": 1, "A": 2, "B": 3})
        indices = list(vocab.token_to_index.values())
        assert len(indices) == len(set(indices))

    def test_frozen_immutability(self) -> None:
        vocab = Vocabulary(name="test", token_to_index={"<PAD>": 0, "<UNK>": 1})
        with pytest.raises(AttributeError):
            vocab.name = "changed"  # type: ignore[misc]

    def test_pad_at_index_zero(self) -> None:
        vocab = Vocabulary(name="test", token_to_index={"<PAD>": 0, "<UNK>": 1})
        assert vocab.encode("<PAD>") == 0

    def test_unk_at_index_one(self) -> None:
        vocab = Vocabulary(name="test", token_to_index={"<PAD>": 0, "<UNK>": 1})
        assert vocab.encode("<UNK>") == 1


class TestPitchTypeVocab:
    """Tests for the PITCH_TYPE_VOCAB constant."""

    def test_pad_at_zero(self) -> None:
        assert PITCH_TYPE_VOCAB.encode("<PAD>") == 0

    def test_unk_at_one(self) -> None:
        assert PITCH_TYPE_VOCAB.encode("<UNK>") == 1

    def test_known_types(self) -> None:
        for pt in ("FF", "SI", "SL", "CH", "CU", "FC", "ST", "SV", "KC", "FS", "KN"):
            assert PITCH_TYPE_VOCAB.encode(pt) >= 2

    def test_unknown_type_maps_to_unk(self) -> None:
        assert PITCH_TYPE_VOCAB.encode("ZZ") == 1

    def test_no_duplicate_indices(self) -> None:
        indices = list(PITCH_TYPE_VOCAB.token_to_index.values())
        assert len(indices) == len(set(indices))


class TestPitchResultVocab:
    """Tests for the PITCH_RESULT_VOCAB constant."""

    def test_pad_at_zero(self) -> None:
        assert PITCH_RESULT_VOCAB.encode("<PAD>") == 0

    def test_unk_at_one(self) -> None:
        assert PITCH_RESULT_VOCAB.encode("<UNK>") == 1

    def test_known_results(self) -> None:
        for r in ("ball", "called_strike", "swinging_strike", "foul", "hit_into_play"):
            assert PITCH_RESULT_VOCAB.encode(r) >= 2

    def test_no_duplicate_indices(self) -> None:
        indices = list(PITCH_RESULT_VOCAB.token_to_index.values())
        assert len(indices) == len(set(indices))


class TestPAEventVocab:
    """Tests for the PA_EVENT_VOCAB constant."""

    def test_pad_at_zero(self) -> None:
        assert PA_EVENT_VOCAB.encode("<PAD>") == 0

    def test_unk_at_one(self) -> None:
        assert PA_EVENT_VOCAB.encode("<UNK>") == 1

    def test_known_events(self) -> None:
        for e in ("single", "double", "triple", "home_run", "strikeout", "walk", "field_out"):
            assert PA_EVENT_VOCAB.encode(e) >= 2

    def test_no_duplicate_indices(self) -> None:
        indices = list(PA_EVENT_VOCAB.token_to_index.values())
        assert len(indices) == len(set(indices))


class TestBBTypeVocab:
    """Tests for the BB_TYPE_VOCAB constant."""

    def test_pad_at_zero(self) -> None:
        assert BB_TYPE_VOCAB.encode("<PAD>") == 0

    def test_unk_at_one(self) -> None:
        assert BB_TYPE_VOCAB.encode("<UNK>") == 1

    def test_known_types(self) -> None:
        for bt in ("ground_ball", "line_drive", "fly_ball", "popup"):
            assert BB_TYPE_VOCAB.encode(bt) >= 2

    def test_size(self) -> None:
        assert BB_TYPE_VOCAB.size == 6

    def test_no_duplicate_indices(self) -> None:
        indices = list(BB_TYPE_VOCAB.token_to_index.values())
        assert len(indices) == len(set(indices))


class TestHandednessVocab:
    """Tests for the HANDEDNESS_VOCAB constant."""

    def test_pad_at_zero(self) -> None:
        assert HANDEDNESS_VOCAB.encode("<PAD>") == 0

    def test_unk_at_one(self) -> None:
        assert HANDEDNESS_VOCAB.encode("<UNK>") == 1

    def test_known_handedness(self) -> None:
        assert HANDEDNESS_VOCAB.encode("R") >= 2
        assert HANDEDNESS_VOCAB.encode("L") >= 2

    def test_size(self) -> None:
        assert HANDEDNESS_VOCAB.size == 4

    def test_no_duplicate_indices(self) -> None:
        indices = list(HANDEDNESS_VOCAB.token_to_index.values())
        assert len(indices) == len(set(indices))
