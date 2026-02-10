"""Tests for ContextualPerformanceModel top-level module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fantasy_baseball_manager.contextual.data.vocab import (
    BB_TYPE_VOCAB,
    HANDEDNESS_VOCAB,
    PA_EVENT_VOCAB,
    PITCH_RESULT_VOCAB,
    PITCH_TYPE_VOCAB,
)
from fantasy_baseball_manager.contextual.model.heads import (
    MaskedGamestateHead,
    PerformancePredictionHead,
)
from fantasy_baseball_manager.contextual.model.model import (
    ContextualPerformanceModel,
)
from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig

from .conftest import make_player_context


def _make_tensorizer(config: ModelConfig) -> Tensorizer:
    return Tensorizer(
        config=config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )


class TestContextualPerformanceModelMGM:
    """Tests with MaskedGamestateHead."""

    def test_forward_output_keys(self, small_config: ModelConfig) -> None:
        head = MaskedGamestateHead(small_config)
        model = ContextualPerformanceModel(small_config, head)
        tensorizer = _make_tensorizer(small_config)

        ctx = make_player_context(n_games=1, pitches_per_game=3)
        single = tensorizer.tensorize_context(ctx)
        batch = tensorizer.collate([single])

        output = model(batch)
        assert "pitch_type_logits" in output
        assert "pitch_result_logits" in output
        assert "transformer_output" in output

    def test_forward_output_shapes(self, small_config: ModelConfig) -> None:
        head = MaskedGamestateHead(small_config)
        model = ContextualPerformanceModel(small_config, head)
        tensorizer = _make_tensorizer(small_config)

        ctx = make_player_context(n_games=1, pitches_per_game=3)
        single = tensorizer.tensorize_context(ctx)
        batch = tensorizer.collate([single])

        output = model(batch)
        # batch=1, seq_len=5 (1 CLS + 1 player + 3 pitches)
        assert output["pitch_type_logits"].shape == (1, 5, small_config.pitch_type_vocab_size)
        assert output["pitch_result_logits"].shape == (1, 5, small_config.pitch_result_vocab_size)
        assert output["transformer_output"].shape == (1, 5, small_config.d_model)

    def test_gradient_flow_end_to_end(self, small_config: ModelConfig) -> None:
        head = MaskedGamestateHead(small_config)
        model = ContextualPerformanceModel(small_config, head)
        tensorizer = _make_tensorizer(small_config)

        ctx = make_player_context(n_games=1, pitches_per_game=3)
        single = tensorizer.tensorize_context(ctx)
        batch = tensorizer.collate([single])

        output = model(batch)
        loss = output["pitch_type_logits"].pow(2).sum() + output["pitch_result_logits"].pow(2).sum()
        loss.backward()

        # Check that embedder parameters received gradients
        for param in model.embedder.parameters():
            if param.requires_grad and param.grad is not None:
                assert param.grad.abs().sum() > 0
                break


class TestContextualPerformanceModelPerf:
    """Tests with PerformancePredictionHead."""

    def test_forward_output_keys(self, small_config: ModelConfig) -> None:
        head = PerformancePredictionHead(small_config, n_targets=small_config.n_batter_targets)
        model = ContextualPerformanceModel(small_config, head)
        tensorizer = _make_tensorizer(small_config)

        ctx = make_player_context(n_games=2, pitches_per_game=3)
        single = tensorizer.tensorize_context(ctx)
        batch = tensorizer.collate([single])

        output = model(batch)
        assert "performance_preds" in output
        assert "transformer_output" in output

    def test_forward_output_shapes(self, small_config: ModelConfig) -> None:
        head = PerformancePredictionHead(small_config, n_targets=small_config.n_batter_targets)
        model = ContextualPerformanceModel(small_config, head)
        tensorizer = _make_tensorizer(small_config)

        ctx = make_player_context(n_games=2, pitches_per_game=3)
        single = tensorizer.tensorize_context(ctx)
        batch = tensorizer.collate([single])

        output = model(batch)
        # CLS embedding → (batch, n_targets), no player token dim
        assert output["performance_preds"].shape == (1, small_config.n_batter_targets)
        # 1 CLS + 2 games x (1 player + 3 pitches) = 9
        assert output["transformer_output"].shape == (1, 9, small_config.d_model)


class TestExtractClsEmbedding:
    """Tests for _extract_cls_embedding."""

    def test_returns_position_zero(self, small_config: ModelConfig) -> None:
        head = MaskedGamestateHead(small_config)
        model = ContextualPerformanceModel(small_config, head)

        batch, seq_len = 2, 6
        hidden = torch.randn(batch, seq_len, small_config.d_model)

        result = model._extract_cls_embedding(hidden)
        assert result.shape == (2, small_config.d_model)
        assert torch.allclose(result[0], hidden[0, 0])
        assert torch.allclose(result[1], hidden[1, 0])


class TestClsEmbedding:
    """Tests for learned CLS embedding (M3 fix)."""

    def test_cls_embedding_differs_from_pad(self, small_config: ModelConfig) -> None:
        """CLS and PAD positions should produce different hidden states."""
        head = MaskedGamestateHead(small_config)
        model = ContextualPerformanceModel(small_config, head)
        tensorizer = _make_tensorizer(small_config)

        ctx = make_player_context(n_games=1, pitches_per_game=3)
        single = tensorizer.tensorize_context(ctx)
        # Pad to seq_len > actual so there's at least one PAD position
        batch = tensorizer.collate([single, single])

        output = model(batch)
        hidden = output["transformer_output"]  # (2, seq_len, d_model)

        # Position 0 is CLS (game_id == -1)
        cls_hidden = hidden[0, 0, :]
        # Find a PAD position (padding_mask == False)
        pad_positions = ~batch.padding_mask[0]
        if pad_positions.any():
            pad_idx = pad_positions.nonzero(as_tuple=True)[0][0].item()
            pad_hidden = hidden[0, pad_idx, :]
            assert not torch.allclose(cls_hidden, pad_hidden, atol=1e-6), (
                "CLS and PAD hidden states should differ"
            )

    def test_cls_embedding_is_learnable(self, small_config: ModelConfig) -> None:
        """cls_embedding should have requires_grad=True and receive gradients."""
        head = PerformancePredictionHead(small_config, n_targets=small_config.n_batter_targets)
        model = ContextualPerformanceModel(small_config, head)
        tensorizer = _make_tensorizer(small_config)

        ctx = make_player_context(n_games=2, pitches_per_game=3)
        single = tensorizer.tensorize_context(ctx)
        batch = tensorizer.collate([single])

        assert model.cls_embedding.requires_grad is True

        output = model(batch)
        loss = output["performance_preds"].pow(2).sum()
        loss.backward()

        assert model.cls_embedding.grad is not None
        assert model.cls_embedding.grad.abs().sum() > 0

    def test_cls_embedding_survives_state_dict(self, small_config: ModelConfig) -> None:
        """cls_embedding should be saved and restored via state_dict."""
        head = MaskedGamestateHead(small_config)
        model1 = ContextualPerformanceModel(small_config, head)
        original_cls = model1.cls_embedding.data.clone()

        state = model1.state_dict()
        assert "cls_embedding" in state

        model2 = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        model2.load_state_dict(state)
        assert torch.equal(model2.cls_embedding.data, original_cls)


class TestSwapHead:
    """Tests for head swapping."""

    def test_swap_head_preserves_transformer_state(self, small_config: ModelConfig) -> None:
        head1 = MaskedGamestateHead(small_config)
        model = ContextualPerformanceModel(small_config, head1)

        # Capture transformer state before swap
        state_before = {k: v.clone() for k, v in model.transformer.state_dict().items()}

        head2 = PerformancePredictionHead(small_config, n_targets=7)
        model.swap_head(head2)

        # Transformer weights unchanged
        state_after = model.transformer.state_dict()
        for key in state_before:
            assert torch.equal(state_before[key], state_after[key])

        # New head is installed
        assert model.head is head2

    def test_swap_head_changes_output(self, small_config: ModelConfig) -> None:
        head1 = MaskedGamestateHead(small_config)
        model = ContextualPerformanceModel(small_config, head1)
        tensorizer = _make_tensorizer(small_config)

        ctx = make_player_context(n_games=1, pitches_per_game=3)
        single = tensorizer.tensorize_context(ctx)
        batch = tensorizer.collate([single])

        out1 = model(batch)
        assert "pitch_type_logits" in out1

        head2 = PerformancePredictionHead(small_config, n_targets=7)
        model.swap_head(head2)

        out2 = model(batch)
        assert "performance_preds" in out2
        assert "pitch_type_logits" not in out2
