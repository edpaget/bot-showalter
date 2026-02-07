"""Tests for SinusoidalPositionalEncoding."""

from __future__ import annotations

import torch

from fantasy_baseball_manager.contextual.model.positional import (
    SinusoidalPositionalEncoding,
)


class TestSinusoidalPositionalEncoding:
    """Tests for the sinusoidal positional encoding module."""

    def test_output_shape_preserved(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=32, max_seq_len=128, dropout=0.0)
        x = torch.randn(2, 10, 32)
        out = pe(x)
        assert out.shape == (2, 10, 32)

    def test_deterministic(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=32, max_seq_len=128, dropout=0.0)
        x = torch.randn(1, 5, 32)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2)

    def test_different_positions_differ(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=32, max_seq_len=128, dropout=0.0)
        x = torch.zeros(1, 10, 32)
        out = pe(x)
        # Different positions should produce different outputs
        assert not torch.allclose(out[0, 0], out[0, 1])
        assert not torch.allclose(out[0, 0], out[0, 5])

    def test_not_in_parameters(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=32, max_seq_len=128, dropout=0.0)
        param_names = [name for name, _ in pe.named_parameters()]
        assert len(param_names) == 0

    def test_encoding_is_registered_buffer(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=32, max_seq_len=128, dropout=0.0)
        buffer_names = [name for name, _ in pe.named_buffers()]
        assert "pe" in buffer_names

    def test_works_with_shorter_sequences(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=64, max_seq_len=2048, dropout=0.0)
        x = torch.randn(3, 5, 64)
        out = pe(x)
        assert out.shape == (3, 5, 64)
