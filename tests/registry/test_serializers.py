"""Tests for model serializers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.registry.serializers import (
    JoblibSerializer,
    ModelSerializer,
    TorchParamsSerializer,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestJoblibSerializer:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(JoblibSerializer(), ModelSerializer)

    def test_extension(self) -> None:
        assert JoblibSerializer().extension == ".joblib"

    def test_round_trip(self, tmp_path: Path) -> None:
        serializer = JoblibSerializer()
        params = {"weights": [1.0, 2.0, 3.0], "bias": 0.5, "nested": {"a": 1}}
        path = tmp_path / "model.joblib"

        serializer.save(params, path)
        loaded = serializer.load(path)

        assert loaded == params

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        serializer = JoblibSerializer()
        with pytest.raises(FileNotFoundError):
            serializer.load(tmp_path / "missing.joblib")


class TestTorchParamsSerializer:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(TorchParamsSerializer(), ModelSerializer)

    def test_extension(self) -> None:
        assert TorchParamsSerializer().extension == ".pt"

    def test_round_trip(self, tmp_path: Path) -> None:
        serializer = TorchParamsSerializer()
        params = {"state_dict": {"layer.weight": [1.0, 2.0]}, "config": {"n_features": 10}}
        path = tmp_path / "model.pt"

        serializer.save(params, path)
        loaded = serializer.load(path)

        assert loaded["config"] == params["config"]
        assert loaded["state_dict"] == params["state_dict"]

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        serializer = TorchParamsSerializer()
        with pytest.raises(FileNotFoundError):
            serializer.load(tmp_path / "missing.pt")
