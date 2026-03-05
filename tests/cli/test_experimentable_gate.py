"""Tests that non-Experimentable models get clear error messages from CLI commands."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain import Ok

runner = CliRunner()


class _NonExperimentableModel:
    """A model that does NOT implement Experimentable."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "dummy"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset()

    @property
    def artifact_type(self) -> str:
        return "file"


class _FakeAssembler:
    def __init__(self, conn: Any, *, statcast_path: Any) -> None:
        pass


def _patch_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mod: str) -> None:
    db_path = tmp_path / "fbm.db"
    seed_conn = create_connection(db_path)
    seed_conn.close()

    monkeypatch.setattr(
        f"{mod}.load_config",
        lambda **kw: SimpleNamespace(data_dir=str(tmp_path), seasons=[2023, 2024]),
    )
    monkeypatch.setattr(
        f"{mod}.create_connection",
        lambda path: create_connection(db_path),
    )
    monkeypatch.setattr(f"{mod}.SqliteDatasetAssembler", _FakeAssembler)
    monkeypatch.setattr(
        f"{mod}.create_model",
        lambda model, assembler: Ok(_NonExperimentableModel()),
    )


class TestMarginalValueGate:
    def test_non_experimentable_model_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _patch_deps(monkeypatch, tmp_path, "fantasy_baseball_manager.cli.commands.marginal_value")
        result = runner.invoke(
            app,
            ["marginal-value", "dummy", "--candidate", "col_a", "--player-type", "batter", "--data-dir", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "Experimentable" in result.output


class TestQuickEvalGate:
    def test_non_experimentable_model_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _patch_deps(monkeypatch, tmp_path, "fantasy_baseball_manager.cli.commands.quick_eval")
        result = runner.invoke(
            app,
            ["quick-eval", "dummy", "--target", "slg", "--data-dir", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "Experimentable" in result.output


class TestCompareFeaturesGate:
    def test_non_experimentable_model_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _patch_deps(monkeypatch, tmp_path, "fantasy_baseball_manager.cli.commands.compare_features")
        result = runner.invoke(
            app,
            [
                "compare-features",
                "dummy",
                "--set-a",
                "default",
                "--set-b",
                "col_a",
                "--player-type",
                "batter",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "Experimentable" in result.output


class TestValidatePreflightGate:
    def test_non_experimentable_model_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _patch_deps(monkeypatch, tmp_path, "fantasy_baseball_manager.cli.commands.validate")
        result = runner.invoke(
            app,
            [
                "validate",
                "preflight",
                "dummy",
                "--candidate-columns",
                "col_a",
                "--player-type",
                "batter",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "Experimentable" in result.output
