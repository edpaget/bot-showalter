"""Tests for quick-eval --experiment auto-logging integration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain import Experiment, Ok, TargetResult
from fantasy_baseball_manager.repos.experiment_repo import SqliteExperimentRepo
from fantasy_baseball_manager.services.quick_eval import QuickEvalResult

runner = CliRunner()


class _FakeModel:
    """Minimal stand-in for _StatcastGBMBase."""

    _batter_columns: tuple[str, ...] = ("col_a", "col_b")
    _pitcher_columns: tuple[str, ...] = ("col_x",)

    def _batter_training_set_builder(self, seasons: list[int]) -> str:
        return "fake_batter_fs"

    def _pitcher_training_set_builder(self, seasons: list[int]) -> str:
        return "fake_pitcher_fs"


class _FakeAssembler:
    def __init__(self, conn: Any, *, statcast_path: Any) -> None:
        pass

    def get_or_materialize(self, fs: str) -> str:
        return "handle"

    def read(self, handle: str) -> list[dict[str, Any]]:
        return [
            {"season": 2023, "col_a": 1.0, "col_b": 2.0, "barrel_rate": 0.1, "slg": 0.400},
            {"season": 2024, "col_a": 1.1, "col_b": 2.1, "barrel_rate": 0.2, "slg": 0.420},
        ]


def _patch_heavy_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Monkeypatch the heavy computation dependencies of quick_eval_cmd."""
    db_path = tmp_path / "fbm.db"
    seed_conn = create_connection(db_path)
    seed_conn.close()

    mod = "fantasy_baseball_manager.cli.commands.quick_eval"

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
        lambda model, assembler: Ok(_FakeModel()),
    )
    # Patch isinstance check for _StatcastGBMBase
    monkeypatch.setattr(
        f"{mod}._StatcastGBMBase",
        _FakeModel,
    )
    return db_path


class TestQuickEvalExperimentLogging:
    def test_logs_experiment_when_flag_provided(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)

        canned = QuickEvalResult(
            target="slg",
            rmse=0.082,
            r_squared=0.75,
            n=100,
            baseline_rmse=0.085,
            delta=-0.003,
            delta_pct=-3.53,
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.quick_eval.quick_eval",
            lambda **kw: canned,
        )

        result = runner.invoke(
            app,
            [
                "quick-eval",
                "test-model",
                "--target",
                "slg",
                "--baseline",
                "0.085",
                "--experiment",
                "barrel rate improves SLG",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Logged experiment #" in result.output

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        experiments = repo.list()
        assert len(experiments) == 1
        exp = experiments[0]
        assert exp.hypothesis == "barrel rate improves SLG"
        assert exp.model == "test-model"
        assert exp.player_type == "batter"
        assert exp.target_results["slg"].rmse == 0.082
        assert exp.target_results["slg"].baseline_rmse == 0.085
        assert exp.target_results["slg"].delta == -0.003
        assert exp.seasons == {"train": [2023], "holdout": [2024]}
        verify_conn.close()

    def test_logs_experiment_with_tags_and_parent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)

        # Seed a parent experiment
        seed_conn = create_connection(db_path)
        parent_repo = SqliteExperimentRepo(seed_conn)
        parent_id = parent_repo.save(
            Experiment(
                timestamp="2026-03-01T00:00:00",
                hypothesis="parent",
                model="test-model",
                player_type="batter",
                feature_diff={"added": [], "removed": []},
                seasons={"train": [2023], "holdout": [2024]},
                params={},
                target_results={"slg": TargetResult(rmse=0.1, baseline_rmse=0.2, delta=-0.1, delta_pct=-50.0)},
                conclusion="ok",
            )
        )
        seed_conn.commit()
        seed_conn.close()

        canned = QuickEvalResult(
            target="slg",
            rmse=0.082,
            r_squared=0.75,
            n=100,
            baseline_rmse=0.085,
            delta=-0.003,
            delta_pct=-3.53,
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.quick_eval.quick_eval",
            lambda **kw: canned,
        )

        result = runner.invoke(
            app,
            [
                "quick-eval",
                "test-model",
                "--target",
                "slg",
                "--baseline",
                "0.085",
                "--experiment",
                "test hypothesis",
                "--tags",
                "feature,batter",
                "--parent-id",
                str(parent_id),
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        child = next(e for e in repo.list() if e.hypothesis == "test hypothesis")
        assert child.tags == ["feature", "batter"]
        assert child.parent_id == parent_id
        verify_conn.close()

    def test_logs_injected_columns_as_feature_diff(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)

        canned = QuickEvalResult(
            target="slg",
            rmse=0.082,
            r_squared=0.75,
            n=100,
            baseline_rmse=0.085,
            delta=-0.003,
            delta_pct=-3.53,
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.quick_eval.quick_eval",
            lambda **kw: canned,
        )

        result = runner.invoke(
            app,
            [
                "quick-eval",
                "test-model",
                "--target",
                "slg",
                "--baseline",
                "0.085",
                "--inject",
                "barrel_rate",
                "--experiment",
                "test inject",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        exp = repo.list()[0]
        assert exp.feature_diff == {"added": ["barrel_rate"], "removed": []}
        verify_conn.close()

    def test_requires_baseline_for_experiment(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)

        canned = QuickEvalResult(
            target="slg",
            rmse=0.082,
            r_squared=0.75,
            n=100,
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.quick_eval.quick_eval",
            lambda **kw: canned,
        )

        result = runner.invoke(
            app,
            [
                "quick-eval",
                "test-model",
                "--target",
                "slg",
                "--experiment",
                "test without baseline",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "baseline" in result.output.lower()

        # No experiment should be logged
        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        assert len(repo.list()) == 0
        verify_conn.close()

    def test_no_experiment_without_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)

        canned = QuickEvalResult(
            target="slg",
            rmse=0.082,
            r_squared=0.75,
            n=100,
            baseline_rmse=0.085,
            delta=-0.003,
            delta_pct=-3.53,
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.quick_eval.quick_eval",
            lambda **kw: canned,
        )

        result = runner.invoke(
            app,
            [
                "quick-eval",
                "test-model",
                "--target",
                "slg",
                "--baseline",
                "0.085",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Logged experiment" not in result.output

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        assert len(repo.list()) == 0
        verify_conn.close()

    def test_auto_generates_conclusion(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)

        canned = QuickEvalResult(
            target="slg",
            rmse=0.082,
            r_squared=0.75,
            n=100,
            baseline_rmse=0.085,
            delta=-0.003,
            delta_pct=-3.53,
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.quick_eval.quick_eval",
            lambda **kw: canned,
        )

        result = runner.invoke(
            app,
            [
                "quick-eval",
                "test-model",
                "--target",
                "slg",
                "--baseline",
                "0.085",
                "--experiment",
                "test conclusion",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        exp = repo.list()[0]
        assert "improved" in exp.conclusion
        assert "slg" in exp.conclusion
        assert "3.53%" in exp.conclusion
        verify_conn.close()

    def test_params_stored_in_experiment(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)

        canned = QuickEvalResult(
            target="slg",
            rmse=0.082,
            r_squared=0.75,
            n=100,
            baseline_rmse=0.085,
            delta=-0.003,
            delta_pct=-3.53,
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.quick_eval.quick_eval",
            lambda **kw: canned,
        )

        result = runner.invoke(
            app,
            [
                "quick-eval",
                "test-model",
                "--target",
                "slg",
                "--baseline",
                "0.085",
                "--param",
                "n_estimators=500",
                "--param",
                "max_depth=6",
                "--experiment",
                "test params",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        exp = repo.list()[0]
        assert exp.params == {"n_estimators": 500, "max_depth": 6}
        verify_conn.close()
