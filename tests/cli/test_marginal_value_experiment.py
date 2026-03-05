"""Tests for marginal-value --experiment auto-logging integration."""

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
from fantasy_baseball_manager.services.quick_eval import MarginalValueResult, TargetDelta

runner = CliRunner()


class _FakeModel:
    """Minimal stand-in satisfying Experimentable protocol."""

    def experiment_player_types(self) -> list[str]:
        return ["batter", "pitcher"]

    def experiment_feature_columns(self, player_type: str) -> list[str]:
        return ["col_a", "col_b"] if player_type == "batter" else ["col_x"]

    def experiment_targets(self, player_type: str) -> list[str]:
        return ["slg", "obp"] if player_type == "batter" else ["era"]

    def experiment_training_data(self, player_type: str, seasons: list[int]) -> dict[int, list[dict[str, Any]]]:
        return {
            2023: [
                {
                    "player_id": 1,
                    "season": 2023,
                    "col_a": 1.0,
                    "col_b": 2.0,
                    "barrel_rate": 0.1,
                    "sprint_speed": 27.0,
                    "slg": 0.400,
                },
            ],
            2024: [
                {
                    "player_id": 2,
                    "season": 2024,
                    "col_a": 1.1,
                    "col_b": 2.1,
                    "barrel_rate": 0.2,
                    "sprint_speed": 28.0,
                    "slg": 0.420,
                },
            ],
        }


class _FakeAssembler:
    def __init__(self, conn: Any, *, statcast_path: Any) -> None:
        pass

    def get_or_materialize(self, fs: str) -> str:
        return "handle"

    def read(self, handle: str) -> list[dict[str, Any]]:
        return [
            {
                "player_id": 1,
                "season": 2023,
                "col_a": 1.0,
                "col_b": 2.0,
                "barrel_rate": 0.1,
                "sprint_speed": 27.0,
                "slg": 0.400,
            },
            {
                "player_id": 2,
                "season": 2024,
                "col_a": 1.1,
                "col_b": 2.1,
                "barrel_rate": 0.2,
                "sprint_speed": 28.0,
                "slg": 0.420,
            },
        ]


def _patch_heavy_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Monkeypatch the heavy computation dependencies of marginal_value_cmd."""
    db_path = tmp_path / "fbm.db"
    seed_conn = create_connection(db_path)
    seed_conn.close()

    mod = "fantasy_baseball_manager.cli.commands.marginal_value"

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
    return db_path


def _canned_marginal_value(**kw: Any) -> MarginalValueResult:
    return MarginalValueResult(
        candidate=kw["candidate_column"],
        deltas=(
            TargetDelta(target="slg", baseline_rmse=0.085, candidate_rmse=0.082, delta=-0.003, delta_pct=-3.53),
            TargetDelta(target="obp", baseline_rmse=0.050, candidate_rmse=0.049, delta=-0.001, delta_pct=-2.00),
        ),
        n_improved=2,
        n_total=2,
        avg_delta_pct=-2.77,
    )


class TestMarginalValueExperimentLogging:
    def test_logs_experiment_per_candidate(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.marginal_value.marginal_value",
            _canned_marginal_value,
        )

        result = runner.invoke(
            app,
            [
                "marginal-value",
                "test-model",
                "--candidate",
                "barrel_rate",
                "--candidate",
                "sprint_speed",
                "--player-type",
                "batter",
                "--experiment",
                "test marginal value logging",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert result.output.count("Logged experiment #") == 2

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        experiments = repo.list()
        assert len(experiments) == 2

        # Both should have the same hypothesis
        for exp in experiments:
            assert exp.hypothesis == "test marginal value logging"
            assert exp.model == "test-model"
            assert exp.player_type == "batter"
            assert len(exp.target_results) == 2
            assert "slg" in exp.target_results
            assert "obp" in exp.target_results

        # Feature diffs should capture the candidate
        candidates = {exp.feature_diff["added"][0] for exp in experiments}
        assert candidates == {"barrel_rate", "sprint_speed"}
        verify_conn.close()

    def test_logs_with_tags_and_parent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.marginal_value.marginal_value",
            _canned_marginal_value,
        )

        result = runner.invoke(
            app,
            [
                "marginal-value",
                "test-model",
                "--candidate",
                "barrel_rate",
                "--player-type",
                "batter",
                "--experiment",
                "test tags",
                "--tags",
                "feature,exploration",
                "--parent-id",
                str(parent_id),
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        child = next(e for e in repo.list() if e.hypothesis == "test tags")
        assert child.tags == ["feature", "exploration"]
        assert child.parent_id == parent_id
        verify_conn.close()

    def test_target_results_map_correctly(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.marginal_value.marginal_value",
            _canned_marginal_value,
        )

        result = runner.invoke(
            app,
            [
                "marginal-value",
                "test-model",
                "--candidate",
                "barrel_rate",
                "--player-type",
                "batter",
                "--experiment",
                "test target mapping",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        exp = repo.list()[0]
        slg = exp.target_results["slg"]
        assert slg.rmse == 0.082
        assert slg.baseline_rmse == 0.085
        assert slg.delta == -0.003
        assert slg.delta_pct == -3.53
        verify_conn.close()

    def test_auto_generates_conclusion(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.marginal_value.marginal_value",
            _canned_marginal_value,
        )

        result = runner.invoke(
            app,
            [
                "marginal-value",
                "test-model",
                "--candidate",
                "barrel_rate",
                "--player-type",
                "batter",
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
        assert "2/2" in exp.conclusion
        assert "barrel_rate" in exp.conclusion
        verify_conn.close()

    def test_no_experiment_without_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.marginal_value.marginal_value",
            _canned_marginal_value,
        )

        result = runner.invoke(
            app,
            [
                "marginal-value",
                "test-model",
                "--candidate",
                "barrel_rate",
                "--player-type",
                "batter",
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

    def test_seasons_stored_correctly(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _patch_heavy_deps(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.marginal_value.marginal_value",
            _canned_marginal_value,
        )

        result = runner.invoke(
            app,
            [
                "marginal-value",
                "test-model",
                "--candidate",
                "barrel_rate",
                "--player-type",
                "batter",
                "--experiment",
                "test seasons",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        exp = repo.list()[0]
        assert exp.seasons == {"train": [2023], "holdout": [2024]}
        verify_conn.close()
