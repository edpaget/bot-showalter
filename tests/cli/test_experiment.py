import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.experiment import Experiment, TargetResult
from fantasy_baseball_manager.repos.experiment_repo import SqliteExperimentRepo

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def _db_monkeypatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Set up a file-backed DB and monkeypatch create_connection to use it."""
    db_path = tmp_path / "fbm.db"
    seed_conn = create_connection(db_path)
    seed_conn.close()
    monkeypatch.setattr(
        "fantasy_baseball_manager.cli.factory.create_connection",
        lambda path: create_connection(db_path),
    )
    return db_path


class TestExperimentLogHelp:
    def test_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["experiment", "log", "--help"])
        assert result.exit_code == 0


class TestExperimentLogCommand:
    def test_full_log_persists(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _db_monkeypatch(monkeypatch, tmp_path)

        target_results = json.dumps(
            {
                "slg": {"rmse": 0.082, "baseline_rmse": 0.085, "delta": -0.003, "delta_pct": -3.53},
            }
        )
        result = runner.invoke(
            app,
            [
                "experiment",
                "log",
                "--hypothesis",
                "adding barrel rate improves SLG",
                "--model",
                "statcast-gbm-preseason",
                "--player-type",
                "batter",
                "--conclusion",
                "barrel rate improved SLG prediction",
                "--feature-diff",
                "+barrel_rate,-old_col",
                "--seasons",
                "2021,2022,2023/2024",
                "--params",
                '{"n_estimators": 500}',
                "--target-results",
                target_results,
                "--tags",
                "feature,batter",
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
        assert exp.hypothesis == "adding barrel rate improves SLG"
        assert exp.model == "statcast-gbm-preseason"
        assert exp.player_type == "batter"
        assert exp.feature_diff == {"added": ["barrel_rate"], "removed": ["old_col"]}
        assert exp.seasons == {"train": [2021, 2022, 2023], "holdout": [2024]}
        assert exp.params == {"n_estimators": 500}
        assert exp.target_results["slg"].rmse == 0.082
        assert exp.conclusion == "barrel rate improved SLG prediction"
        assert exp.tags == ["feature", "batter"]
        verify_conn.close()

    def test_tags_parsed_from_comma_separated(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _db_monkeypatch(monkeypatch, tmp_path)

        target_results = json.dumps({"slg": {"rmse": 0.1, "baseline_rmse": 0.2, "delta": -0.1, "delta_pct": -50.0}})
        result = runner.invoke(
            app,
            [
                "experiment",
                "log",
                "--hypothesis",
                "test",
                "--model",
                "m",
                "--player-type",
                "batter",
                "--conclusion",
                "ok",
                "--feature-diff",
                "+a",
                "--seasons",
                "2023/2024",
                "--params",
                "{}",
                "--target-results",
                target_results,
                "--tags",
                "alpha,beta,gamma",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        exp = repo.list()[0]
        assert exp.tags == ["alpha", "beta", "gamma"]
        verify_conn.close()

    def test_feature_diff_parsed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _db_monkeypatch(monkeypatch, tmp_path)

        target_results = json.dumps({"slg": {"rmse": 0.1, "baseline_rmse": 0.2, "delta": -0.1, "delta_pct": -50.0}})
        result = runner.invoke(
            app,
            [
                "experiment",
                "log",
                "--hypothesis",
                "test",
                "--model",
                "m",
                "--player-type",
                "batter",
                "--conclusion",
                "ok",
                "--feature-diff",
                "+col1,+col2,-col3",
                "--seasons",
                "2023/2024",
                "--params",
                "{}",
                "--target-results",
                target_results,
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(verify_conn)
        exp = repo.list()[0]
        assert exp.feature_diff == {"added": ["col1", "col2"], "removed": ["col3"]}
        verify_conn.close()

    def test_parent_id_stored(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = _db_monkeypatch(monkeypatch, tmp_path)

        # Create a parent experiment first
        seed_conn = create_connection(db_path)
        repo = SqliteExperimentRepo(seed_conn)
        parent_id = repo.save(
            Experiment(
                timestamp="2026-03-01T00:00:00",
                hypothesis="parent",
                model="m",
                player_type="batter",
                feature_diff={"added": [], "removed": []},
                seasons={"train": [2023], "holdout": [2024]},
                params={},
                target_results={"x": TargetResult(rmse=0.1, baseline_rmse=0.2, delta=-0.1, delta_pct=-50.0)},
                conclusion="ok",
            )
        )
        seed_conn.commit()
        seed_conn.close()

        target_results = json.dumps({"slg": {"rmse": 0.1, "baseline_rmse": 0.2, "delta": -0.1, "delta_pct": -50.0}})
        result = runner.invoke(
            app,
            [
                "experiment",
                "log",
                "--hypothesis",
                "child",
                "--model",
                "m",
                "--player-type",
                "batter",
                "--conclusion",
                "ok",
                "--feature-diff",
                "+a",
                "--seasons",
                "2023/2024",
                "--params",
                "{}",
                "--target-results",
                target_results,
                "--parent-id",
                str(parent_id),
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        verify_repo = SqliteExperimentRepo(verify_conn)
        experiments = verify_repo.list()
        child = next(e for e in experiments if e.hypothesis == "child")
        assert child.parent_id == parent_id
        verify_conn.close()

    def test_missing_required_fields(self) -> None:
        result = runner.invoke(app, ["experiment", "log"])
        assert result.exit_code != 0

    def test_malformed_target_results_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _db_monkeypatch(monkeypatch, tmp_path)

        result = runner.invoke(
            app,
            [
                "experiment",
                "log",
                "--hypothesis",
                "test",
                "--model",
                "m",
                "--player-type",
                "batter",
                "--conclusion",
                "ok",
                "--feature-diff",
                "+a",
                "--seasons",
                "2023/2024",
                "--params",
                "{}",
                "--target-results",
                "{not valid json",
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code != 0

    def test_malformed_params_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _db_monkeypatch(monkeypatch, tmp_path)

        target_results = json.dumps({"slg": {"rmse": 0.1, "baseline_rmse": 0.2, "delta": -0.1, "delta_pct": -50.0}})
        result = runner.invoke(
            app,
            [
                "experiment",
                "log",
                "--hypothesis",
                "test",
                "--model",
                "m",
                "--player-type",
                "batter",
                "--conclusion",
                "ok",
                "--feature-diff",
                "+a",
                "--seasons",
                "2023/2024",
                "--params",
                "not json",
                "--target-results",
                target_results,
                "--data-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code != 0
