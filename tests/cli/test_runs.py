from datetime import UTC, datetime
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.model_run import ModelRunRecord
from fantasy_baseball_manager.repos.model_run_repo import SqliteModelRunRepo

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def _seed_model_run(conn: object, system: str = "marcel", version: str = "v1") -> None:
    repo = SqliteModelRunRepo(SingleConnectionProvider(conn))  # type: ignore[arg-type]
    repo.upsert(
        ModelRunRecord(
            system=system,
            version=version,
            config_json={"data_dir": "./data", "seasons": [2023]},
            artifact_type="none",
            artifact_path=None,
            git_commit="abc123",
            tags_json={"env": "test"},
            metrics_json={"rmse": 0.5},
            created_at=datetime.now(UTC).isoformat(),
        )
    )


class TestRunsListCommand:
    def test_runs_list_command_exists(self) -> None:
        result = runner.invoke(app, ["runs", "list", "--help"])
        assert result.exit_code == 0

    def test_runs_list_shows_records(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_model_run(db_conn, system="marcel", version="v1")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "list", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "marcel" in result.output
        assert "v1" in result.output

    def test_runs_list_filter_by_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_model_run(db_conn, system="marcel", version="v1")
        _seed_model_run(db_conn, system="steamer", version="v1")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "list", "--model", "marcel", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "marcel" in result.output
        assert "steamer" not in result.output

    def test_runs_list_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "list", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "No runs found" in result.output


class TestRunsShowCommand:
    def test_runs_show_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_model_run(db_conn, system="marcel", version="v1")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "show", "marcel/v1", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "marcel" in result.output
        assert "v1" in result.output
        assert "abc123" in result.output

    def test_runs_show_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "show", "nonexistent/v1", "--data-dir", "./data"])
        assert result.exit_code != 0


class TestRunsDeleteCommand:
    def test_runs_delete_command(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = tmp_path / "fbm.db"
        seed_conn = create_connection(db_path)
        _seed_model_run(seed_conn, system="marcel", version="v1")
        seed_conn.commit()
        seed_conn.close()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        result = runner.invoke(app, ["runs", "delete", "marcel/v1", "--yes", "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output

        # Verify with a SECOND connection — proves deletion was committed
        verify_conn = create_connection(db_path)
        repo = SqliteModelRunRepo(SingleConnectionProvider(verify_conn))
        assert repo.get("marcel", "v1") is None
        verify_conn.close()

    def test_runs_delete_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "delete", "nonexistent/v1", "--yes", "--data-dir", "./data"])
        assert result.exit_code != 0
