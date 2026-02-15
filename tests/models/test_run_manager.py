from __future__ import annotations

import builtins
import subprocess
from pathlib import Path

import pytest

from fantasy_baseball_manager.domain.model_run import ModelRunRecord
from fantasy_baseball_manager.models.protocols import ModelConfig
from fantasy_baseball_manager.models.run_manager import RunContext, RunManager


class FakeModelRunRepo:
    def __init__(self) -> None:
        self._records: list[ModelRunRecord] = []
        self._next_id = 1

    def upsert(self, record: ModelRunRecord) -> int:
        row_id = self._next_id
        self._next_id += 1
        self._records.append(record)
        return row_id

    def get(self, system: str, version: str) -> ModelRunRecord | None:
        for r in self._records:
            if r.system == system and r.version == version:
                return r
        return None

    def list(self, system: str | None = None) -> builtins.list[ModelRunRecord]:
        if system is not None:
            return [r for r in self._records if r.system == system]
        return builtins.list(self._records)

    def delete(self, system: str, version: str) -> None:
        self._records = [r for r in self._records if not (r.system == system and r.version == version)]


class _FakeModel:
    @property
    def name(self) -> str:
        return "fake"

    @property
    def description(self) -> str:
        return "Fake model"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"train"})

    @property
    def artifact_type(self) -> str:
        return "none"

    def train(self, config: ModelConfig) -> None:
        pass


class _FakeModelWithDir(_FakeModel):
    @property
    def artifact_type(self) -> str:
        return "directory"


class TestRunContext:
    def test_run_dir_created(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "artifacts" / "fake" / "v1"
        run_dir.mkdir(parents=True)
        ctx = RunContext(system="fake", version="v1", run_dir=run_dir)
        assert ctx.run_dir == run_dir
        assert ctx.run_dir.exists()

    def test_log_metric(self) -> None:
        ctx = RunContext(system="fake", version="v1", run_dir=Path("/tmp"))
        ctx.log_metric("rmse", 0.5)
        assert ctx.metrics == {"rmse": 0.5}

    def test_log_metric_overwrites(self) -> None:
        ctx = RunContext(system="fake", version="v1", run_dir=Path("/tmp"))
        ctx.log_metric("rmse", 0.5)
        ctx.log_metric("rmse", 0.3)
        assert ctx.metrics == {"rmse": 0.3}

    def test_metrics_initially_empty(self) -> None:
        ctx = RunContext(system="fake", version="v1", run_dir=Path("/tmp"))
        assert ctx.metrics == {}

    def test_system_and_version_properties(self) -> None:
        ctx = RunContext(system="marcel", version="v2.0", run_dir=Path("/tmp"))
        assert ctx.system == "marcel"
        assert ctx.version == "v2.0"


class TestRunManager:
    def test_begin_run_creates_context(self, tmp_path: Path) -> None:
        repo = FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        model = _FakeModelWithDir()
        config = ModelConfig(version="v1")

        ctx = mgr.begin_run(model, config)

        assert ctx.system == "fake"
        assert ctx.version == "v1"
        expected_dir = tmp_path / "fake" / "v1"
        assert ctx.run_dir == expected_dir
        assert ctx.run_dir.exists()

    def test_begin_run_no_artifact_dir_for_none_type(self, tmp_path: Path) -> None:
        repo = FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        model = _FakeModel()  # artifact_type == "none"
        config = ModelConfig(version="v1")

        ctx = mgr.begin_run(model, config)

        assert ctx.run_dir == tmp_path / "fake" / "v1"
        assert not ctx.run_dir.exists()

    def test_finalize_run_upserts_record(self, tmp_path: Path) -> None:
        repo = FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1", tags={"env": "test"})
        ctx = RunContext(system="fake", version="v1", run_dir=tmp_path / "fake" / "v1")

        mgr.finalize_run(ctx, config)

        assert len(repo._records) == 1
        record = repo._records[0]
        assert record.system == "fake"
        assert record.version == "v1"
        assert record.tags_json == {"env": "test"}

    def test_finalize_run_captures_git_commit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

        fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123\n")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: fake_result,
        )

        repo = FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")
        ctx = RunContext(system="fake", version="v1", run_dir=tmp_path / "fake" / "v1")

        mgr.finalize_run(ctx, config)

        assert repo._records[0].git_commit == "abc123"

    def test_finalize_run_git_commit_none_on_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

        def _raise(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", _raise)

        repo = FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")
        ctx = RunContext(system="fake", version="v1", run_dir=tmp_path / "fake" / "v1")

        mgr.finalize_run(ctx, config)

        assert repo._records[0].git_commit is None

    def test_finalize_run_returns_record_id(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=1),
        )

        repo = FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")
        ctx = RunContext(system="fake", version="v1", run_dir=tmp_path / "fake" / "v1")

        row_id = mgr.finalize_run(ctx, config)

        assert row_id == 1

    def test_finalize_run_with_metrics(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=1),
        )

        repo = FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")
        ctx = RunContext(system="fake", version="v1", run_dir=tmp_path / "fake" / "v1")
        ctx.log_metric("rmse", 0.5)

        mgr.finalize_run(ctx, config)

        assert repo._records[0].metrics_json == {"rmse": 0.5}

    def test_version_required(self, tmp_path: Path) -> None:
        repo = FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        model = _FakeModel()
        config = ModelConfig()  # version is None

        with pytest.raises(ValueError, match="version"):
            mgr.begin_run(model, config)

    def test_delete_run_removes_record(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=1),
        )

        repo = FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")
        ctx = RunContext(system="fake", version="v1", run_dir=tmp_path / "fake" / "v1")
        mgr.finalize_run(ctx, config)
        assert len(repo._records) == 1

        mgr.delete_run("fake", "v1")

        assert len(repo._records) == 0

    def test_delete_run_removes_artifact_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=1),
        )

        repo = FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")

        artifact_dir = tmp_path / "fake" / "v1"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "model.pkl").write_text("data")

        ctx = RunContext(system="fake", version="v1", run_dir=artifact_dir)
        mgr.finalize_run(ctx, config)

        mgr.delete_run("fake", "v1")

        assert not artifact_dir.exists()

    def test_delete_run_no_artifact_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=1),
        )

        repo = FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")
        ctx = RunContext(system="fake", version="v1", run_dir=tmp_path / "fake" / "v1")
        mgr.finalize_run(ctx, config)

        # No error even though artifact dir doesn't exist
        mgr.delete_run("fake", "v1")

        assert len(repo._records) == 0
