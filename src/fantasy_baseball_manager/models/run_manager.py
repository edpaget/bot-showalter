from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fantasy_baseball_manager.domain.model_run import ArtifactType, ModelRunRecord
from fantasy_baseball_manager.models.protocols import ModelConfig, ProjectionModel
from fantasy_baseball_manager.repos.protocols import ModelRunRepo


class RunContext:
    def __init__(self, system: str, version: str, run_dir: Path, artifact_type: str) -> None:
        self._system = system
        self._version = version
        self._run_dir = run_dir
        self._artifact_type = artifact_type
        self._metrics: dict[str, float] = {}

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def system(self) -> str:
        return self._system

    @property
    def version(self) -> str:
        return self._version

    @property
    def artifact_type(self) -> str:
        return self._artifact_type

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def log_metric(self, key: str, value: float) -> None:
        self._metrics[key] = value


class RunManager:
    def __init__(self, model_run_repo: ModelRunRepo, artifacts_root: Path) -> None:
        self._repo = model_run_repo
        self._artifacts_root = artifacts_root

    def begin_run(self, model: ProjectionModel, config: ModelConfig) -> RunContext:
        if config.version is None:
            raise ValueError("config.version is required to begin a model run")

        run_dir = self._artifacts_root / model.name / config.version

        if model.artifact_type != ArtifactType.NONE.value:
            run_dir.mkdir(parents=True, exist_ok=True)

        return RunContext(system=model.name, version=config.version, run_dir=run_dir, artifact_type=model.artifact_type)

    def finalize_run(self, context: RunContext, config: ModelConfig) -> int:
        git_commit = self._capture_git_commit()

        config_json: dict[str, Any] = {
            "data_dir": config.data_dir,
            "artifacts_dir": config.artifacts_dir,
            "seasons": config.seasons,
            "model_params": config.model_params,
        }

        metrics = context.metrics or None

        record = ModelRunRecord(
            system=context.system,
            version=context.version,
            config_json=config_json,
            artifact_type=context.artifact_type,
            artifact_path=str(context.run_dir),
            git_commit=git_commit,
            tags_json=config.tags if config.tags else None,
            metrics_json=metrics,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        return self._repo.upsert(record)

    def delete_run(self, system: str, version: str) -> None:
        record = self._repo.get(system, version)
        if record is not None and record.artifact_path is not None:
            artifact_path = Path(record.artifact_path)
            if artifact_path.is_dir():
                shutil.rmtree(artifact_path)
        self._repo.delete(system, version)

    @staticmethod
    def _capture_git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except FileNotFoundError, OSError:
            return None
