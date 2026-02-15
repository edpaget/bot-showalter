import enum
from dataclasses import dataclass
from typing import Any


class ArtifactType(enum.Enum):
    NONE = "none"
    FILE = "file"
    DIRECTORY = "directory"


@dataclass(frozen=True)
class ModelRunRecord:
    system: str
    version: str
    config_json: dict[str, Any]
    artifact_type: str
    created_at: str
    operation: str = "train"
    train_dataset_id: int | None = None
    validation_dataset_id: int | None = None
    holdout_dataset_id: int | None = None
    metrics_json: dict[str, Any] | None = None
    artifact_path: str | None = None
    git_commit: str | None = None
    tags_json: dict[str, str] | None = None
    id: int | None = None
