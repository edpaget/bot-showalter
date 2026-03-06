"""Training metadata: save/load/validate to prevent prediction data leakage."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used at runtime in function bodies

logger = logging.getLogger(__name__)

_METADATA_FILENAME = "training_metadata.json"


@dataclass(frozen=True)
class TrainingMetadata:
    train_seasons: list[int]
    holdout_seasons: list[int]


def save_training_metadata(
    artifact_path: Path,
    train_seasons: list[int],
    holdout_seasons: list[int],
) -> None:
    data = {
        "train_seasons": sorted(train_seasons),
        "holdout_seasons": sorted(holdout_seasons),
    }
    (artifact_path / _METADATA_FILENAME).write_text(json.dumps(data))


def load_training_metadata(artifact_path: Path) -> TrainingMetadata | None:
    path = artifact_path / _METADATA_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return TrainingMetadata(
        train_seasons=data["train_seasons"],
        holdout_seasons=data["holdout_seasons"],
    )


def validate_no_leakage(artifact_path: Path, prediction_seasons: list[int]) -> None:
    metadata = load_training_metadata(artifact_path)
    if metadata is None:
        logger.warning("No training metadata found at %s — skipping leakage check", artifact_path)
        return
    if not prediction_seasons:
        return
    all_seen = set(metadata.train_seasons) | set(metadata.holdout_seasons)
    overlap = set(prediction_seasons) & all_seen
    if overlap:
        msg = (
            f"Data leakage: prediction seasons {sorted(overlap)} overlap with "
            f"training metadata (train={metadata.train_seasons}, "
            f"holdout={metadata.holdout_seasons})"
        )
        raise ValueError(msg)
