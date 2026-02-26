from typing import TYPE_CHECKING, Any

import joblib

if TYPE_CHECKING:
    from pathlib import Path


def save_models(models: dict[str, Any], path: Path) -> None:
    joblib.dump(models, path)


def load_models(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = joblib.load(path)
    return result
