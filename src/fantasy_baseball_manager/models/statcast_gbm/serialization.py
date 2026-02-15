from pathlib import Path
from typing import Any

import joblib


def save_models(models: dict[str, Any], path: Path) -> None:
    joblib.dump(models, path)


def load_models(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = joblib.load(path)
    return result
