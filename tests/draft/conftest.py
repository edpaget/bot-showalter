import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the cache DB at a temporary directory so tests never read/write the real cache."""
    monkeypatch.setenv("FANTASY__CACHE__DB_PATH", str(tmp_path / "test_cache.db"))
    # Also clear any league config that might leak from a real config.yaml
    for key in list(os.environ):
        if key.startswith("FANTASY__") and key != "FANTASY__CACHE__DB_PATH":
            monkeypatch.delenv(key)
