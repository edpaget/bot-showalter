import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the cache DB at a temporary directory so tests never read/write the real cache."""
    monkeypatch.setenv("FANTASY__CACHE__DB_PATH", str(tmp_path / "test_cache.db"))
    for key in list(os.environ):
        if key.startswith("FANTASY__") and key != "FANTASY__CACHE__DB_PATH":
            monkeypatch.delenv(key)
