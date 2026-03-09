"""Root conftest — configure xdist worker count and test environment."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.models.registry import _REGISTRY

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Generator

# Disable OpenMP multi-threading in tests.  scikit-learn's
# HistGradientBoostingRegressor spawns OpenMP threads whose management
# overhead dwarfs the actual computation on small test datasets (30 rows).
os.environ.setdefault("OMP_NUM_THREADS", "1")


def pytest_xdist_auto_num_workers() -> int:
    """Use half the available cores so concurrent test runs don't starve each other."""
    return max(1, (os.cpu_count() or 1) // 2)


@pytest.fixture
def conn() -> Generator[sqlite3.Connection]:
    connection = create_connection(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def isolated_model_registry() -> Generator[None]:
    """Provide an empty model registry that restores original state on teardown."""
    saved = _REGISTRY.copy()
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)
