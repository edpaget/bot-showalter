"""Root conftest — configure xdist worker count and test environment."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Generator

import pytest

from fantasy_baseball_manager.db.connection import create_connection

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
