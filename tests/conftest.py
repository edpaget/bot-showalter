"""Root conftest — configure xdist worker count."""

from __future__ import annotations

import os


def pytest_xdist_auto_num_workers() -> int:
    """Use half the available cores so concurrent test runs don't starve each other."""
    return max(1, (os.cpu_count() or 1) // 2)
