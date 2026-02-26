from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.db.connection import create_connection

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Generator


@pytest.fixture
def conn() -> Generator[sqlite3.Connection]:
    connection = create_connection(":memory:")
    yield connection
    connection.close()
