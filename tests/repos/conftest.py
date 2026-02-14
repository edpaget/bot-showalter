import sqlite3
from collections.abc import Generator

import pytest

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection


@pytest.fixture
def conn() -> Generator[sqlite3.Connection]:
    connection = create_connection(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def statcast_conn() -> Generator[sqlite3.Connection]:
    connection = create_statcast_connection(":memory:")
    yield connection
    connection.close()
