import sqlite3
from collections.abc import Generator
from typing import Any

import pandas as pd
import pytest

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection


class FakeDataSource:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @property
    def source_type(self) -> str:
        return "test"

    @property
    def source_detail(self) -> str:
        return "fake"

    def fetch(self, **params: Any) -> pd.DataFrame:
        return self._df


class ErrorDataSource:
    @property
    def source_type(self) -> str:
        return "test"

    @property
    def source_detail(self) -> str:
        return "error"

    def fetch(self, **params: Any) -> pd.DataFrame:
        raise RuntimeError("fetch failed")


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
